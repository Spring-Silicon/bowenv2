use crate::EpisodeId;
use crate::admission::{AdaptiveAdmissionSchedule, AdmissionDecision, AdmissionSmoothingConfig};
use crate::pool::{Admission, AdmissionResult, CompletedSearchEpisode, CompletedTask, WorkerPool};
use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::internal;
use gz_engine::{EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine};
use gz_eval::{EvalOutput, EvalRequest, Evaluator, eval_error_to_engine_error, validate_outputs};
use gz_eval_service::{BackendOutputs, FeatureEvalBackend, ModelGeneration};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, FeatureSchema, FeatureSchemaHash,
    OpponentStateFeatures, PositionFeatures, encode_feature_row,
};
use gz_measurer::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasureLedgerSnapshot, MeasuredSymmetricGame,
    MeasurerAdmission, MeasurerRunSummary, ReplayMeasurer,
};
use gz_replay::{ReplayError, ReplayStore};
use gz_search::{
    EngineIdentity, GumbelEpisode, GumbelMcts, GumbelValueMode, SearchAction, SymmetricActorTrace,
    SymmetricEpisode, WorkToken,
};
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const EVAL_PIPELINE_DEPTH: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadedOrchestratorConfig {
    pub workers_per_lane: NonZeroUsize,
    pub max_batch: NonZeroUsize,
    pub flush_after: Duration,
    pub admission_stagger: Duration,
    pub admission_smoothing: Option<AdmissionSmoothingConfig>,
}

pub struct ThreadedGumbelOrchestrator<E, V> {
    engines: Vec<E>,
    evaluator: V,
    search: GumbelMcts,
    config: ThreadedOrchestratorConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaneEpisodes<G, C> {
    pub lane: usize,
    /// Completed batch-path episodes. Engine handles inside each episode are
    /// opaque identifiers only; the lane has already released them.
    pub episodes: Vec<OrchestratedEpisode<G, C>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedRun<G, C> {
    /// Batch-path episode equality surface. Engine handles inside returned
    /// episodes have already been released and must not be dereferenced.
    pub lanes: Vec<LaneEpisodes<G, C>>,
    pub batch_sizes: Vec<usize>,
}

pub struct ReplayRuntime<'a> {
    pub store: &'a ReplayStore,
    pub backpressure: Option<ReplayBackpressure>,
    /// Break equal-reward games by episode length (shorter wins) before
    /// the coin flip: whittlezero's duration tiebreak, discrete form.
    pub length_tiebreak: bool,
}

pub struct FeaturizedRuntime<X, B> {
    pub extractors: Vec<X>,
    /// One batcher thread per backend; lanes are assigned round-robin
    /// (lane % backends.len()). Multiple evaluator processes parallelize
    /// the per-batch host work (decode/stage/encode runs on one thread
    /// per process) and keep the GPU's kernel queue dense.
    pub backends: Vec<B>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayBackpressure {
    pub max_row_backlog: NonZeroU64,
    pub gate_poll: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayLaneSummary {
    pub lane: usize,
    pub episodes_completed: u64,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub search_contexts: u64,
    pub replay_rows: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedReplayRun {
    pub lanes: Vec<ReplayLaneSummary>,
    pub batch_sizes: Vec<usize>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub search_contexts: u64,
    pub replay_rows: u64,
    pub measure_ledger: MeasureLedgerSnapshot,
}

struct EvalJob {
    lane: usize,
    slot: usize,
    token: WorkToken,
    request: EvalRequest,
}

struct FeaturizedEvalJob {
    lane: usize,
    slot: usize,
    token: WorkToken,
    row: FeatureRow,
    action_count: u32,
    model: ModelGeneration,
}

struct EvalReply {
    slot: usize,
    token: WorkToken,
    output: EvalOutput,
}

#[derive(Default)]
struct EvalPressure {
    outstanding: AtomicUsize,
    reserved: AtomicUsize,
    capacity_work: AtomicU64,
    capacity_busy_ns: AtomicU64,
}

impl EvalPressure {
    fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }

    fn reserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    fn capacity_totals(&self) -> (u64, u64) {
        // record_capacity publishes busy time before work. Loading work first
        // means observing a new work total also observes its matching duration.
        let work = self.capacity_work.load(Ordering::Acquire);
        let busy_ns = self.capacity_busy_ns.load(Ordering::Acquire);
        (work, busy_ns)
    }

    fn reserve(&self, count: usize) {
        self.reserved.fetch_add(count, Ordering::AcqRel);
    }

    fn cancel_reservations(&self, count: usize) {
        atomic_saturating_sub(&self.reserved, count);
    }

    fn submit(&self, reserved: bool) {
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        if reserved {
            atomic_saturating_sub(&self.reserved, 1);
        }
    }

    fn cancel_submission(&self) {
        atomic_saturating_sub(&self.outstanding, 1);
    }

    fn complete(&self, count: usize) {
        atomic_saturating_sub(&self.outstanding, count);
    }

    fn complete_current_batch(&self, count: usize, capacity_work: usize, busy: Duration) {
        self.complete(count);
        let busy_ns = busy.as_nanos().min(u128::from(u64::MAX)) as u64;
        atomic_saturating_add_u64(&self.capacity_busy_ns, busy_ns.max(1));
        atomic_saturating_add_u64(
            &self.capacity_work,
            u64::try_from(capacity_work).unwrap_or(u64::MAX),
        );
    }
}

fn atomic_saturating_sub(value: &AtomicUsize, count: usize) {
    let _ = value.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_sub(count))
    });
}

fn atomic_saturating_add_u64(value: &AtomicU64, count: u64) {
    let _ = value.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_add(count))
    });
}

struct SharedAdmissionShaper {
    started: Instant,
    schedule: Mutex<AdaptiveAdmissionSchedule>,
    pressure: Arc<EvalPressure>,
    bootstrap_grants: AtomicUsize,
    paced_grants: AtomicUsize,
    max_waiting: AtomicUsize,
    waiting_lanes: Vec<std::sync::atomic::AtomicBool>,
    next_stats_ms: AtomicUsize,
}

impl SharedAdmissionShaper {
    fn new(
        lanes: usize,
        workers_per_lane: NonZeroUsize,
        evaluator_processes: usize,
        max_batch: NonZeroUsize,
        config: AdmissionSmoothingConfig,
        pressure: Arc<EvalPressure>,
    ) -> EngineResult<Self> {
        let lanes = NonZeroUsize::new(lanes).ok_or_else(|| internal("zero lanes"))?;
        let evaluator_processes = NonZeroUsize::new(evaluator_processes)
            .ok_or_else(|| internal("zero evaluator processes"))?;
        let total_workers = lanes
            .get()
            .checked_mul(workers_per_lane.get())
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| internal("worker count overflow"))?;
        let target_outstanding = max_batch
            .get()
            .checked_mul(EVAL_PIPELINE_DEPTH)
            .and_then(|target| target.checked_mul(evaluator_processes.get()))
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| internal("evaluator pressure target overflow"))?;
        let schedule = AdaptiveAdmissionSchedule::new(
            lanes,
            total_workers,
            evaluator_processes,
            target_outstanding,
            config,
        )
        .map_err(|_| internal("invalid admission smoothing config"))?;
        let lane_count = lanes.get();
        Ok(Self {
            started: Instant::now(),
            schedule: Mutex::new(schedule),
            pressure,
            bootstrap_grants: AtomicUsize::new(0),
            paced_grants: AtomicUsize::new(0),
            max_waiting: AtomicUsize::new(0),
            waiting_lanes: (0..lane_count)
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect(),
            next_stats_ms: AtomicUsize::new(30_000),
        })
    }

    fn request(&self, lane: usize, idle_workers: usize) -> EngineResult<AdmissionDecision> {
        if idle_workers == 0 && !self.waiting_lanes[lane].load(Ordering::Acquire) {
            return Ok(AdmissionDecision::default());
        }
        let mut schedule = self
            .schedule
            .lock()
            .map_err(|_| internal("admission shaper lock poisoned"))?;
        let (capacity_work, capacity_busy_ns) = self.pressure.capacity_totals();
        let outstanding = self.pressure.outstanding();
        let decision = schedule.request(
            self.started.elapsed(),
            lane,
            idle_workers,
            capacity_work,
            capacity_busy_ns,
            outstanding,
        );
        let waiting = schedule.total_waiting();
        self.waiting_lanes[lane].store(schedule.lane_waiting(lane), Ordering::Release);
        self.pressure.reserve(decision.limit);
        self.bootstrap_grants
            .fetch_add(decision.bootstrap_grants, Ordering::Relaxed);
        self.paced_grants
            .fetch_add(decision.paced_grants, Ordering::Relaxed);
        self.max_waiting.fetch_max(waiting, Ordering::Relaxed);
        self.report_stats(&schedule, waiting)?;
        Ok(decision)
    }

    fn observe_episode_work(&self, evaluations: u64) -> EngineResult<()> {
        self.schedule
            .lock()
            .map_err(|_| internal("admission shaper lock poisoned"))?
            .observe_episode_work(evaluations);
        Ok(())
    }

    fn finish_admission(
        &self,
        lane: usize,
        decision: AdmissionDecision,
        admitted: usize,
        roots_exhausted: bool,
    ) -> EngineResult<()> {
        let unused = decision.limit.saturating_sub(admitted);
        self.pressure.cancel_reservations(unused);
        let mut schedule = self
            .schedule
            .lock()
            .map_err(|_| internal("admission shaper lock poisoned"))?;
        schedule.restore_unused(lane, unused, decision.bootstrap_grants > 0);
        if roots_exhausted {
            schedule.clear_lane(lane);
        }
        self.waiting_lanes[lane].store(schedule.lane_waiting(lane), Ordering::Release);
        Ok(())
    }

    fn clear_lane(&self, lane: usize) -> EngineResult<()> {
        self.schedule
            .lock()
            .map_err(|_| internal("admission shaper lock poisoned"))?
            .clear_lane(lane);
        self.waiting_lanes[lane].store(false, Ordering::Release);
        Ok(())
    }

    fn report_stats(
        &self,
        schedule: &AdaptiveAdmissionSchedule,
        waiting: usize,
    ) -> EngineResult<()> {
        const STATS_INTERVAL_MS: usize = 30_000;
        let elapsed_ms = usize::try_from(self.started.elapsed().as_millis()).unwrap_or(usize::MAX);
        let next = self.next_stats_ms.load(Ordering::Relaxed);
        if elapsed_ms < next
            || self
                .next_stats_ms
                .compare_exchange(
                    next,
                    elapsed_ms.saturating_add(STATS_INTERVAL_MS),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_err()
        {
            return Ok(());
        }
        let eval_capacity_milli = schedule
            .eval_capacity_ema()
            .map_or(0, |capacity| (capacity * 1_000.0).round() as u64);
        let episode_work_milli = schedule
            .episode_eval_work_ema()
            .map_or(0, |work| (work * 1_000.0).round() as u64);
        let gap_us = schedule
            .admission_gap()
            .map_or(0, |gap| gap.as_micros().min(u128::from(u64::MAX)) as u64);
        let pressure_gain_milli = (schedule.pressure_gain() * 1_000.0).round() as u64;
        eprintln!(
            "event=admission_stats outstanding={} reserved={} waiting={} max_waiting={} bootstrap_grants={} paced_grants={} eval_capacity_milli={} episode_work_milli={} pressure_gain_milli={} gap_us={}",
            self.pressure.outstanding(),
            self.pressure.reserved(),
            waiting,
            self.max_waiting.load(Ordering::Relaxed),
            self.bootstrap_grants.load(Ordering::Relaxed),
            self.paced_grants.load(Ordering::Relaxed),
            eval_capacity_milli,
            episode_work_milli,
            pressure_gain_milli,
            gap_us,
        );
        Ok(())
    }
}

fn build_admission_shaper(
    lanes: usize,
    evaluator_processes: usize,
    config: ThreadedOrchestratorConfig,
    pressure: Arc<EvalPressure>,
) -> EngineResult<Option<Arc<SharedAdmissionShaper>>> {
    let Some(smoothing) = config.admission_smoothing else {
        return Ok(None);
    };
    if !config.admission_stagger.is_zero() {
        return Err(internal(
            "fixed and adaptive admission pacing are mutually exclusive",
        ));
    }
    SharedAdmissionShaper::new(
        lanes,
        config.workers_per_lane,
        evaluator_processes,
        config.max_batch,
        smoothing,
        pressure,
    )
    .map(Arc::new)
    .map(Some)
}

struct ModelLeaseRegistry {
    state: Mutex<ModelLeaseState>,
}

struct FeaturizedBatcherContext {
    model_registry: Arc<ModelLeaseRegistry>,
    eval_pressure: Option<Arc<EvalPressure>>,
}

struct ModelLeaseState {
    current: ModelGeneration,
    generations: Vec<ModelGenerationState>,
    releasable: VecDeque<ModelGeneration>,
}

struct ModelGenerationState {
    model: ModelGeneration,
    users: usize,
}

struct ModelLease {
    registry: Arc<ModelLeaseRegistry>,
    model: ModelGeneration,
}

impl ModelLeaseRegistry {
    fn new(current: ModelGeneration) -> EngineResult<Self> {
        if current.id == 0 {
            return Err(internal("zero model generation"));
        }
        Ok(Self {
            state: Mutex::new(ModelLeaseState {
                current,
                generations: vec![ModelGenerationState {
                    model: current,
                    users: 0,
                }],
                releasable: VecDeque::new(),
            }),
        })
    }

    fn acquire_current(self: &Arc<Self>) -> ModelLease {
        let mut state = self.state.lock().expect("model lease registry poisoned");
        let current = state.current;
        let generation = state
            .generations
            .iter_mut()
            .find(|generation| generation.model == current)
            .expect("current model generation is registered");
        generation.users = generation
            .users
            .checked_add(1)
            .expect("model lease count overflowed");
        ModelLease {
            registry: Arc::clone(self),
            model: current,
        }
    }

    fn publish(&self, model: ModelGeneration) -> EngineResult<()> {
        if model.id == 0 {
            return Err(internal("zero model generation"));
        }
        let mut state = self.state.lock().expect("model lease registry poisoned");
        if state.current == model {
            return Ok(());
        }
        if state.generations.iter().any(|generation| {
            generation.model.id == model.id && generation.model.version != model.version
        }) {
            return Err(internal("model generation id changed version"));
        }
        if state.generations.iter().any(|generation| {
            generation.model.version == model.version && generation.model.id != model.id
        }) {
            return Err(internal("model version has multiple resident generations"));
        }
        if state
            .generations
            .iter()
            .all(|generation| generation.model != model)
        {
            if state.generations.len() >= 2 {
                return Err(internal("too many resident model generations"));
            }
            state
                .generations
                .push(ModelGenerationState { model, users: 0 });
        }
        let previous = state.current;
        state.current = model;
        if state
            .generations
            .iter()
            .any(|generation| generation.model == previous && generation.users == 0)
        {
            queue_model_release(&mut state, previous);
        }
        Ok(())
    }

    fn take_releasable(&self) -> Vec<ModelGeneration> {
        let mut state = self.state.lock().expect("model lease registry poisoned");
        let models = state.releasable.drain(..).collect::<Vec<_>>();
        state
            .generations
            .retain(|generation| !models.contains(&generation.model));
        models
    }
}

impl Drop for ModelLease {
    fn drop(&mut self) {
        let mut state = self
            .registry
            .state
            .lock()
            .expect("model lease registry poisoned");
        let Some(generation) = state
            .generations
            .iter_mut()
            .find(|generation| generation.model == self.model)
        else {
            return;
        };
        assert!(generation.users > 0, "model lease count underflowed");
        generation.users -= 1;
        let became_unused = generation.users == 0;
        if became_unused && state.current != self.model {
            queue_model_release(&mut state, self.model);
        }
    }
}

fn queue_model_release(state: &mut ModelLeaseState, model: ModelGeneration) {
    if !state.releasable.contains(&model) {
        state.releasable.push_back(model);
    }
}

struct EpisodeModelLeases {
    registry: Arc<ModelLeaseRegistry>,
    episodes: HashMap<EpisodeId, ModelLease>,
}

impl EpisodeModelLeases {
    fn new(registry: Arc<ModelLeaseRegistry>) -> Self {
        Self {
            registry,
            episodes: HashMap::new(),
        }
    }

    fn ensure(&mut self, episode_id: EpisodeId) -> EngineResult<ModelGeneration> {
        if let Some(lease) = self.episodes.get(&episode_id) {
            return Ok(lease.model);
        }
        let acquired = self.registry.acquire_current();
        let model = acquired.model;
        self.episodes.insert(episode_id, acquired);
        Ok(model)
    }

    fn release(&mut self, episode_id: EpisodeId) {
        self.episodes.remove(&episode_id);
    }
}

enum ReplayJob {
    Symmetric {
        game: Box<MeasuredSymmetricGame>,
        ack: SyncSender<EngineResult<MeasurerAdmission>>,
    },
}

impl<E, V> ThreadedGumbelOrchestrator<E, V>
where
    E: GraphEngine + Send,
    E::Graph: Send,
    E::Candidate: Send,
    V: Evaluator + Send,
{
    pub const fn new(
        engines: Vec<E>,
        evaluator: V,
        search: GumbelMcts,
        config: ThreadedOrchestratorConfig,
    ) -> Self {
        Self {
            engines,
            evaluator,
            search,
            config,
        }
    }

    pub fn run<R>(self, root_sources: Vec<R>) -> EngineResult<ThreadedRun<E::Graph, E::Candidate>>
    where
        R: RootSource<E> + Send,
    {
        if self.search.config().value_mode == GumbelValueMode::SymmetricSelfplay {
            return Err(internal(
                "symmetric selfplay requires featurized replay output",
            ));
        }
        let lanes = self.engines.len();
        if root_sources.len() != lanes {
            return Err(internal("lane count mismatch"));
        }
        validate_engine_identities(&self.engines)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let intake_capacity = lanes * workers_per_lane;
        // Intake can hold every possible outstanding eval at once. The batcher
        // never waits on a lane while holding jobs, so this bounded channel
        // cannot form a steady-state send cycle.
        let (intake_tx, intake_rx) = sync_channel(intake_capacity);
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for _ in 0..lanes {
            // A lane can have at most one outstanding eval per worker. This
            // capacity lets the batcher route all lane replies without blocking.
            let (tx, rx) = sync_channel(workers_per_lane);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let eval_pressure = Arc::new(EvalPressure::default());
        let admission_shaper =
            build_admission_shaper(lanes, 1, config, Arc::clone(&eval_pressure))?;
        let search = &self.search;
        let evaluator = self.evaluator;
        let engines = self.engines;

        let (batch_result, lane_results) = std::thread::scope(|scope| {
            let batch_pressure = Arc::clone(&eval_pressure);
            let batch_handle = scope.spawn(move || {
                run_batcher(evaluator, intake_rx, reply_txs, config, batch_pressure)
            });
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, ((engine, roots), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_tx.clone();
                let eval_pressure = Arc::clone(&eval_pressure);
                let admission_shaper = admission_shaper.clone();
                lane_handles.push(scope.spawn(move || {
                    run_lane_pipeline(
                        engine,
                        roots,
                        LaneRuntime {
                            lane,
                            lanes,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            pool_capacity: config.workers_per_lane,
                            worker_id_base: (lane * config.workers_per_lane.get()) as u64,
                            admission_stagger: config.admission_stagger,
                            admission_shaper,
                            eval_pressure,
                            intake_tx,
                            reply_rx,
                        },
                        CollectMode::new(),
                    )
                }));
            }

            drop(intake_tx);

            let lane_results = lane_handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err(internal("worker blocked")))
                })
                .collect::<Vec<_>>();
            let batch_result = batch_handle
                .join()
                .unwrap_or_else(|_| Err(internal("eval backend unavailable")));

            (batch_result, lane_results)
        });

        let batch_sizes = batch_result?;
        let mut lanes = Vec::with_capacity(lane_results.len());

        for result in lane_results {
            lanes.push(result?);
        }

        Ok(ThreadedRun { lanes, batch_sizes })
    }

    pub fn run_featurized_with_replay<R, X, B>(
        self,
        root_sources: Vec<R>,
        featurized: FeaturizedRuntime<X, B>,
        replay: ReplayRuntime<'_>,
    ) -> EngineResult<ThreadedReplayRun>
    where
        R: RootSource<E> + Send,
        X: FeatureExtractor<E> + Send,
        B: FeatureEvalBackend + Send,
    {
        if self.search.config().value_mode != GumbelValueMode::SymmetricSelfplay {
            return Err(internal("replay selfplay requires symmetric search"));
        }
        if !replay.length_tiebreak {
            return Err(internal(
                "symmetric selfplay requires the episode length tiebreak",
            ));
        }

        let lanes = self.engines.len();
        if root_sources.len() != lanes || featurized.extractors.len() != lanes {
            return Err(internal("lane count mismatch"));
        }
        validate_engine_identities(&self.engines)?;
        let schema_hash = validate_feature_schemas::<E, X>(&featurized.extractors)?;
        validate_backend_count(featurized.backends.len(), lanes)?;
        let data_mode = if self.search.config().mask_stop {
            gz_replay::ReplayDataMode::SymmetricSelfplay
        } else {
            gz_replay::ReplayDataMode::SymmetricSelfplayStop
        };
        replay
            .store
            .ensure_data_mode(data_mode)
            .map_err(map_replay_error)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let worker_capacity = lanes
            .checked_mul(workers_per_lane)
            .ok_or_else(|| internal("worker count overflow"))?;
        let evals_per_worker = if self.search.symmetric_wave_batching() {
            self.search
                .config()
                .max_considered_actions
                .get()
                .min(self.search.config().simulations.get())
        } else {
            1
        };
        let intake_capacity = worker_capacity
            .checked_mul(evals_per_worker)
            .ok_or_else(|| internal("wave eval capacity overflow"))?;
        let backend_count = featurized.backends.len();
        let mut intake_txs = Vec::with_capacity(backend_count);
        let mut intake_rxs = Vec::with_capacity(backend_count);
        for _ in 0..backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            intake_txs.push(tx);
            intake_rxs.push(rx);
        }
        let (replay_tx, replay_rx) = sync_channel(worker_capacity);
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);
        let reply_capacity = workers_per_lane
            .checked_mul(evals_per_worker)
            .ok_or_else(|| internal("wave reply capacity overflow"))?;
        for _ in 0..lanes {
            let (tx, rx) = sync_channel(reply_capacity);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let eval_pressure = Arc::new(EvalPressure::default());
        let admission_shaper =
            build_admission_shaper(lanes, backend_count, config, Arc::clone(&eval_pressure))?;
        let search = &self.search;
        let backends = featurized.backends;
        let model_registries = backends
            .iter()
            .map(|backend| ModelLeaseRegistry::new(backend.model_generation()).map(Arc::new))
            .collect::<EngineResult<Vec<_>>>()?;
        let extractors = featurized.extractors;
        let engines = self.engines;
        let store = replay.store;
        let backpressure = replay.backpressure;
        let length_tiebreak = replay.length_tiebreak;
        let feature_schema = first_schema::<E, X>(&extractors, schema_hash)?;
        store
            .ensure_feature_schema(feature_schema.config())
            .map_err(map_replay_error)?;
        validate_collator_capacity(
            &FeatureCollator::new(feature_schema.clone(), config.max_batch),
            config,
        )?;
        let _ = self.evaluator;

        let (batch_results, sink_result, lane_results) = std::thread::scope(|scope| {
            let mut batch_handles = Vec::with_capacity(backend_count);
            for ((backend, intake_rx), model_registry) in backends
                .into_iter()
                .zip(intake_rxs)
                .zip(model_registries.iter().cloned())
            {
                let batch_capacity = backend.batch_capacity().unwrap_or(config.max_batch);
                let collator = FeatureCollator::new(feature_schema.clone(), batch_capacity);
                let reply_txs = reply_txs.clone();
                let eval_pressure = Arc::clone(&eval_pressure);
                batch_handles.push(scope.spawn(move || {
                    run_featurized_batcher(
                        backend,
                        collator,
                        intake_rx,
                        reply_txs,
                        config,
                        FeaturizedBatcherContext {
                            model_registry,
                            eval_pressure: Some(eval_pressure),
                        },
                    )
                }));
            }
            drop(reply_txs);
            let sink_handle =
                scope.spawn(move || run_replay_sink(store, replay_rx, length_tiebreak));
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, ((((engine, roots), extractor), reply_rx), model_registry)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(extractors)
                .zip(reply_rxs)
                .zip(
                    (0..lanes)
                        .map(|lane| Arc::clone(&model_registries[lane % model_registries.len()])),
                )
                .enumerate()
            {
                let intake_tx = intake_txs[lane % backend_count].clone();
                let replay_tx = replay_tx.clone();
                let eval_pressure = Arc::clone(&eval_pressure);
                let admission_shaper = admission_shaper.clone();
                lane_handles.push(scope.spawn(move || {
                    run_lane_pipeline(
                        engine,
                        roots,
                        LaneRuntime {
                            lane,
                            lanes,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            pool_capacity: config.workers_per_lane,
                            worker_id_base: (lane * config.workers_per_lane.get()) as u64,
                            admission_stagger: config.admission_stagger,
                            admission_shaper,
                            eval_pressure,
                            intake_tx,
                            reply_rx,
                        },
                        FeaturizedReplayMode::new(
                            lane,
                            extractor,
                            replay_tx,
                            store,
                            backpressure,
                            model_registry,
                        ),
                    )
                }));
            }

            drop(intake_txs);
            drop(replay_tx);

            let lane_results = lane_handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err(internal("worker blocked")))
                })
                .collect::<Vec<_>>();
            let batch_results = batch_handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err(internal("eval backend unavailable")))
                })
                .collect::<Vec<_>>();
            let sink_result = sink_handle
                .join()
                .unwrap_or_else(|_| Err(internal("replay sink failed")));

            (batch_results, sink_result, lane_results)
        });

        let mut batch_sizes = Vec::new();
        for result in batch_results {
            batch_sizes.extend(result?);
        }
        let measurer_summary = sink_result?;
        let mut lane_summaries = Vec::with_capacity(lane_results.len());
        let mut search_contexts = 0;
        for result in lane_results {
            let mut result = result?;
            merge_lane_measurer_summary(&mut result, &measurer_summary);
            search_contexts += result.search_contexts;
            lane_summaries.push(result);
        }

        Ok(ThreadedReplayRun {
            lanes: lane_summaries,
            batch_sizes,
            episodes_appended: measurer_summary.episodes_appended,
            episodes_dropped: measurer_summary.episodes_dropped,
            search_contexts,
            replay_rows: measurer_summary.replay_rows,
            measure_ledger: measurer_summary.measure_ledger,
        })
    }
}

struct LaneRuntime<'a, J> {
    lane: usize,
    lanes: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    pool_capacity: NonZeroUsize,
    worker_id_base: u64,
    admission_stagger: Duration,
    admission_shaper: Option<Arc<SharedAdmissionShaper>>,
    eval_pressure: Arc<EvalPressure>,
    intake_tx: SyncSender<J>,
    reply_rx: Receiver<EvalReply>,
}

struct SymmetricFeatureRows<C> {
    p1: Vec<Vec<u8>>,
    p2: Vec<Vec<u8>>,
    candidates: Vec<C>,
}

trait LaneMode<E>
where
    E: GraphEngine,
{
    type Job;
    type Output;

    #[allow(clippy::too_many_arguments)]
    fn admit_roots<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        search: &GumbelMcts,
        identity: EngineIdentity,
        next_episode_id: &mut u64,
        limit: usize,
        pressure_reserved: bool,
    ) -> EngineResult<AdmissionResult>
    where
        R: RootSource<E>,
    {
        let mut admission = Admission {
            search,
            identity,
            symmetric_selfplay: search.config().value_mode == GumbelValueMode::SymmetricSelfplay,
            pressure_reserved,
            next_episode_id,
        };
        pool.admit_limited(engine, roots, &mut admission, limit, |engine, id, root| {
            self.admit_episode(engine, id, root)
        })
    }

    fn gate_open(&self) -> bool {
        true
    }

    fn gate_poll(&self) -> Option<Duration> {
        None
    }

    fn admit_episode(
        &mut self,
        engine: &mut E,
        episode_id: EpisodeId,
        root: E::Graph,
    ) -> EngineResult<()> {
        let _ = (engine, episode_id, root);
        Ok(())
    }

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<CompletedTask<E::Graph, E::Candidate>>>;

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()>;

    fn complete(
        &mut self,
        engine: &mut E,
        search: &GumbelMcts,
        completed: CompletedTask<E::Graph, E::Candidate>,
    ) -> EngineResult<Option<u64>>;

    fn finish(self, lane: usize) -> Self::Output;
}

fn run_lane_pipeline<E, R, M>(
    mut engine: E,
    mut roots: R,
    runtime: LaneRuntime<'_, M::Job>,
    mut mode: M,
) -> EngineResult<M::Output>
where
    E: GraphEngine,
    R: RootSource<E>,
    M: LaneMode<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let mut pool = WorkerPool::new(runtime.pool_capacity, runtime.worker_id_base);
    let mut roots_exhausted = false;
    let mut next_episode_id = (runtime.lane as u64) << 32;
    let mut admission_pacer = AdmissionPacer::new(
        runtime.lane,
        runtime.lanes,
        runtime.workers_per_lane.get(),
        runtime.admission_stagger,
    );
    loop {
        let mut adaptive_retry_after = None;
        if !roots_exhausted {
            let gate_open = mode.gate_open();
            if gate_open {
                let learner_slots = available_learner_slots(&pool, runtime.workers_per_lane.get());
                if let Some(shaper) = &runtime.admission_shaper {
                    let decision = shaper.request(runtime.lane, learner_slots)?;
                    adaptive_retry_after = decision.retry_after;
                    if decision.limit > 0 {
                        let result = match mode.admit_roots(
                            &mut pool,
                            &mut engine,
                            &mut roots,
                            runtime.search,
                            identity,
                            &mut next_episode_id,
                            decision.limit,
                            true,
                        ) {
                            Ok(result) => result,
                            Err(error) => {
                                shaper.finish_admission(runtime.lane, decision, 0, false)?;
                                return Err(error);
                            }
                        };
                        roots_exhausted = result.roots_exhausted;
                        shaper.finish_admission(
                            runtime.lane,
                            decision,
                            result.admitted,
                            roots_exhausted,
                        )?;
                    } else if !pool.active()
                        && let Some(sleep) = decision.retry_after
                    {
                        std::thread::sleep(sleep);
                    }
                } else if learner_slots > 0 && admission_pacer.ready() {
                    let result = mode.admit_roots(
                        &mut pool,
                        &mut engine,
                        &mut roots,
                        runtime.search,
                        identity,
                        &mut next_episode_id,
                        admission_pacer.limit().min(learner_slots),
                        false,
                    )?;
                    roots_exhausted = result.roots_exhausted;
                    admission_pacer.record(result.admitted);
                } else if !pool.active()
                    && let Some(sleep) = admission_pacer.sleep_until_ready()
                {
                    std::thread::sleep(sleep);
                }
            } else {
                if let Some(shaper) = &runtime.admission_shaper {
                    shaper.clear_lane(runtime.lane)?;
                }
            }
            if !gate_open
                && !pool.active()
                && let Some(gate_poll) = mode.gate_poll()
            {
                // The gate limits admission only. In-flight episodes always
                // finish, so backlog can overshoot by at most total workers
                // times rows per episode. This sleep is the throttled-idle
                // path that prevents a fully gated lane from busy-spinning.
                std::thread::sleep(gate_poll);
            }
        }

        for completed in mode.drive(&mut engine, &mut pool)? {
            let episode_work = mode.complete(&mut engine, runtime.search, completed)?;
            if let (Some(shaper), Some(evaluations)) = (&runtime.admission_shaper, episode_work) {
                shaper.observe_episode_work(evaluations)?;
            }
        }

        mode.send_parked(
            runtime.lane,
            &mut pool,
            &runtime.intake_tx,
            &runtime.eval_pressure,
        )?;

        if roots_exhausted && !pool.active() {
            if let Some(shaper) = &runtime.admission_shaper {
                shaper.clear_lane(runtime.lane)?;
            }
            return Ok(mode.finish(runtime.lane));
        }

        let reply_wait = adaptive_retry_after.filter(|_| {
            !roots_exhausted && available_learner_slots(&pool, runtime.workers_per_lane.get()) > 0
        });
        if pool.has_parked() {
            receive_replies(&mut engine, &mut pool, &runtime.reply_rx, reply_wait)?;
        }
    }
}

fn available_learner_slots<G, C>(pool: &WorkerPool<G, C>, workers_per_lane: usize) -> usize
where
    G: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
{
    workers_per_lane
        .saturating_sub(pool.active_count())
        .min(pool.idle_count())
}

struct AdmissionPacer {
    stagger: Duration,
    next: Instant,
    resume_offset: Duration,
}

impl AdmissionPacer {
    fn new(lane: usize, lanes: usize, workers_per_lane: usize, stagger: Duration) -> Self {
        let now = Instant::now();
        if stagger.is_zero() {
            return Self {
                stagger,
                next: now,
                resume_offset: Duration::ZERO,
            };
        }
        let offset = spread_duration(stagger, lane, lanes);
        eprintln!(
            "event=admission_pacer lane={lane} interval_ms={} first_delay_ms={} cohort_span_ms={}",
            stagger.as_millis(),
            offset.as_millis(),
            stagger.as_millis() * workers_per_lane as u128,
        );
        Self {
            stagger,
            next: now + offset,
            resume_offset: offset,
        }
    }

    fn ready(&mut self) -> bool {
        if self.stagger.is_zero() {
            return true;
        }
        let now = Instant::now();
        if now.saturating_duration_since(self.next) >= self.stagger {
            // Do not repay missed admissions in a burst after a closed gate
            // or a fully occupied lane. Reapply the lane's global phase.
            self.next = now + self.resume_offset;
        }
        now >= self.next
    }

    fn limit(&self) -> usize {
        if self.stagger.is_zero() {
            usize::MAX
        } else {
            1
        }
    }

    fn record(&mut self, admitted: usize) {
        if self.stagger.is_zero() || admitted == 0 {
            return;
        }
        self.next = Instant::now() + self.stagger;
    }

    fn sleep_until_ready(&self) -> Option<Duration> {
        if self.stagger.is_zero() {
            return None;
        }
        Some(self.next.saturating_duration_since(Instant::now())).filter(|sleep| !sleep.is_zero())
    }
}

fn spread_duration(duration: Duration, index: usize, count: usize) -> Duration {
    if count <= 1 || index == 0 || duration.is_zero() {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos().saturating_mul(index as u128) / count as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}

struct CollectMode<G, C> {
    episodes: Vec<OrchestratedEpisode<G, C>>,
}

impl<G, C> CollectMode<G, C> {
    fn new() -> Self {
        Self {
            episodes: Vec::new(),
        }
    }
}

impl<E> LaneMode<E> for CollectMode<E::Graph, E::Candidate>
where
    E: GraphEngine,
{
    type Job = EvalJob;
    type Output = LaneEpisodes<E::Graph, E::Candidate>;

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<CompletedTask<E::Graph, E::Candidate>>> {
        pool.drive(engine, "worker blocked", None)
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()> {
        send_plain_parked(lane, pool, intake_tx, eval_pressure)
    }

    fn complete(
        &mut self,
        engine: &mut E,
        _search: &GumbelMcts,
        completed: CompletedTask<E::Graph, E::Candidate>,
    ) -> EngineResult<Option<u64>> {
        let completed = completed.into_gumbel()?;
        let evaluations = completed.evaluations;
        release_episode_handles(engine, &completed.episode, &[])?;
        self.episodes.push(completed);
        Ok(Some(evaluations))
    }

    fn finish(self, lane: usize) -> Self::Output {
        LaneEpisodes {
            lane,
            episodes: self.episodes,
        }
    }
}

struct FeaturizedReplayMode<'a, X> {
    extractor: X,
    replay_tx: SyncSender<ReplayJob>,
    store: &'a ReplayStore,
    backpressure: Option<ReplayBackpressure>,
    admitted_at: HashMap<EpisodeId, Instant>,
    summary: ReplayLaneSummary,
    model_leases: EpisodeModelLeases,
}

impl<'a, X> FeaturizedReplayMode<'a, X> {
    fn new(
        lane: usize,
        extractor: X,
        replay_tx: SyncSender<ReplayJob>,
        store: &'a ReplayStore,
        backpressure: Option<ReplayBackpressure>,
        model_registry: Arc<ModelLeaseRegistry>,
    ) -> Self {
        Self {
            extractor,
            replay_tx,
            store,
            backpressure,
            admitted_at: HashMap::new(),
            summary: ReplayLaneSummary {
                lane,
                episodes_completed: 0,
                episodes_appended: 0,
                episodes_dropped: 0,
                search_contexts: 0,
                replay_rows: 0,
            },
            model_leases: EpisodeModelLeases::new(model_registry),
        }
    }
}

impl<E, X> LaneMode<E> for FeaturizedReplayMode<'_, X>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    type Job = FeaturizedEvalJob;
    type Output = ReplayLaneSummary;

    fn gate_open(&self) -> bool {
        replay_gate_open(self.store, self.backpressure)
    }

    fn gate_poll(&self) -> Option<Duration> {
        self.backpressure.map(|backpressure| backpressure.gate_poll)
    }

    fn admit_episode(
        &mut self,
        _engine: &mut E,
        episode_id: EpisodeId,
        _root: E::Graph,
    ) -> EngineResult<()> {
        self.model_leases.ensure(episode_id)?;
        self.admitted_at.insert(episode_id, Instant::now());
        Ok(())
    }

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<CompletedTask<E::Graph, E::Candidate>>> {
        pool.drive(engine, "worker blocked", Some(&mut self.extractor))
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()> {
        send_featurized_parked(lane, pool, intake_tx, eval_pressure, &mut self.model_leases)
    }

    fn complete(
        &mut self,
        engine: &mut E,
        search: &GumbelMcts,
        completed: CompletedTask<E::Graph, E::Candidate>,
    ) -> EngineResult<Option<u64>> {
        let episode_id = completed.episode_id;
        let episode = match completed.episode {
            CompletedSearchEpisode::Symmetric(episode) => *episode,
            CompletedSearchEpisode::Gumbel(episode) => {
                release_episode_handles(engine, &episode, &[])?;
                self.model_leases.release(episode_id);
                return Err(internal("non-symmetric episode in symmetric replay"));
            }
        };

        if let Some(admitted_at) = self.admitted_at.remove(&episode_id) {
            self.store
                .observe_episode_latency(admitted_at.elapsed().as_secs_f64());
        }
        let feature_rows =
            match feature_rows_for_symmetric_episode(engine, &mut self.extractor, search, &episode)
            {
                Ok(rows) => rows,
                Err(error) => {
                    release_symmetric_episode_handles(engine, &episode, &[])?;
                    self.model_leases.release(episode_id);
                    return Err(error);
                }
            };
        let game = measured_symmetric_game(self.summary.lane, &episode, &feature_rows);
        self.summary.episodes_completed += 1;
        self.summary.search_contexts += symmetric_search_contexts(&episode);

        let append = append_symmetric_replay_job(&self.replay_tx, game);
        let release = release_symmetric_episode_handles(engine, &episode, &feature_rows.candidates);
        self.model_leases.release(episode_id);
        release?;
        append?;

        Ok(Some(completed.evaluations))
    }

    fn finish(mut self, lane: usize) -> Self::Output {
        self.summary.lane = lane;
        self.summary
    }
}

fn send_featurized_parked<G, C>(
    lane: usize,
    pool: &mut WorkerPool<G, C>,
    intake_tx: &SyncSender<FeaturizedEvalJob>,
    eval_pressure: &EvalPressure,
    model_leases: &mut EpisodeModelLeases,
) -> EngineResult<()>
where
    G: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
{
    for parked in pool.take_unsent_parked() {
        let row = parked.row.ok_or_else(|| internal("missing feature row"))?;
        let model = model_leases.ensure(parked.episode_id)?;
        if parked.pressure_reserved {
            pool.consume_pressure_reservation(parked.slot, parked.token)?;
        }
        eval_pressure.submit(parked.pressure_reserved);
        if intake_tx
            .send(FeaturizedEvalJob {
                lane,
                slot: parked.slot,
                token: parked.token,
                row,
                action_count: parked.action_count,
                model,
            })
            .is_err()
        {
            eval_pressure.cancel_submission();
            return Err(internal("eval backend unavailable"));
        }
    }
    Ok(())
}

fn symmetric_search_contexts<G, C>(episode: &SymmetricEpisode<G, C>) -> u64 {
    episode
        .p1
        .root_stats
        .iter()
        .chain(&episode.p2.root_stats)
        .map(|stats| stats.portable_contexts as u64)
        .sum()
}

fn merge_lane_measurer_summary(lane: &mut ReplayLaneSummary, measurer: &MeasurerRunSummary) {
    let Some(measured) = measurer.lanes.get(lane.lane) else {
        return;
    };
    lane.episodes_appended = measured.episodes_appended;
    lane.episodes_dropped = measured.episodes_dropped;
    lane.replay_rows = measured.replay_rows;
}

fn send_plain_parked<G, C>(
    lane: usize,
    pool: &mut WorkerPool<G, C>,
    intake_tx: &SyncSender<EvalJob>,
    eval_pressure: &EvalPressure,
) -> EngineResult<()>
where
    G: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
{
    for parked in pool.take_unsent_parked() {
        if parked.pressure_reserved {
            pool.consume_pressure_reservation(parked.slot, parked.token)?;
        }
        eval_pressure.submit(parked.pressure_reserved);
        if intake_tx
            .send(EvalJob {
                lane,
                slot: parked.slot,
                token: parked.token,
                request: parked.request,
            })
            .is_err()
        {
            eval_pressure.cancel_submission();
            return Err(internal("eval backend unavailable"));
        }
    }
    Ok(())
}

fn feature_rows_for_symmetric_episode<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    episode: &SymmetricEpisode<E::Graph, E::Candidate>,
) -> EngineResult<SymmetricFeatureRows<E::Candidate>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let mut candidates = Vec::new();
    let rows = (|| {
        let p1 = feature_rows_for_symmetric_actor(
            engine,
            extractor,
            search,
            &episode.p1,
            &episode.p2,
            false,
            &mut candidates,
        )?;
        let p2 = feature_rows_for_symmetric_actor(
            engine,
            extractor,
            search,
            &episode.p2,
            &episode.p1,
            true,
            &mut candidates,
        )?;
        Ok(SymmetricFeatureRows {
            p1,
            p2,
            candidates: std::mem::take(&mut candidates),
        })
    })();
    match rows {
        Ok(rows) => Ok(rows),
        Err(error) => {
            engine.release(&[], &candidates)?;
            Err(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn feature_rows_for_symmetric_actor<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    actor: &SymmetricActorTrace<E::Graph, E::Candidate>,
    opponent: &SymmetricActorTrace<E::Graph, E::Candidate>,
    opponent_after_turn: bool,
    created_candidates: &mut Vec<E::Candidate>,
) -> EngineResult<Vec<Vec<u8>>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let schema = extractor.schema().clone();
    let mut rows = Vec::with_capacity(actor.steps.len());
    let mut candidates = Vec::new();
    for (index, step) in actor.steps.iter().enumerate() {
        candidates.clear();
        engine.candidates(
            step.before,
            search.config().candidate_options,
            &mut candidates,
        )?;
        created_candidates.extend(candidates.iter().copied());
        let (actor_step, actor_inactive) =
            symmetric_position_state(actor, index, false, search.config());
        let mut position = replay_position_features(
            episode_position_config(search),
            extractor.schema(),
            actor_step,
        )?;
        if actor_inactive {
            position.budget_step = -position.budget_step.abs();
        }
        position.opponent_present = true;
        let mut row = extractor
            .extract(engine, step.before, &candidates, position)
            .map_err(|_| internal("feature extraction failed"))?;

        let requested_opponent_index = index + usize::from(opponent_after_turn);
        let opponent_index = requested_opponent_index.min(opponent.steps.len());
        let opponent_graph = symmetric_actor_state(opponent, opponent_index);
        let (opponent_step, opponent_inactive) = symmetric_position_state(
            opponent,
            opponent_index,
            requested_opponent_index > opponent.steps.len(),
            search.config(),
        );
        let mut opponent_position = replay_position_features(
            episode_position_config(search),
            extractor.schema(),
            opponent_step,
        )?;
        if opponent_inactive {
            opponent_position.budget_step = -opponent_position.budget_step.abs();
        }
        let opponent_row = extractor
            .extract(engine, opponent_graph, &[], opponent_position)
            .map_err(|_| internal("opponent feature extraction failed"))?;
        row.opponent = Some(OpponentStateFeatures {
            node_count: opponent_row.node_count,
            node_tokens: opponent_row.node_tokens,
            node_attrs: opponent_row.node_attrs,
            edges: opponent_row.edges,
            position: opponent_row.position,
        });
        let expected_actions = if search.config().mask_stop {
            step.legal_actions.len().saturating_add(1)
        } else {
            step.legal_actions.len()
        };
        if row.actions.len() != expected_actions {
            return Err(internal("symmetric feature row action count mismatch"));
        }

        let mut bytes = Vec::new();
        encode_feature_row(&row, &schema, &mut bytes)
            .map_err(|_| internal("feature row encoding failed"))?;
        rows.push(bytes);
    }
    Ok(rows)
}

fn symmetric_actor_state<G: Copy, C>(actor: &SymmetricActorTrace<G, C>, index: usize) -> G {
    if index == 0 {
        actor.root
    } else {
        actor
            .steps
            .get(index - 1)
            .map_or(actor.final_graph, |step| step.after)
    }
}

fn symmetric_position_state<G, C>(
    actor: &SymmetricActorTrace<G, C>,
    decision_count: usize,
    observed_after_trace: bool,
    config: gz_search::GumbelMctsConfig,
) -> (usize, bool) {
    let rewrites = actor.steps[..decision_count]
        .iter()
        .filter(|step| matches!(step.action, SearchAction::Candidate(_)))
        .count();
    let at_trace_end = decision_count == actor.steps.len();
    let inactive = at_trace_end
        && (actor.stopped || rewrites >= config.max_steps || actor.blocked && observed_after_trace);
    (rewrites, inactive)
}

#[allow(clippy::too_many_arguments)]
fn measured_symmetric_game<G: Copy, C: Copy>(
    lane: usize,
    episode: &SymmetricEpisode<G, C>,
    rows: &SymmetricFeatureRows<C>,
) -> MeasuredSymmetricGame {
    MeasuredSymmetricGame {
        lane,
        p1_artifact: symmetric_artifact(&episode.p1, &rows.p1, episode.search_config_hash),
        p2_artifact: symmetric_artifact(&episode.p2, &rows.p2, episode.search_config_hash),
    }
}

fn symmetric_artifact<G: Copy, C>(
    actor: &SymmetricActorTrace<G, C>,
    feature_rows: &[Vec<u8>],
    search_config_hash: gz_engine::SearchConfigHash,
) -> CompletedEpisodeArtifact {
    CompletedEpisodeArtifact {
        root: actor.root_context,
        final_graph: actor.final_context,
        final_measure: gz_engine::MeasureSummary::from(&actor.final_measure),
        stop_selected: actor.stopped,
        search_config_hash,
        steps: actor
            .steps
            .iter()
            .map(|step| CompletedEpisodeStep {
                before: step.step_ref.before,
                after: step.step_ref.after,
                selected_action: step.selected_action,
                legal_actions: step.legal_actions.clone(),
                policy_target: step.policy_target.clone(),
                root_value: Some(step.root_value),
                root_search_value: Some(step.root_search_value),
                model_version: Some(step.model_version),
            })
            .collect(),
        feature_rows: Some(feature_rows.to_vec()),
    }
}

struct EpisodePositionConfig {
    max_steps: usize,
    export_position: bool,
}

fn episode_position_config(search: &GumbelMcts) -> EpisodePositionConfig {
    let config = search.config();
    EpisodePositionConfig {
        max_steps: config.max_steps,
        export_position: config.export_position,
    }
}

fn replay_position_features(
    config: EpisodePositionConfig,
    _schema: &FeatureSchema,
    index: usize,
) -> EngineResult<PositionFeatures> {
    let (root_step, budget_fraction, budget_step) = if config.export_position {
        let budget_step = if config.max_steps == 0 {
            0.0
        } else {
            1.0 / config.max_steps as f32
        };
        let budget_fraction = if config.max_steps == 0 {
            1.0
        } else {
            config.max_steps.saturating_sub(index) as f32 / config.max_steps as f32
        };
        (
            u32::try_from(index).map_err(|_| internal("root step overflow"))?,
            budget_fraction,
            budget_step,
        )
    } else {
        (0, 0.0, 0.0)
    };
    Ok(PositionFeatures {
        root_step,
        leaf_depth: 0,
        budget_fraction,
        budget_step,
        opponent_reward: 0.0,
        opponent_present: false,
    })
}

fn release_episode_handles<E>(
    engine: &mut E,
    episode: &GumbelEpisode<E::Graph, E::Candidate>,
    extra_candidates: &[E::Candidate],
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_created_handles(
        engine,
        &episode.created_graphs,
        &episode.created_candidates,
        extra_candidates,
    )
}

fn release_symmetric_episode_handles<E>(
    engine: &mut E,
    episode: &SymmetricEpisode<E::Graph, E::Candidate>,
    extra_candidates: &[E::Candidate],
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_created_handles(
        engine,
        &episode.created_graphs,
        &episode.created_candidates,
        extra_candidates,
    )
}

fn release_created_handles<E>(
    engine: &mut E,
    created_graphs: &[E::Graph],
    created_candidates: &[E::Candidate],
    extra_candidates: &[E::Candidate],
) -> EngineResult<()>
where
    E: GraphEngine,
{
    if extra_candidates.is_empty() {
        return engine.release(created_graphs, created_candidates);
    }

    let mut candidates = Vec::with_capacity(created_candidates.len() + extra_candidates.len());
    candidates.extend_from_slice(created_candidates);
    candidates.extend_from_slice(extra_candidates);
    engine.release(created_graphs, &candidates)
}

fn append_symmetric_replay_job(
    replay_tx: &SyncSender<ReplayJob>,
    game: MeasuredSymmetricGame,
) -> EngineResult<MeasurerAdmission> {
    let (ack, done) = sync_channel(1);
    replay_tx
        .send(ReplayJob::Symmetric {
            game: Box::new(game),
            ack,
        })
        .map_err(|_| internal("replay sink failed"))?;
    done.recv().map_err(|_| internal("replay sink failed"))?
}

/// Resumes every pending reply.
fn receive_replies<E>(
    engine: &mut E,
    pool: &mut WorkerPool<E::Graph, E::Candidate>,
    reply_rx: &Receiver<EvalReply>,
    wait: Option<Duration>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    let reply = match wait {
        Some(wait) => match reply_rx.recv_timeout(wait) {
            Ok(reply) => reply,
            Err(RecvTimeoutError::Timeout) => return Ok(()),
            Err(RecvTimeoutError::Disconnected) => {
                return Err(internal("eval backend unavailable"));
            }
        },
        None => reply_rx
            .recv()
            .map_err(|_| internal("eval backend unavailable"))?,
    };
    pool.resume(engine, reply.slot, reply.token, reply.output)?;

    loop {
        match reply_rx.try_recv() {
            Ok(reply) => {
                pool.resume(engine, reply.slot, reply.token, reply.output)?;
            }
            Err(TryRecvError::Empty) => return Ok(()),
            Err(TryRecvError::Disconnected) => return Err(internal("eval backend unavailable")),
        }
    }
}

fn replay_gate_open(store: &ReplayStore, backpressure: Option<ReplayBackpressure>) -> bool {
    let Some(backpressure) = backpressure else {
        return true;
    };
    let counters = store.counters();
    let backlog = counters
        .produced_rows
        .saturating_sub(counters.consumed_rows);

    backlog <= backpressure.max_row_backlog.get()
}

fn run_batcher<V>(
    mut evaluator: V,
    intake_rx: Receiver<EvalJob>,
    reply_txs: Vec<SyncSender<EvalReply>>,
    config: ThreadedOrchestratorConfig,
    eval_pressure: Arc<EvalPressure>,
) -> EngineResult<Vec<usize>>
where
    V: Evaluator,
{
    let mut batch_sizes = Vec::new();

    loop {
        let first = match intake_rx.recv() {
            Ok(job) => job,
            Err(_) => return Ok(batch_sizes),
        };
        let mut batch = vec![first];
        let deadline = Instant::now() + config.flush_after;

        while batch.len() < config.max_batch.get() {
            let now = Instant::now();
            let remaining = deadline.saturating_duration_since(now);
            match intake_rx.recv_timeout(remaining) {
                Ok(job) => batch.push(job),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }

        let requests = batch
            .iter()
            .map(|job| job.request.clone())
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(requests.len());
        let capacity_started = Instant::now();
        evaluator
            .evaluate_batch(&requests, &mut outputs)
            .map_err(eval_error_to_engine_error)?;
        let capacity_busy = capacity_started.elapsed();
        validate_outputs(&requests, &outputs).map_err(eval_error_to_engine_error)?;
        let completed = batch.len();
        batch_sizes.push(completed);

        for (job, output) in batch.into_iter().zip(outputs) {
            let _ = reply_txs[job.lane].send(EvalReply {
                slot: job.slot,
                token: job.token,
                output,
            });
        }
        eval_pressure.complete_current_batch(completed, completed, capacity_busy);
    }
}

/// Batches eval jobs and keeps one submitted batch in flight: while batch
/// N runs on the backend, batch N+1 is collected and submitted before N's
/// outputs are received, so a pipelining backend (the evaluator process)
/// overlaps its request read and staging with GPU compute. Non-pipelining
/// backends compute at submit and the loop degenerates to the historical
/// serial behavior.
///
/// Liveness: while a batch is in flight, collection is bounded by the
/// flush window and may come up empty (every parked eval can be inside
/// the in-flight batch, and new jobs only arrive after its replies), so
/// the loop always progresses to receive-and-route.
fn run_featurized_batcher<B>(
    mut backend: B,
    mut collator: FeatureCollator,
    intake_rx: Receiver<FeaturizedEvalJob>,
    reply_txs: Vec<SyncSender<EvalReply>>,
    config: ThreadedOrchestratorConfig,
    context: FeaturizedBatcherContext,
) -> EngineResult<Vec<usize>>
where
    B: FeatureEvalBackend,
{
    type Routing = BatcherRouting;
    let max_batch = collator.batch_capacity().get();

    // Up to PIPELINE_DEPTH submitted batches ride the backend at once
    // (the evaluator moves outputs off its static buffers at launch, so
    // its GPU queue holds a batch while the previous one drains); replies
    // are FIFO. Depth 3: one computing, one staged behind it, one in the
    // socket buffer, so the server never starves between client drains.
    // Machine-parsed by the trainer driver (eval fill metrics); field
    // changes must update its parser. Counters are cumulative: the
    // driver computes rates and window means from deltas.
    const STATS_INTERVAL: Duration = Duration::from_secs(30);

    let mut batch_sizes = Vec::new();
    let mut batch = Vec::with_capacity(max_batch);
    let mut rows = Vec::with_capacity(max_batch);
    let mut action_counts = Vec::with_capacity(max_batch);
    let mut bytes = Vec::new();
    let mut deferred: VecDeque<FeaturizedEvalJob> = VecDeque::with_capacity(max_batch);
    let mut in_flight: VecDeque<(Routing, gz_eval_service::PendingBatch, ModelGeneration)> =
        VecDeque::with_capacity(EVAL_PIPELINE_DEPTH);
    let mut capacity_accounted_at = None;
    let mut intake_open = true;
    let mut stats_batches: usize = 0;
    let mut last_stats = Instant::now();

    while intake_open || !in_flight.is_empty() || !deferred.is_empty() {
        release_releasable_models(&mut backend, &context.model_registry)?;
        batch.clear();
        let mut batch_model = None;
        if in_flight.len() < EVAL_PIPELINE_DEPTH && (intake_open || !deferred.is_empty()) {
            if let Some(first) = deferred.pop_front() {
                batch_model = Some(first.model);
                batch.push(first);
                let queued = deferred.len();
                for _ in 0..queued {
                    let job = deferred
                        .pop_front()
                        .expect("deferred eval queue length changed");
                    if batch.len() < max_batch && Some(job.model) == batch_model {
                        batch.push(job);
                    } else {
                        deferred.push_back(job);
                    }
                }
            }
            // Fill toward a FULL batch. The evaluator's buffers (and its
            // CUDA-graph forward) are capacity-shaped, so a half batch
            // costs the same GPU time as a full one: padding rows are
            // pure waste. While the backend holds work, a partial batch
            // therefore waits -- each flush-window timeout drains the
            // oldest reply instead, and the workers that unblocks come
            // straight back with new evals to finish the fill. Only a
            // backend about to go idle flushes a partial batch.
            loop {
                if batch.len() >= max_batch {
                    break;
                }
                if batch.is_empty() && in_flight.is_empty() {
                    // Nothing anywhere: block for work.
                    match intake_rx.recv() {
                        Ok(job) => {
                            batch_model = Some(job.model);
                            batch.push(job);
                        }
                        Err(_) => {
                            intake_open = false;
                            break;
                        }
                    }
                    continue;
                }
                if !intake_open {
                    break;
                }
                match intake_rx.recv_timeout(config.flush_after) {
                    Ok(job) => {
                        if batch_model.is_none() {
                            batch_model = Some(job.model);
                            batch.push(job);
                        } else if Some(job.model) == batch_model {
                            batch.push(job);
                        } else {
                            deferred.push_back(job);
                            if deferred.len() >= max_batch {
                                break;
                            }
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if in_flight.is_empty() {
                            // Backend idle: ship what we have now.
                            break;
                        }
                        drain_oldest(
                            &mut backend,
                            &mut in_flight,
                            &reply_txs,
                            &mut batch_sizes,
                            &context,
                            max_batch,
                            EvalCapacityAccounting {
                                pressure: context.eval_pressure.as_deref(),
                                accounted_at: &mut capacity_accounted_at,
                            },
                        )?;
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        intake_open = false;
                        break;
                    }
                }
            }
        }

        let submitted = if batch.is_empty() {
            false
        } else {
            let model = batch_model.ok_or_else(|| internal("missing eval batch model"))?;
            let mut routing: Routing = Vec::with_capacity(batch.len());
            rows.clear();
            action_counts.clear();
            for job in batch.drain(..) {
                if job.model != model {
                    return Err(internal("mixed model generations in eval batch"));
                }
                routing.push((job.lane, job.slot, job.token, job.action_count));
                action_counts.push(job.action_count);
                rows.push(job.row);
            }
            collator
                .collate_into(&rows, &mut bytes)
                .map_err(|_| internal("feature collation failed"))?;
            if in_flight.is_empty() {
                capacity_accounted_at = Some(Instant::now());
            }
            release_releasable_models(&mut backend, &context.model_registry)?;
            let pending = backend
                .submit_for_model(model, &bytes, &action_counts)
                .map_err(|_| internal("feature eval backend failed"))?;
            in_flight.push_back((routing, pending, model));
            true
        };

        // Drain the oldest reply when the pipeline is full, when this
        // round collected nothing (idle lanes are waiting on replies),
        // or when intake closed and only the tail remains.
        let must_drain =
            in_flight.len() >= EVAL_PIPELINE_DEPTH || (!submitted && !in_flight.is_empty());
        if must_drain {
            drain_oldest(
                &mut backend,
                &mut in_flight,
                &reply_txs,
                &mut batch_sizes,
                &context,
                max_batch,
                EvalCapacityAccounting {
                    pressure: context.eval_pressure.as_deref(),
                    accounted_at: &mut capacity_accounted_at,
                },
            )?;
        }
        if last_stats.elapsed() >= STATS_INTERVAL && batch_sizes.len() > stats_batches {
            stats_batches = batch_sizes.len();
            let stats_rows: u64 = batch_sizes.iter().map(|&size| size as u64).sum();
            last_stats = Instant::now();
            eprintln!("event=eval_stats role=current batches={stats_batches} rows={stats_rows}");
        }
    }

    release_releasable_models(&mut backend, &context.model_registry)?;
    Ok(batch_sizes)
}

type BatcherRouting = Vec<(usize, usize, WorkToken, u32)>;

struct EvalCapacityAccounting<'a> {
    pressure: Option<&'a EvalPressure>,
    accounted_at: &'a mut Option<Instant>,
}

fn drain_oldest<B>(
    backend: &mut B,
    in_flight: &mut VecDeque<(
        BatcherRouting,
        gz_eval_service::PendingBatch,
        ModelGeneration,
    )>,
    reply_txs: &[SyncSender<EvalReply>],
    batch_sizes: &mut Vec<usize>,
    context: &FeaturizedBatcherContext,
    max_batch: usize,
    capacity: EvalCapacityAccounting<'_>,
) -> EngineResult<()>
where
    B: FeatureEvalBackend,
{
    let Some((routing, pending, model)) = in_flight.pop_front() else {
        return Ok(());
    };
    let capacity_work = backend.capacity_work(routing.len(), max_batch);
    let outputs = backend
        .receive(pending)
        .map_err(|_| internal("feature eval backend failed"))?;
    let completed_at = Instant::now();
    if outputs.model_version != model.version {
        return Err(internal("evaluator served the wrong model version"));
    }
    let counts = routing
        .iter()
        .map(|&(_, _, _, action_count)| action_count)
        .collect::<Vec<_>>();
    validate_backend_outputs(&outputs, &counts)?;
    context.model_registry.publish(outputs.active_generation)?;
    let completed = routing.len();
    batch_sizes.push(completed);

    for ((lane, slot, token, _), row) in routing.into_iter().zip(outputs.rows) {
        let _ = reply_txs[lane].send(EvalReply {
            slot,
            token,
            output: EvalOutput {
                model_version: outputs.model_version,
                policy_logits: row.policy_logits,
                value: row.value,
            },
        });
    }
    if let Some(eval_pressure) = capacity.pressure {
        let capacity_started = capacity
            .accounted_at
            .take()
            .ok_or_else(|| internal("missing evaluator capacity clock"))?;
        let capacity_busy = completed_at.saturating_duration_since(capacity_started);
        eval_pressure.complete_current_batch(completed, capacity_work, capacity_busy);
        if !in_flight.is_empty() {
            *capacity.accounted_at = Some(completed_at);
        }
    }
    release_releasable_models(backend, &context.model_registry)?;
    Ok(())
}

fn release_releasable_models<B>(
    backend: &mut B,
    model_registry: &ModelLeaseRegistry,
) -> EngineResult<()>
where
    B: FeatureEvalBackend,
{
    for model in model_registry.take_releasable() {
        backend
            .release_model_generation(model)
            .map_err(|_| internal("feature eval backend failed"))?;
    }
    Ok(())
}

fn validate_backend_outputs(outputs: &BackendOutputs, action_counts: &[u32]) -> EngineResult<()> {
    if outputs.rows.len() != action_counts.len() {
        return Err(internal("eval output count mismatch"));
    }
    for (row, &action_count) in outputs.rows.iter().zip(action_counts) {
        if row.policy_logits.len() != action_count as usize {
            return Err(internal("eval output length mismatch"));
        }
        if !row.value.is_finite() || row.policy_logits.iter().any(|value| !value.is_finite()) {
            return Err(internal("invalid eval output"));
        }
    }
    Ok(())
}

fn run_replay_sink(
    store: &ReplayStore,
    replay_rx: Receiver<ReplayJob>,
    length_tiebreak: bool,
) -> EngineResult<MeasurerRunSummary> {
    let mut measurer = ReplayMeasurer::new(store, length_tiebreak);
    // Machine-parsed by the trainer driver (measure ledger metrics);
    // field changes must update its parser. Counters are cumulative.
    const STATS_INTERVAL: Duration = Duration::from_secs(30);
    let mut last_stats = Instant::now();

    while let Ok(job) = replay_rx.recv() {
        let (result, ack) = match job {
            ReplayJob::Symmetric { game, ack } => (
                measurer.admit_symmetric(*game).map_err(map_replay_error),
                ack,
            ),
        };
        let failed = result.as_ref().err().cloned();
        let _ = ack.send(result);
        if let Some(error) = failed {
            return Err(error);
        }
        if last_stats.elapsed() >= STATS_INTERVAL {
            last_stats = Instant::now();
            let stats = measurer.stats();
            eprintln!(
                "event=measure_stats appended={} dropped={} finals={} distinct={}",
                stats.episodes_appended,
                stats.episodes_dropped,
                stats.finals,
                stats.distinct_finals,
            );
        }
    }

    Ok(measurer.finish())
}

fn map_replay_error(error: ReplayError) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(1),
        message: ErrorMessage::new(format!("replay sink failed: {error}"))
            .expect("replay error message is bounded"),
    }
}

fn validate_engine_identities<E>(engines: &[E]) -> EngineResult<()>
where
    E: GraphEngine,
{
    let Some(first) = engines.first().map(EngineIdentity::from_engine) else {
        return Ok(());
    };
    for engine in &engines[1..] {
        if EngineIdentity::from_engine(engine) != first {
            return Err(internal("engine identity mismatch"));
        }
    }
    Ok(())
}

fn validate_feature_schemas<E, X>(extractors: &[X]) -> EngineResult<FeatureSchemaHash>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let Some(first) = extractors.first() else {
        return Err(internal("missing feature schema"));
    };
    let hash = first.schema().hash();
    for extractor in &extractors[1..] {
        if extractor.schema().hash() != hash {
            return Err(internal("feature schema mismatch"));
        }
    }
    Ok(hash)
}

fn first_schema<E, X>(extractors: &[X], hash: FeatureSchemaHash) -> EngineResult<FeatureSchema>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let schema = extractors
        .first()
        .ok_or_else(|| internal("missing feature schema"))?
        .schema();
    if schema.hash() != hash {
        return Err(internal("feature schema mismatch"));
    }
    Ok(schema.clone())
}

fn validate_backend_count(backends: usize, lanes: usize) -> EngineResult<()> {
    if backends == 0 {
        return Err(internal("no eval backends"));
    }
    if backends > lanes {
        return Err(internal("more eval backends than lanes"));
    }
    Ok(())
}

fn validate_collator_capacity(
    collator: &FeatureCollator,
    config: ThreadedOrchestratorConfig,
) -> EngineResult<()> {
    if collator.batch_capacity() != config.max_batch {
        return Err(internal("feature batch capacity mismatch"));
    }
    Ok(())
}
