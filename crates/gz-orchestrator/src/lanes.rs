use crate::EpisodeId;
use crate::admission::{AdaptiveAdmissionSchedule, AdmissionDecision, AdmissionSmoothingConfig};
use crate::pool::{Admission, AdmissionResult, WorkerPool};
use crate::project::{artifact_from_episode, projected_reference};
use crate::reference::{
    ArenaRolloutClaim, EpisodeRolloutClaim, PolicyModel, Reference, ReferenceProvider,
    RolloutOutcome,
};
use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::internal;
use gz_engine::{
    CandidateOptions, EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine, ModelVersion,
};
use gz_eval::{EvalOutput, EvalRequest, Evaluator, eval_error_to_engine_error, validate_outputs};
use gz_eval_service::{BackendOutputs, FeatureEvalBackend};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, FeatureSchema, FeatureSchemaHash,
    OpponentBatchRef, OpponentStateFeatures, PositionFeatures, encode_feature_row,
};
use gz_measurer::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasureLedgerSnapshot, MeasuredCompetitiveGame,
    MeasuredEpisode, MeasurerAdmission, MeasurerAdmissionStatus, MeasurerError, MeasurerRunSummary,
    ProjectedReference, ProjectionMode, ReplayMeasurer, ValueTargetConfig,
};
use gz_replay::{ReplayError, ReplayReferenceKind, ReplayStore};
use gz_search::{
    EngineIdentity, EvalModel, GumbelEpisode, GumbelEpisodeContext, GumbelMcts,
    GumbelOpponentContext, GumbelPlayer, GumbelStep, GumbelStopReason, WorkToken,
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

pub struct ReplayRuntime<'a, P> {
    pub store: &'a ReplayStore,
    pub providers: Vec<P>,
    pub backpressure: Option<ReplayBackpressure>,
    /// Break equal-reward games by episode length (shorter wins) before
    /// the coin flip: whittlezero's duration tiebreak, discrete form.
    pub length_tiebreak: bool,
    /// Terminal learner/reference outcome label written to every value row.
    pub value_target: ValueTargetConfig,
}

pub struct FeaturizedRuntime<X, B> {
    pub extractors: Vec<X>,
    /// One batcher thread per backend; lanes are assigned round-robin
    /// (lane % backends.len()). Multiple evaluator processes parallelize
    /// the per-batch host work (decode/stage/encode runs on one thread
    /// per process) and keep the GPU's kernel queue dense.
    pub backends: Vec<B>,
    /// Historical-incumbent evaluators used only by generated-root policy
    /// rollouts. Empty for every legacy/fixed-root path.
    pub reference_backends: Vec<B>,
    /// Pinned arena-challenger evaluators. These are separate from the live
    /// learner so learner checkpoint swaps cannot invalidate an arena gate.
    pub challenger_backends: Vec<B>,
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
    pub reference_steps: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedReplayRun {
    pub lanes: Vec<ReplayLaneSummary>,
    pub batch_sizes: Vec<usize>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub search_contexts: u64,
    pub replay_rows: u64,
    pub reference_steps: u64,
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
    opponent_ref: OpponentBatchRef,
}

struct EvalReply {
    slot: usize,
    token: WorkToken,
    output: EvalOutput,
    route: EvalRoute,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvalRoute {
    Current,
    Incumbent,
    Challenger,
}

enum ReplayJob {
    Episode {
        episode: Box<MeasuredEpisode>,
        ack: SyncSender<EngineResult<MeasurerAdmission>>,
    },
    Competitive {
        game: Box<MeasuredCompetitiveGame>,
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

    pub fn run<R>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
    ) -> EngineResult<ThreadedRun<E::Graph, E::Candidate>>
    where
        R: RootSource<E> + Send,
    {
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
                            context,
                            intake_tx,
                            reference_intake_tx: None,
                            challenger_intake_tx: None,
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

    pub fn run_with_replay<R, P>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
        replay: ReplayRuntime<'_, P>,
    ) -> EngineResult<ThreadedReplayRun>
    where
        R: RootSource<E> + Send,
        P: ReferenceProvider<E> + Send,
    {
        let lanes = self.engines.len();
        if root_sources.len() != lanes || replay.providers.len() != lanes {
            return Err(internal("lane count mismatch"));
        }
        ensure_replay_data_mode::<E, P>(replay.store, &replay.providers, replay.value_target)?;
        validate_engine_identities(&self.engines)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let intake_capacity = lanes * workers_per_lane;
        let (intake_tx, intake_rx) = sync_channel(intake_capacity);
        let (replay_tx, replay_rx) = sync_channel(intake_capacity);
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for _ in 0..lanes {
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
        let providers = replay.providers;
        let store = replay.store;
        let backpressure = replay.backpressure;
        let length_tiebreak = replay.length_tiebreak;
        let value_target = replay.value_target;

        let (batch_result, sink_result, lane_results) = std::thread::scope(|scope| {
            let batch_pressure = Arc::clone(&eval_pressure);
            let batch_handle = scope.spawn(move || {
                run_batcher(evaluator, intake_rx, reply_txs, config, batch_pressure)
            });
            let sink_handle = scope
                .spawn(move || run_replay_sink(store, replay_rx, length_tiebreak, value_target));
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, (((engine, roots), provider), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(providers)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_tx.clone();
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
                            context,
                            intake_tx,
                            reference_intake_tx: None,
                            challenger_intake_tx: None,
                            reply_rx,
                        },
                        ReplayMode::new(
                            lane,
                            lanes,
                            provider,
                            replay_tx,
                            store,
                            backpressure,
                            value_target,
                        ),
                    )
                }));
            }

            drop(intake_tx);
            drop(replay_tx);

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
            let sink_result = sink_handle
                .join()
                .unwrap_or_else(|_| Err(internal("replay sink failed")));

            (batch_result, sink_result, lane_results)
        });

        let batch_sizes = batch_result?;
        let measurer_summary = sink_result?;
        let mut lanes = Vec::with_capacity(lane_results.len());
        let mut search_contexts = 0;
        let mut reference_steps = 0;

        for result in lane_results {
            let mut result = result?;
            merge_lane_measurer_summary(&mut result, &measurer_summary);
            search_contexts += result.search_contexts;
            reference_steps += result.reference_steps;
            lanes.push(result);
        }

        Ok(ThreadedReplayRun {
            lanes,
            batch_sizes,
            episodes_appended: measurer_summary.episodes_appended,
            episodes_dropped: measurer_summary.episodes_dropped,
            search_contexts,
            replay_rows: measurer_summary.replay_rows,
            reference_steps,
            measure_ledger: measurer_summary.measure_ledger,
        })
    }

    pub fn run_featurized<R, X, B>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
        featurized: FeaturizedRuntime<X, B>,
    ) -> EngineResult<ThreadedRun<E::Graph, E::Candidate>>
    where
        R: RootSource<E> + Send,
        X: FeatureExtractor<E> + Send,
        B: FeatureEvalBackend + Send,
    {
        let lanes = self.engines.len();
        if root_sources.len() != lanes || featurized.extractors.len() != lanes {
            return Err(internal("lane count mismatch"));
        }
        validate_engine_identities(&self.engines)?;
        let schema_hash = validate_feature_schemas::<E, X>(&featurized.extractors)?;
        validate_backend_count(featurized.backends.len(), lanes)?;
        validate_reference_backend_count(featurized.reference_backends.len(), lanes)?;
        validate_reference_backend_count(featurized.challenger_backends.len(), lanes)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let intake_capacity = lanes * workers_per_lane;
        let backend_count = featurized.backends.len();
        let mut intake_txs = Vec::with_capacity(backend_count);
        let mut intake_rxs = Vec::with_capacity(backend_count);
        for _ in 0..backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            intake_txs.push(tx);
            intake_rxs.push(rx);
        }
        let reference_backend_count = featurized.reference_backends.len();
        let mut reference_intake_txs = Vec::with_capacity(reference_backend_count);
        let mut reference_intake_rxs = Vec::with_capacity(reference_backend_count);
        for _ in 0..reference_backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            reference_intake_txs.push(tx);
            reference_intake_rxs.push(rx);
        }
        let challenger_backend_count = featurized.challenger_backends.len();
        let mut challenger_intake_txs = Vec::with_capacity(challenger_backend_count);
        let mut challenger_intake_rxs = Vec::with_capacity(challenger_backend_count);
        for _ in 0..challenger_backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            challenger_intake_txs.push(tx);
            challenger_intake_rxs.push(rx);
        }
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for _ in 0..lanes {
            let (tx, rx) = sync_channel(workers_per_lane);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let eval_pressure = Arc::new(EvalPressure::default());
        let admission_shaper =
            build_admission_shaper(lanes, backend_count, config, Arc::clone(&eval_pressure))?;
        let search = &self.search;
        let backends = featurized.backends;
        let reference_backends = featurized.reference_backends;
        let challenger_backends = featurized.challenger_backends;
        let extractors = featurized.extractors;
        let engines = self.engines;
        let feature_schema = first_schema::<E, X>(&extractors, schema_hash)?;
        validate_collator_capacity(
            &FeatureCollator::new(feature_schema.clone(), config.max_batch),
            config,
        )?;
        let _ = self.evaluator;

        let (batch_results, lane_results) = std::thread::scope(|scope| {
            let mut batch_handles = Vec::with_capacity(backend_count);
            for (backend, intake_rx) in backends.into_iter().zip(intake_rxs) {
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
                        EvalRoute::Current,
                        Some(eval_pressure),
                    )
                }));
            }
            for (backend, intake_rx) in reference_backends.into_iter().zip(reference_intake_rxs) {
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
                        EvalRoute::Incumbent,
                        Some(eval_pressure),
                    )
                }));
            }
            for (backend, intake_rx) in challenger_backends.into_iter().zip(challenger_intake_rxs) {
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
                        EvalRoute::Challenger,
                        Some(eval_pressure),
                    )
                }));
            }
            drop(reply_txs);
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, (((engine, roots), extractor), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(extractors)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_txs[lane % backend_count].clone();
                let reference_intake_tx = (!reference_intake_txs.is_empty())
                    .then(|| reference_intake_txs[lane % reference_intake_txs.len()].clone());
                let challenger_intake_tx = (!challenger_intake_txs.is_empty())
                    .then(|| challenger_intake_txs[lane % challenger_intake_txs.len()].clone());
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
                            context,
                            intake_tx,
                            reference_intake_tx,
                            challenger_intake_tx,
                            reply_rx,
                        },
                        FeaturizedCollectMode::new(extractor),
                    )
                }));
            }

            drop(intake_txs);
            drop(reference_intake_txs);
            drop(challenger_intake_txs);

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

            (batch_results, lane_results)
        });

        let mut batch_sizes = Vec::new();
        for result in batch_results {
            batch_sizes.extend(result?);
        }
        let mut lanes = Vec::with_capacity(lane_results.len());

        for result in lane_results {
            lanes.push(result?);
        }

        Ok(ThreadedRun { lanes, batch_sizes })
    }

    pub fn run_featurized_with_replay<R, X, B, P>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
        featurized: FeaturizedRuntime<X, B>,
        replay: ReplayRuntime<'_, P>,
    ) -> EngineResult<ThreadedReplayRun>
    where
        R: RootSource<E> + Send,
        X: FeatureExtractor<E> + Send,
        B: FeatureEvalBackend + Send,
        P: ReferenceProvider<E> + Send,
    {
        let lanes = self.engines.len();
        if root_sources.len() != lanes
            || featurized.extractors.len() != lanes
            || replay.providers.len() != lanes
        {
            return Err(internal("lane count mismatch"));
        }
        ensure_replay_data_mode::<E, P>(replay.store, &replay.providers, replay.value_target)?;
        validate_engine_identities(&self.engines)?;
        let schema_hash = validate_feature_schemas::<E, X>(&featurized.extractors)?;
        validate_backend_count(featurized.backends.len(), lanes)?;
        validate_reference_backend_count(featurized.reference_backends.len(), lanes)?;
        validate_reference_backend_count(featurized.challenger_backends.len(), lanes)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let arena_parallelism = replay
            .providers
            .first()
            .map_or(0, ReferenceProvider::<E>::arena_parallelism);
        if replay
            .providers
            .iter()
            .any(|provider| provider.arena_parallelism() != arena_parallelism)
        {
            return Err(internal("arena parallelism mismatch"));
        }
        let coordinator_capacity = workers_per_lane.max(arena_parallelism);
        let intake_capacity = lanes
            .checked_mul(workers_per_lane)
            .and_then(|capacity| {
                capacity.checked_add(coordinator_capacity.saturating_sub(workers_per_lane))
            })
            .ok_or_else(|| internal("worker count overflow"))?;
        let pool_capacities = (0..lanes)
            .map(|lane| {
                NonZeroUsize::new(if lane == 0 {
                    coordinator_capacity
                } else {
                    workers_per_lane
                })
                .expect("worker configuration is nonzero")
            })
            .collect::<Vec<_>>();
        let mut next_worker_id = 0u64;
        let mut worker_id_bases = Vec::with_capacity(lanes);
        for capacity in &pool_capacities {
            worker_id_bases.push(next_worker_id);
            let capacity =
                u64::try_from(capacity.get()).map_err(|_| internal("worker count overflow"))?;
            next_worker_id = next_worker_id
                .checked_add(capacity)
                .ok_or_else(|| internal("worker count overflow"))?;
        }
        let backend_count = featurized.backends.len();
        let mut intake_txs = Vec::with_capacity(backend_count);
        let mut intake_rxs = Vec::with_capacity(backend_count);
        for _ in 0..backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            intake_txs.push(tx);
            intake_rxs.push(rx);
        }
        let reference_backend_count = featurized.reference_backends.len();
        let mut reference_intake_txs = Vec::with_capacity(reference_backend_count);
        let mut reference_intake_rxs = Vec::with_capacity(reference_backend_count);
        for _ in 0..reference_backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            reference_intake_txs.push(tx);
            reference_intake_rxs.push(rx);
        }
        let challenger_backend_count = featurized.challenger_backends.len();
        let mut challenger_intake_txs = Vec::with_capacity(challenger_backend_count);
        let mut challenger_intake_rxs = Vec::with_capacity(challenger_backend_count);
        for _ in 0..challenger_backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            challenger_intake_txs.push(tx);
            challenger_intake_rxs.push(rx);
        }
        let (replay_tx, replay_rx) = sync_channel(intake_capacity);
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for &capacity in &pool_capacities {
            let (tx, rx) = sync_channel(capacity.get());
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let eval_pressure = Arc::new(EvalPressure::default());
        let admission_shaper =
            build_admission_shaper(lanes, backend_count, config, Arc::clone(&eval_pressure))?;
        let search = &self.search;
        let backends = featurized.backends;
        let reference_backends = featurized.reference_backends;
        let challenger_backends = featurized.challenger_backends;
        let extractors = featurized.extractors;
        let engines = self.engines;
        let providers = replay.providers;
        let store = replay.store;
        let backpressure = replay.backpressure;
        let length_tiebreak = replay.length_tiebreak;
        let value_target = replay.value_target;
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
            for (backend, intake_rx) in backends.into_iter().zip(intake_rxs) {
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
                        EvalRoute::Current,
                        Some(eval_pressure),
                    )
                }));
            }
            for (backend, intake_rx) in reference_backends.into_iter().zip(reference_intake_rxs) {
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
                        EvalRoute::Incumbent,
                        Some(eval_pressure),
                    )
                }));
            }
            for (backend, intake_rx) in challenger_backends.into_iter().zip(challenger_intake_rxs) {
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
                        EvalRoute::Challenger,
                        Some(eval_pressure),
                    )
                }));
            }
            drop(reply_txs);
            let sink_handle = scope
                .spawn(move || run_replay_sink(store, replay_rx, length_tiebreak, value_target));
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, ((((engine, roots), extractor), provider), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(extractors)
                .zip(providers)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_txs[lane % backend_count].clone();
                let reference_intake_tx = (!reference_intake_txs.is_empty())
                    .then(|| reference_intake_txs[lane % reference_intake_txs.len()].clone());
                let challenger_intake_tx = (!challenger_intake_txs.is_empty())
                    .then(|| challenger_intake_txs[lane % challenger_intake_txs.len()].clone());
                let replay_tx = replay_tx.clone();
                let eval_pressure = Arc::clone(&eval_pressure);
                let admission_shaper = admission_shaper.clone();
                let pool_capacity = pool_capacities[lane];
                let worker_id_base = worker_id_bases[lane];
                lane_handles.push(scope.spawn(move || {
                    run_lane_pipeline(
                        engine,
                        roots,
                        LaneRuntime {
                            lane,
                            lanes,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            pool_capacity,
                            worker_id_base,
                            admission_stagger: config.admission_stagger,
                            admission_shaper,
                            eval_pressure,
                            context,
                            intake_tx,
                            reference_intake_tx,
                            challenger_intake_tx,
                            reply_rx,
                        },
                        FeaturizedReplayMode::new(
                            lane,
                            lanes,
                            extractor,
                            provider.sampled_tree_mode(),
                            provider.sampled_trajectory_mode(),
                            provider.per_root_policy_mode(),
                            provider,
                            replay_tx,
                            store,
                            backpressure,
                            value_target,
                        ),
                    )
                }));
            }

            drop(intake_txs);
            drop(reference_intake_txs);
            drop(challenger_intake_txs);
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
        let mut lanes = Vec::with_capacity(lane_results.len());
        let mut search_contexts = 0;
        let mut reference_steps = 0;

        for result in lane_results {
            let mut result = result?;
            merge_lane_measurer_summary(&mut result, &measurer_summary);
            search_contexts += result.search_contexts;
            reference_steps += result.reference_steps;
            lanes.push(result);
        }

        Ok(ThreadedReplayRun {
            lanes,
            batch_sizes,
            episodes_appended: measurer_summary.episodes_appended,
            episodes_dropped: measurer_summary.episodes_dropped,
            search_contexts,
            replay_rows: measurer_summary.replay_rows,
            reference_steps,
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
    context: GumbelEpisodeContext,
    intake_tx: SyncSender<J>,
    reference_intake_tx: Option<SyncSender<J>>,
    challenger_intake_tx: Option<SyncSender<J>>,
    reply_rx: Receiver<EvalReply>,
}

struct EpisodeFeatureRows<C> {
    rows: Vec<Vec<u8>>,
    candidates: Vec<C>,
}

struct CompetitiveFeatureRows<C> {
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

    fn begin(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        context: GumbelEpisodeContext,
    ) {
        let _ = (search, identity, context);
    }

    fn before_root_admission<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        next_episode_id: &mut u64,
    ) -> EngineResult<()>
    where
        R: RootSource<E>,
    {
        let _ = (pool, engine, roots, next_episode_id);
        Ok(())
    }

    fn admit_ready_tasks(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        search: &GumbelMcts,
        identity: EngineIdentity,
        _context: GumbelEpisodeContext,
        limit: usize,
    ) -> EngineResult<()> {
        let _ = (pool, search, identity, limit);
        Ok(())
    }

    fn learner_admission_slots(
        &self,
        pool: &WorkerPool<E::Graph, E::Candidate>,
        workers_per_lane: usize,
    ) -> usize {
        available_learner_slots(pool, workers_per_lane)
    }

    fn has_pending_tasks(&self) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_roots<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        search: &GumbelMcts,
        identity: EngineIdentity,
        context: GumbelEpisodeContext,
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
            context,
            sampled_tree: false,
            pressure_reserved,
            next_episode_id,
        };
        pool.admit_limited(
            engine,
            roots,
            &mut admission,
            limit,
            |engine, id, root, context| self.episode_context(engine, id, root, context),
        )
    }

    fn gate_open(&self) -> bool {
        true
    }

    /// Whether learner episodes may be admitted right now. False while
    /// a rollout-backed provider has no reference yet: the seed rollout
    /// (admitted in before_root_admission) plays first, so no episode
    /// is ever admitted unlabeled.
    fn admission_open(&self) -> bool {
        true
    }

    fn gate_poll(&self) -> Option<Duration> {
        None
    }

    fn episode_context(
        &mut self,
        engine: &mut E,
        episode_id: EpisodeId,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisodeContext> {
        let _ = (engine, episode_id, root);
        Ok(context)
    }

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>>;

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
        reference_intake_tx: Option<&SyncSender<Self::Job>>,
        challenger_intake_tx: Option<&SyncSender<Self::Job>>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()>;

    fn observe_version(&mut self, version: ModelVersion) {
        let _ = version;
    }

    fn complete(
        &mut self,
        engine: &mut E,
        search: &GumbelMcts,
        completed: OrchestratedEpisode<E::Graph, E::Candidate>,
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
    mode.begin(runtime.search, identity, runtime.context);

    loop {
        let mut adaptive_retry_after = None;
        if !roots_exhausted {
            mode.before_root_admission(&mut pool, &mut engine, &mut roots, &mut next_episode_id)?;
        }
        let ready_slots = mode.learner_admission_slots(&pool, runtime.workers_per_lane.get());
        mode.admit_ready_tasks(
            &mut pool,
            runtime.search,
            identity,
            runtime.context,
            ready_slots,
        )?;
        if !roots_exhausted {
            let admission_open = mode.gate_open() && mode.admission_open();
            if admission_open {
                let learner_slots =
                    mode.learner_admission_slots(&pool, runtime.workers_per_lane.get());
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
                            runtime.context,
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
                        runtime.context,
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
            if !admission_open
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
            runtime.reference_intake_tx.as_ref(),
            runtime.challenger_intake_tx.as_ref(),
            &runtime.eval_pressure,
        )?;

        if roots_exhausted && !pool.active() && !mode.has_pending_tasks() {
            if let Some(shaper) = &runtime.admission_shaper {
                shaper.clear_lane(runtime.lane)?;
            }
            return Ok(mode.finish(runtime.lane));
        }

        let reply_wait = adaptive_retry_after.filter(|_| {
            !roots_exhausted
                && mode.learner_admission_slots(&pool, runtime.workers_per_lane.get()) > 0
        });
        if pool.has_parked()
            && let Some(version) =
                receive_replies(&mut engine, &mut pool, &runtime.reply_rx, reply_wait)?
        {
            mode.observe_version(version);
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
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>> {
        pool.drive(engine, "worker blocked", None, |_, _, _| {})
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
        _reference_intake_tx: Option<&SyncSender<Self::Job>>,
        _challenger_intake_tx: Option<&SyncSender<Self::Job>>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()> {
        send_plain_parked(lane, pool, intake_tx, eval_pressure)
    }

    fn complete(
        &mut self,
        engine: &mut E,
        _search: &GumbelMcts,
        completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<Option<u64>> {
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

struct FeaturizedCollectMode<X, G, C> {
    extractor: X,
    episodes: Vec<OrchestratedEpisode<G, C>>,
}

impl<X, G, C> FeaturizedCollectMode<X, G, C> {
    fn new(extractor: X) -> Self {
        Self {
            extractor,
            episodes: Vec::new(),
        }
    }
}

impl<E, X> LaneMode<E> for FeaturizedCollectMode<X, E::Graph, E::Candidate>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    type Job = FeaturizedEvalJob;
    type Output = LaneEpisodes<E::Graph, E::Candidate>;

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>> {
        pool.drive(
            engine,
            "worker blocked",
            Some(&mut self.extractor),
            |_, _, _| {},
        )
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
        _reference_intake_tx: Option<&SyncSender<Self::Job>>,
        _challenger_intake_tx: Option<&SyncSender<Self::Job>>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()> {
        send_featurized_parked(lane, pool, intake_tx, eval_pressure)
    }

    fn complete(
        &mut self,
        engine: &mut E,
        _search: &GumbelMcts,
        completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<Option<u64>> {
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

struct ReplayMode<'a, P> {
    lane: usize,
    lanes: usize,
    provider: P,
    replay_tx: SyncSender<ReplayJob>,
    store: &'a ReplayStore,
    backpressure: Option<ReplayBackpressure>,
    value_target: ValueTargetConfig,
    references: HashMap<EpisodeId, Option<Reference>>,
    admitted_at: HashMap<EpisodeId, Instant>,
    summary: ReplayLaneSummary,
    rollout: Option<OpponentRollout>,
}

impl<'a, P> ReplayMode<'a, P> {
    fn new(
        lane: usize,
        lanes: usize,
        provider: P,
        replay_tx: SyncSender<ReplayJob>,
        store: &'a ReplayStore,
        backpressure: Option<ReplayBackpressure>,
        value_target: ValueTargetConfig,
    ) -> Self {
        Self {
            lane,
            lanes,
            provider,
            replay_tx,
            store,
            backpressure,
            value_target,
            references: HashMap::new(),
            admitted_at: HashMap::new(),
            summary: ReplayLaneSummary {
                lane,
                episodes_completed: 0,
                episodes_appended: 0,
                episodes_dropped: 0,
                search_contexts: 0,
                replay_rows: 0,
                reference_steps: 0,
            },
            rollout: None,
        }
    }
}

impl<E, P> LaneMode<E> for ReplayMode<'_, P>
where
    E: GraphEngine,
    P: ReferenceProvider<E>,
{
    type Job = EvalJob;
    type Output = ReplayLaneSummary;

    fn begin(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        context: GumbelEpisodeContext,
    ) {
        let arena_parallelism = self.provider.arena_parallelism();
        self.rollout = Some(OpponentRollout::new(
            search,
            identity,
            context,
            self.lane,
            self.lanes,
            arena_parallelism,
        ));
    }

    fn before_root_admission<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        next_episode_id: &mut u64,
    ) -> EngineResult<()>
    where
        R: RootSource<E>,
    {
        let mut rollout = self
            .rollout
            .take()
            .ok_or_else(|| internal("missing opponent rollout"))?;
        let result = rollout.try_admit(pool, engine, roots, &mut self.provider, next_episode_id);
        self.rollout = Some(rollout);
        result
    }

    fn gate_open(&self) -> bool {
        replay_gate_open(self.store, self.backpressure)
    }

    fn admission_open(&self) -> bool {
        self.provider.admission_ready()
    }

    fn learner_admission_slots(
        &self,
        pool: &WorkerPool<E::Graph, E::Candidate>,
        workers_per_lane: usize,
    ) -> usize {
        if self
            .rollout
            .as_ref()
            .is_some_and(OpponentRollout::arena_active)
        {
            0
        } else {
            available_learner_slots(pool, workers_per_lane)
        }
    }

    fn gate_poll(&self) -> Option<Duration> {
        self.backpressure.map(|backpressure| backpressure.gate_poll)
    }

    fn episode_context(
        &mut self,
        engine: &mut E,
        episode_id: EpisodeId,
        root: E::Graph,
        mut context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisodeContext> {
        let reference = self.provider.reference(engine, root)?;
        context.opponent = reference.as_ref().map(opponent_context);
        self.references.insert(episode_id, reference);
        self.admitted_at.insert(episode_id, Instant::now());
        Ok(context)
    }

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>> {
        pool.drive(engine, "worker blocked", None, |_, _, _| {})
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
        _reference_intake_tx: Option<&SyncSender<Self::Job>>,
        _challenger_intake_tx: Option<&SyncSender<Self::Job>>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()> {
        send_plain_parked(lane, pool, intake_tx, eval_pressure)
    }

    fn observe_version(&mut self, version: ModelVersion) {
        if let Some(rollout) = &mut self.rollout {
            rollout.observe_version(version);
        }
    }

    fn complete(
        &mut self,
        engine: &mut E,
        search: &GumbelMcts,
        mut completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<Option<u64>> {
        let mut rollout = self
            .rollout
            .take()
            .ok_or_else(|| internal("missing opponent rollout"))?;
        if rollout.intercept(engine, &mut self.provider, &completed)? {
            self.rollout = Some(rollout);
            return Ok(None);
        }
        self.rollout = Some(rollout);

        let reference = self
            .references
            .remove(&completed.episode_id)
            .ok_or_else(|| internal("missing replay reference"))?;
        if let Some(admitted_at) = self.admitted_at.remove(&completed.episode_id) {
            self.store
                .observe_episode_latency(admitted_at.elapsed().as_secs_f64());
        }
        self.summary.episodes_completed += 1;
        self.summary.search_contexts += episode_search_contexts(&completed.episode);
        self.summary.reference_steps += reference
            .as_ref()
            .map_or(0, |reference| reference.steps.len() as u64);
        let root_reward = match value_target_root_reward(
            engine,
            completed.episode.root,
            search.config().measure_options,
            self.value_target,
        ) {
            Ok(reward) => reward,
            Err(error) => {
                release_episode_handles(engine, &completed.episode, &[])?;
                return Err(error);
            }
        };

        let episode = measured_episode(
            self.lane,
            completed.episode_id.value(),
            &completed.episode,
            root_reward,
            reference.as_ref(),
            None,
            self.provider.expects_reference(),
        );
        let append = append_replay_job(&self.replay_tx, episode);
        release_episode_handles(engine, &completed.episode, &[])?;
        let admission = append?;
        if should_observe_admission(admission)
            && let Some(reward) = admission.learner_reward
        {
            self.provider.observe(reward);
        }

        clear_replayed_episode_trace(&mut completed.episode);
        Ok(Some(completed.evaluations))
    }

    fn finish(mut self, lane: usize) -> Self::Output {
        self.summary.lane = lane;
        self.summary
    }
}

struct FeaturizedReplayMode<'a, X, P, G> {
    extractor: X,
    replay: ReplayMode<'a, P>,
    candidate_options: CandidateOptions,
    export_position: bool,
    sampled_tree: bool,
    sampled_trajectory: Option<SampledTrajectoryState<G>>,
    per_root_policy: Option<PerRootPolicyState<G>>,
    root_evaluations: HashMap<EpisodeId, u64>,
}

struct SampledTrajectoryState<G> {
    search: Option<GumbelMcts>,
    in_flight: HashMap<EpisodeId, bool>,
    ready: VecDeque<ReadySampledLearner<G>>,
}

impl<G> Default for SampledTrajectoryState<G> {
    fn default() -> Self {
        Self {
            search: None,
            in_flight: HashMap::new(),
            ready: VecDeque::new(),
        }
    }
}

struct ReadySampledLearner<G> {
    episode_id: EpisodeId,
    root: G,
    owned_root: bool,
    reference: Reference,
}

struct PerRootPolicyState<G> {
    search: Option<GumbelMcts>,
    in_flight: HashMap<EpisodeId, PerRootPrelude>,
    ready: VecDeque<ReadySampledLearner<G>>,
    retry: VecDeque<RetryPolicyPrelude<G>>,
}

impl<G> Default for PerRootPolicyState<G> {
    fn default() -> Self {
        Self {
            search: None,
            in_flight: HashMap::new(),
            ready: VecDeque::new(),
            retry: VecDeque::new(),
        }
    }
}

struct RetryPolicyPrelude<G> {
    episode_id: EpisodeId,
    root: G,
    owned_root: bool,
    pressure_reserved: bool,
}

struct PerRootPrelude {
    claim: EpisodeRolloutClaim,
    owned_root: bool,
}

impl<'a, X, P, G> FeaturizedReplayMode<'a, X, P, G> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        lane: usize,
        lanes: usize,
        extractor: X,
        sampled_tree: bool,
        sampled_trajectory: bool,
        per_root_policy: bool,
        provider: P,
        replay_tx: SyncSender<ReplayJob>,
        store: &'a ReplayStore,
        backpressure: Option<ReplayBackpressure>,
        value_target: ValueTargetConfig,
    ) -> Self {
        Self {
            extractor,
            replay: ReplayMode::new(
                lane,
                lanes,
                provider,
                replay_tx,
                store,
                backpressure,
                value_target,
            ),
            candidate_options: CandidateOptions::default(),
            export_position: true,
            sampled_tree,
            sampled_trajectory: sampled_trajectory.then(SampledTrajectoryState::default),
            per_root_policy: per_root_policy.then(PerRootPolicyState::default),
            root_evaluations: HashMap::new(),
        }
    }
}

impl<E, X, P> LaneMode<E> for FeaturizedReplayMode<'_, X, P, E::Graph>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
    P: ReferenceProvider<E>,
{
    type Job = FeaturizedEvalJob;
    type Output = ReplayLaneSummary;

    fn begin(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        context: GumbelEpisodeContext,
    ) {
        self.replay.begin(search, identity, context);
        self.candidate_options = search.config().candidate_options;
        self.export_position = search.config().export_position;
        if let Some(sampled) = &mut self.sampled_trajectory {
            sampled.search = Some(search.categorical_policy_rollout());
        }
        if let Some(per_root) = &mut self.per_root_policy {
            per_root.search = Some(search.policy_rollout());
        }
    }

    fn admit_ready_tasks(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        search: &GumbelMcts,
        identity: EngineIdentity,
        _context: GumbelEpisodeContext,
        limit: usize,
    ) -> EngineResult<()> {
        let mut remaining = limit.min(pool.idle_count());
        while remaining > 0 {
            let retry = self
                .per_root_policy
                .as_mut()
                .and_then(|per_root| per_root.retry.pop_front());
            let Some(retry) = retry else {
                break;
            };
            let latest = self
                .replay
                .rollout
                .as_ref()
                .and_then(OpponentRollout::latest_version);
            let Some(claim) = self.replay.provider.claim_per_root_policy(latest) else {
                self.per_root_policy
                    .as_mut()
                    .expect("per-root policy state exists")
                    .retry
                    .push_front(retry);
                break;
            };
            let per_root = self
                .per_root_policy
                .as_mut()
                .expect("per-root policy state exists");
            let policy_search = per_root
                .search
                .as_ref()
                .ok_or_else(|| internal("missing per-root policy search"))?;
            if !pool.admit_direct(
                policy_search,
                identity,
                retry.root,
                GumbelEpisodeContext::default(),
                retry.episode_id,
                false,
                retry.pressure_reserved,
            ) {
                per_root.retry.push_front(retry);
                break;
            }
            per_root.in_flight.insert(
                retry.episode_id,
                PerRootPrelude {
                    claim,
                    owned_root: retry.owned_root,
                },
            );
            remaining -= 1;
        }

        while remaining > 0 {
            let (ready, pressure_reserved) = if let Some(per_root) = &mut self.per_root_policy {
                (per_root.ready.pop_front(), false)
            } else if let Some(sampled) = &mut self.sampled_trajectory {
                (sampled.ready.pop_front(), false)
            } else {
                (None, false)
            };
            let Some(ready) = ready else {
                break;
            };
            let learner_context = GumbelEpisodeContext {
                noise_seed: crate::root::episode_noise_seed(ready.episode_id.value()),
                opponent: Some(opponent_context(&ready.reference)),
            };
            let admitted = pool.admit_direct(
                search,
                identity,
                ready.root,
                learner_context,
                ready.episode_id,
                ready.owned_root,
                pressure_reserved,
            );
            if !admitted {
                if let Some(per_root) = &mut self.per_root_policy {
                    per_root.ready.push_front(ready);
                } else {
                    self.sampled_trajectory
                        .as_mut()
                        .expect("sampled trajectory state exists")
                        .ready
                        .push_front(ready);
                }
                break;
            }
            self.replay
                .references
                .insert(ready.episode_id, Some(ready.reference));
            remaining -= 1;
        }
        Ok(())
    }

    fn learner_admission_slots(
        &self,
        pool: &WorkerPool<E::Graph, E::Candidate>,
        workers_per_lane: usize,
    ) -> usize {
        if self
            .replay
            .rollout
            .as_ref()
            .is_some_and(OpponentRollout::arena_active)
        {
            0
        } else {
            available_learner_slots(pool, workers_per_lane)
        }
    }

    fn has_pending_tasks(&self) -> bool {
        self.sampled_trajectory
            .as_ref()
            .is_some_and(|sampled| !sampled.in_flight.is_empty() || !sampled.ready.is_empty())
            || self.per_root_policy.as_ref().is_some_and(|per_root| {
                !per_root.in_flight.is_empty()
                    || !per_root.ready.is_empty()
                    || !per_root.retry.is_empty()
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_roots<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        search: &GumbelMcts,
        identity: EngineIdentity,
        context: GumbelEpisodeContext,
        next_episode_id: &mut u64,
        limit: usize,
        pressure_reserved: bool,
    ) -> EngineResult<AdmissionResult>
    where
        R: RootSource<E>,
    {
        if self.sampled_trajectory.is_none() && self.per_root_policy.is_none() {
            let mut admission = Admission {
                search,
                identity,
                context,
                sampled_tree: self.sampled_tree,
                pressure_reserved,
                next_episode_id,
            };
            return pool.admit_limited(
                engine,
                roots,
                &mut admission,
                limit,
                |engine, id, root, context| self.episode_context(engine, id, root, context),
            );
        }
        if limit == 0 {
            return Ok(AdmissionResult {
                roots_exhausted: false,
                admitted: 0,
            });
        }

        let mut admitted = 0;
        while admitted < limit && pool.idle_count() > 0 {
            let owned_root = roots.episode_roots_are_owned();
            let Some(root) = roots.next_root(engine)? else {
                return Ok(AdmissionResult {
                    roots_exhausted: true,
                    admitted,
                });
            };
            let episode_id = EpisodeId::new(*next_episode_id);
            if let Some(per_root) = &mut self.per_root_policy {
                let latest = self
                    .replay
                    .rollout
                    .as_ref()
                    .and_then(OpponentRollout::latest_version);
                let Some(claim) = self.replay.provider.claim_per_root_policy(latest) else {
                    per_root.retry.push_back(RetryPolicyPrelude {
                        episode_id,
                        root,
                        owned_root,
                        pressure_reserved,
                    });
                    self.replay.admitted_at.insert(episode_id, Instant::now());
                    *next_episode_id += 1;
                    admitted += 1;
                    continue;
                };
                let policy_search = per_root
                    .search
                    .as_ref()
                    .ok_or_else(|| internal("missing per-root policy search"))?;
                if !pool.admit_direct(
                    policy_search,
                    identity,
                    root,
                    GumbelEpisodeContext::default(),
                    episode_id,
                    false,
                    pressure_reserved,
                ) {
                    return Err(internal("per-root policy admission lost idle slot"));
                }
                per_root
                    .in_flight
                    .insert(episode_id, PerRootPrelude { claim, owned_root });
            } else {
                let sampled = self
                    .sampled_trajectory
                    .as_mut()
                    .expect("sampled trajectory mode checked");
                let sampled_search = sampled
                    .search
                    .as_ref()
                    .ok_or_else(|| internal("missing categorical policy search"))?;
                if !pool.admit_direct_categorical(
                    sampled_search,
                    identity,
                    root,
                    GumbelEpisodeContext {
                        noise_seed: sampled_trajectory_seed(episode_id),
                        opponent: None,
                    },
                    episode_id,
                    false,
                    pressure_reserved,
                ) {
                    return Err(internal("sampled trajectory admission lost idle slot"));
                }
                sampled.in_flight.insert(episode_id, owned_root);
            }
            self.replay.admitted_at.insert(episode_id, Instant::now());
            *next_episode_id += 1;
            admitted += 1;
        }
        Ok(AdmissionResult {
            roots_exhausted: false,
            admitted,
        })
    }

    fn before_root_admission<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        next_episode_id: &mut u64,
    ) -> EngineResult<()>
    where
        R: RootSource<E>,
    {
        self.replay
            .before_root_admission(pool, engine, roots, next_episode_id)
    }

    fn gate_open(&self) -> bool {
        self.replay.gate_open()
    }

    fn admission_open(&self) -> bool {
        self.replay.provider.admission_ready()
    }

    fn gate_poll(&self) -> Option<Duration> {
        self.replay.gate_poll()
    }

    fn episode_context(
        &mut self,
        engine: &mut E,
        episode_id: EpisodeId,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisodeContext> {
        let reference = self.replay.provider.reference_with_features(
            engine,
            root,
            &mut self.extractor,
            self.candidate_options,
            self.export_position,
        )?;
        let mut context = context;
        context.opponent = reference.as_ref().map(opponent_context);
        self.replay.references.insert(episode_id, reference);
        self.replay.admitted_at.insert(episode_id, Instant::now());
        Ok(context)
    }

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>> {
        let references = &self.replay.references;
        pool.drive(
            engine,
            "worker blocked",
            Some(&mut self.extractor),
            |episode_id, root_step, row| {
                attach_reference_opponent(references, episode_id, root_step, row);
            },
        )
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
        reference_intake_tx: Option<&SyncSender<Self::Job>>,
        challenger_intake_tx: Option<&SyncSender<Self::Job>>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()> {
        let rollout = self.replay.rollout.as_ref();
        let per_root = self.per_root_policy.as_ref();
        send_featurized_parked_routed(
            lane,
            pool,
            intake_tx,
            reference_intake_tx,
            challenger_intake_tx,
            eval_pressure,
            |episode_id| {
                rollout
                    .and_then(|rollout| rollout.eval_route(episode_id))
                    .or_else(|| {
                        per_root
                            .and_then(|state| state.in_flight.get(&episode_id))
                            .map(|prelude| policy_eval_route(prelude.claim.model))
                    })
                    .unwrap_or(EvalRoute::Current)
            },
        )
    }

    fn observe_version(&mut self, version: ModelVersion) {
        self.replay.observe_version(version);
    }

    fn complete(
        &mut self,
        engine: &mut E,
        search: &GumbelMcts,
        mut completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<Option<u64>> {
        let mut rollout = self
            .replay
            .rollout
            .take()
            .ok_or_else(|| internal("missing opponent rollout"))?;
        if rollout.intercept_with_features(
            engine,
            &mut self.replay.provider,
            &completed,
            &mut self.extractor,
        )? {
            self.replay.rollout = Some(rollout);
            return Ok(None);
        }
        self.replay.rollout = Some(rollout);

        let per_root_claim = self
            .per_root_policy
            .as_mut()
            .and_then(|per_root| per_root.in_flight.remove(&completed.episode_id));
        if let Some(prelude) = per_root_claim {
            let evaluations = self
                .root_evaluations
                .entry(completed.episode_id)
                .or_default();
            *evaluations = evaluations.saturating_add(completed.evaluations);
            let measure = &completed.episode.final_measure;
            let reward = (measure.measured && measure.valid)
                .then_some(measure.scalar_reward)
                .flatten()
                .filter(|reward| reward.is_finite());
            let projection = match reward {
                Some(final_reward) => reference_steps_for_gumbel_episode_with_features(
                    engine,
                    &mut self.extractor,
                    search,
                    &completed.episode,
                    final_reward,
                ),
                None => Ok((Vec::new(), Vec::new())),
            };
            let (steps, feature_candidates) = match projection {
                Ok(projected) => projected,
                Err(error) => {
                    release_episode_handles(engine, &completed.episode, &[])?;
                    return Err(error);
                }
            };
            release_episode_handles(engine, &completed.episode, &feature_candidates)?;
            let outcome = reward.map(|final_reward| RolloutOutcome {
                final_reward,
                final_graph: completed.episode.final_context,
                steps,
                search_config_hash: completed.episode.search_config_hash,
                model_version: rollout_model_version(&completed.episode),
            });
            if let Some(reference) = self
                .replay
                .provider
                .finish_per_root_policy(prelude.claim, outcome)
            {
                self.per_root_policy
                    .as_mut()
                    .expect("per-root policy state exists")
                    .ready
                    .push_back(ReadySampledLearner {
                        episode_id: completed.episode_id,
                        root: completed.episode.root,
                        owned_root: prelude.owned_root,
                        reference,
                    });
            } else {
                self.per_root_policy
                    .as_mut()
                    .expect("per-root policy state exists")
                    .retry
                    .push_back(RetryPolicyPrelude {
                        episode_id: completed.episode_id,
                        root: completed.episode.root,
                        owned_root: prelude.owned_root,
                        pressure_reserved: false,
                    });
            }
            return Ok(None);
        }

        let sampled_prelude = self
            .sampled_trajectory
            .as_mut()
            .and_then(|sampled| sampled.in_flight.remove(&completed.episode_id));
        if let Some(owned_root) = sampled_prelude {
            let evaluations = self
                .root_evaluations
                .entry(completed.episode_id)
                .or_default();
            *evaluations = evaluations.saturating_add(completed.evaluations);
            let measure = &completed.episode.final_measure;
            let reward = (measure.measured && measure.valid)
                .then_some(measure.scalar_reward)
                .flatten()
                .filter(|reward| reward.is_finite());
            let projection = match reward {
                Some(final_reward) => reference_steps_for_gumbel_episode_with_features(
                    engine,
                    &mut self.extractor,
                    search,
                    &completed.episode,
                    final_reward,
                ),
                None => Ok((Vec::new(), Vec::new())),
            };
            let (steps, feature_candidates) = match projection {
                Ok(projected) => projected,
                Err(error) => {
                    release_episode_handles(engine, &completed.episode, &[])?;
                    return Err(error);
                }
            };
            release_episode_handles(engine, &completed.episode, &feature_candidates)?;
            let outcome = reward.map(|final_reward| RolloutOutcome {
                final_reward,
                final_graph: completed.episode.final_context,
                steps,
                search_config_hash: completed.episode.search_config_hash,
                model_version: rollout_model_version(&completed.episode),
            });
            if let Some(reference) = self.replay.provider.finish_sampled_trajectory(outcome) {
                self.sampled_trajectory
                    .as_mut()
                    .expect("sampled trajectory state exists")
                    .ready
                    .push_back(ReadySampledLearner {
                        episode_id: completed.episode_id,
                        root: completed.episode.root,
                        owned_root,
                        reference,
                    });
            } else {
                if owned_root {
                    engine.release(&[completed.episode.root], &[])?;
                }
                self.replay.admitted_at.remove(&completed.episode_id);
                self.root_evaluations.remove(&completed.episode_id);
            }
            return Ok(None);
        }

        if completed.episode.competitive.is_some() {
            if !self.sampled_tree {
                release_episode_handles(engine, &completed.episode, &[])?;
                return Err(internal("competitive episode outside sampled-tree mode"));
            }
            if self
                .replay
                .references
                .remove(&completed.episode_id)
                .is_none()
            {
                release_episode_handles(engine, &completed.episode, &[])?;
                return Err(internal("missing replay reference"));
            }
            if let Some(admitted_at) = self.replay.admitted_at.remove(&completed.episode_id) {
                self.replay
                    .store
                    .observe_episode_latency(admitted_at.elapsed().as_secs_f64());
            }
            let root_reward = match value_target_root_reward(
                engine,
                completed.episode.root,
                search.config().measure_options,
                self.replay.value_target,
            ) {
                Ok(reward) => reward,
                Err(error) => {
                    release_episode_handles(engine, &completed.episode, &[])?;
                    return Err(error);
                }
            };
            let feature_rows = match feature_rows_for_competitive_episode(
                engine,
                &mut self.extractor,
                search,
                &completed.episode,
            ) {
                Ok(rows) => rows,
                Err(error) => {
                    release_episode_handles(engine, &completed.episode, &[])?;
                    return Err(error);
                }
            };
            let game = match measured_competitive_game(
                self.replay.lane,
                completed.episode_id.value(),
                &completed.episode,
                root_reward,
                &feature_rows,
            ) {
                Ok(game) => game,
                Err(error) => {
                    release_episode_handles(engine, &completed.episode, &feature_rows.candidates)?;
                    return Err(error);
                }
            };
            self.replay.summary.episodes_completed += 1;
            self.replay.summary.search_contexts += episode_search_contexts(&completed.episode);
            self.replay.summary.reference_steps += completed
                .episode
                .competitive
                .as_deref()
                .map_or(0, |trace| trace.opponent_steps.len() as u64);

            let append = append_competitive_replay_job(&self.replay.replay_tx, game);
            release_episode_handles(engine, &completed.episode, &feature_rows.candidates)?;
            let admission = append?;
            if should_observe_admission(admission)
                && let Some(reward) = admission.learner_reward
            {
                self.replay.provider.observe(reward);
            }
            clear_replayed_episode_trace(&mut completed.episode);
            let evaluations = self
                .root_evaluations
                .remove(&completed.episode_id)
                .unwrap_or(0)
                .saturating_add(completed.evaluations);
            return Ok(Some(evaluations));
        }

        let reference = self
            .replay
            .references
            .remove(&completed.episode_id)
            .ok_or_else(|| internal("missing replay reference"))?;
        if let Some(admitted_at) = self.replay.admitted_at.remove(&completed.episode_id) {
            self.replay
                .store
                .observe_episode_latency(admitted_at.elapsed().as_secs_f64());
        }
        let feature_rows = feature_rows_for_episode(
            engine,
            &mut self.extractor,
            search,
            &completed.episode,
            reference.as_ref(),
        )?;
        let root_reward = match value_target_root_reward(
            engine,
            completed.episode.root,
            search.config().measure_options,
            self.replay.value_target,
        ) {
            Ok(reward) => reward,
            Err(error) => {
                release_episode_handles(engine, &completed.episode, &feature_rows.candidates)?;
                return Err(error);
            }
        };
        self.replay.summary.episodes_completed += 1;
        self.replay.summary.search_contexts += episode_search_contexts(&completed.episode);
        self.replay.summary.reference_steps += reference
            .as_ref()
            .map_or(0, |reference| reference.steps.len() as u64);

        let episode = measured_episode(
            self.replay.lane,
            completed.episode_id.value(),
            &completed.episode,
            root_reward,
            reference.as_ref(),
            Some(&feature_rows.rows),
            self.replay.provider.expects_reference(),
        );
        let append = append_replay_job(&self.replay.replay_tx, episode);
        release_episode_handles(engine, &completed.episode, &feature_rows.candidates)?;
        let admission = append?;
        if should_observe_admission(admission)
            && let Some(reward) = admission.learner_reward
        {
            self.replay.provider.observe(reward);
        }

        clear_replayed_episode_trace(&mut completed.episode);
        let evaluations = self
            .root_evaluations
            .remove(&completed.episode_id)
            .unwrap_or(0)
            .saturating_add(completed.evaluations);
        Ok(Some(evaluations))
    }

    fn finish(mut self, lane: usize) -> Self::Output {
        self.replay.summary.lane = lane;
        self.replay.summary
    }
}

fn send_featurized_parked<G, C>(
    lane: usize,
    pool: &mut WorkerPool<G, C>,
    intake_tx: &SyncSender<FeaturizedEvalJob>,
    eval_pressure: &EvalPressure,
) -> EngineResult<()>
where
    G: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
{
    send_featurized_parked_routed(lane, pool, intake_tx, None, None, eval_pressure, |_| {
        EvalRoute::Current
    })
}

fn send_featurized_parked_routed<G, C, F>(
    lane: usize,
    pool: &mut WorkerPool<G, C>,
    intake_tx: &SyncSender<FeaturizedEvalJob>,
    reference_intake_tx: Option<&SyncSender<FeaturizedEvalJob>>,
    challenger_intake_tx: Option<&SyncSender<FeaturizedEvalJob>>,
    eval_pressure: &EvalPressure,
    mut route: F,
) -> EngineResult<()>
where
    G: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
    F: FnMut(EpisodeId) -> EvalRoute,
{
    for parked in pool.take_unsent_parked() {
        let row = parked.row.ok_or_else(|| internal("missing feature row"))?;
        let opponent_ref =
            parked
                .request
                .position
                .opponent
                .map_or_else(OpponentBatchRef::default, |opponent| OpponentBatchRef {
                    trajectory_id: opponent.trajectory_id,
                    row: opponent.row,
                });
        let episode_route = route(parked.episode_id);
        let eval_route = match parked.model {
            EvalModel::Episode => episode_route,
            EvalModel::Current => EvalRoute::Current,
            EvalModel::Incumbent => EvalRoute::Incumbent,
        };
        let destination = match eval_route {
            EvalRoute::Current => intake_tx,
            EvalRoute::Incumbent => {
                reference_intake_tx.ok_or_else(|| internal("missing incumbent eval backend"))?
            }
            EvalRoute::Challenger => {
                challenger_intake_tx.ok_or_else(|| internal("missing challenger eval backend"))?
            }
        };
        if parked.pressure_reserved {
            pool.consume_pressure_reservation(parked.slot, parked.token)?;
        }
        eval_pressure.submit(parked.pressure_reserved);
        if destination
            .send(FeaturizedEvalJob {
                lane,
                slot: parked.slot,
                token: parked.token,
                row,
                action_count: parked.action_count,
                opponent_ref,
            })
            .is_err()
        {
            eval_pressure.cancel_submission();
            return Err(internal("eval backend unavailable"));
        }
    }
    Ok(())
}

const fn policy_eval_route(model: PolicyModel) -> EvalRoute {
    match model {
        PolicyModel::Current => EvalRoute::Current,
        PolicyModel::Incumbent => EvalRoute::Incumbent,
        PolicyModel::Challenger => EvalRoute::Challenger,
    }
}

fn episode_search_contexts<G, C>(episode: &GumbelEpisode<G, C>) -> u64 {
    episode
        .root_stats
        .iter()
        .map(|stats| stats.portable_contexts as u64)
        .sum()
}

fn measured_episode<G, C>(
    lane: usize,
    episode_id: u64,
    episode: &GumbelEpisode<G, C>,
    root_reward: f32,
    reference: Option<&Reference>,
    feature_rows: Option<&[Vec<u8>]>,
    expects_reference: bool,
) -> MeasuredEpisode {
    MeasuredEpisode {
        lane,
        episode_id,
        artifact: artifact_from_episode(episode, feature_rows),
        root_reward,
        reference: reference.map(projected_reference),
        mode: if expects_reference {
            ProjectionMode::RequireReference
        } else {
            ProjectionMode::AllowUnlabeled
        },
    }
}

fn should_observe_admission(admission: MeasurerAdmission) -> bool {
    matches!(
        admission.status,
        MeasurerAdmissionStatus::Appended { .. }
            | MeasurerAdmissionStatus::Dropped {
                reason: MeasurerError::MissingReference
            }
    )
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

fn attach_reference_opponent(
    references: &HashMap<EpisodeId, Option<Reference>>,
    episode_id: EpisodeId,
    root_step: u32,
    row: &mut FeatureRow,
) {
    let Some(Some(reference)) = references.get(&episode_id) else {
        return;
    };
    attach_opponent_step(reference, root_step as usize, row);
}

fn attach_opponent_step(reference: &Reference, step_index: usize, row: &mut FeatureRow) {
    let Some(step) = aligned_reference_step(reference, step_index) else {
        return;
    };
    row.opponent = step.features.clone();
}

fn aligned_reference_step(
    reference: &Reference,
    step_index: usize,
) -> Option<&crate::reference::ReferenceStep> {
    if reference.steps.is_empty() {
        return None;
    }
    reference
        .steps
        .get(step_index)
        .or_else(|| reference.steps.last())
}

fn feature_rows_for_episode<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    episode: &GumbelEpisode<E::Graph, E::Candidate>,
    reference: Option<&Reference>,
) -> EngineResult<EpisodeFeatureRows<E::Candidate>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let schema = extractor.schema().clone();
    let mut out = Vec::with_capacity(episode.steps.len());
    let mut candidates = Vec::new();
    let mut created_candidates = Vec::new();

    for (index, step) in episode.steps.iter().enumerate() {
        candidates.clear();
        engine.candidates(
            step.before,
            search.config().candidate_options,
            &mut candidates,
        )?;
        created_candidates.extend(candidates.iter().copied());
        // Mirror the eval-side export gate: rows must train the model on
        // the same position inputs it served with.
        let position = replay_position_features(search, extractor.schema(), index, reference)?;
        let mut row = extractor
            .extract(engine, step.before, &candidates, position)
            .map_err(|_| internal("feature extraction failed"))?;
        if let Some(reference) = reference {
            attach_opponent_step(reference, index, &mut row);
        }
        if row.actions.len() != step.legal_actions.len() {
            return Err(internal("feature row action count mismatch"));
        }

        let mut bytes = Vec::new();
        encode_feature_row(&row, &schema, &mut bytes)
            .map_err(|_| internal("feature row encoding failed"))?;
        out.push(bytes);
    }

    Ok(EpisodeFeatureRows {
        rows: out,
        candidates: created_candidates,
    })
}

fn feature_rows_for_competitive_episode<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    episode: &GumbelEpisode<E::Graph, E::Candidate>,
) -> EngineResult<CompetitiveFeatureRows<E::Candidate>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let (p1, p2) = competitive_actors(episode)?;
    let mut candidates = Vec::new();
    let rows = (|| {
        let p1_rows = feature_rows_for_competitive_actor(
            engine,
            extractor,
            search,
            p1,
            p2,
            false,
            &mut candidates,
        )?;
        let p2_rows = feature_rows_for_competitive_actor(
            engine,
            extractor,
            search,
            p2,
            p1,
            true,
            &mut candidates,
        )?;
        Ok(CompetitiveFeatureRows {
            p1: p1_rows,
            p2: p2_rows,
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
fn feature_rows_for_competitive_actor<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    actor: CompetitiveActor<'_, E::Graph, E::Candidate>,
    opponent: CompetitiveActor<'_, E::Graph, E::Candidate>,
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
        let mut position = replay_position_features(search, extractor.schema(), index, None)?;
        position.opponent_present = true;
        let mut row = extractor
            .extract(engine, step.before, &candidates, position)
            .map_err(|_| internal("feature extraction failed"))?;

        let opponent_index = (index + usize::from(opponent_after_turn)).min(opponent.steps.len());
        let (opponent_graph, _) = competitive_actor_state(opponent, opponent_index);
        let opponent_position =
            replay_position_features(search, extractor.schema(), opponent_index, None)?;
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
        if row.actions.len() != step.legal_actions.len() {
            return Err(internal("feature row action count mismatch"));
        }

        let mut bytes = Vec::new();
        encode_feature_row(&row, &schema, &mut bytes)
            .map_err(|_| internal("feature row encoding failed"))?;
        rows.push(bytes);
    }
    Ok(rows)
}

#[derive(Clone, Copy)]
struct CompetitiveActor<'a, G, C> {
    root: G,
    final_graph: G,
    root_context: gz_engine::ReplayGraphContext,
    final_context: gz_engine::ReplayGraphContext,
    steps: &'a [GumbelStep<G, C>],
    final_measure: &'a gz_engine::MeasureResult<G>,
    stop_reason: GumbelStopReason,
}

fn competitive_actors<G, C>(
    episode: &GumbelEpisode<G, C>,
) -> EngineResult<(CompetitiveActor<'_, G, C>, CompetitiveActor<'_, G, C>)>
where
    G: Copy,
{
    let trace = episode
        .competitive
        .as_deref()
        .ok_or_else(|| internal("missing sampled-tree competitive trace"))?;
    let learner = CompetitiveActor {
        root: episode.root,
        final_graph: episode.final_graph,
        root_context: episode.root_context,
        final_context: episode.final_context,
        steps: &episode.steps,
        final_measure: &episode.final_measure,
        stop_reason: episode.stop_reason,
    };
    let opponent = CompetitiveActor {
        root: trace.opponent_root,
        final_graph: trace.opponent_final_graph,
        root_context: trace.opponent_root_context,
        final_context: trace.opponent_final_context,
        steps: &trace.opponent_steps,
        final_measure: &trace.opponent_final_measure,
        stop_reason: trace.opponent_stop_reason,
    };
    Ok(match trace.learner_player {
        GumbelPlayer::One => (learner, opponent),
        GumbelPlayer::Two => (opponent, learner),
    })
}

fn competitive_actor_state<G: Copy, C>(
    actor: CompetitiveActor<'_, G, C>,
    index: usize,
) -> (G, gz_engine::ReplayGraphContext) {
    if index == 0 {
        return (actor.root, actor.root_context);
    }
    actor
        .steps
        .get(index - 1)
        .map_or((actor.final_graph, actor.final_context), |step| {
            (step.after, step.step_ref.after)
        })
}

fn measured_competitive_game<G: Copy, C: Copy>(
    lane: usize,
    game_id: u64,
    episode: &GumbelEpisode<G, C>,
    root_reward: f32,
    rows: &CompetitiveFeatureRows<C>,
) -> EngineResult<MeasuredCompetitiveGame> {
    let trace = episode
        .competitive
        .as_deref()
        .ok_or_else(|| internal("missing sampled-tree competitive trace"))?;
    let (p1, p2) = competitive_actors(episode)?;
    let p1_is_learner = trace.learner_player == GumbelPlayer::One;
    Ok(MeasuredCompetitiveGame {
        lane,
        game_id,
        learner_is_p1: p1_is_learner,
        root_reward,
        p1_artifact: competitive_artifact(p1, &rows.p1, episode.search_config_hash),
        p1_reference: competitive_reference(
            p2,
            if p1_is_learner {
                ReplayReferenceKind::GatedPolicy
            } else {
                ReplayReferenceKind::Gumbel
            },
            episode.search_config_hash,
        ),
        p2_artifact: competitive_artifact(p2, &rows.p2, episode.search_config_hash),
        p2_reference: competitive_reference(
            p1,
            if p1_is_learner {
                ReplayReferenceKind::Gumbel
            } else {
                ReplayReferenceKind::GatedPolicy
            },
            episode.search_config_hash,
        ),
    })
}

fn value_target_root_reward<E>(
    engine: &mut E,
    root: E::Graph,
    options: gz_engine::MeasureOptions,
    value_target: ValueTargetConfig,
) -> EngineResult<f32>
where
    E: GraphEngine,
{
    if matches!(value_target, ValueTargetConfig::Sign) {
        return Ok(0.0);
    }
    let measure = engine.measure(root, options)?;
    if !measure.measured || !measure.valid {
        return Err(internal("root graph was not measured"));
    }
    measure
        .scalar_reward
        .filter(|reward| reward.is_finite())
        .ok_or_else(|| internal("root graph has no finite scalar reward"))
}

fn competitive_artifact<G: Copy, C>(
    actor: CompetitiveActor<'_, G, C>,
    feature_rows: &[Vec<u8>],
    search_config_hash: gz_engine::SearchConfigHash,
) -> CompletedEpisodeArtifact {
    CompletedEpisodeArtifact {
        root: actor.root_context,
        final_graph: actor.final_context,
        final_measure: gz_engine::MeasureSummary::from(actor.final_measure),
        stop_selected: matches!(actor.stop_reason, GumbelStopReason::SelectedStop),
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
                model_version: Some(step.model_version),
            })
            .collect(),
        feature_rows: Some(feature_rows.to_vec()),
    }
}

fn competitive_reference<G, C>(
    actor: CompetitiveActor<'_, G, C>,
    kind: ReplayReferenceKind,
    search_config_hash: gz_engine::SearchConfigHash,
) -> ProjectedReference {
    ProjectedReference {
        kind,
        final_reward: actor.final_measure.scalar_reward.unwrap_or(0.0),
        final_graph: Some(actor.final_context),
        ref_id: None,
        search_config_hash: Some(search_config_hash),
        model_version: homogeneous_model_version(actor.steps),
        step_count: actor.steps.len() + 1,
    }
}

fn homogeneous_model_version<G, C>(steps: &[GumbelStep<G, C>]) -> Option<ModelVersion> {
    let version = steps.first()?.model_version;
    steps
        .iter()
        .all(|step| step.model_version == version)
        .then_some(version)
}

fn opponent_context(reference: &Reference) -> GumbelOpponentContext {
    GumbelOpponentContext {
        // Zero is the transient eval protocol's explicit "not cacheable"
        // sentinel. Registry-backed references carry process-unique ids;
        // other providers stay uncached rather than aliasing each other.
        trajectory_id: reference.ref_id.unwrap_or(0),
        row_count: reference.steps.len() as u32,
        final_reward: reference.final_reward,
    }
}

fn sampled_trajectory_seed(episode_id: EpisodeId) -> u64 {
    const SAMPLE_TRAJECTORY_SALT: u64 = 0x7361_6d70_5f74_726a; // "samp_trj"
    crate::root::episode_noise_seed(episode_id.value() ^ SAMPLE_TRAJECTORY_SALT)
}

fn replay_position_features(
    search: &GumbelMcts,
    schema: &FeatureSchema,
    index: usize,
    reference: Option<&Reference>,
) -> EngineResult<PositionFeatures> {
    let (root_step, budget_fraction, budget_step) = if search.config().export_position {
        let (budget_fraction, budget_step) = search.root_budget(index);
        (
            u32::try_from(index).map_err(|_| internal("root step overflow"))?,
            budget_fraction,
            budget_step,
        )
    } else {
        (0, 0.0, 0.0)
    };
    let scale = schema.config().opponent_reward_scale;
    let opponent_reward = reference.map_or(0.0, |reference| reference.final_reward / scale);

    Ok(PositionFeatures {
        root_step,
        leaf_depth: 0,
        budget_fraction,
        budget_step,
        opponent_reward,
        opponent_present: reference.is_some(),
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
    if extra_candidates.is_empty() {
        return engine.release(&episode.created_graphs, &episode.created_candidates);
    }

    let mut candidates =
        Vec::with_capacity(episode.created_candidates.len() + extra_candidates.len());
    candidates.extend_from_slice(&episode.created_candidates);
    candidates.extend_from_slice(extra_candidates);
    engine.release(&episode.created_graphs, &candidates)
}

fn reference_steps_for_gumbel_episode<G, C>(
    episode: &GumbelEpisode<G, C>,
) -> Vec<crate::reference::ReferenceStep> {
    let mut steps = Vec::with_capacity(episode.steps.len() + 1);
    match episode.steps.first() {
        Some(step) => steps.push(crate::reference::ReferenceStep {
            context: step.step_ref.before,
            features: None,
        }),
        None => steps.push(crate::reference::ReferenceStep {
            context: episode.final_context,
            features: None,
        }),
    }
    steps.extend(
        episode
            .steps
            .iter()
            .map(|step| crate::reference::ReferenceStep {
                context: step.step_ref.after,
                features: None,
            }),
    );
    steps
}

fn reference_steps_for_gumbel_episode_with_features<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    episode: &GumbelEpisode<E::Graph, E::Candidate>,
    final_reward: f32,
) -> EngineResult<(Vec<crate::reference::ReferenceStep>, Vec<E::Candidate>)>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let mut candidates = Vec::new();
    let steps = (|| {
        let mut steps = Vec::with_capacity(episode.steps.len() + 1);

        match episode.steps.first() {
            Some(step) => steps.push(reference_step_with_features(
                engine,
                extractor,
                search,
                GumbelReferenceStepInput {
                    graph: step.before,
                    context: step.step_ref.before,
                    index: 0,
                    final_reward,
                },
                &mut candidates,
            )?),
            None => steps.push(reference_step_with_features(
                engine,
                extractor,
                search,
                GumbelReferenceStepInput {
                    graph: episode.final_graph,
                    context: episode.final_context,
                    index: 0,
                    final_reward,
                },
                &mut candidates,
            )?),
        }

        for (index, step) in episode.steps.iter().enumerate() {
            steps.push(reference_step_with_features(
                engine,
                extractor,
                search,
                GumbelReferenceStepInput {
                    graph: step.after,
                    context: step.step_ref.after,
                    index: index + 1,
                    final_reward,
                },
                &mut candidates,
            )?);
        }

        Ok(steps)
    })();

    match steps {
        Ok(steps) => Ok((steps, candidates)),
        Err(error) => {
            engine.release(&[], &candidates)?;
            Err(error)
        }
    }
}

struct GumbelReferenceStepInput<G> {
    graph: G,
    context: gz_engine::ReplayGraphContext,
    index: usize,
    final_reward: f32,
}

fn reference_step_with_features<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    input: GumbelReferenceStepInput<E::Graph>,
    created_candidates: &mut Vec<E::Candidate>,
) -> EngineResult<crate::reference::ReferenceStep>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let mut candidates = Vec::new();
    engine.candidates(
        input.graph,
        search.config().candidate_options,
        &mut candidates,
    )?;
    created_candidates.extend(candidates.iter().copied());
    let position = replay_position_features(search, extractor.schema(), input.index, None)?;
    let scale = extractor.schema().config().opponent_reward_scale;
    let row = extractor
        .extract(
            engine,
            input.graph,
            &candidates,
            PositionFeatures {
                opponent_reward: input.final_reward / scale,
                opponent_present: true,
                ..position
            },
        )
        .map_err(|_| internal("reference feature extraction failed"))?;

    Ok(crate::reference::ReferenceStep {
        context: input.context,
        features: Some(OpponentStateFeatures {
            node_count: row.node_count,
            node_tokens: row.node_tokens,
            node_attrs: row.node_attrs,
            edges: row.edges,
            position: row.position,
        }),
    })
}

/// Drives opponent rollout episodes for rollout-based reference providers
/// (the policy opponent). Tracks the newest model version seen on eval
/// replies. It prioritizes the greedy checkpoint challenge, then fills the
/// accepted checkpoint's trajectory pool with categorical policy rollouts.
/// Rollout episodes never reach the replay store or the run summary.
struct OpponentRollout {
    greedy_search: GumbelMcts,
    sample_search: GumbelMcts,
    identity: EngineIdentity,
    latest_version: Option<ModelVersion>,
    in_flight: Option<InFlightOpponentRollout>,
    arena_in_flight: HashMap<EpisodeId, OpponentRolloutKind>,
    arena_partition: Option<(usize, usize)>,
}

#[derive(Clone, Copy)]
struct InFlightOpponentRollout {
    episode_id: EpisodeId,
    kind: OpponentRolloutKind,
}

#[derive(Clone, Copy)]
enum OpponentRolloutKind {
    Challenge,
    Sample(ModelVersion),
    Arena {
        claim: ArenaRolloutClaim,
        root_reward: f32,
    },
}

impl OpponentRollout {
    fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        _context: GumbelEpisodeContext,
        lane: usize,
        lanes: usize,
        arena_parallelism: usize,
    ) -> Self {
        let arena_partition = if arena_parallelism == 0 {
            Some((lane, lanes))
        } else if lane == 0 {
            Some((0, 1))
        } else {
            None
        };
        Self {
            greedy_search: search.policy_rollout(),
            sample_search: search.policy_sample_rollout(),
            identity,
            latest_version: None,
            in_flight: None,
            arena_in_flight: HashMap::new(),
            arena_partition,
        }
    }

    fn observe_version(&mut self, version: ModelVersion) {
        self.latest_version = Some(version);
    }

    const fn latest_version(&self) -> Option<ModelVersion> {
        self.latest_version
    }

    fn arena_active(&self) -> bool {
        !self.arena_in_flight.is_empty()
    }

    fn eval_route(&self, episode_id: EpisodeId) -> Option<EvalRoute> {
        let kind = self
            .in_flight
            .filter(|flight| flight.episode_id == episode_id)
            .map(|flight| flight.kind)
            .or_else(|| self.arena_in_flight.get(&episode_id).copied())?;
        Some(match kind {
            OpponentRolloutKind::Arena { claim, .. } => policy_eval_route(claim.model),
            OpponentRolloutKind::Challenge => EvalRoute::Current,
            OpponentRolloutKind::Sample(_) => EvalRoute::Incumbent,
        })
    }

    /// Runs before root admission so a busy pool cannot starve the
    /// rollout: the freed slot goes to the rollout first.
    fn try_admit<E, R, P>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        provider: &mut P,
        next_episode_id: &mut u64,
    ) -> EngineResult<()>
    where
        E: GraphEngine,
        R: RootSource<E>,
        P: ReferenceProvider<E>,
    {
        // Arena roots are independent and share one pinned model. Fill every
        // idle lane slot before admitting learner work so all roots advance
        // through the dedicated evaluator in capacity-sized waves.
        if let Some((lane, lanes)) = self.arena_partition {
            while pool.idle_count() > 0 {
                let Some(claim) = provider.claim_arena_rollout(self.latest_version, lane, lanes)
                else {
                    break;
                };
                if !self.admit(
                    pool,
                    engine,
                    roots,
                    provider,
                    next_episode_id,
                    OpponentRolloutKind::Arena {
                        claim,
                        root_reward: 0.0,
                    },
                )? {
                    break;
                }
            }
        }

        if self.in_flight.is_some() || pool.idle_count() == 0 {
            return Ok(());
        }

        // latest_version None at cold start does not block: providers
        // that seed their reference claim and the rollout's own eval
        // replies name the version it played under.
        let kind = if provider.claim_rollout(self.latest_version) {
            Some(OpponentRolloutKind::Challenge)
        } else {
            provider
                .claim_sample_rollout(self.latest_version)
                .map(OpponentRolloutKind::Sample)
        };
        let Some(kind) = kind else {
            return Ok(());
        };
        let _ = self.admit(pool, engine, roots, provider, next_episode_id, kind)?;
        Ok(())
    }

    fn admit<E, R, P>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        provider: &mut P,
        next_episode_id: &mut u64,
        mut kind: OpponentRolloutKind,
    ) -> EngineResult<bool>
    where
        E: GraphEngine,
        R: RootSource<E>,
        P: ReferenceProvider<E>,
    {
        let root = match kind {
            OpponentRolloutKind::Arena { claim, .. } => provider.arena_root(engine, claim.index),
            OpponentRolloutKind::Challenge | OpponentRolloutKind::Sample(_) => {
                roots.fixed_root(engine)
            }
        };
        let root = match root {
            Ok(Some(root)) => root,
            Ok(None) => {
                finish_opponent_rollout(provider, kind, None, None);
                return Ok(false);
            }
            Err(error) => {
                finish_opponent_rollout(provider, kind, None, None);
                return Err(error);
            }
        };
        if let OpponentRolloutKind::Arena {
            ref mut root_reward,
            ..
        } = kind
        {
            let measure = match engine.measure(root, self.greedy_search.config().measure_options) {
                Ok(measure) => measure,
                Err(error) => {
                    finish_opponent_rollout(provider, kind, None, None);
                    return Err(error);
                }
            };
            let Some(reward) = (measure.measured && measure.valid)
                .then_some(measure.scalar_reward)
                .flatten()
                .filter(|reward| reward.is_finite())
            else {
                finish_opponent_rollout(provider, kind, None, None);
                return Ok(false);
            };
            *root_reward = reward;
        }

        let episode_id = EpisodeId::new(*next_episode_id);
        let search = match kind {
            OpponentRolloutKind::Challenge | OpponentRolloutKind::Arena { .. } => {
                &self.greedy_search
            }
            OpponentRolloutKind::Sample(_) => &self.sample_search,
        };
        let admitted = pool.admit_direct(
            search,
            self.identity,
            root,
            GumbelEpisodeContext {
                noise_seed: match kind {
                    OpponentRolloutKind::Challenge | OpponentRolloutKind::Arena { .. } => 0,
                    OpponentRolloutKind::Sample(_) => {
                        const SAMPLE_SALT: u64 = 0x7265_665f_7361_6d70; // "ref_samp"
                        crate::root::episode_noise_seed(episode_id.value() ^ SAMPLE_SALT)
                    }
                },
                opponent: None,
            },
            episode_id,
            false,
            false,
        );
        if admitted {
            *next_episode_id += 1;
            if matches!(kind, OpponentRolloutKind::Arena { .. }) {
                self.arena_in_flight.insert(episode_id, kind);
            } else {
                self.in_flight = Some(InFlightOpponentRollout { episode_id, kind });
            }
        } else {
            finish_opponent_rollout(provider, kind, None, None);
        }
        Ok(admitted)
    }

    /// Claims a completed rollout episode: releases its handles and
    /// reports the outcome to the provider. Returns true when the episode
    /// was a rollout and must not be projected, appended, or counted.
    fn intercept<E, P>(
        &mut self,
        engine: &mut E,
        provider: &mut P,
        completed: &OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<bool>
    where
        E: GraphEngine,
        P: ReferenceProvider<E>,
    {
        let Some(kind) = self.take_in_flight(completed.episode_id) else {
            return Ok(false);
        };
        release_episode_handles(engine, &completed.episode, &[])?;

        let measure = &completed.episode.final_measure;
        let reward = if measure.measured && measure.valid {
            measure.scalar_reward.filter(|reward| reward.is_finite())
        } else {
            None
        };
        let outcome = reward.map(|final_reward| RolloutOutcome {
            final_reward,
            final_graph: completed.episode.final_context,
            steps: reference_steps_for_gumbel_episode(&completed.episode),
            search_config_hash: completed.episode.search_config_hash,
            model_version: rollout_model_version(&completed.episode),
        });
        let score = arena_score(kind, reward);
        finish_opponent_rollout(provider, kind, score, outcome);
        Ok(true)
    }

    fn intercept_with_features<E, P, X>(
        &mut self,
        engine: &mut E,
        provider: &mut P,
        completed: &OrchestratedEpisode<E::Graph, E::Candidate>,
        extractor: &mut X,
    ) -> EngineResult<bool>
    where
        E: GraphEngine,
        P: ReferenceProvider<E>,
        X: FeatureExtractor<E>,
    {
        let Some(kind) = self.take_in_flight(completed.episode_id) else {
            return Ok(false);
        };

        let measure = &completed.episode.final_measure;
        let reward = if measure.measured && measure.valid {
            measure.scalar_reward.filter(|reward| reward.is_finite())
        } else {
            None
        };

        let (steps, feature_candidates) = match (kind, reward) {
            (OpponentRolloutKind::Arena { .. }, Some(_)) => (
                reference_steps_for_gumbel_episode(&completed.episode),
                Vec::new(),
            ),
            (_, Some(final_reward)) => reference_steps_for_gumbel_episode_with_features(
                engine,
                extractor,
                match kind {
                    OpponentRolloutKind::Challenge => &self.greedy_search,
                    OpponentRolloutKind::Sample(_) => &self.sample_search,
                    OpponentRolloutKind::Arena { .. } => unreachable!(),
                },
                &completed.episode,
                final_reward,
            )?,
            (_, None) => (Vec::new(), Vec::new()),
        };
        release_episode_handles(engine, &completed.episode, &feature_candidates)?;

        let outcome = reward.map(|final_reward| RolloutOutcome {
            final_reward,
            final_graph: completed.episode.final_context,
            steps,
            search_config_hash: completed.episode.search_config_hash,
            model_version: rollout_model_version(&completed.episode),
        });
        let score = arena_score(kind, reward);
        finish_opponent_rollout(provider, kind, score, outcome);
        Ok(true)
    }

    fn take_in_flight(&mut self, episode_id: EpisodeId) -> Option<OpponentRolloutKind> {
        if self
            .in_flight
            .is_some_and(|in_flight| in_flight.episode_id == episode_id)
        {
            return self.in_flight.take().map(|in_flight| in_flight.kind);
        }
        self.arena_in_flight.remove(&episode_id)
    }
}

fn finish_opponent_rollout<E, P>(
    provider: &mut P,
    kind: OpponentRolloutKind,
    score: Option<f32>,
    outcome: Option<RolloutOutcome>,
) where
    E: GraphEngine,
    P: ReferenceProvider<E>,
{
    match kind {
        OpponentRolloutKind::Challenge => provider.finish_rollout(outcome),
        OpponentRolloutKind::Sample(version) => {
            provider.finish_sample_rollout(version, outcome);
        }
        OpponentRolloutKind::Arena { claim, .. } => {
            provider.finish_arena_rollout(claim, score, outcome);
        }
    }
}

fn arena_score(kind: OpponentRolloutKind, final_reward: Option<f32>) -> Option<f32> {
    let OpponentRolloutKind::Arena { root_reward, .. } = kind else {
        return None;
    };
    final_reward.map(|reward| (reward - root_reward) / root_reward.abs().max(1.0))
}

/// A policy trajectory belongs to one checkpoint only. A hot-swap that
/// lands between roots makes the rollout unusable as either a challenger
/// or a trajectory-pool sample.
fn rollout_model_version<G, C>(episode: &GumbelEpisode<G, C>) -> Option<ModelVersion> {
    let version = episode.steps.first()?.model_version;
    episode
        .steps
        .iter()
        .all(|step| step.model_version == version)
        .then_some(version)
}

fn clear_replayed_episode_trace<G, C>(episode: &mut GumbelEpisode<G, C>) {
    // Drop the backing buffers, not just the elements: clear() keeps
    // capacity, and created_candidates alone reaches millions of ids per
    // episode (~20 MB). Completed episodes are retained for the run
    // summary, so kept capacity is a per-episode leak on unbounded runs.
    episode.steps = Vec::new();
    episode.root_stats = Vec::new();
    episode.created_graphs = Vec::new();
    episode.created_candidates = Vec::new();
    episode.competitive = None;
}

fn append_replay_job(
    replay_tx: &SyncSender<ReplayJob>,
    episode: MeasuredEpisode,
) -> EngineResult<MeasurerAdmission> {
    let (ack, done) = sync_channel(1);
    replay_tx
        .send(ReplayJob::Episode {
            episode: Box::new(episode),
            ack,
        })
        .map_err(|_| internal("replay sink failed"))?;
    done.recv().map_err(|_| internal("replay sink failed"))?
}

fn append_competitive_replay_job(
    replay_tx: &SyncSender<ReplayJob>,
    game: MeasuredCompetitiveGame,
) -> EngineResult<MeasurerAdmission> {
    let (ack, done) = sync_channel(1);
    replay_tx
        .send(ReplayJob::Competitive {
            game: Box::new(game),
            ack,
        })
        .map_err(|_| internal("replay sink failed"))?;
    done.recv().map_err(|_| internal("replay sink failed"))?
}

/// Resumes every pending reply; returns the newest model version seen so
/// callers can drive version-triggered opponent rollouts.
fn receive_replies<E>(
    engine: &mut E,
    pool: &mut WorkerPool<E::Graph, E::Candidate>,
    reply_rx: &Receiver<EvalReply>,
    wait: Option<Duration>,
) -> EngineResult<Option<ModelVersion>>
where
    E: GraphEngine,
{
    let reply = match wait {
        Some(wait) => match reply_rx.recv_timeout(wait) {
            Ok(reply) => reply,
            Err(RecvTimeoutError::Timeout) => return Ok(None),
            Err(RecvTimeoutError::Disconnected) => {
                return Err(internal("eval backend unavailable"));
            }
        },
        None => reply_rx
            .recv()
            .map_err(|_| internal("eval backend unavailable"))?,
    };
    let mut version = (reply.route == EvalRoute::Current).then_some(reply.output.model_version);
    pool.resume(engine, reply.slot, reply.token, reply.output)?;

    loop {
        match reply_rx.try_recv() {
            Ok(reply) => {
                if reply.route == EvalRoute::Current {
                    version = Some(reply.output.model_version);
                }
                pool.resume(engine, reply.slot, reply.token, reply.output)?;
            }
            Err(TryRecvError::Empty) => return Ok(version),
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
                route: EvalRoute::Current,
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
    route: EvalRoute,
    eval_pressure: Option<Arc<EvalPressure>>,
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
    let mut opponent_refs = Vec::with_capacity(max_batch);
    let mut bytes = Vec::new();
    let mut in_flight: std::collections::VecDeque<(Routing, gz_eval_service::PendingBatch)> =
        std::collections::VecDeque::with_capacity(EVAL_PIPELINE_DEPTH);
    let mut capacity_accounted_at = None;
    let mut intake_open = true;
    let mut stats_batches: usize = 0;
    let mut last_stats = Instant::now();

    while intake_open || !in_flight.is_empty() {
        batch.clear();
        if intake_open && in_flight.len() < EVAL_PIPELINE_DEPTH {
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
                        Ok(job) => batch.push(job),
                        Err(_) => {
                            intake_open = false;
                            break;
                        }
                    }
                    continue;
                }
                match intake_rx.recv_timeout(config.flush_after) {
                    Ok(job) => batch.push(job),
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
                            route,
                            max_batch,
                            EvalCapacityAccounting {
                                pressure: eval_pressure.as_deref(),
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
            let mut routing: Routing = Vec::with_capacity(batch.len());
            rows.clear();
            action_counts.clear();
            opponent_refs.clear();
            for job in batch.drain(..) {
                routing.push((job.lane, job.slot, job.token, job.action_count));
                action_counts.push(job.action_count);
                opponent_refs.push(job.opponent_ref);
                rows.push(job.row);
            }
            collator
                .collate_with_opponent_refs(&rows, &opponent_refs, &mut bytes)
                .map_err(|_| internal("feature collation failed"))?;
            if route == EvalRoute::Current && in_flight.is_empty() {
                capacity_accounted_at = Some(Instant::now());
            }
            let pending = backend
                .submit(&bytes, &action_counts)
                .map_err(|_| internal("feature eval backend failed"))?;
            in_flight.push_back((routing, pending));
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
                route,
                max_batch,
                EvalCapacityAccounting {
                    pressure: eval_pressure.as_deref(),
                    accounted_at: &mut capacity_accounted_at,
                },
            )?;
        }
        if last_stats.elapsed() >= STATS_INTERVAL && batch_sizes.len() > stats_batches {
            stats_batches = batch_sizes.len();
            let stats_rows: u64 = batch_sizes.iter().map(|&size| size as u64).sum();
            last_stats = Instant::now();
            let role = match route {
                EvalRoute::Current => "current",
                EvalRoute::Incumbent => "incumbent",
                EvalRoute::Challenger => "challenger",
            };
            eprintln!("event=eval_stats role={role} batches={stats_batches} rows={stats_rows}");
        }
    }

    Ok(batch_sizes)
}

type BatcherRouting = Vec<(usize, usize, WorkToken, u32)>;

struct EvalCapacityAccounting<'a> {
    pressure: Option<&'a EvalPressure>,
    accounted_at: &'a mut Option<Instant>,
}

fn drain_oldest<B>(
    backend: &mut B,
    in_flight: &mut std::collections::VecDeque<(BatcherRouting, gz_eval_service::PendingBatch)>,
    reply_txs: &[SyncSender<EvalReply>],
    batch_sizes: &mut Vec<usize>,
    route: EvalRoute,
    max_batch: usize,
    capacity: EvalCapacityAccounting<'_>,
) -> EngineResult<()>
where
    B: FeatureEvalBackend,
{
    let Some((routing, pending)) = in_flight.pop_front() else {
        return Ok(());
    };
    let capacity_work = backend.capacity_work(routing.len(), max_batch);
    let outputs = backend
        .receive(pending)
        .map_err(|_| internal("feature eval backend failed"))?;
    let completed_at = Instant::now();
    let counts = routing
        .iter()
        .map(|&(_, _, _, action_count)| action_count)
        .collect::<Vec<_>>();
    validate_backend_outputs(&outputs, &counts)?;
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
            route,
        });
    }
    if let Some(eval_pressure) = capacity.pressure {
        match route {
            EvalRoute::Current => {
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
            EvalRoute::Incumbent | EvalRoute::Challenger => eval_pressure.complete(completed),
        }
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
    value_target: ValueTargetConfig,
) -> EngineResult<MeasurerRunSummary> {
    let mut measurer = ReplayMeasurer::with_value_target(store, length_tiebreak, value_target);
    // Machine-parsed by the trainer driver (measure ledger metrics);
    // field changes must update its parser. Counters are cumulative.
    const STATS_INTERVAL: Duration = Duration::from_secs(30);
    let mut last_stats = Instant::now();

    while let Ok(job) = replay_rx.recv() {
        let (result, ack) = match job {
            ReplayJob::Episode { episode, ack } => {
                (measurer.admit(*episode).map_err(map_replay_error), ack)
            }
            ReplayJob::Competitive { game, ack } => (
                measurer.admit_competitive(*game).map_err(map_replay_error),
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

fn ensure_replay_data_mode<E, P>(
    store: &ReplayStore,
    providers: &[P],
    value_target: ValueTargetConfig,
) -> EngineResult<()>
where
    E: GraphEngine,
    P: ReferenceProvider<E>,
{
    let sampled_tree = providers
        .first()
        .is_some_and(ReferenceProvider::<E>::sampled_tree_mode);
    if providers
        .iter()
        .any(|provider| provider.sampled_tree_mode() != sampled_tree)
    {
        return Err(internal("sampled-tree provider mode mismatch"));
    }
    if !value_target.is_valid() {
        return Err(internal("invalid value target configuration"));
    }
    let data_mode = match value_target {
        ValueTargetConfig::Sign if sampled_tree => gz_replay::ReplayDataMode::SampledTree,
        ValueTargetConfig::Sign => gz_replay::ReplayDataMode::Standard,
        ValueTargetConfig::Graded { reward_scale } => {
            gz_replay::ReplayDataMode::graded(sampled_tree, reward_scale)
                .map_err(map_replay_error)?
        }
    };
    store.ensure_data_mode(data_mode).map_err(map_replay_error)
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

fn validate_reference_backend_count(backends: usize, lanes: usize) -> EngineResult<()> {
    if backends > lanes {
        return Err(internal("more reference eval backends than lanes"));
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
