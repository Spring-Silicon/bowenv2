use crate::EpisodeId;
use crate::admission::{
    AdmissionPacer, AdmissionSmoothingConfig, EVAL_PIPELINE_DEPTH, EvalPressure,
    SharedAdmissionShaper, build_admission_shaper,
};
use crate::internal;
use crate::leases::{EpisodeModelLeases, ModelLeaseRegistry};
use crate::pool::{Admission, AdmissionResult, CompletedTask, WorkerPool};
use crate::root::RootSource;
use gz_engine::{EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine};
use gz_eval::EvalOutput;
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
    EngineIdentity, GumbelMcts, GumbelValueMode, SearchAction, SymmetricActorTrace,
    SymmetricEpisode, WorkToken,
};
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadedOrchestratorConfig {
    pub workers_per_lane: NonZeroUsize,
    pub max_batch: NonZeroUsize,
    pub flush_after: Duration,
    pub admission_stagger: Duration,
    pub admission_smoothing: Option<AdmissionSmoothingConfig>,
}

pub struct ThreadedGumbelOrchestrator<E> {
    engines: Vec<E>,
    search: GumbelMcts,
    config: ThreadedOrchestratorConfig,
}

pub struct ReplayRuntime<'a> {
    pub store: &'a ReplayStore,
    pub backpressure: Option<ReplayBackpressure>,
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
    pub replay_rows: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedReplayRun {
    pub lanes: Vec<ReplayLaneSummary>,
    pub batch_sizes: Vec<usize>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub replay_rows: u64,
    pub measure_ledger: MeasureLedgerSnapshot,
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

struct FeaturizedBatcherContext {
    model_registry: Arc<ModelLeaseRegistry>,
    eval_pressure: Option<Arc<EvalPressure>>,
}

enum ReplayJob {
    Symmetric {
        game: Box<MeasuredSymmetricGame>,
        ack: SyncSender<EngineResult<MeasurerAdmission>>,
    },
}

impl<E> ThreadedGumbelOrchestrator<E>
where
    E: GraphEngine + Send,
    E::Graph: Send,
    E::Candidate: Send,
{
    pub const fn new(
        engines: Vec<E>,
        search: GumbelMcts,
        config: ThreadedOrchestratorConfig,
    ) -> Self {
        Self {
            engines,
            search,
            config,
        }
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
        let evals_per_worker = self
            .search
            .config()
            .max_considered_actions
            .get()
            .min(self.search.config().simulations.get());
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
        let admission_shaper = build_admission_shaper(
            lanes,
            backend_count,
            config.workers_per_lane,
            config.max_batch,
            config.admission_stagger,
            config.admission_smoothing,
            Arc::clone(&eval_pressure),
        )?;
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
        let feature_schema = first_schema::<E, X>(&extractors, schema_hash)?;
        store
            .ensure_feature_schema(feature_schema.config())
            .map_err(map_replay_error)?;
        validate_collator_capacity(
            &FeatureCollator::new(feature_schema.clone(), config.max_batch),
            config,
        )?;
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
            let sink_handle = scope.spawn(move || run_replay_sink(store, replay_rx));
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
        for result in lane_results {
            let mut result = result?;
            merge_lane_measurer_summary(&mut result, &measurer_summary);
            lane_summaries.push(result);
        }

        Ok(ThreadedReplayRun {
            lanes: lane_summaries,
            batch_sizes,
            episodes_appended: measurer_summary.episodes_appended,
            episodes_dropped: measurer_summary.episodes_dropped,
            replay_rows: measurer_summary.replay_rows,
            measure_ledger: measurer_summary.measure_ledger,
        })
    }
}

struct LaneRuntime<'a> {
    lane: usize,
    lanes: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    pool_capacity: NonZeroUsize,
    admission_stagger: Duration,
    admission_shaper: Option<Arc<SharedAdmissionShaper>>,
    eval_pressure: Arc<EvalPressure>,
    intake_tx: SyncSender<FeaturizedEvalJob>,
    reply_rx: Receiver<EvalReply>,
}

struct SymmetricFeatureRows<C> {
    p1: Vec<Vec<u8>>,
    p2: Vec<Vec<u8>>,
    candidates: Vec<C>,
}

#[allow(clippy::too_many_arguments)]
fn admit_roots<E, R, X>(
    mode: &mut FeaturizedReplayMode<'_, X>,
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
    E: GraphEngine,
    R: RootSource<E>,
{
    let mut admission = Admission {
        search,
        identity,
        pressure_reserved,
        next_episode_id,
    };
    pool.admit_limited(engine, roots, &mut admission, limit, |_, id, _| {
        mode.admit_episode(id)
    })
}

fn run_lane_pipeline<E, R, X>(
    mut engine: E,
    mut roots: R,
    runtime: LaneRuntime<'_>,
    mut mode: FeaturizedReplayMode<'_, X>,
) -> EngineResult<ReplayLaneSummary>
where
    E: GraphEngine,
    R: RootSource<E>,
    X: FeatureExtractor<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let mut pool = WorkerPool::new(runtime.pool_capacity);
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
                        let result = match admit_roots(
                            &mut mode,
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
                    let result = admit_roots(
                        &mut mode,
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
                replay_rows: 0,
            },
            model_leases: EpisodeModelLeases::new(model_registry),
        }
    }

    fn gate_open(&self) -> bool {
        replay_gate_open(self.store, self.backpressure)
    }

    fn gate_poll(&self) -> Option<Duration> {
        self.backpressure.map(|backpressure| backpressure.gate_poll)
    }

    fn admit_episode(&mut self, episode_id: EpisodeId) -> EngineResult<()> {
        self.model_leases.ensure(episode_id)?;
        self.admitted_at.insert(episode_id, Instant::now());
        Ok(())
    }

    fn drive<E>(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<CompletedTask<E::Graph, E::Candidate>>>
    where
        E: GraphEngine,
        X: FeatureExtractor<E>,
    {
        pool.drive(engine, &mut self.extractor)
    }

    fn send_parked<G, C>(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<G, C>,
        intake_tx: &SyncSender<FeaturizedEvalJob>,
        eval_pressure: &EvalPressure,
    ) -> EngineResult<()>
    where
        G: Copy + Eq + std::hash::Hash,
        C: Copy + Eq + std::hash::Hash,
    {
        send_featurized_parked(lane, pool, intake_tx, eval_pressure, &mut self.model_leases)
    }

    fn complete<E>(
        &mut self,
        engine: &mut E,
        search: &GumbelMcts,
        completed: CompletedTask<E::Graph, E::Candidate>,
    ) -> EngineResult<Option<u64>>
    where
        E: GraphEngine,
        X: FeatureExtractor<E>,
    {
        let episode_id = completed.episode_id;
        let episode = completed.episode;

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

        let append = append_symmetric_replay_job(&self.replay_tx, game);
        let release = release_symmetric_episode_handles(engine, &episode, &feature_rows.candidates);
        self.model_leases.release(episode_id);
        release?;
        append?;

        Ok(Some(completed.evaluations))
    }

    fn finish(mut self, lane: usize) -> ReplayLaneSummary {
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
                row: parked.row,
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

fn merge_lane_measurer_summary(lane: &mut ReplayLaneSummary, measurer: &MeasurerRunSummary) {
    let Some(measured) = measurer.lanes.get(lane.lane) else {
        return;
    };
    lane.episodes_appended = measured.episodes_appended;
    lane.episodes_dropped = measured.episodes_dropped;
    lane.replay_rows = measured.replay_rows;
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
) -> EngineResult<MeasurerRunSummary> {
    let mut measurer = ReplayMeasurer::new(store);
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
