use crate::EpisodeId;
use crate::pool::{Admission, WorkerPool};
use crate::project::project_episode;
use crate::reference::{Reference, ReferenceProvider};
use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::internal;
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::{EvalOutput, EvalRequest, Evaluator, eval_error_to_engine_error, validate_outputs};
use gz_replay::{ReplayEpisodeRecord, ReplayError, ReplayRow, ReplayStore};
use gz_search::{EngineIdentity, GumbelEpisodeContext, GumbelMcts, WorkToken};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadedOrchestratorConfig {
    pub workers_per_lane: NonZeroUsize,
    pub max_batch: NonZeroUsize,
    pub flush_after: Duration,
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
    pub episodes: Vec<OrchestratedEpisode<G, C>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedRun<G, C> {
    pub lanes: Vec<LaneEpisodes<G, C>>,
    pub batch_sizes: Vec<usize>,
}

pub struct ReplayRuntime<'a, P> {
    pub store: &'a ReplayStore,
    pub providers: Vec<P>,
    pub backpressure: Option<ReplayBackpressure>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayBackpressure {
    pub max_row_backlog: NonZeroU64,
    pub gate_poll: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedReplayRun<G, C> {
    pub run: ThreadedRun<G, C>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
}

struct EvalJob {
    lane: usize,
    slot: usize,
    token: WorkToken,
    request: EvalRequest,
}

struct EvalReply {
    slot: usize,
    token: WorkToken,
    output: EvalOutput,
}

struct ReplayJob {
    record: ReplayEpisodeRecord,
    rows: Vec<ReplayRow>,
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
        let search = &self.search;
        let evaluator = self.evaluator;
        let engines = self.engines;

        let (batch_result, lane_results) = std::thread::scope(|scope| {
            let batch_handle =
                scope.spawn(move || run_batcher(evaluator, intake_rx, reply_txs, config));
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, ((engine, roots), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_tx.clone();
                lane_handles.push(scope.spawn(move || {
                    run_lane(
                        engine,
                        roots,
                        LaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                        },
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
    ) -> EngineResult<ThreadedReplayRun<E::Graph, E::Candidate>>
    where
        R: RootSource<E> + Send,
        P: ReferenceProvider<E> + Send,
    {
        let lanes = self.engines.len();
        if root_sources.len() != lanes || replay.providers.len() != lanes {
            return Err(internal("lane count mismatch"));
        }

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
        let search = &self.search;
        let evaluator = self.evaluator;
        let engines = self.engines;
        let providers = replay.providers;
        let store = replay.store;
        let backpressure = replay.backpressure;

        let (batch_result, sink_result, lane_results) = std::thread::scope(|scope| {
            let batch_handle =
                scope.spawn(move || run_batcher(evaluator, intake_rx, reply_txs, config));
            let sink_handle = scope.spawn(move || run_replay_sink(store, replay_rx));
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
                lane_handles.push(scope.spawn(move || {
                    run_lane_with_replay(
                        engine,
                        roots,
                        provider,
                        ReplayLaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                            replay_tx,
                            store,
                            backpressure,
                        },
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
        let episodes_appended = sink_result?;
        let mut lanes = Vec::with_capacity(lane_results.len());
        let mut episodes_dropped = 0;

        for result in lane_results {
            let result = result?;
            episodes_dropped += result.episodes_dropped;
            lanes.push(result.lane);
        }

        Ok(ThreadedReplayRun {
            run: ThreadedRun { lanes, batch_sizes },
            episodes_appended,
            episodes_dropped,
        })
    }
}

struct LaneRuntime<'a> {
    lane: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    context: GumbelEpisodeContext,
    intake_tx: SyncSender<EvalJob>,
    reply_rx: Receiver<EvalReply>,
}

struct ReplayLaneRuntime<'a> {
    lane: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    context: GumbelEpisodeContext,
    intake_tx: SyncSender<EvalJob>,
    reply_rx: Receiver<EvalReply>,
    replay_tx: SyncSender<ReplayJob>,
    store: &'a ReplayStore,
    backpressure: Option<ReplayBackpressure>,
}

struct ReplayLaneResult<G, C> {
    lane: LaneEpisodes<G, C>,
    episodes_dropped: u64,
}

fn run_lane<E, R>(
    mut engine: E,
    mut roots: R,
    runtime: LaneRuntime<'_>,
) -> EngineResult<LaneEpisodes<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    R: RootSource<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let worker_id_base = (runtime.lane * runtime.workers_per_lane.get()) as u64;
    let mut pool = WorkerPool::new(runtime.workers_per_lane, worker_id_base);
    let mut episodes = Vec::new();
    let mut roots_exhausted = false;
    let mut next_episode_id = (runtime.lane as u64) << 32;

    loop {
        if !roots_exhausted {
            let mut admission = Admission {
                search: runtime.search,
                identity,
                context: runtime.context,
                next_episode_id: &mut next_episode_id,
            };
            roots_exhausted = pool.admit(&mut engine, &mut roots, &mut admission)?.1;
        }

        episodes.extend(pool.drive(&mut engine, "worker blocked")?);

        for parked in pool.take_unsent_parked() {
            runtime
                .intake_tx
                .send(EvalJob {
                    lane: runtime.lane,
                    slot: parked.slot,
                    token: parked.token,
                    request: parked.request,
                })
                .map_err(|_| internal("eval backend unavailable"))?;
        }

        if roots_exhausted && !pool.active() {
            return Ok(LaneEpisodes {
                lane: runtime.lane,
                episodes,
            });
        }

        if pool.has_parked() {
            let reply = runtime
                .reply_rx
                .recv()
                .map_err(|_| internal("eval backend unavailable"))?;
            pool.resume(reply.slot, reply.token, reply.output)?;

            loop {
                match runtime.reply_rx.try_recv() {
                    Ok(reply) => pool.resume(reply.slot, reply.token, reply.output)?,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        return Err(internal("eval backend unavailable"));
                    }
                }
            }
        }
    }
}

fn run_lane_with_replay<E, R, P>(
    mut engine: E,
    mut roots: R,
    mut provider: P,
    runtime: ReplayLaneRuntime<'_>,
) -> EngineResult<ReplayLaneResult<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    R: RootSource<E>,
    P: ReferenceProvider<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let worker_id_base = (runtime.lane * runtime.workers_per_lane.get()) as u64;
    let mut pool = WorkerPool::new(runtime.workers_per_lane, worker_id_base);
    let mut episodes = Vec::new();
    let mut references = HashMap::<EpisodeId, Option<Reference<E::Graph>>>::new();
    let mut roots_exhausted = false;
    let mut next_episode_id = (runtime.lane as u64) << 32;
    let mut episodes_dropped = 0;

    loop {
        if !roots_exhausted {
            if replay_gate_open(runtime.store, runtime.backpressure) {
                let mut admission = Admission {
                    search: runtime.search,
                    identity,
                    context: runtime.context,
                    next_episode_id: &mut next_episode_id,
                };
                let (admitted, exhausted) = pool.admit(&mut engine, &mut roots, &mut admission)?;
                roots_exhausted = exhausted;

                for (episode_id, root) in admitted {
                    references.insert(episode_id, provider.reference(&mut engine, root)?);
                }
            } else if !pool.active()
                && let Some(backpressure) = runtime.backpressure
            {
                // The gate limits admission only. In-flight episodes always
                // finish, so backlog can overshoot by at most total workers
                // times rows per episode. This sleep is the throttled-idle
                // path that prevents a fully gated lane from busy-spinning.
                std::thread::sleep(backpressure.gate_poll);
            }
        }

        for completed in pool.drive(&mut engine, "worker blocked")? {
            let reference = references
                .remove(&completed.episode_id)
                .ok_or_else(|| internal("missing replay reference"))?;

            if let Some((record, rows)) = project_episode(&completed.episode, reference.as_ref()) {
                runtime
                    .replay_tx
                    .send(ReplayJob { record, rows })
                    .map_err(|_| internal("replay sink failed"))?;
            } else {
                episodes_dropped += 1;
            }

            episodes.push(completed);
        }

        for parked in pool.take_unsent_parked() {
            runtime
                .intake_tx
                .send(EvalJob {
                    lane: runtime.lane,
                    slot: parked.slot,
                    token: parked.token,
                    request: parked.request,
                })
                .map_err(|_| internal("eval backend unavailable"))?;
        }

        if roots_exhausted && !pool.active() {
            return Ok(ReplayLaneResult {
                lane: LaneEpisodes {
                    lane: runtime.lane,
                    episodes,
                },
                episodes_dropped,
            });
        }

        if pool.has_parked() {
            let reply = runtime
                .reply_rx
                .recv()
                .map_err(|_| internal("eval backend unavailable"))?;
            pool.resume(reply.slot, reply.token, reply.output)?;

            loop {
                match runtime.reply_rx.try_recv() {
                    Ok(reply) => pool.resume(reply.slot, reply.token, reply.output)?,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        return Err(internal("eval backend unavailable"));
                    }
                }
            }
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
        evaluator
            .evaluate_batch(&requests, &mut outputs)
            .map_err(eval_error_to_engine_error)?;
        validate_outputs(&requests, &outputs).map_err(eval_error_to_engine_error)?;
        batch_sizes.push(batch.len());

        for (job, output) in batch.into_iter().zip(outputs) {
            let _ = reply_txs[job.lane].send(EvalReply {
                slot: job.slot,
                token: job.token,
                output,
            });
        }
    }
}

fn run_replay_sink(store: &ReplayStore, replay_rx: Receiver<ReplayJob>) -> EngineResult<u64> {
    let mut episodes_appended = 0;

    while let Ok(job) = replay_rx.recv() {
        store
            .append_episode(&job.record, &job.rows)
            .map_err(map_replay_error)?;
        episodes_appended += 1;
    }

    Ok(episodes_appended)
}

fn map_replay_error(_error: ReplayError) -> gz_engine::EngineError {
    internal("replay sink failed")
}

#[cfg(test)]
mod tests {
    use super::map_replay_error;
    use gz_replay::ReplayError;

    #[test]
    fn replay_errors_map_to_sink_failure() {
        let error = map_replay_error(ReplayError::InvalidRecord);

        assert!(error.to_string().contains("replay sink failed"));
    }
}
