use super::projection::{
    append_symmetric_replay_job, feature_rows_for_symmetric_episode, measured_symmetric_game,
    release_symmetric_episode_handles,
};
use super::*;

pub(super) struct LaneRuntime<'a> {
    pub(super) lane: usize,
    pub(super) lanes: usize,
    pub(super) search: &'a GumbelMcts,
    pub(super) workers_per_lane: NonZeroUsize,
    pub(super) pool_capacity: NonZeroUsize,
    pub(super) admission_stagger: Duration,
    pub(super) admission_shaper: Option<Arc<SharedAdmissionShaper>>,
    pub(super) eval_pressure: Arc<EvalPressure>,
    pub(super) intake_tx: SyncSender<FeaturizedEvalJob>,
    pub(super) reply_rx: Receiver<EvalReply>,
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

pub(super) fn run_lane_pipeline<E, R, X>(
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

pub(super) struct FeaturizedReplayMode<'a, X> {
    extractor: X,
    replay_tx: SyncSender<ReplayJob>,
    store: &'a ReplayStore,
    backpressure: Option<ReplayBackpressure>,
    admitted_at: HashMap<EpisodeId, Instant>,
    summary: ReplayLaneSummary,
    model_leases: EpisodeModelLeases,
}

impl<'a, X> FeaturizedReplayMode<'a, X> {
    pub(super) fn new(
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

pub(super) fn merge_lane_measurer_summary(
    lane: &mut ReplayLaneSummary,
    measurer: &MeasurerRunSummary,
) {
    let Some(measured) = measurer.lanes.get(lane.lane) else {
        return;
    };
    lane.episodes_appended = measured.episodes_appended;
    lane.episodes_dropped = measured.episodes_dropped;
    lane.replay_rows = measured.replay_rows;
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
