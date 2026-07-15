use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::{internal, service_engine_work};
use crate::{EpisodeId, WorkerId};
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::{EvalOutput, EvalRequest};
use gz_features::{FeatureExtractor, FeatureRow, PositionFeatures};
use gz_search::{
    CategoricalPolicyEpisodeTask, EngineIdentity, EvalModel, GumbelEpisode, GumbelEpisodeTask,
    GumbelHandleBatch, GumbelMcts, SampledTreeEpisodeTask, SearchPoll, SearchWork,
    SearchWorkResult, WorkToken,
};
use std::hash::Hash;
use std::num::NonZeroUsize;

pub(crate) struct WorkerPool<G, C> {
    slots: Vec<Slot<G, C>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ParkedEval {
    pub episode_id: EpisodeId,
    pub slot: usize,
    pub token: WorkToken,
    pub request: EvalRequest,
    pub row: Option<FeatureRow>,
    pub action_count: u32,
    pub model: EvalModel,
    pub pressure_reserved: bool,
}

pub(crate) struct Admission<'a> {
    pub search: &'a GumbelMcts,
    pub identity: EngineIdentity,
    pub context: gz_search::GumbelEpisodeContext,
    pub sampled_tree: bool,
    pub pressure_reserved: bool,
    pub next_episode_id: &'a mut u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionResult {
    pub roots_exhausted: bool,
    pub admitted: usize,
}

struct Slot<G, C> {
    worker_id: WorkerId,
    state: SlotState<G, C>,
}

struct ActiveEpisode<G, C> {
    task: EpisodeTask<G, C>,
    episode_id: EpisodeId,
    evaluations: u64,
    pressure_reserved: bool,
}

#[allow(clippy::large_enum_variant)]
enum SlotState<G, C> {
    Idle,
    Running(ActiveEpisode<G, C>),
    Parked {
        episode: ActiveEpisode<G, C>,
        token: WorkToken,
        request: EvalRequest,
        row: Option<FeatureRow>,
        action_count: u32,
        model: EvalModel,
        sent: bool,
    },
}

impl<G, C> SlotState<G, C> {
    fn take(&mut self) -> Self {
        std::mem::replace(self, Self::Idle)
    }

    fn take_running(&mut self) -> Option<ActiveEpisode<G, C>> {
        match self.take() {
            Self::Running(episode) => Some(episode),
            other => {
                *self = other;
                None
            }
        }
    }

    fn take_parked(&mut self) -> Option<ActiveEpisode<G, C>> {
        match self.take() {
            Self::Parked { episode, .. } => Some(episode),
            other => {
                *self = other;
                None
            }
        }
    }

    fn parked_token(&self) -> Option<WorkToken> {
        match self {
            Self::Parked { token, .. } => Some(*token),
            _ => None,
        }
    }
}

impl<G, C> WorkerPool<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub(crate) fn new(workers: NonZeroUsize, worker_id_base: u64) -> Self {
        let slots = (0..workers.get())
            .map(|index| Slot {
                worker_id: WorkerId::new(worker_id_base + index as u64),
                state: SlotState::Idle,
            })
            .collect();
        Self { slots }
    }

    pub(crate) fn admit<E, R, F>(
        &mut self,
        engine: &mut E,
        roots: &mut R,
        admission: &mut Admission<'_>,
        episode_context: F,
    ) -> EngineResult<bool>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        R: RootSource<E>,
        F: FnMut(
            &mut E,
            EpisodeId,
            G,
            gz_search::GumbelEpisodeContext,
        ) -> EngineResult<gz_search::GumbelEpisodeContext>,
    {
        Ok(self
            .admit_limited(engine, roots, admission, usize::MAX, episode_context)?
            .roots_exhausted)
    }

    pub(crate) fn admit_limited<E, R, F>(
        &mut self,
        engine: &mut E,
        roots: &mut R,
        admission: &mut Admission<'_>,
        limit: usize,
        mut episode_context: F,
    ) -> EngineResult<AdmissionResult>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        R: RootSource<E>,
        F: FnMut(
            &mut E,
            EpisodeId,
            G,
            gz_search::GumbelEpisodeContext,
        ) -> EngineResult<gz_search::GumbelEpisodeContext>,
    {
        if limit == 0 {
            return Ok(AdmissionResult {
                roots_exhausted: false,
                admitted: 0,
            });
        }

        let mut admitted = 0;
        for slot in &mut self.slots {
            if admitted >= limit {
                break;
            }
            if !matches!(slot.state, SlotState::Idle) {
                continue;
            }

            let Some(root) = roots.next_root(engine)? else {
                return Ok(AdmissionResult {
                    roots_exhausted: true,
                    admitted,
                });
            };

            let episode_id = EpisodeId::new(*admission.next_episode_id);
            *admission.next_episode_id += 1;
            let context = episode_context(
                engine,
                episode_id,
                root,
                gz_search::GumbelEpisodeContext {
                    noise_seed: crate::root::episode_noise_seed(episode_id.value()),
                    ..admission.context
                },
            )?;
            let mut task = if admission.sampled_tree {
                EpisodeTask::SampledTree(SampledTreeEpisodeTask::new(
                    admission.search,
                    admission.identity,
                    root,
                    context,
                ))
            } else {
                EpisodeTask::Gumbel(GumbelEpisodeTask::new(
                    admission.search,
                    admission.identity,
                    root,
                    context,
                ))
            };
            if roots.episode_roots_are_owned() {
                task.track_owned_root();
            }
            slot.state = SlotState::Running(ActiveEpisode {
                task,
                episode_id,
                evaluations: 0,
                pressure_reserved: admission.pressure_reserved,
            });
            admitted += 1;
        }

        Ok(AdmissionResult {
            roots_exhausted: false,
            admitted,
        })
    }

    /// Admits one episode outside the root source -- the opponent rollout
    /// path. The caller supplies the root, search config, context, and
    /// episode id. Returns false when no worker slot is idle.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit_direct(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: gz_search::GumbelEpisodeContext,
        episode_id: EpisodeId,
        owned_root: bool,
        pressure_reserved: bool,
    ) -> bool {
        for slot in &mut self.slots {
            if !matches!(slot.state, SlotState::Idle) {
                continue;
            }

            let mut task =
                EpisodeTask::Gumbel(GumbelEpisodeTask::new(search, identity, root, context));
            if owned_root {
                task.track_owned_root();
            }
            slot.state = SlotState::Running(ActiveEpisode {
                task,
                episode_id,
                evaluations: 0,
                pressure_reserved,
            });
            return true;
        }

        false
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit_direct_categorical(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: gz_search::GumbelEpisodeContext,
        episode_id: EpisodeId,
        owned_root: bool,
        pressure_reserved: bool,
    ) -> bool {
        for slot in &mut self.slots {
            if !matches!(slot.state, SlotState::Idle) {
                continue;
            }
            let mut task = EpisodeTask::Categorical(CategoricalPolicyEpisodeTask::new(
                search, identity, root, context,
            ));
            if owned_root {
                task.track_owned_root();
            }
            slot.state = SlotState::Running(ActiveEpisode {
                task,
                episode_id,
                evaluations: 0,
                pressure_reserved,
            });
            return true;
        }
        false
    }

    pub(crate) fn drive<E, F>(
        &mut self,
        engine: &mut E,
        blocked_message: &'static str,
        mut extractor: Option<&mut dyn FeatureExtractor<E>>,
        mut decorate_row: F,
    ) -> EngineResult<Vec<OrchestratedEpisode<G, C>>>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        F: FnMut(EpisodeId, u32, &mut FeatureRow),
    {
        let mut completed = Vec::new();

        for slot in &mut self.slots {
            while let Some(mut episode) = slot.state.take_running() {
                let poll = match episode.task.poll() {
                    Ok(poll) => poll,
                    Err(error) => {
                        release_task_all(engine, &mut episode.task)?;
                        return Err(error);
                    }
                };
                release_task_releasable(engine, &mut episode.task)?;

                match poll {
                    SearchPoll::Work(work) => {
                        let token = work.token();
                        let result = match service_engine_work(engine, &work) {
                            Ok(result) => result,
                            Err(error) => {
                                release_task_all(engine, &mut episode.task)?;
                                return Err(error);
                            }
                        };
                        if let Some(result) = result {
                            if let Err(error) = episode.task.resume(token, result) {
                                release_task_all(engine, &mut episode.task)?;
                                return Err(error);
                            }
                            release_task_releasable(engine, &mut episode.task)?;
                            slot.state = SlotState::Running(episode);
                            continue;
                        }

                        let SearchWork::Eval(work) = work else {
                            release_task_all(engine, &mut episode.task)?;
                            return Err(internal("unsupported search work"));
                        };
                        episode.evaluations = episode.evaluations.saturating_add(1);
                        let action_count = match u32::try_from(work.request.actions.len()) {
                            Ok(action_count) => action_count,
                            Err(_) => {
                                release_task_all(engine, &mut episode.task)?;
                                return Err(internal("action count overflow"));
                            }
                        };
                        let row = match extractor.as_deref_mut() {
                            Some(extractor) => {
                                let scale = extractor.schema().config().opponent_reward_scale;
                                let position = position_features(
                                    work.request.position,
                                    scale,
                                    work.opponent.is_some(),
                                );
                                match extractor.extract(
                                    engine,
                                    work.graph,
                                    &work.candidates,
                                    position,
                                ) {
                                    Ok(mut row) => {
                                        if let Some(opponent) = work.opponent.as_deref() {
                                            let opponent_position =
                                                position_features(opponent.position, scale, false);
                                            let opponent_row = extractor
                                                .extract(
                                                    engine,
                                                    opponent.graph,
                                                    &[],
                                                    opponent_position,
                                                )
                                                .map_err(|_| {
                                                    internal("opponent feature extraction failed")
                                                });
                                            let opponent_row = match opponent_row {
                                                Ok(row) => row,
                                                Err(error) => {
                                                    release_task_all(engine, &mut episode.task)?;
                                                    return Err(error);
                                                }
                                            };
                                            row.opponent = Some(opponent_state(opponent_row));
                                        }
                                        // Pair evals attach the opponent state at
                                        // the row the search aligned to (real root
                                        // step + leaf depth, advanced to the
                                        // opponent's horizon for STOP re-evals) --
                                        // never the request's exported root_step,
                                        // which export_position zeroes. The task's
                                        // real step is the fallback for references
                                        // without per-step states.
                                        let root_step =
                                            match u32::try_from(episode.task.step_index()) {
                                                Ok(root_step) => root_step,
                                                Err(_) => {
                                                    release_task_all(engine, &mut episode.task)?;
                                                    return Err(internal("root step overflow"));
                                                }
                                            };
                                        let opponent_row = work
                                            .request
                                            .position
                                            .opponent_row()
                                            .unwrap_or(root_step);
                                        if work.opponent.is_none() {
                                            decorate_row(
                                                episode.episode_id,
                                                opponent_row,
                                                &mut row,
                                            );
                                        }
                                        Some(row)
                                    }
                                    Err(_) => {
                                        release_task_all(engine, &mut episode.task)?;
                                        return Err(internal("feature extraction failed"));
                                    }
                                }
                            }
                            None => None,
                        };
                        slot.state = SlotState::Parked {
                            episode,
                            token,
                            request: work.request,
                            row,
                            action_count,
                            model: work.model,
                            sent: false,
                        };
                    }
                    SearchPoll::Blocked => {
                        release_task_all(engine, &mut episode.task)?;
                        return Err(internal(blocked_message));
                    }
                    SearchPoll::Done(result) => {
                        completed.push(OrchestratedEpisode {
                            worker_id: slot.worker_id,
                            episode_id: episode.episode_id,
                            evaluations: episode.evaluations,
                            episode: result,
                        });
                    }
                }
            }
        }

        Ok(completed)
    }

    pub(crate) fn parked(&self) -> Vec<ParkedEval> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match &slot.state {
                SlotState::Parked {
                    episode,
                    token,
                    request,
                    row,
                    action_count,
                    model,
                    ..
                } => Some(ParkedEval {
                    episode_id: episode.episode_id,
                    slot: index,
                    token: *token,
                    request: request.clone(),
                    row: row.clone(),
                    action_count: *action_count,
                    model: *model,
                    pressure_reserved: episode.pressure_reserved,
                }),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn take_unsent_parked(&mut self) -> Vec<ParkedEval> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(index, slot)| match &mut slot.state {
                SlotState::Parked {
                    episode,
                    token,
                    request,
                    row,
                    action_count,
                    model,
                    sent,
                    ..
                } if !*sent => {
                    *sent = true;
                    Some(ParkedEval {
                        episode_id: episode.episode_id,
                        slot: index,
                        token: *token,
                        request: request.clone(),
                        row: row.clone(),
                        action_count: *action_count,
                        model: *model,
                        pressure_reserved: episode.pressure_reserved,
                    })
                }
                _ => None,
            })
            .collect()
    }

    pub(crate) fn resume<E>(
        &mut self,
        engine: &mut E,
        slot_index: usize,
        token: WorkToken,
        output: EvalOutput,
    ) -> EngineResult<()>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
    {
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| internal("unknown work token"))?;

        let Some(expected) = slot.state.parked_token() else {
            return Err(internal("resume without pending work"));
        };
        if expected != token {
            return Err(internal("unknown work token"));
        }

        let mut episode = slot
            .state
            .take_parked()
            .expect("token check ensures the slot is parked");
        if let Err(error) = episode.task.resume(token, SearchWorkResult::Eval(output)) {
            release_task_all(engine, &mut episode.task)?;
            return Err(error);
        }
        release_task_releasable(engine, &mut episode.task)?;
        slot.state = SlotState::Running(episode);
        Ok(())
    }

    pub(crate) fn consume_pressure_reservation(
        &mut self,
        slot_index: usize,
        token: WorkToken,
    ) -> EngineResult<()> {
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| internal("unknown pressure reservation slot"))?;
        let SlotState::Parked {
            episode,
            token: expected,
            ..
        } = &mut slot.state
        else {
            return Err(internal("pressure reservation without pending work"));
        };
        if *expected != token {
            return Err(internal("pressure reservation token mismatch"));
        }
        episode.pressure_reserved = false;
        Ok(())
    }

    pub(crate) fn has_running(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| matches!(slot.state, SlotState::Running(_)))
    }

    pub(crate) fn has_parked(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| matches!(slot.state, SlotState::Parked { .. }))
    }

    pub(crate) fn active(&self) -> bool {
        self.has_running() || self.has_parked()
    }

    pub(crate) fn active_count(&self) -> usize {
        self.slots.len() - self.idle_count()
    }

    pub(crate) fn idle_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| matches!(slot.state, SlotState::Idle))
            .count()
    }
}

fn release_task_releasable<E>(
    engine: &mut E,
    task: &mut EpisodeTask<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_handles(engine, task.take_releasable())
}

fn release_task_all<E>(
    engine: &mut E,
    task: &mut EpisodeTask<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_handles(engine, task.take_all_handles())
}

#[allow(clippy::large_enum_variant)]
enum EpisodeTask<G, C> {
    Gumbel(GumbelEpisodeTask<G, C>),
    Categorical(CategoricalPolicyEpisodeTask<G, C>),
    SampledTree(SampledTreeEpisodeTask<G, C>),
}

impl<G, C> EpisodeTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelEpisode<G, C>>> {
        match self {
            Self::Gumbel(task) => task.poll(),
            Self::Categorical(task) => task.poll(),
            Self::SampledTree(task) => task.poll(),
        }
    }

    fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        match self {
            Self::Gumbel(task) => task.resume(token, result),
            Self::Categorical(task) => task.resume(token, result),
            Self::SampledTree(task) => task.resume(token, result),
        }
    }

    fn step_index(&self) -> usize {
        match self {
            Self::Gumbel(task) => task.step_index(),
            Self::Categorical(task) => task.step_index(),
            Self::SampledTree(task) => task.step_index(),
        }
    }

    fn take_releasable(&mut self) -> GumbelHandleBatch<G, C> {
        match self {
            Self::Gumbel(task) => task.take_releasable(),
            Self::Categorical(task) => task.take_releasable(),
            Self::SampledTree(task) => task.take_releasable(),
        }
    }

    fn track_owned_root(&mut self) {
        match self {
            Self::Gumbel(task) => task.track_owned_root(),
            Self::Categorical(task) => task.track_owned_root(),
            Self::SampledTree(task) => task.track_owned_root(),
        }
    }

    fn take_all_handles(&mut self) -> GumbelHandleBatch<G, C> {
        match self {
            Self::Gumbel(task) => task.take_all_handles(),
            Self::Categorical(task) => task.take_all_handles(),
            Self::SampledTree(task) => task.take_all_handles(),
        }
    }
}

fn release_handles<E>(
    engine: &mut E,
    handles: GumbelHandleBatch<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    if handles.is_empty() {
        return Ok(());
    }
    engine.release(&handles.graphs, &handles.candidates)
}

fn position_features(
    position: gz_eval::EvalPositionContext,
    opponent_reward_scale: f32,
    dynamic_opponent: bool,
) -> PositionFeatures {
    let opponent = position.opponent;
    PositionFeatures {
        root_step: position.root_step,
        leaf_depth: position.leaf_depth,
        budget_fraction: position.budget_fraction,
        budget_step: position.budget_step,
        opponent_reward: opponent.map_or(0.0, |opponent| {
            opponent.final_reward / opponent_reward_scale
        }),
        opponent_present: opponent.is_some() || dynamic_opponent,
    }
}

fn opponent_state(row: FeatureRow) -> gz_features::OpponentStateFeatures {
    gz_features::OpponentStateFeatures {
        node_count: row.node_count,
        node_tokens: row.node_tokens,
        node_attrs: row.node_attrs,
        edges: row.edges,
        position: row.position,
    }
}
