use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::{internal, service_engine_work};
use crate::{EpisodeId, WorkerId};
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::{EvalOutput, EvalRequest};
use gz_features::{FeatureExtractor, FeatureRow, PositionFeatures};
use gz_search::{
    EngineIdentity, EvalModel, GumbelEpisode, GumbelEpisodeTask, GumbelMcts, PolicyRollout,
    PolicyRolloutContext, PolicyRolloutEpisode, PolicyRolloutEpisodeTask, SampledTreeEpisodeTask,
    SearchHandleBatch, SearchPoll, SearchWork, SearchWorkResult, SymmetricEpisode,
    SymmetricSelfplayEpisodeTask, WorkToken,
};
use std::hash::Hash;
use std::num::NonZeroUsize;

pub(crate) struct WorkerPool<G, C> {
    slots: Vec<Slot<G, C>>,
}

pub(crate) struct CompletedTask<G, C> {
    pub worker_id: WorkerId,
    pub episode_id: EpisodeId,
    pub evaluations: u64,
    pub episode: CompletedSearchEpisode<G, C>,
}

pub(crate) enum CompletedSearchEpisode<G, C> {
    Gumbel(GumbelEpisode<G, C>),
    PolicyRollout(PolicyRolloutEpisode<G, C>),
    Symmetric(Box<SymmetricEpisode<G, C>>),
}

type EpisodePoll<G, C> = SearchPoll<G, C, Box<CompletedSearchEpisode<G, C>>>;

impl<G, C> CompletedTask<G, C> {
    pub(crate) fn into_gumbel(self) -> EngineResult<OrchestratedEpisode<G, C>> {
        let CompletedSearchEpisode::Gumbel(episode) = self.episode else {
            return Err(internal(
                "policy rollout completed outside sampled-trajectory mode",
            ));
        };
        Ok(OrchestratedEpisode {
            worker_id: self.worker_id,
            episode_id: self.episode_id,
            evaluations: self.evaluations,
            episode,
        })
    }
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
    pub symmetric_selfplay: bool,
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
        evals: Vec<ParkedEvalState>,
    },
}

struct ParkedEvalState {
    token: WorkToken,
    request: Option<EvalRequest>,
    row: Option<FeatureRow>,
    action_count: u32,
    model: EvalModel,
    sent: bool,
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

    fn has_parked_token(&self, token: WorkToken) -> bool {
        match self {
            Self::Parked { evals, .. } => evals.iter().any(|eval| eval.token == token),
            _ => false,
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
            let mut task = if admission.symmetric_selfplay {
                EpisodeTask::Symmetric(SymmetricSelfplayEpisodeTask::with_wave_batching(
                    admission.search,
                    admission.identity,
                    root,
                    context,
                    admission.search.symmetric_wave_batching(),
                ))
            } else if admission.sampled_tree {
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
    pub(crate) fn admit_direct_policy_rollout(
        &mut self,
        search: &PolicyRollout,
        identity: EngineIdentity,
        root: G,
        context: PolicyRolloutContext,
        episode_id: EpisodeId,
        owned_root: bool,
        pressure_reserved: bool,
    ) -> bool {
        for slot in &mut self.slots {
            if !matches!(slot.state, SlotState::Idle) {
                continue;
            }
            let mut task = EpisodeTask::PolicyRollout(PolicyRolloutEpisodeTask::new(
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
    ) -> EngineResult<Vec<CompletedTask<G, C>>>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        F: FnMut(EpisodeId, u32, &mut FeatureRow),
    {
        let mut completed = Vec::new();

        for slot in &mut self.slots {
            let Some(mut episode) = slot.state.take_running() else {
                continue;
            };
            let mut parked_evals = Vec::new();
            loop {
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
                        parked_evals.push(ParkedEvalState {
                            token,
                            request: Some(work.request),
                            row,
                            action_count,
                            model: work.model,
                            sent: false,
                        });
                    }
                    SearchPoll::Blocked => {
                        if parked_evals.is_empty() {
                            release_task_all(engine, &mut episode.task)?;
                            return Err(internal(blocked_message));
                        }
                        slot.state = SlotState::Parked {
                            episode,
                            evals: parked_evals,
                        };
                        break;
                    }
                    SearchPoll::Done(result) => {
                        if !parked_evals.is_empty() {
                            release_task_all(engine, &mut episode.task)?;
                            return Err(internal("search completed with pending evaluations"));
                        }
                        completed.push(CompletedTask {
                            worker_id: slot.worker_id,
                            episode_id: episode.episode_id,
                            evaluations: episode.evaluations,
                            episode: *result,
                        });
                        break;
                    }
                }
            }
        }

        Ok(completed)
    }

    pub(crate) fn parked(&self) -> Vec<ParkedEval> {
        let mut parked = Vec::new();
        for (index, slot) in self.slots.iter().enumerate() {
            let SlotState::Parked { episode, evals } = &slot.state else {
                continue;
            };
            let mut pressure_reserved = episode.pressure_reserved;
            for eval in evals {
                let Some(request) = &eval.request else {
                    continue;
                };
                parked.push(ParkedEval {
                    episode_id: episode.episode_id,
                    slot: index,
                    token: eval.token,
                    request: request.clone(),
                    row: eval.row.clone(),
                    action_count: eval.action_count,
                    model: eval.model,
                    pressure_reserved,
                });
                pressure_reserved = false;
            }
        }
        parked
    }

    pub(crate) fn take_unsent_parked(&mut self) -> Vec<ParkedEval> {
        let mut parked = Vec::new();
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let SlotState::Parked { episode, evals } = &mut slot.state else {
                continue;
            };
            let mut pressure_reserved = episode.pressure_reserved;
            for eval in evals.iter_mut().filter(|eval| !eval.sent) {
                eval.sent = true;
                let request = eval
                    .request
                    .take()
                    .expect("unsent parked eval retains its request");
                parked.push(ParkedEval {
                    episode_id: episode.episode_id,
                    slot: index,
                    token: eval.token,
                    request,
                    row: eval.row.take(),
                    action_count: eval.action_count,
                    model: eval.model,
                    pressure_reserved,
                });
                pressure_reserved = false;
            }
        }
        parked
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

        if !slot.state.has_parked_token(token) {
            return Err(internal("resume without pending work"));
        }
        let SlotState::Parked { episode, evals } = &mut slot.state else {
            unreachable!("token check ensures the slot is parked");
        };
        let index = evals
            .iter()
            .position(|eval| eval.token == token)
            .expect("token check ensures the eval exists");
        evals.swap_remove(index);
        if let Err(error) = episode.task.resume(token, SearchWorkResult::Eval(output)) {
            release_task_all(engine, &mut episode.task)?;
            slot.state = SlotState::Idle;
            return Err(error);
        }
        release_task_releasable(engine, &mut episode.task)?;
        if evals.is_empty() {
            let SlotState::Parked { episode, .. } = slot.state.take() else {
                unreachable!("slot remains parked until its final eval reply");
            };
            slot.state = SlotState::Running(episode);
        }
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
        let SlotState::Parked { episode, evals, .. } = &mut slot.state else {
            return Err(internal("pressure reservation without pending work"));
        };
        if !evals.iter().any(|eval| eval.token == token) {
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
    PolicyRollout(PolicyRolloutEpisodeTask<G, C>),
    SampledTree(SampledTreeEpisodeTask<G, C>),
    Symmetric(SymmetricSelfplayEpisodeTask<G, C>),
}

impl<G, C> EpisodeTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    fn poll(&mut self) -> EngineResult<EpisodePoll<G, C>> {
        Ok(match self {
            Self::Gumbel(task) => task
                .poll()?
                .map_done(|episode| Box::new(CompletedSearchEpisode::Gumbel(episode))),
            Self::PolicyRollout(task) => task
                .poll()?
                .map_done(|episode| Box::new(CompletedSearchEpisode::PolicyRollout(episode))),
            Self::SampledTree(task) => task
                .poll()?
                .map_done(|episode| Box::new(CompletedSearchEpisode::Gumbel(episode))),
            Self::Symmetric(task) => task
                .poll()?
                .map_done(|episode| Box::new(CompletedSearchEpisode::Symmetric(Box::new(episode)))),
        })
    }

    fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        match self {
            Self::Gumbel(task) => task.resume(token, result),
            Self::PolicyRollout(task) => task.resume(token, result),
            Self::SampledTree(task) => task.resume(token, result),
            Self::Symmetric(task) => task.resume(token, result),
        }
    }

    fn step_index(&self) -> usize {
        match self {
            Self::Gumbel(task) => task.step_index(),
            Self::PolicyRollout(task) => task.step_index(),
            Self::SampledTree(task) => task.step_index(),
            Self::Symmetric(task) => task.step_index(),
        }
    }

    fn take_releasable(&mut self) -> SearchHandleBatch<G, C> {
        match self {
            Self::Gumbel(task) => task.take_releasable(),
            Self::PolicyRollout(task) => task.take_releasable(),
            Self::SampledTree(task) => task.take_releasable(),
            Self::Symmetric(task) => task.take_releasable(),
        }
    }

    fn track_owned_root(&mut self) {
        match self {
            Self::Gumbel(task) => task.track_owned_root(),
            Self::PolicyRollout(task) => task.track_owned_root(),
            Self::SampledTree(task) => task.track_owned_root(),
            Self::Symmetric(task) => task.track_owned_root(),
        }
    }

    fn take_all_handles(&mut self) -> SearchHandleBatch<G, C> {
        match self {
            Self::Gumbel(task) => task.take_all_handles(),
            Self::PolicyRollout(task) => task.take_all_handles(),
            Self::SampledTree(task) => task.take_all_handles(),
            Self::Symmetric(task) => task.take_all_handles(),
        }
    }
}

fn release_handles<E>(
    engine: &mut E,
    handles: SearchHandleBatch<E::Graph, E::Candidate>,
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
