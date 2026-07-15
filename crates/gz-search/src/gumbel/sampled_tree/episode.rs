use super::super::schedule::budget_fraction;
use super::super::{
    GumbelCompetitiveTrace, GumbelEpisode, GumbelEpisodeContext, GumbelHandleBatch, GumbelMcts,
    GumbelMctsConfig, GumbelPlayer, GumbelRootStats, GumbelStep, GumbelStopReason,
};
use super::root::{SampledTreeRootResult, SampledTreeRootTask};
use crate::support::{internal, step_ref};
use crate::work::{
    ApplyWork, EngineIdentity, EvalModel, EvalWork, ExpandResult, ExpandWork, MeasureWork,
    SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    ApplyResult, EngineResult, MeasureResult, PortableCandidateRef, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash,
};
use gz_eval::{
    EvalAction, EvalOutput, EvalPositionContext, EvalRequest, eval_error_to_engine_error,
};
use std::collections::HashSet;
use std::hash::Hash;

const ROLE_SALT: u64 = 0x7361_6d70_5f72_6f6c;

pub struct SampledTreeEpisodeTask<G, C> {
    config: GumbelMctsConfig,
    reference_mask_stop: bool,
    search_config_hash: SearchConfigHash,
    identity: EngineIdentity,
    context: GumbelEpisodeContext,
    learner_player: GumbelPlayer,
    p1: ActorTrace<G, C>,
    p2: ActorTrace<G, C>,
    next_player: GumbelPlayer,
    learner_visited: HashSet<ReplayGraphContext>,
    path_graphs: Vec<G>,
    turn_graphs: Vec<G>,
    turn_candidates: Vec<C>,
    releasable: GumbelHandleBatch<G, C>,
    next_token: u64,
    pending: Option<Pending<G, C>>,
    state: State<G, C>,
}

impl<G, C> SampledTreeEpisodeTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelEpisodeContext,
    ) -> Self {
        assert!(
            !search.config().tree_reuse,
            "sampled-tree does not support tree reuse"
        );
        let config = search.config();
        let learner_player = if mixed_seed(context.noise_seed ^ ROLE_SALT) & 1 == 0 {
            GumbelPlayer::One
        } else {
            GumbelPlayer::Two
        };
        Self {
            config,
            reference_mask_stop: search.reference_mask_stop(),
            search_config_hash: crate::sampled_tree_search_config_hash(
                search.search_config_hash(),
                search.reference_mask_stop(),
            ),
            identity,
            context,
            learner_player,
            p1: ActorTrace::new(root),
            p2: ActorTrace::new(root),
            next_player: GumbelPlayer::One,
            learner_visited: HashSet::new(),
            path_graphs: Vec::new(),
            turn_graphs: Vec::new(),
            turn_candidates: Vec::new(),
            releasable: GumbelHandleBatch::default(),
            next_token: 0,
            pending: None,
            state: State::Turn,
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelEpisode<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }
        loop {
            let state = std::mem::replace(&mut self.state, State::Done);
            match state {
                State::Turn => self.start_turn()?,
                State::LearnerRoot(mut task) => {
                    let poll = match task.poll() {
                        Ok(poll) => poll,
                        Err(error) => {
                            self.append_releasable(task.take_all_handles());
                            return Err(error);
                        }
                    };
                    self.append_releasable(task.take_releasable());
                    match poll {
                        SearchPoll::Work(work) => {
                            let outer = self.next_token();
                            let inner = work.token();
                            let work = retokenize(work, outer);
                            self.pending = Some(Pending::LearnerRoot { outer, inner, task });
                            return Ok(SearchPoll::Work(work));
                        }
                        SearchPoll::Blocked => {
                            self.state = State::LearnerRoot(task);
                            return Ok(SearchPoll::Blocked);
                        }
                        SearchPoll::Done(result) => self.finish_learner_turn(result)?,
                    }
                }
                State::GreedyExpand => {
                    let player = self.next_player;
                    let graph = self.actor(player).current;
                    let token = self.next_token();
                    self.pending = Some(Pending::GreedyExpand { token, player });
                    return Ok(SearchPoll::Work(SearchWork::Expand(ExpandWork {
                        token,
                        graph,
                        options: self.config.candidate_options,
                    })));
                }
                State::GreedyEval { player, root } => {
                    let request = EvalRequest::with_position(
                        root.context,
                        root.eval_actions.clone(),
                        self.actor_position(self.actor(player).moves),
                    )
                    .map_err(|_| internal("invalid sampled-tree greedy eval request"))?;
                    let token = self.next_token();
                    let work = EvalWork {
                        token,
                        graph: self.actor(player).current,
                        candidates: root
                            .candidates
                            .iter()
                            .map(|entry| entry.candidate)
                            .collect(),
                        request: request.clone(),
                        measure_options: self.config.measure_options,
                        model: EvalModel::Incumbent,
                        opponent: None,
                    };
                    self.pending = Some(Pending::GreedyEval {
                        token,
                        player,
                        root,
                        request: Box::new(request),
                    });
                    return Ok(SearchPoll::Work(SearchWork::Eval(work)));
                }
                State::GreedyChoose(mut choice) => {
                    let Some(action) = choice.ranking.get(choice.cursor).copied() else {
                        self.finish_greedy_stop(choice)?;
                        continue;
                    };
                    choice.cursor += 1;
                    let stop = choice.root.candidates.len();
                    if action == stop {
                        self.finish_greedy_stop(choice)?;
                        continue;
                    }
                    let token = self.next_token();
                    let graph = self.actor(choice.player).current;
                    let candidate = choice.root.candidates[action].candidate;
                    self.pending = Some(Pending::GreedyApply {
                        token,
                        choice,
                        action,
                    });
                    return Ok(SearchPoll::Work(SearchWork::Apply(ApplyWork {
                        token,
                        graph,
                        candidate,
                    })));
                }
                State::MeasureP1 => {
                    let token = self.next_token();
                    self.pending = Some(Pending::MeasureP1 { token });
                    return Ok(SearchPoll::Work(SearchWork::Measure(MeasureWork {
                        token,
                        graph: self.p1.current,
                        options: self.config.measure_options,
                    })));
                }
                State::MeasureP2(p1_measure) => {
                    let token = self.next_token();
                    self.pending = Some(Pending::MeasureP2 { token, p1_measure });
                    return Ok(SearchPoll::Work(SearchWork::Measure(MeasureWork {
                        token,
                        graph: self.p2.current,
                        options: self.config.measure_options,
                    })));
                }
                State::DoneResult(episode) => {
                    self.state = State::Done;
                    return Ok(SearchPoll::Done(*episode));
                }
                State::Done => return Err(internal("poll after done")),
            }
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| internal("resume without pending work"))?;
        if pending.token() != token {
            self.pending = Some(pending);
            return Err(internal("unknown work token"));
        }
        let pending = match pending {
            Pending::LearnerRoot {
                inner, mut task, ..
            } => {
                if let Err(error) = task.resume(inner, result) {
                    self.append_releasable(task.take_all_handles());
                    return Err(error);
                }
                self.append_releasable(task.take_releasable());
                self.state = State::LearnerRoot(task);
                return Ok(());
            }
            pending => pending,
        };
        self.track_handles(&result);
        match (pending, result) {
            (Pending::GreedyExpand { player, .. }, SearchWorkResult::Expand(result)) => {
                self.resume_greedy_expand(player, result)
            }
            (
                Pending::GreedyEval {
                    player,
                    root,
                    request,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => self.resume_greedy_eval(player, root, *request, output),
            (Pending::GreedyApply { choice, action, .. }, SearchWorkResult::Apply(applied)) => {
                self.resume_greedy_apply(choice, action, applied)
            }
            (Pending::MeasureP1 { .. }, SearchWorkResult::Measure(measure)) => {
                self.state = State::MeasureP2(measure);
                Ok(())
            }
            (Pending::MeasureP2 { p1_measure, .. }, SearchWorkResult::Measure(p2_measure)) => {
                self.finish_episode(p1_measure, p2_measure)
            }
            (pending, _) => {
                self.pending = Some(pending);
                Err(internal("mismatched work result"))
            }
        }
    }

    pub fn step_index(&self) -> usize {
        self.actor(self.learner_player).moves
    }

    pub fn take_releasable(&mut self) -> GumbelHandleBatch<G, C> {
        self.drain_root_releasable(false);
        std::mem::take(&mut self.releasable)
    }

    pub fn track_owned_root(&mut self) {
        self.path_graphs.push(self.p1.root);
    }

    pub fn take_all_handles(&mut self) -> GumbelHandleBatch<G, C> {
        self.drain_root_releasable(true);
        let mut handles = self.take_releasable();
        handles.graphs.append(&mut self.path_graphs);
        handles.graphs.append(&mut self.turn_graphs);
        handles.candidates.append(&mut self.turn_candidates);
        handles
    }

    fn start_turn(&mut self) -> EngineResult<()> {
        if !self.p1.active(self.config.max_steps) && !self.p2.active(self.config.max_steps) {
            self.state = State::MeasureP1;
            return Ok(());
        }
        let player = self.next_player;
        if !self.actor(player).active(self.config.max_steps) {
            self.next_player = player.opponent();
            self.state = State::Turn;
            return Ok(());
        }
        if player == self.learner_player {
            let learner = self.actor(player);
            let opponent = self.actor(player.opponent());
            let root_search = GumbelMcts::new(self.config)
                .with_policy_rollout_mask_stop(self.reference_mask_stop);
            let task = SampledTreeRootTask::new(
                &root_search,
                self.identity,
                self.learner_player,
                learner.current,
                opponent.current,
                opponent.context,
                learner.moves,
                opponent.moves,
                opponent.stopped,
                self.context.noise_seed,
                self.learner_visited.clone(),
            );
            self.state = State::LearnerRoot(task);
        } else {
            self.state = State::GreedyExpand;
        }
        Ok(())
    }

    fn finish_learner_turn(&mut self, result: SampledTreeRootResult<G, C>) -> EngineResult<()> {
        let player = self.learner_player;
        let max_steps = self.config.max_steps;
        let before_context = result.step.step_ref.before;
        if self.actor(player.opponent()).context.is_none()
            && self.actor(player.opponent()).current == result.step.before
        {
            self.actor_mut(player.opponent()).context = Some(before_context);
        }
        let actor = self.actor_mut(player);
        actor.root_context.get_or_insert(before_context);
        actor.context = Some(result.selected_after_context);
        actor.current = result.selected_after;
        actor.moves += 1;
        actor.steps.push(result.step);
        actor.root_stats.push(result.stats);
        if result.selected_stop {
            actor.stopped = true;
            actor.stop_reason = GumbelStopReason::SelectedStop;
        } else if actor.moves >= max_steps {
            actor.stop_reason = GumbelStopReason::MaxSteps;
        }
        if self.config.no_backtrack {
            self.learner_visited.insert(before_context);
        }
        if !result.selected_stop {
            self.path_graphs.push(result.selected_after);
        }
        self.next_player = player.opponent();
        self.state = State::Turn;
        Ok(())
    }

    fn resume_greedy_expand(
        &mut self,
        player: GumbelPlayer,
        result: ExpandResult<C>,
    ) -> EngineResult<()> {
        let context = self.identity.context(result.graph_hash);
        let actor = self.actor_mut(player);
        if let Some(expected) = actor.context
            && expected != context
        {
            return Err(internal("sampled-tree greedy context mismatch"));
        }
        actor.context = Some(context);
        actor.root_context.get_or_insert(context);
        let mut candidates = Vec::with_capacity(result.candidates.len());
        let mut eval_actions = Vec::with_capacity(result.candidates.len() + 1);
        for candidate in result.candidates {
            let candidate_ref = PortableCandidateRef::new(context, candidate.candidate_hash);
            eval_actions.push(EvalAction::candidate(
                candidate_ref,
                candidate.kind,
                candidate.tags,
                candidate.static_prior,
            ));
            candidates.push(GreedyCandidate {
                candidate: candidate.candidate,
                action_ref: PortableSearchActionRef::candidate(candidate_ref),
                summary: SearchCandidateSummary {
                    kind: candidate.kind,
                    tags: candidate.tags,
                    static_prior: candidate.static_prior,
                },
            });
        }
        eval_actions.push(EvalAction::stop(context));
        self.state = State::GreedyEval {
            player,
            root: GreedyRoot {
                context,
                candidates,
                eval_actions,
            },
        };
        Ok(())
    }

    fn resume_greedy_eval(
        &mut self,
        player: GumbelPlayer,
        root: GreedyRoot<C>,
        request: EvalRequest,
        output: EvalOutput,
    ) -> EngineResult<()> {
        output
            .validate_for(&request)
            .map_err(eval_error_to_engine_error)?;
        let stop = root.candidates.len();
        let mut ranking = (0..output.policy_logits.len()).collect::<Vec<_>>();
        ranking.sort_by(|&left, &right| {
            let left_score =
                if self.reference_mask_stop && !root.candidates.is_empty() && left == stop {
                    f32::NEG_INFINITY
                } else {
                    output.policy_logits[left]
                };
            let right_score =
                if self.reference_mask_stop && !root.candidates.is_empty() && right == stop {
                    f32::NEG_INFINITY
                } else {
                    output.policy_logits[right]
                };
            right_score
                .total_cmp(&left_score)
                .then_with(|| left.cmp(&right))
        });
        if self.reference_mask_stop && !root.candidates.is_empty() {
            ranking.retain(|action| *action != stop);
        }
        self.state = State::GreedyChoose(Box::new(GreedyChoice {
            player,
            root,
            output,
            ranking,
            cursor: 0,
        }));
        Ok(())
    }

    fn resume_greedy_apply(
        &mut self,
        choice: Box<GreedyChoice<C>>,
        action: usize,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<()> {
        if applied.rejected.is_some() {
            self.state = State::GreedyChoose(choice);
            return Ok(());
        }
        let after_context = self.identity.context(applied.after_hash);
        let player = choice.player;
        let max_steps = self.config.max_steps;
        let before = self.actor(player).current;
        let before_context = self
            .actor(player)
            .context
            .ok_or_else(|| internal("missing sampled-tree greedy context"))?;
        let entry = choice.root.candidates[action];
        let step = greedy_step(
            before,
            applied.after,
            before_context,
            after_context,
            SearchAction::Candidate(entry.candidate),
            entry.action_ref,
            Some(entry.summary),
            action,
            &choice,
        )?;
        let actor = self.actor_mut(player);
        actor.current = applied.after;
        actor.context = Some(after_context);
        actor.moves += 1;
        actor.steps.push(step);
        if actor.moves >= max_steps {
            actor.stop_reason = GumbelStopReason::MaxSteps;
        }
        self.partition_turn_graphs(Some(applied.after));
        self.release_turn_candidates();
        self.next_player = player.opponent();
        self.state = State::Turn;
        Ok(())
    }

    fn finish_greedy_stop(&mut self, choice: Box<GreedyChoice<C>>) -> EngineResult<()> {
        let player = choice.player;
        let actor = self.actor(player);
        let context = actor
            .context
            .ok_or_else(|| internal("missing sampled-tree greedy context"))?;
        let stop = choice.root.candidates.len();
        let step = greedy_step(
            actor.current,
            actor.current,
            context,
            context,
            SearchAction::Stop,
            PortableSearchActionRef::stop(context),
            None,
            stop,
            &choice,
        )?;
        let actor = self.actor_mut(player);
        actor.moves += 1;
        actor.steps.push(step);
        actor.stopped = true;
        actor.stop_reason = GumbelStopReason::SelectedStop;
        self.partition_turn_graphs(None);
        self.release_turn_candidates();
        self.next_player = player.opponent();
        self.state = State::Turn;
        Ok(())
    }

    fn finish_episode(
        &mut self,
        p1_measure: MeasureResult<G>,
        p2_measure: MeasureResult<G>,
    ) -> EngineResult<()> {
        let (learner, opponent, learner_measure, opponent_measure) =
            if self.learner_player == GumbelPlayer::One {
                (&mut self.p1, &mut self.p2, p1_measure, p2_measure)
            } else {
                (&mut self.p2, &mut self.p1, p2_measure, p1_measure)
            };
        let learner_context = learner
            .context
            .unwrap_or_else(|| self.identity.context(learner_measure.graph_hash));
        let opponent_context = opponent
            .context
            .unwrap_or_else(|| self.identity.context(opponent_measure.graph_hash));
        let learner_root_context = learner.root_context.unwrap_or(learner_context);
        let opponent_root_context = opponent.root_context.unwrap_or(opponent_context);
        let competitive = GumbelCompetitiveTrace {
            learner_player: self.learner_player,
            opponent_root: opponent.root,
            opponent_final_graph: opponent.current,
            opponent_root_context,
            opponent_final_context: opponent_context,
            opponent_steps: std::mem::take(&mut opponent.steps),
            opponent_final_measure: opponent_measure,
            opponent_stop_reason: opponent.stop_reason,
        };
        self.state = State::DoneResult(Box::new(GumbelEpisode {
            root: learner.root,
            final_graph: learner.current,
            root_context: learner_root_context,
            final_context: learner_context,
            steps: std::mem::take(&mut learner.steps),
            root_stats: std::mem::take(&mut learner.root_stats),
            created_graphs: std::mem::take(&mut self.path_graphs),
            created_candidates: std::mem::take(&mut self.turn_candidates),
            final_measure: learner_measure,
            stop_reason: learner.stop_reason,
            search_config_hash: self.search_config_hash,
            competitive: Some(Box::new(competitive)),
        }));
        Ok(())
    }

    fn track_handles(&mut self, result: &SearchWorkResult<G, C>) {
        match result {
            SearchWorkResult::Expand(result) => self.turn_candidates.extend(
                result
                    .candidates
                    .iter()
                    .map(|candidate| candidate.candidate),
            ),
            SearchWorkResult::Apply(result) => self.turn_graphs.push(result.after),
            SearchWorkResult::Measure(_) | SearchWorkResult::Eval(_) => {}
        }
    }

    fn append_releasable(&mut self, mut handles: GumbelHandleBatch<G, C>) {
        self.releasable.graphs.append(&mut handles.graphs);
        self.releasable.candidates.append(&mut handles.candidates);
    }

    fn drain_root_releasable(&mut self, all: bool) {
        let handles = match &mut self.state {
            State::LearnerRoot(task) => Some(if all {
                task.take_all_handles()
            } else {
                task.take_releasable()
            }),
            _ => match &mut self.pending {
                Some(Pending::LearnerRoot { task, .. }) => Some(if all {
                    task.take_all_handles()
                } else {
                    task.take_releasable()
                }),
                _ => None,
            },
        };
        if let Some(handles) = handles {
            self.append_releasable(handles);
        }
    }

    fn partition_turn_graphs(&mut self, selected: Option<G>) {
        let mut selected = selected;
        for graph in self.turn_graphs.drain(..) {
            if selected == Some(graph) {
                self.path_graphs.push(graph);
                selected = None;
            } else {
                self.releasable.graphs.push(graph);
            }
        }
    }

    fn release_turn_candidates(&mut self) {
        self.releasable.candidates.append(&mut self.turn_candidates);
    }

    fn actor(&self, player: GumbelPlayer) -> &ActorTrace<G, C> {
        match player {
            GumbelPlayer::One => &self.p1,
            GumbelPlayer::Two => &self.p2,
        }
    }

    fn actor_mut(&mut self, player: GumbelPlayer) -> &mut ActorTrace<G, C> {
        match player {
            GumbelPlayer::One => &mut self.p1,
            GumbelPlayer::Two => &mut self.p2,
        }
    }

    fn actor_position(&self, step: usize) -> EvalPositionContext {
        if !self.config.export_position {
            return EvalPositionContext {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 0.0,
                budget_step: 0.0,
                opponent: None,
            };
        }
        EvalPositionContext {
            root_step: step as u32,
            leaf_depth: 0,
            budget_fraction: budget_fraction(self.config.max_steps, step),
            budget_step: 1.0 / self.config.max_steps.max(1) as f32,
            opponent: None,
        }
    }

    fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }
}

#[allow(clippy::too_many_arguments)]
fn greedy_step<G: Copy, C: Copy>(
    before: G,
    after: G,
    before_context: ReplayGraphContext,
    after_context: ReplayGraphContext,
    action: SearchAction<C>,
    action_ref: PortableSearchActionRef,
    summary: Option<SearchCandidateSummary>,
    selected: usize,
    choice: &GreedyChoice<C>,
) -> EngineResult<GumbelStep<G, C>> {
    let mut legal_actions = choice
        .root
        .candidates
        .iter()
        .map(|entry| entry.action_ref)
        .collect::<Vec<_>>();
    legal_actions.push(PortableSearchActionRef::stop(before_context));
    Ok(GumbelStep {
        before,
        after,
        action,
        step_ref: step_ref(before_context, action_ref, after_context)?,
        selected_action: action_ref,
        selected_candidate: summary,
        engine_candidate_count: choice.root.candidates.len(),
        action_count: legal_actions.len(),
        selected_rank: selected,
        policy_target: vec![0.0; legal_actions.len()],
        legal_actions,
        considered_action_indices: Vec::new(),
        root_value: choice.output.value,
        root_search_value: choice.output.value,
        root_q_max: choice.output.value,
        model_version: choice.output.model_version,
    })
}

fn retokenize<G, C>(work: SearchWork<G, C>, token: WorkToken) -> SearchWork<G, C> {
    match work {
        SearchWork::Expand(mut work) => {
            work.token = token;
            SearchWork::Expand(work)
        }
        SearchWork::Apply(mut work) => {
            work.token = token;
            SearchWork::Apply(work)
        }
        SearchWork::Measure(mut work) => {
            work.token = token;
            SearchWork::Measure(work)
        }
        SearchWork::Eval(mut work) => {
            work.token = token;
            SearchWork::Eval(work)
        }
    }
}

fn mixed_seed(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct ActorTrace<G, C> {
    root: G,
    current: G,
    context: Option<ReplayGraphContext>,
    root_context: Option<ReplayGraphContext>,
    moves: usize,
    stopped: bool,
    stop_reason: GumbelStopReason,
    steps: Vec<GumbelStep<G, C>>,
    root_stats: Vec<GumbelRootStats>,
}

impl<G: Copy, C> ActorTrace<G, C> {
    fn new(root: G) -> Self {
        Self {
            root,
            current: root,
            context: None,
            root_context: None,
            moves: 0,
            stopped: false,
            stop_reason: GumbelStopReason::MaxSteps,
            steps: Vec::new(),
            root_stats: Vec::new(),
        }
    }

    fn active(&self, max_steps: usize) -> bool {
        !self.stopped && self.moves < max_steps
    }
}

#[derive(Clone, Copy)]
struct GreedyCandidate<C> {
    candidate: C,
    action_ref: PortableSearchActionRef,
    summary: SearchCandidateSummary,
}

struct GreedyRoot<C> {
    context: ReplayGraphContext,
    candidates: Vec<GreedyCandidate<C>>,
    eval_actions: Vec<EvalAction>,
}

struct GreedyChoice<C> {
    player: GumbelPlayer,
    root: GreedyRoot<C>,
    output: EvalOutput,
    ranking: Vec<usize>,
    cursor: usize,
}

#[allow(clippy::large_enum_variant)]
enum State<G, C> {
    Turn,
    LearnerRoot(SampledTreeRootTask<G, C>),
    GreedyExpand,
    GreedyEval {
        player: GumbelPlayer,
        root: GreedyRoot<C>,
    },
    GreedyChoose(Box<GreedyChoice<C>>),
    MeasureP1,
    MeasureP2(MeasureResult<G>),
    DoneResult(Box<GumbelEpisode<G, C>>),
    Done,
}

#[allow(clippy::large_enum_variant)]
enum Pending<G, C> {
    LearnerRoot {
        outer: WorkToken,
        inner: WorkToken,
        task: SampledTreeRootTask<G, C>,
    },
    GreedyExpand {
        token: WorkToken,
        player: GumbelPlayer,
    },
    GreedyEval {
        token: WorkToken,
        player: GumbelPlayer,
        root: GreedyRoot<C>,
        request: Box<EvalRequest>,
    },
    GreedyApply {
        token: WorkToken,
        choice: Box<GreedyChoice<C>>,
        action: usize,
    },
    MeasureP1 {
        token: WorkToken,
    },
    MeasureP2 {
        token: WorkToken,
        p1_measure: MeasureResult<G>,
    },
}

impl<G, C> Pending<G, C> {
    fn token(&self) -> WorkToken {
        match self {
            Self::LearnerRoot { outer, .. } => *outer,
            Self::GreedyExpand { token, .. }
            | Self::GreedyEval { token, .. }
            | Self::GreedyApply { token, .. }
            | Self::MeasureP1 { token }
            | Self::MeasureP2 { token, .. } => *token,
        }
    }
}
