use super::super::super::schedule::{
    GumbelRng, budget_fraction, considered_actions, considered_visit_sequence, overlap_noise_scale,
    root_seed, sample_root_gumbels, softmax,
};
use super::super::super::{
    GumbelHandleBatch, GumbelMcts, GumbelPlayer, GumbelRootStats, GumbelStep, GumbelValueMode,
};
use super::{
    ActionEdge, Board, CandidateEntry, Expansion, Node, PLAYER_SALT, Pending, RootCandidate, Run,
    State, SymmetricRootAction, SymmetricRootResult, SymmetricSelfplayRootTask, WaveRun,
    candidate_ref,
};
use crate::support::{internal, step_ref};
use crate::work::{
    EngineIdentity, EvalOpponentWork, EvalWork, ExpandResult, ExpandWork, SearchPoll, SearchWork,
    SearchWorkResult, WorkToken,
};
use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{EngineResult, PortableCandidateRef, PortableSearchActionRef, ReplayGraphContext};
use gz_eval::{
    EvalAction, EvalOutput, EvalPositionContext, EvalRequest, eval_error_to_engine_error,
};
use std::collections::HashSet;
use std::hash::Hash;

impl<G, C> SymmetricSelfplayRootTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        graphs: [G; 2],
        contexts: [Option<ReplayGraphContext>; 2],
        rewrites: [usize; 2],
        inactive: [bool; 2],
        stopped: [bool; 2],
        player: GumbelPlayer,
        noise_seed: u64,
        visited: [HashSet<ReplayGraphContext>; 2],
    ) -> Self {
        assert_eq!(
            search.config().value_mode,
            GumbelValueMode::SymmetricSelfplay,
            "symmetric task requires symmetric value mode"
        );
        let board = Board {
            graphs,
            contexts,
            rewrites,
            inactive,
            stopped,
            player,
        };
        Self {
            config: search.config(),
            identity,
            noise_seed,
            visited,
            root_context: contexts[player.index()],
            root_candidates: Vec::new(),
            nodes: Vec::new(),
            created_graphs: Vec::new(),
            speculative_graphs: Vec::new(),
            created_candidates: Vec::new(),
            releasable: GumbelHandleBatch::default(),
            reused: None,
            eval_count: 0,
            carried_nodes: 0,
            carried_root_visits: 0,
            next_token: 0,
            pending: None,
            state: State::Resolve { board },
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, SymmetricRootResult<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }
        let state = std::mem::replace(&mut self.state, State::Done);
        match state {
            State::Resolve { mut board } => {
                while !board.active(self.config.max_steps, board.player) {
                    if board.terminal(self.config.max_steps) {
                        return Err(internal("symmetric root started terminal"));
                    }
                    board.player = board.player.opponent();
                }
                let token = self.next_token();
                self.pending = Some(Pending::Expand { token, board });
                Ok(SearchPoll::Work(SearchWork::Expand(ExpandWork {
                    token,
                    graph: board.current_graph(),
                    options: self.config.candidate_options,
                })))
            }
            State::Eval { board, expansion } => self.poll_eval(board, expansion),
            State::WaveRunning(wave) => self.poll_wave(wave),
            State::DoneResult(result) => {
                self.state = State::Done;
                Ok(SearchPoll::Done(result))
            }
            State::Done => Err(internal("poll after symmetric root completion")),
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        let state = std::mem::replace(&mut self.state, State::Done);
        match state {
            State::WaveRunning(mut wave) if wave.has_pending(token) => {
                let resumed = self.resume_wave(&mut wave, token, result);
                self.state = State::WaveRunning(wave);
                return resumed;
            }
            state => self.state = state,
        }
        let pending = self
            .pending
            .take()
            .ok_or_else(|| internal("resume without pending symmetric work"))?;
        if pending.token() != token {
            self.pending = Some(pending);
            return Err(internal("unknown symmetric work token"));
        }
        match (pending, result) {
            (Pending::Expand { board, .. }, SearchWorkResult::Expand(result)) => {
                self.resume_expand(board, result)
            }
            (
                Pending::Eval {
                    board,
                    expansion,
                    request,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => self.resume_eval(board, expansion, request, output),
            (pending, _) => {
                self.pending = Some(pending);
                Err(internal("mismatched symmetric work result"))
            }
        }
    }

    pub fn take_releasable(&mut self) -> GumbelHandleBatch<G, C> {
        std::mem::take(&mut self.releasable)
    }

    pub fn take_all_handles(&mut self) -> GumbelHandleBatch<G, C> {
        self.release_all_created();
        if let Some(mut reused) = self.reused.take() {
            self.releasable.graphs.append(&mut reused.created_graphs);
            self.releasable
                .candidates
                .append(&mut reused.created_candidates);
        }
        self.take_releasable()
    }

    pub(in super::super) fn take_reused_task(
        &mut self,
        visited: [HashSet<ReplayGraphContext>; 2],
    ) -> EngineResult<Option<Self>> {
        let Some(mut reused) = self.reused.take() else {
            return Ok(None);
        };
        let Some(root) = reused.nodes.first() else {
            self.releasable.graphs.append(&mut reused.created_graphs);
            self.releasable
                .candidates
                .append(&mut reused.created_candidates);
            return Err(internal("empty symmetric reused tree"));
        };
        let root_context = root.context;
        let root_candidates = root.candidate_metadata.clone();
        let mut task = Self {
            config: self.config,
            identity: self.identity,
            noise_seed: self.noise_seed,
            visited,
            root_context: Some(root_context),
            root_candidates,
            nodes: std::mem::take(&mut reused.nodes),
            created_graphs: std::mem::take(&mut reused.created_graphs),
            speculative_graphs: Vec::new(),
            created_candidates: std::mem::take(&mut reused.created_candidates),
            releasable: GumbelHandleBatch::default(),
            reused: None,
            eval_count: 0,
            carried_nodes: reused.carried_nodes,
            carried_root_visits: reused.carried_root_visits,
            next_token: 0,
            pending: None,
            state: State::Done,
        };
        task.refresh_reused_no_backtrack_masks();
        let run = match task.start_run() {
            Ok(run) => run,
            Err(error) => {
                let mut handles = task.take_all_handles();
                self.releasable.graphs.append(&mut handles.graphs);
                self.releasable.candidates.append(&mut handles.candidates);
                return Err(error);
            }
        };
        task.state = State::WaveRunning(WaveRun::new(run));
        Ok(Some(task))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn align_root_board(
        &mut self,
        graphs: [G; 2],
        mut contexts: [Option<ReplayGraphContext>; 2],
        rewrites: [usize; 2],
        inactive: [bool; 2],
        stopped: [bool; 2],
        player: GumbelPlayer,
    ) -> EngineResult<[Option<ReplayGraphContext>; 2]> {
        let root = self
            .nodes
            .first_mut()
            .ok_or_else(|| internal("missing symmetric reused root"))?;
        if root.board.graphs != graphs
            || root.board.rewrites != rewrites
            || root.board.inactive != inactive
            || root.board.stopped != stopped
            || root.board.player != player
        {
            return Err(internal("symmetric reused board mismatch"));
        }
        for (index, context) in contexts.iter_mut().enumerate() {
            match (root.board.contexts[index], *context) {
                (Some(carried), Some(live)) if carried != live => {
                    return Err(internal("symmetric reused board context mismatch"));
                }
                (Some(carried), None) => *context = Some(carried),
                (None, Some(live)) => root.board.contexts[index] = Some(live),
                _ => {}
            }
        }
        if root.board.contexts[player.index()] != Some(root.context) {
            return Err(internal("symmetric reused root context mismatch"));
        }
        self.root_context = Some(root.context);
        Ok(contexts)
    }

    fn refresh_reused_no_backtrack_masks(&mut self) {
        if !self.config.no_backtrack {
            return;
        }
        let root_contexts = self
            .nodes
            .first()
            .map(|root| root.board.contexts)
            .unwrap_or([None, None]);
        for node in &mut self.nodes {
            let player = node.board.player.index();
            let visited = &self.visited[player];
            let current = node.board.contexts[player];
            for (action, edge) in node.edges.iter_mut().enumerate() {
                if edge.after_context.is_some_and(|context| {
                    visited.contains(&context)
                        || root_contexts[player] == Some(context)
                        || current == Some(context)
                }) {
                    node.masked[action] = true;
                    node.priors[action] = 0.0;
                    edge.visits = 0;
                    edge.value_sum = 0.0;
                }
            }
        }
    }

    fn poll_eval(
        &mut self,
        board: Board<G>,
        expansion: Expansion<C>,
    ) -> EngineResult<SearchPoll<G, C, SymmetricRootResult<G, C>>> {
        let player = board.player;
        let request = EvalRequest::with_position(
            expansion.context,
            expansion.eval_actions.clone(),
            self.board_position(board, player),
        )
        .map_err(|_| internal("invalid symmetric eval request"))?;
        let opponent = player.opponent();
        let token = self.next_token();
        let work = EvalWork {
            token,
            graph: board.graphs[player.index()],
            candidates: expansion
                .candidates
                .iter()
                .map(|candidate| candidate.handle)
                .collect(),
            request: request.clone(),
            measure_options: self.config.measure_options,
            opponent: Some(Box::new(EvalOpponentWork {
                graph: board.graphs[opponent.index()],
                position: self.board_position(board, opponent),
            })),
        };
        self.pending = Some(Pending::Eval {
            token,
            board,
            expansion,
            request,
        });
        Ok(SearchPoll::Work(SearchWork::Eval(work)))
    }

    fn resume_expand(&mut self, mut board: Board<G>, result: ExpandResult<C>) -> EngineResult<()> {
        self.created_candidates.extend(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.candidate),
        );
        let context = self.identity.context(result.graph_hash);
        let player = board.player;
        if let Some(expected) = board.contexts[player.index()]
            && expected != context
        {
            return Err(internal("symmetric graph context mismatch"));
        }
        board.contexts[player.index()] = Some(context);
        if board.graphs[0] == board.graphs[1] {
            board.contexts[player.opponent().index()].get_or_insert(context);
        }
        self.root_context = Some(context);

        if result.candidates.is_empty() {
            self.release_all_created();
            self.state = State::DoneResult(SymmetricRootResult::Pass { player, context });
            return Ok(());
        }

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
            candidates.push(CandidateEntry {
                handle: candidate.candidate,
                hash: candidate.candidate_hash,
                summary: SearchCandidateSummary {
                    kind: candidate.kind,
                    tags: candidate.tags,
                    static_prior: candidate.static_prior,
                },
            });
        }
        eval_actions.push(EvalAction::stop(context));
        self.state = State::Eval {
            board,
            expansion: Expansion {
                context,
                candidates,
                eval_actions,
            },
        };
        Ok(())
    }

    fn resume_eval(
        &mut self,
        board: Board<G>,
        expansion: Expansion<C>,
        request: EvalRequest,
        output: EvalOutput,
    ) -> EngineResult<()> {
        output
            .validate_for(&request)
            .map_err(eval_error_to_engine_error)?;
        let candidate_count = expansion.candidates.len();
        let action_count = if self.config.mask_stop {
            candidate_count
        } else {
            candidate_count + 1
        };
        let logits = output.policy_logits[..action_count].to_vec();
        let priors = softmax(&logits);
        let candidate_metadata = expansion
            .candidates
            .iter()
            .map(|candidate| RootCandidate {
                hash: candidate.hash,
                summary: candidate.summary,
            })
            .collect::<Vec<_>>();
        let node_index = self.nodes.len();
        if node_index == 0 {
            self.root_candidates = candidate_metadata.clone();
        }
        self.nodes.push(Node {
            board,
            context: expansion.context,
            candidates: expansion
                .candidates
                .into_iter()
                .map(|candidate| candidate.handle)
                .collect(),
            candidate_metadata,
            logits,
            priors,
            value: output.value,
            model_version: output.model_version,
            masked: vec![false; action_count],
            edges: (0..action_count).map(|_| ActionEdge::default()).collect(),
            pass: None,
        });
        self.eval_count += 1;
        let run = self.start_run()?;
        self.state = State::WaveRunning(WaveRun::new(run));
        Ok(())
    }

    fn start_run(&self) -> EngineResult<Run> {
        let root = self
            .nodes
            .first()
            .ok_or_else(|| internal("missing symmetric root node"))?;
        let player_seed = match root.board.player {
            GumbelPlayer::One => 0,
            GumbelPlayer::Two => PLAYER_SALT,
        };
        let mut rng = GumbelRng::new(root_seed(
            self.config.seed ^ self.noise_seed ^ player_seed,
            root.board.rewrites[root.board.player.index()] as u32,
        ));
        let mut base_scores = root
            .logits
            .iter()
            .zip(&root.masked)
            .map(
                |(logit, masked)| {
                    if *masked { f32::NEG_INFINITY } else { *logit }
                },
            )
            .collect::<Vec<_>>();
        let scale = if self.config.gumbel_noise_overlap >= 0.0 {
            overlap_noise_scale(
                &base_scores,
                self.config.max_considered_actions.get(),
                self.config.gumbel_noise_overlap,
                self.config.gumbel_scale,
            )
        } else {
            self.config.gumbel_scale
        };
        for (score, noise) in
            base_scores
                .iter_mut()
                .zip(sample_root_gumbels(root.logits.len(), scale, &mut rng))
        {
            *score += noise;
        }
        let considered = considered_actions(&base_scores, self.config.max_considered_actions.get())
            .into_iter()
            .filter(|&action| !root.masked[action])
            .collect::<Vec<_>>();
        let schedule = considered_visit_sequence(considered.len(), self.config.simulations.get());
        let root_visit_baseline = root.edges.iter().map(|edge| edge.visits).collect();
        Ok(Run {
            base_scores,
            considered,
            schedule,
            root_visit_baseline,
            schedule_index: 0,
            simulations: 0,
        })
    }

    pub(super) fn finish_root(
        &mut self,
        run: Run,
    ) -> EngineResult<SearchPoll<G, C, SymmetricRootResult<G, C>>> {
        let root = self
            .nodes
            .first()
            .ok_or_else(|| internal("missing symmetric root node"))?;
        if root.edges.len() != run.root_visit_baseline.len() {
            return Err(internal("symmetric root visit baseline length mismatch"));
        }
        if root
            .edges
            .iter()
            .zip(&run.root_visit_baseline)
            .any(|(edge, baseline)| edge.visits < *baseline)
        {
            return Err(internal("symmetric root visits fell below their baseline"));
        }
        let fresh_root_visits = u32::try_from(run.simulations)
            .map_err(|_| internal("symmetric simulation count overflow"))?;
        let baseline_visits = run
            .root_visit_baseline
            .iter()
            .try_fold(0_u32, |total, visits| {
                total
                    .checked_add(*visits)
                    .ok_or_else(|| internal("symmetric root visit baseline overflow"))
            })?;
        let expected_visits = baseline_visits
            .checked_add(fresh_root_visits)
            .ok_or_else(|| internal("symmetric root visit total overflow"))?;
        if root.total_visits() != expected_visits {
            return Err(internal(
                "symmetric root visit ledger did not add the configured budget",
            ));
        }
        let scores = self.root_scores(0, &run.base_scores);
        let mut selectable = run
            .considered
            .iter()
            .copied()
            .filter(|&action| !root.masked[action])
            .collect::<Vec<_>>();
        if selectable.is_empty() {
            selectable = (0..root.action_count())
                .filter(|&action| !root.masked[action])
                .collect();
        }
        let Some(selected) = selectable.into_iter().max_by(|&left, &right| {
            let left_fresh = root.edges[left].visits - run.root_visit_baseline[left];
            let right_fresh = root.edges[right].visits - run.root_visit_baseline[right];
            left_fresh
                .cmp(&right_fresh)
                .then_with(|| scores[left].total_cmp(&scores[right]))
                .then_with(|| right.cmp(&left))
        }) else {
            let player = root.board.player;
            let context = root.context;
            self.release_all_created();
            self.state = State::Done;
            return Ok(SearchPoll::Done(SymmetricRootResult::Pass {
                player,
                context,
            }));
        };
        let selected_stop = root.is_stop(selected);
        let (selected_after, selected_after_context, action_ref, action, selected_candidate) =
            if selected_stop {
                (
                    root.board.current_graph(),
                    root.context,
                    PortableSearchActionRef::stop(root.context),
                    SearchAction::Stop,
                    None,
                )
            } else {
                let edge = &root.edges[selected];
                let selected_after = edge
                    .after_graph
                    .ok_or_else(|| internal("selected symmetric action was not applied"))?;
                let selected_after_context = edge
                    .after_context
                    .ok_or_else(|| internal("selected symmetric action has no context"))?;
                (
                    selected_after,
                    selected_after_context,
                    candidate_ref(root.context, self.root_candidates[selected].hash),
                    SearchAction::Candidate(root.candidates[selected]),
                    Some(self.root_candidates[selected].summary),
                )
            };
        let player = root.board.player;
        let successor = if selected_stop {
            self.stopped_board(0)
        } else {
            let mut board = root.board;
            board.graphs[player.index()] = selected_after;
            board.contexts[player.index()] = Some(selected_after_context);
            board.rewrites[player.index()] += 1;
            board.player = player.opponent();
            board
        };
        let root_search_value = root.search_value();
        let root_q_max = root
            .edges
            .iter()
            .filter_map(|edge| (edge.visits > 0).then_some(edge.q()))
            .reduce(f32::max)
            .unwrap_or(root.value);
        let step = GumbelStep {
            before: root.board.current_graph(),
            after: selected_after,
            action,
            step_ref: step_ref(root.context, action_ref, selected_after_context)?,
            selected_action: action_ref,
            selected_candidate,
            engine_candidate_count: root.candidates.len(),
            action_count: root.action_count(),
            selected_rank: selected,
            legal_actions: root.action_refs(&self.root_candidates),
            policy_target: self.improved_policy(0),
            considered_action_indices: run.considered,
            root_value: root.value,
            root_search_value,
            root_q_max,
            model_version: root.model_version,
        };
        self.prepare_reuse(selected, self.normalize_active_player(successor))?;
        if !selected_stop {
            self.transfer_selected_graph(selected_after)?;
        }
        self.release_all_created();
        let result = SymmetricRootResult::Action(Box::new(SymmetricRootAction {
            step,
            player,
            selected_after,
            selected_after_context,
            stats: GumbelRootStats {
                simulations: run.simulations,
                expanded_nodes: self.nodes.len(),
                eval_count: self.eval_count,
                carried_nodes: self.carried_nodes,
                carried_root_visits: self.carried_root_visits,
            },
        }));
        self.state = State::Done;
        Ok(SearchPoll::Done(result))
    }

    fn release_all_created(&mut self) {
        self.releasable.graphs.append(&mut self.created_graphs);
        self.releasable.graphs.append(&mut self.speculative_graphs);
        self.releasable
            .candidates
            .append(&mut self.created_candidates);
    }

    fn position(&self, step: usize) -> EvalPositionContext {
        if !self.config.export_position {
            return EvalPositionContext {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 0.0,
                budget_step: 0.0,
            };
        }
        EvalPositionContext {
            root_step: step as u32,
            leaf_depth: 0,
            budget_fraction: budget_fraction(self.config.max_steps, step),
            budget_step: 1.0 / self.config.max_steps.max(1) as f32,
        }
    }

    pub(super) fn board_position(
        &self,
        board: Board<G>,
        player: GumbelPlayer,
    ) -> EvalPositionContext {
        let mut position = self.position(board.rewrites[player.index()]);
        if !board.active(self.config.max_steps, player) {
            // Preserve the true rewrite count for the length tiebreak while
            // exposing retirement independently in the budget-step sign.
            position.budget_step = -position.budget_step.abs();
        }
        position
    }

    pub(super) fn stopped_board(&self, node_index: usize) -> Board<G> {
        let mut board = self.nodes[node_index].board;
        let player = board.player;
        board.inactive[player.index()] = true;
        board.stopped[player.index()] = true;
        board.player = player.opponent();
        board
    }

    fn normalize_active_player(&self, mut board: Board<G>) -> Board<G> {
        while !board.active(self.config.max_steps, board.player)
            && !board.terminal(self.config.max_steps)
        {
            board.player = board.player.opponent();
        }
        board
    }

    pub(super) fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }
}
