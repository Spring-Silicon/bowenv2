use super::super::schedule::{
    GumbelRng, considered_actions, considered_visit_sequence, overlap_noise_scale, root_seed,
    sample_root_gumbels, softmax,
};
use super::super::{
    GumbelHandleBatch, GumbelMcts, GumbelMctsConfig, GumbelPlayer, GumbelRootStats, GumbelStep,
    GumbelValueMode,
};
use crate::support::{internal, step_ref};
use crate::work::{
    ApplyWork, EngineIdentity, EvalOpponentWork, EvalWork, ExpandResult, ExpandWork, MeasureWork,
    SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    ApplyResult, CandidateHash, EngineResult, MeasureResult, PortableCandidateRef,
    PortableSearchActionRef, ReplayGraphContext,
};
use gz_eval::{
    EvalAction, EvalOutput, EvalPositionContext, EvalRequest, eval_error_to_engine_error,
};
use std::collections::HashSet;
use std::hash::Hash;

const PLAYER_SALT: u64 = 0x7379_6d6d_5f70_6c79;

pub enum SymmetricRootResult<G, C> {
    Action(Box<SymmetricRootAction<G, C>>),
    Pass {
        player: GumbelPlayer,
        context: ReplayGraphContext,
    },
}

pub struct SymmetricRootAction<G, C> {
    pub step: GumbelStep<G, C>,
    pub player: GumbelPlayer,
    pub selected_after: G,
    pub selected_after_context: ReplayGraphContext,
    pub stats: GumbelRootStats,
}

pub struct SymmetricSelfplayRootTask<G, C> {
    config: GumbelMctsConfig,
    identity: EngineIdentity,
    noise_seed: u64,
    visited: [HashSet<ReplayGraphContext>; 2],
    root_context: Option<ReplayGraphContext>,
    root_candidates: Vec<RootCandidate>,
    nodes: Vec<Node<G, C>>,
    created_graphs: Vec<G>,
    speculative_graphs: Vec<G>,
    created_candidates: Vec<C>,
    releasable: GumbelHandleBatch<G, C>,
    reused: Option<ReusedTree<G, C>>,
    eval_count: usize,
    portable_contexts: usize,
    carried_nodes: usize,
    carried_root_visits: u32,
    next_token: u64,
    wave_batching: bool,
    pending: Option<Pending<G, C>>,
    state: State<G, C>,
}

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
        Self::with_wave_batching(
            search, identity, graphs, contexts, rewrites, inactive, stopped, player, noise_seed,
            visited, false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_wave_batching(
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
        wave_batching: bool,
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
            portable_contexts: 0,
            carried_nodes: 0,
            carried_root_visits: 0,
            next_token: 0,
            wave_batching,
            pending: None,
            state: State::Resolve {
                board,
                attach: None,
                run: None,
            },
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, SymmetricRootResult<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }
        loop {
            let state = std::mem::replace(&mut self.state, State::Done);
            match state {
                State::Resolve {
                    mut board,
                    mut attach,
                    mut run,
                } => {
                    while !board.active(self.config.max_steps, board.player) {
                        if board.terminal(self.config.max_steps) {
                            let terminal_run = run
                                .take()
                                .ok_or_else(|| internal("symmetric root started terminal"))?;
                            self.state = State::MeasureP1 {
                                board,
                                attach,
                                run: terminal_run,
                            };
                            break;
                        }
                        board.player = board.player.opponent();
                        if let Some(attach) = &mut attach {
                            attach.turns = attach.turns.saturating_add(1);
                        }
                    }
                    if matches!(self.state, State::MeasureP1 { .. }) {
                        continue;
                    }
                    let token = self.next_token();
                    self.pending = Some(Pending::Expand {
                        token,
                        board,
                        attach,
                        run,
                    });
                    return Ok(SearchPoll::Work(SearchWork::Expand(ExpandWork {
                        token,
                        graph: board.current_graph(),
                        options: self.config.candidate_options,
                    })));
                }
                State::Eval {
                    board,
                    attach,
                    run,
                    expansion,
                } => return self.poll_eval(board, attach, run, expansion),
                State::Running(mut run) => {
                    if run.descent.is_none() && !self.start_descent(&mut run) {
                        return self.finish_root(run);
                    }
                    self.continue_descent(run)?;
                }
                State::WaveRunning(wave) => return self.poll_wave(wave),
                State::MeasureP1 { board, attach, run } => {
                    let token = self.next_token();
                    self.pending = Some(Pending::MeasureP1 {
                        token,
                        board,
                        attach,
                        run,
                    });
                    return Ok(SearchPoll::Work(SearchWork::Measure(MeasureWork {
                        token,
                        graph: board.graphs[0],
                        options: self.config.measure_options,
                    })));
                }
                State::MeasureP2 {
                    board,
                    attach,
                    run,
                    p1_measure,
                } => {
                    let token = self.next_token();
                    self.pending = Some(Pending::MeasureP2 {
                        token,
                        board,
                        attach,
                        run,
                        p1_measure,
                    });
                    return Ok(SearchPoll::Work(SearchWork::Measure(MeasureWork {
                        token,
                        graph: board.graphs[1],
                        options: self.config.measure_options,
                    })));
                }
                State::Work(work) => return Ok(SearchPoll::Work(work)),
                State::DoneResult(result) => {
                    self.state = State::Done;
                    return Ok(SearchPoll::Done(result));
                }
                State::Done => return Err(internal("poll after symmetric root completion")),
            }
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        if self.wave_batching {
            let state = std::mem::replace(&mut self.state, State::Done);
            match state {
                State::WaveRunning(mut wave) if wave.has_pending(token) => {
                    let resumed = self.resume_wave(&mut wave, token, result);
                    self.state = State::WaveRunning(wave);
                    return resumed;
                }
                state => self.state = state,
            }
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
            (
                Pending::Expand {
                    board, attach, run, ..
                },
                SearchWorkResult::Expand(result),
            ) => self.resume_expand(board, attach, run, result),
            (
                Pending::Eval {
                    board,
                    attach,
                    run,
                    expansion,
                    request,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => self.resume_eval(board, attach, run, expansion, request, output),
            (
                Pending::Apply {
                    run, node, action, ..
                },
                SearchWorkResult::Apply(applied),
            ) => self.resume_apply(run, node, action, applied),
            (
                Pending::MeasureP1 {
                    board, attach, run, ..
                },
                SearchWorkResult::Measure(p1_measure),
            ) => {
                self.state = State::MeasureP2 {
                    board,
                    attach,
                    run,
                    p1_measure,
                };
                Ok(())
            }
            (
                Pending::MeasureP2 {
                    board,
                    attach,
                    mut run,
                    p1_measure,
                    ..
                },
                SearchWorkResult::Measure(p2_measure),
            ) => {
                let value = terminal_value(&board, &p1_measure, &p2_measure)?;
                self.attach_and_backup(&mut run, attach, BranchTarget::Terminal(value))?;
                self.state = State::Running(run);
                Ok(())
            }
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

    pub(super) fn take_reused_task(
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
            portable_contexts: 0,
            carried_nodes: reused.carried_nodes,
            carried_root_visits: reused.carried_root_visits,
            next_token: 0,
            wave_batching: self.wave_batching,
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
        task.state = if task.wave_batching {
            State::WaveRunning(WaveRun::new(run))
        } else {
            State::Running(run)
        };
        Ok(Some(task))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn align_root_board(
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
        attach: Option<Attach>,
        run: Option<Run>,
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
            attach,
            run,
            expansion,
            request,
        });
        Ok(SearchPoll::Work(SearchWork::Eval(work)))
    }

    fn resume_expand(
        &mut self,
        mut board: Board<G>,
        mut attach: Option<Attach>,
        run: Option<Run>,
        result: ExpandResult<C>,
    ) -> EngineResult<()> {
        self.created_candidates.extend(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.candidate),
        );
        let context = self.identity.context(result.graph_hash);
        self.portable_contexts += 1;
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
        if attach.is_none() {
            self.root_context = Some(context);
        }

        if result.candidates.is_empty() {
            if run.is_none() {
                self.release_all_created();
                self.state = State::DoneResult(SymmetricRootResult::Pass { player, context });
                return Ok(());
            }
            board.inactive[player.index()] = true;
            board.player = player.opponent();
            if let Some(attach) = &mut attach {
                attach.turns = attach.turns.saturating_add(1);
            }
            self.state = State::Resolve { board, attach, run };
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
            attach,
            run,
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
        attach: Option<Attach>,
        mut run: Option<Run>,
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
        if let Some(mut run) = run.take() {
            self.attach_and_backup(&mut run, attach, BranchTarget::Node(node_index))?;
            self.state = State::Running(run);
        } else {
            let run = self.start_run()?;
            self.state = if self.wave_batching {
                State::WaveRunning(WaveRun::new(run))
            } else {
                State::Running(run)
            };
        }
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
            descent: None,
        })
    }

    fn start_descent(&self, run: &mut Run) -> bool {
        let Some(&target) = run.schedule.get(run.schedule_index) else {
            return false;
        };
        self.replenish_considered(run);
        let root = &self.nodes[0];
        let scores = self.root_scores(0, &run.base_scores);
        let action = run
            .considered
            .iter()
            .copied()
            .filter(|&action| {
                !root.masked[action]
                    && root.edges[action].visits
                        == run.root_visit_baseline[action].saturating_add(target)
            })
            .max_by(|&left, &right| {
                scores[left]
                    .total_cmp(&scores[right])
                    .then_with(|| right.cmp(&left))
            })
            .or_else(|| {
                run.considered
                    .iter()
                    .copied()
                    .filter(|&action| !root.masked[action])
                    .max_by(|&left, &right| {
                        scores[left]
                            .total_cmp(&scores[right])
                            .then_with(|| right.cmp(&left))
                    })
            });
        let Some(action) = action else {
            return false;
        };
        let mut seen = self.visited.clone();
        for player in [GumbelPlayer::One, GumbelPlayer::Two] {
            if let Some(context) = root.board.contexts[player.index()] {
                seen[player.index()].insert(context);
            }
        }
        run.descent = Some(Descent {
            node: 0,
            path: Vec::new(),
            seen,
            forced: Some(action),
        });
        true
    }

    fn poll_wave(
        &mut self,
        mut wave: WaveRun<G, C>,
    ) -> EngineResult<SearchPoll<G, C, SymmetricRootResult<G, C>>> {
        loop {
            if wave.active.is_empty() && !self.launch_wave(&mut wave) {
                return self.finish_root(wave.run);
            }
            if wave.root_apply_preflight_ready() {
                self.resolve_root_apply_preflight(&mut wave)?;
            }

            let defer_completions = wave.any_pending();
            let mut progressed = false;
            for index in 0..wave.active.len() {
                match self.advance_wave_simulation(&mut wave.active[index], defer_completions)? {
                    WaveAdvance::Work(work) => {
                        self.state = State::WaveRunning(wave);
                        return Ok(SearchPoll::Work(work));
                    }
                    WaveAdvance::Progressed => progressed = true,
                    WaveAdvance::Waiting => {}
                }
            }

            if wave.any_pending() {
                self.state = State::WaveRunning(wave);
                return Ok(SearchPoll::Blocked);
            }

            if wave
                .active
                .iter()
                .all(|simulation| matches!(simulation.state, WaveSimulationState::Outcome { .. }))
            {
                for simulation in wave.active.drain(..) {
                    let WaveSimulationState::Outcome { descent, value } = simulation.state else {
                        unreachable!("checked wave outcome state");
                    };
                    self.backup_descent(&descent, value);
                    wave.run.simulations += 1;
                    wave.run.schedule_index += 1;
                }
                continue;
            }

            if !progressed {
                self.state = State::WaveRunning(wave);
                return Err(internal("symmetric wave made no progress"));
            }
        }
    }

    fn launch_wave(&self, wave: &mut WaveRun<G, C>) -> bool {
        let Some(&target) = wave.run.schedule.get(wave.run.schedule_index) else {
            return false;
        };
        self.replenish_considered(&mut wave.run);
        let width = wave.run.schedule[wave.run.schedule_index..]
            .iter()
            .take_while(|&&visit| visit == target)
            .count();
        let root = &self.nodes[0];
        let mut virtual_visits = root
            .edges
            .iter()
            .map(|edge| edge.visits)
            .collect::<Vec<_>>();

        let mut actions = Vec::with_capacity(width);
        for _ in 0..width {
            let Some(action) = self.wave_root_action(&wave.run, target, &virtual_visits) else {
                break;
            };
            if actions.contains(&action) {
                break;
            }
            actions.push(action);
            virtual_visits[action] = virtual_visits[action].saturating_add(1);
        }

        // STOP performs no engine apply. Keep it at a wave boundary so the
        // speculative first-visit preflight remains candidate-only and the
        // sequential schedule order is preserved exactly.
        if let Some(stop) = actions.iter().position(|&action| root.is_stop(action)) {
            actions.truncate(stop.max(1));
        }

        let unexpanded = actions
            .iter()
            .filter(|&&action| root.edges[action].branch.is_none())
            .count();
        if unexpanded > 0 && unexpanded < actions.len() {
            actions.truncate(1);
        }
        wave.root_apply_preflight = actions.len() > 1 && unexpanded == actions.len();
        wave.active
            .extend(actions.into_iter().map(|action| WaveSimulation {
                state: WaveSimulationState::Running(self.new_descent(action)),
            }));

        !wave.active.is_empty()
    }

    fn replenish_considered(&self, run: &mut Run) {
        let root = &self.nodes[0];
        let desired = root
            .masked
            .iter()
            .filter(|masked| !**masked)
            .count()
            .min(self.config.max_considered_actions.get());
        let current = run
            .considered
            .iter()
            .filter(|&&action| !root.masked[action])
            .count();
        if current >= desired {
            return;
        }

        let mut replacements = (0..root.action_count())
            .filter(|&action| !root.masked[action] && !run.considered.contains(&action))
            .collect::<Vec<_>>();
        replacements.sort_by(|&left, &right| {
            run.base_scores[right]
                .total_cmp(&run.base_scores[left])
                .then_with(|| left.cmp(&right))
        });
        run.considered
            .extend(replacements.into_iter().take(desired - current));
    }

    fn resolve_root_apply_preflight(&mut self, wave: &mut WaveRun<G, C>) -> EngineResult<()> {
        let all_safe = wave.active.iter().all(|simulation| {
            let WaveSimulationState::RootApplyReady(ready) = &simulation.state else {
                return false;
            };
            self.root_apply_safe(ready)
        });
        let ready = wave
            .active
            .iter_mut()
            .map(|simulation| {
                let state = std::mem::replace(&mut simulation.state, WaveSimulationState::Vacant);
                let WaveSimulationState::RootApplyReady(ready) = state else {
                    return Err(internal("incomplete symmetric root apply preflight"));
                };
                Ok(ready)
            })
            .collect::<EngineResult<Vec<_>>>()?;
        wave.root_apply_preflight = false;

        if all_safe {
            // Distinct successful root applies are independent; install them
            // in schedule order before their branches advance concurrently.
            for (simulation, ready) in wave.active.iter_mut().zip(ready) {
                self.consume_speculative_graph(ready.applied.after)?;
                simulation.state =
                    self.resume_wave_apply(ready.descent, ready.node, ready.action, ready.applied)?;
            }
            return Ok(());
        }

        // A rejection can retarget the first simulation. Discard later
        // speculation and resume through the existing sequential path.
        let mut ready = ready.into_iter();
        let first = ready
            .next()
            .ok_or_else(|| internal("empty symmetric root apply preflight"))?;
        for discarded in ready {
            self.discard_speculative_graph(discarded.applied.after)?;
        }
        wave.active.truncate(1);
        self.consume_speculative_graph(first.applied.after)?;
        wave.active[0].state =
            self.resume_wave_apply(first.descent, first.node, first.action, first.applied)?;
        Ok(())
    }

    fn root_apply_safe(&self, ready: &RootApplyReady<G, C>) -> bool {
        if ready.applied.rejected.is_some() {
            return false;
        }
        let player = self.nodes[ready.node].board.player;
        let context = self.identity.context(ready.applied.after_hash);
        !self.config.no_backtrack || !ready.descent.seen[player.index()].contains(&context)
    }

    fn consume_speculative_graph(&mut self, graph: G) -> EngineResult<()> {
        let index = self
            .speculative_graphs
            .iter()
            .position(|candidate| *candidate == graph)
            .ok_or_else(|| internal("untracked speculative symmetric graph"))?;
        self.speculative_graphs.swap_remove(index);
        Ok(())
    }

    fn discard_speculative_graph(&mut self, graph: G) -> EngineResult<()> {
        self.consume_speculative_graph(graph)?;
        self.releasable.graphs.push(graph);
        Ok(())
    }

    fn wave_root_action(&self, run: &Run, target: u32, visits: &[u32]) -> Option<usize> {
        let root = &self.nodes[0];
        let scale = (self.config.c_visit + visits.iter().copied().max().unwrap_or(0) as f32)
            * self.config.c_scale;
        let completed_q = self.completed_q(0);
        let scores = run
            .base_scores
            .iter()
            .zip(completed_q)
            .map(|(base, q)| base + scale * q)
            .collect::<Vec<_>>();
        run.considered
            .iter()
            .copied()
            .filter(|&action| {
                !root.masked[action]
                    && visits[action] == run.root_visit_baseline[action].saturating_add(target)
            })
            .max_by(|&left, &right| {
                scores[left]
                    .total_cmp(&scores[right])
                    .then_with(|| right.cmp(&left))
            })
            .or_else(|| {
                run.considered
                    .iter()
                    .copied()
                    .filter(|&action| !root.masked[action])
                    .max_by(|&left, &right| {
                        scores[left]
                            .total_cmp(&scores[right])
                            .then_with(|| right.cmp(&left))
                    })
            })
    }

    fn new_descent(&self, action: usize) -> Descent {
        let root = &self.nodes[0];
        let mut seen = self.visited.clone();
        for player in [GumbelPlayer::One, GumbelPlayer::Two] {
            if let Some(context) = root.board.contexts[player.index()] {
                seen[player.index()].insert(context);
            }
        }
        Descent {
            node: 0,
            path: Vec::new(),
            seen,
            forced: Some(action),
        }
    }

    fn advance_wave_simulation(
        &mut self,
        simulation: &mut WaveSimulation<G, C>,
        defer_completions: bool,
    ) -> EngineResult<WaveAdvance<G, C>> {
        let state = std::mem::replace(&mut simulation.state, WaveSimulationState::Vacant);
        match state {
            WaveSimulationState::Running(mut descent) => {
                let node_index = descent.node;
                let action = match descent.forced.take() {
                    Some(action) => Some(action),
                    None => self.select_nonroot(node_index),
                };
                let Some(action) = action else {
                    if let Some(branch) = self.nodes[node_index].pass {
                        descent.path.push(PathStep::Transform { flip: branch.flip });
                        simulation.state = wave_target_state(descent, branch.target);
                    } else {
                        let mut board = self.nodes[node_index].board;
                        let player = board.player;
                        board.inactive[player.index()] = true;
                        board.player = player.opponent();
                        simulation.state = WaveSimulationState::Resolve {
                            descent,
                            board,
                            attach: Attach {
                                slot: AttachSlot::Pass { node: node_index },
                                turns: 1,
                            },
                        };
                    }
                    return Ok(WaveAdvance::Progressed);
                };

                if let Some(branch) = self.nodes[node_index].edges[action].branch {
                    let player = self.nodes[node_index].board.player;
                    if let Some(context) = self.nodes[node_index].edges[action].after_context {
                        if self.config.no_backtrack
                            && descent.seen[player.index()].contains(&context)
                        {
                            self.mask_action(node_index, action);
                            simulation.state = WaveSimulationState::Running(descent);
                            return Ok(WaveAdvance::Progressed);
                        }
                        descent.seen[player.index()].insert(context);
                    }
                    descent.path.push(PathStep::Decision {
                        node: node_index,
                        action,
                        flip: branch.flip,
                    });
                    simulation.state = wave_target_state(descent, branch.target);
                    return Ok(WaveAdvance::Progressed);
                }

                if self.nodes[node_index].is_stop(action) {
                    let board = self.stopped_board(node_index);
                    simulation.state = WaveSimulationState::Resolve {
                        descent,
                        board,
                        attach: Attach {
                            slot: AttachSlot::Action {
                                node: node_index,
                                action,
                            },
                            turns: 1,
                        },
                    };
                    return Ok(WaveAdvance::Progressed);
                }

                let token = self.next_token();
                let node = &self.nodes[node_index];
                let work = SearchWork::Apply(ApplyWork {
                    token,
                    graph: node.board.current_graph(),
                    candidate: node.candidates[action],
                });
                simulation.state = WaveSimulationState::ApplyPending {
                    token,
                    descent,
                    node: node_index,
                    action,
                };
                Ok(WaveAdvance::Work(work))
            }
            WaveSimulationState::Resolve {
                descent,
                mut board,
                mut attach,
            } => {
                while !board.active(self.config.max_steps, board.player) {
                    if board.terminal(self.config.max_steps) {
                        let token = self.next_token();
                        let work = SearchWork::Measure(MeasureWork {
                            token,
                            graph: board.graphs[0],
                            options: self.config.measure_options,
                        });
                        simulation.state = WaveSimulationState::MeasureP1Pending {
                            token,
                            descent,
                            board,
                            attach,
                        };
                        return Ok(WaveAdvance::Work(work));
                    }
                    board.player = board.player.opponent();
                    attach.turns = attach.turns.saturating_add(1);
                }
                let token = self.next_token();
                let work = SearchWork::Expand(ExpandWork {
                    token,
                    graph: board.current_graph(),
                    options: self.config.candidate_options,
                });
                simulation.state = WaveSimulationState::ExpandPending {
                    token,
                    descent,
                    board,
                    attach,
                };
                Ok(WaveAdvance::Work(work))
            }
            WaveSimulationState::EvalReady {
                descent,
                board,
                attach,
                expansion,
            } => {
                let player = board.player;
                let request = EvalRequest::with_position(
                    expansion.context,
                    expansion.eval_actions.clone(),
                    self.board_position(board, player),
                )
                .map_err(|_| internal("invalid symmetric wave eval request"))?;
                let opponent = player.opponent();
                let token = self.next_token();
                let work = SearchWork::Eval(EvalWork {
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
                });
                simulation.state = WaveSimulationState::EvalPending {
                    token,
                    descent,
                    board,
                    attach,
                    expansion,
                    request,
                };
                Ok(WaveAdvance::Work(work))
            }
            WaveSimulationState::EvalComplete {
                descent,
                board,
                attach,
                expansion,
                request,
                output,
            } => {
                if defer_completions {
                    simulation.state = WaveSimulationState::EvalComplete {
                        descent,
                        board,
                        attach,
                        expansion,
                        request,
                        output,
                    };
                    return Ok(WaveAdvance::Waiting);
                }
                simulation.state =
                    self.complete_wave_eval(descent, board, attach, expansion, request, output)?;
                Ok(WaveAdvance::Progressed)
            }
            WaveSimulationState::MeasureP2Ready {
                descent,
                board,
                attach,
                p1_measure,
            } => {
                let token = self.next_token();
                let work = SearchWork::Measure(MeasureWork {
                    token,
                    graph: board.graphs[1],
                    options: self.config.measure_options,
                });
                simulation.state = WaveSimulationState::MeasureP2Pending {
                    token,
                    descent,
                    board,
                    attach,
                    p1_measure,
                };
                Ok(WaveAdvance::Work(work))
            }
            WaveSimulationState::Outcome { descent, value } => {
                simulation.state = WaveSimulationState::Outcome { descent, value };
                Ok(WaveAdvance::Waiting)
            }
            state @ WaveSimulationState::RootApplyReady(_) => {
                simulation.state = state;
                Ok(WaveAdvance::Waiting)
            }
            state @ (WaveSimulationState::ApplyPending { .. }
            | WaveSimulationState::ExpandPending { .. }
            | WaveSimulationState::EvalPending { .. }
            | WaveSimulationState::MeasureP1Pending { .. }
            | WaveSimulationState::MeasureP2Pending { .. }) => {
                simulation.state = state;
                Ok(WaveAdvance::Waiting)
            }
            WaveSimulationState::Vacant => Err(internal("vacant symmetric wave simulation")),
        }
    }

    fn resume_wave(
        &mut self,
        wave: &mut WaveRun<G, C>,
        token: WorkToken,
        result: SearchWorkResult<G, C>,
    ) -> EngineResult<()> {
        let root_apply_preflight = wave.root_apply_preflight;
        let simulation = wave
            .active
            .iter_mut()
            .find(|simulation| simulation.state.pending_token() == Some(token))
            .ok_or_else(|| internal("unknown symmetric wave token"))?;
        let state = std::mem::replace(&mut simulation.state, WaveSimulationState::Vacant);
        simulation.state = match (state, result) {
            (
                WaveSimulationState::ApplyPending {
                    descent,
                    node,
                    action,
                    ..
                },
                SearchWorkResult::Apply(applied),
            ) => {
                if root_apply_preflight {
                    self.speculative_graphs.push(applied.after);
                    WaveSimulationState::RootApplyReady(RootApplyReady {
                        descent,
                        node,
                        action,
                        applied,
                    })
                } else {
                    self.resume_wave_apply(descent, node, action, applied)?
                }
            }
            (
                WaveSimulationState::ExpandPending {
                    descent,
                    board,
                    attach,
                    ..
                },
                SearchWorkResult::Expand(expansion),
            ) => self.resume_wave_expand(descent, board, attach, expansion)?,
            (
                WaveSimulationState::EvalPending {
                    descent,
                    board,
                    attach,
                    expansion,
                    request,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => WaveSimulationState::EvalComplete {
                descent,
                board,
                attach,
                expansion,
                request,
                output,
            },
            (
                WaveSimulationState::MeasureP1Pending {
                    descent,
                    board,
                    attach,
                    ..
                },
                SearchWorkResult::Measure(p1_measure),
            ) => WaveSimulationState::MeasureP2Ready {
                descent,
                board,
                attach,
                p1_measure,
            },
            (
                WaveSimulationState::MeasureP2Pending {
                    mut descent,
                    board,
                    attach,
                    p1_measure,
                    ..
                },
                SearchWorkResult::Measure(p2_measure),
            ) => {
                let value = terminal_value(&board, &p1_measure, &p2_measure)?;
                let value =
                    self.attach_target(&mut descent, attach, BranchTarget::Terminal(value))?;
                WaveSimulationState::Outcome { descent, value }
            }
            (state, _) => {
                simulation.state = state;
                return Err(internal("mismatched symmetric wave result"));
            }
        };
        Ok(())
    }

    fn resume_wave_apply(
        &mut self,
        mut descent: Descent,
        node_index: usize,
        action: usize,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<WaveSimulationState<G, C>> {
        if applied.rejected.is_some() {
            self.releasable.graphs.push(applied.after);
            self.mask_action(node_index, action);
            return Ok(WaveSimulationState::Running(descent));
        }
        let player = self.nodes[node_index].board.player;
        let context = self.identity.context(applied.after_hash);
        self.portable_contexts += 1;
        if self.config.no_backtrack && descent.seen[player.index()].contains(&context) {
            self.releasable.graphs.push(applied.after);
            self.mask_action(node_index, action);
            return Ok(WaveSimulationState::Running(descent));
        }
        descent.seen[player.index()].insert(context);
        self.created_graphs.push(applied.after);
        let edge = &mut self.nodes[node_index].edges[action];
        edge.after_graph = Some(applied.after);
        edge.after_context = Some(context);
        let mut board = self.nodes[node_index].board;
        board.graphs[player.index()] = applied.after;
        board.contexts[player.index()] = Some(context);
        board.rewrites[player.index()] += 1;
        board.player = player.opponent();
        Ok(WaveSimulationState::Resolve {
            descent,
            board,
            attach: Attach {
                slot: AttachSlot::Action {
                    node: node_index,
                    action,
                },
                turns: 1,
            },
        })
    }

    fn resume_wave_expand(
        &mut self,
        descent: Descent,
        mut board: Board<G>,
        mut attach: Attach,
        result: ExpandResult<C>,
    ) -> EngineResult<WaveSimulationState<G, C>> {
        self.created_candidates.extend(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.candidate),
        );
        let context = self.identity.context(result.graph_hash);
        self.portable_contexts += 1;
        let player = board.player;
        if let Some(expected) = board.contexts[player.index()]
            && expected != context
        {
            return Err(internal("symmetric wave graph context mismatch"));
        }
        board.contexts[player.index()] = Some(context);
        if board.graphs[0] == board.graphs[1] {
            board.contexts[player.opponent().index()].get_or_insert(context);
        }

        if result.candidates.is_empty() {
            board.inactive[player.index()] = true;
            board.player = player.opponent();
            attach.turns = attach.turns.saturating_add(1);
            return Ok(WaveSimulationState::Resolve {
                descent,
                board,
                attach,
            });
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
        Ok(WaveSimulationState::EvalReady {
            descent,
            board,
            attach,
            expansion: Expansion {
                context,
                candidates,
                eval_actions,
            },
        })
    }

    fn complete_wave_eval(
        &mut self,
        mut descent: Descent,
        board: Board<G>,
        attach: Attach,
        expansion: Expansion<C>,
        request: EvalRequest,
        output: EvalOutput,
    ) -> EngineResult<WaveSimulationState<G, C>> {
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
        let value = self.attach_target(&mut descent, attach, BranchTarget::Node(node_index))?;
        Ok(WaveSimulationState::Outcome { descent, value })
    }

    fn continue_descent(&mut self, mut run: Run) -> EngineResult<()> {
        let descent = run
            .descent
            .as_mut()
            .ok_or_else(|| internal("missing symmetric descent"))?;
        let node_index = descent.node;
        let action = match descent.forced.take() {
            Some(action) => Some(action),
            None => self.select_nonroot(node_index),
        };
        let Some(action) = action else {
            if let Some(branch) = self.nodes[node_index].pass {
                descent.path.push(PathStep::Transform { flip: branch.flip });
                return self.follow_target(run, branch.target);
            }
            let mut board = self.nodes[node_index].board;
            let player = board.player;
            board.inactive[player.index()] = true;
            board.player = player.opponent();
            self.state = State::Resolve {
                board,
                attach: Some(Attach {
                    slot: AttachSlot::Pass { node: node_index },
                    turns: 1,
                }),
                run: Some(run),
            };
            return Ok(());
        };

        if let Some(branch) = self.nodes[node_index].edges[action].branch {
            let player = self.nodes[node_index].board.player;
            if let Some(context) = self.nodes[node_index].edges[action].after_context {
                if self.config.no_backtrack && descent.seen[player.index()].contains(&context) {
                    self.mask_action(node_index, action);
                    self.state = State::Running(run);
                    return Ok(());
                }
                descent.seen[player.index()].insert(context);
            }
            descent.path.push(PathStep::Decision {
                node: node_index,
                action,
                flip: branch.flip,
            });
            return self.follow_target(run, branch.target);
        }
        if self.nodes[node_index].is_stop(action) {
            let board = self.stopped_board(node_index);
            self.state = State::Resolve {
                board,
                attach: Some(Attach {
                    slot: AttachSlot::Action {
                        node: node_index,
                        action,
                    },
                    turns: 1,
                }),
                run: Some(run),
            };
            return Ok(());
        }
        let token = self.next_token();
        self.pending = Some(Pending::Apply {
            token,
            run,
            node: node_index,
            action,
        });
        let node = &self.nodes[node_index];
        self.state = State::Work(SearchWork::Apply(ApplyWork {
            token,
            graph: node.board.current_graph(),
            candidate: node.candidates[action],
        }));
        Ok(())
    }

    fn resume_apply(
        &mut self,
        mut run: Run,
        node_index: usize,
        action: usize,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<()> {
        if applied.rejected.is_some() {
            self.releasable.graphs.push(applied.after);
            self.mask_action(node_index, action);
            self.state = State::Running(run);
            return Ok(());
        }
        let player = self.nodes[node_index].board.player;
        let context = self.identity.context(applied.after_hash);
        self.portable_contexts += 1;
        let Some(descent) = run.descent.as_mut() else {
            self.releasable.graphs.push(applied.after);
            return Err(internal("missing symmetric descent"));
        };
        if self.config.no_backtrack && descent.seen[player.index()].contains(&context) {
            self.releasable.graphs.push(applied.after);
            self.mask_action(node_index, action);
            self.state = State::Running(run);
            return Ok(());
        }
        descent.seen[player.index()].insert(context);
        self.created_graphs.push(applied.after);
        let edge = &mut self.nodes[node_index].edges[action];
        edge.after_graph = Some(applied.after);
        edge.after_context = Some(context);
        let mut board = self.nodes[node_index].board;
        board.graphs[player.index()] = applied.after;
        board.contexts[player.index()] = Some(context);
        board.rewrites[player.index()] += 1;
        board.player = player.opponent();
        self.state = State::Resolve {
            board,
            attach: Some(Attach {
                slot: AttachSlot::Action {
                    node: node_index,
                    action,
                },
                turns: 1,
            }),
            run: Some(run),
        };
        Ok(())
    }

    fn follow_target(&mut self, mut run: Run, target: BranchTarget) -> EngineResult<()> {
        match target {
            BranchTarget::Node(node) => {
                run.descent
                    .as_mut()
                    .ok_or_else(|| internal("missing symmetric descent"))?
                    .node = node;
                self.state = State::Running(run);
            }
            BranchTarget::Terminal(value) => {
                self.backup(&mut run, value)?;
                self.state = State::Running(run);
            }
        }
        Ok(())
    }

    fn attach_and_backup(
        &mut self,
        run: &mut Run,
        attach: Option<Attach>,
        target: BranchTarget,
    ) -> EngineResult<()> {
        let attach = attach.ok_or_else(|| internal("missing symmetric branch attachment"))?;
        let descent = run
            .descent
            .as_mut()
            .ok_or_else(|| internal("missing symmetric descent"))?;
        let value = self.attach_target(descent, attach, target)?;
        self.backup(run, value)
    }

    fn attach_target(
        &mut self,
        descent: &mut Descent,
        attach: Attach,
        target: BranchTarget,
    ) -> EngineResult<f32> {
        let branch = Branch {
            target,
            flip: attach.turns % 2 == 1,
        };
        match attach.slot {
            AttachSlot::Action { node, action } => {
                let edge = &mut self.nodes[node].edges[action];
                if edge.branch.is_some() {
                    return Err(internal("symmetric action branch already installed"));
                }
                edge.branch = Some(branch);
                descent.path.push(PathStep::Decision {
                    node,
                    action,
                    flip: branch.flip,
                });
            }
            AttachSlot::Pass { node } => {
                if self.nodes[node].pass.is_some() {
                    return Err(internal("symmetric pass branch already installed"));
                }
                self.nodes[node].pass = Some(branch);
                descent.path.push(PathStep::Transform { flip: branch.flip });
            }
        }
        let value = match target {
            BranchTarget::Node(node) => self.nodes[node].value,
            BranchTarget::Terminal(value) => value,
        };
        Ok(value)
    }

    fn backup(&mut self, run: &mut Run, value: f32) -> EngineResult<()> {
        let descent = run
            .descent
            .as_ref()
            .ok_or_else(|| internal("missing symmetric descent"))?;
        self.backup_descent(descent, value);
        run.simulations += 1;
        run.schedule_index += 1;
        run.descent = None;
        Ok(())
    }

    fn backup_descent(&mut self, descent: &Descent, mut value: f32) {
        for step in descent.path.iter().copied().rev() {
            match step {
                PathStep::Decision { node, action, flip } => {
                    if flip {
                        value = -value;
                    }
                    let edge = &mut self.nodes[node].edges[action];
                    edge.visits += 1;
                    edge.value_sum += value;
                }
                PathStep::Transform { flip } => {
                    if flip {
                        value = -value;
                    }
                }
            }
        }
    }

    fn mask_action(&mut self, node: usize, action: usize) {
        self.nodes[node].masked[action] = true;
        self.nodes[node].priors[action] = 0.0;
    }

    fn select_nonroot(&self, node_index: usize) -> Option<usize> {
        let node = &self.nodes[node_index];
        let policy = self.improved_policy(node_index);
        let total = node.total_visits() as f32;
        (0..node.action_count())
            .filter(|&action| !node.masked[action])
            .max_by(|&left, &right| {
                let left_score = policy[left] - node.edges[left].visits as f32 / (1.0 + total);
                let right_score = policy[right] - node.edges[right].visits as f32 / (1.0 + total);
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right.cmp(&left))
            })
    }

    fn completed_q(&self, node_index: usize) -> Vec<f32> {
        let node = &self.nodes[node_index];
        let visits = node.total_visits();
        let mixed = if visits == 0 {
            node.value
        } else {
            let mut mass = 0.0;
            let mut weighted = 0.0;
            for (prior, edge) in node.priors.iter().zip(&node.edges) {
                if edge.visits > 0 {
                    mass += prior;
                    weighted += prior * edge.q();
                }
            }
            if mass > 0.0 {
                (node.value + visits as f32 * weighted / mass) / (1.0 + visits as f32)
            } else {
                node.value
            }
        };
        node.edges
            .iter()
            .map(|edge| if edge.visits > 0 { edge.q() } else { mixed })
            .collect()
    }

    fn improved_policy(&self, node_index: usize) -> Vec<f32> {
        let node = &self.nodes[node_index];
        let scale = (self.config.c_visit + node.max_visits() as f32) * self.config.c_scale;
        let scores = node
            .logits
            .iter()
            .zip(self.completed_q(node_index))
            .zip(&node.masked)
            .map(|((logit, q), masked)| {
                if *masked {
                    f32::NEG_INFINITY
                } else {
                    logit + scale * q
                }
            })
            .collect::<Vec<_>>();
        softmax(&scores)
    }

    fn root_scores(&self, node_index: usize, base: &[f32]) -> Vec<f32> {
        let node = &self.nodes[node_index];
        let scale = (self.config.c_visit + node.max_visits() as f32) * self.config.c_scale;
        base.iter()
            .zip(self.completed_q(node_index))
            .zip(&node.masked)
            .map(|((base, q), masked)| {
                if *masked {
                    f32::NEG_INFINITY
                } else {
                    base + scale * q
                }
            })
            .collect()
    }

    fn prepare_reuse(&mut self, selected: usize, expected: Board<G>) -> EngineResult<()> {
        if !self.config.tree_reuse {
            return Ok(());
        }
        if self.reused.is_some() {
            return Err(internal("symmetric reused tree already prepared"));
        }
        let Some(branch) = self.nodes[0].edges[selected].branch else {
            return Ok(());
        };
        let BranchTarget::Node(root_index) = branch.target else {
            return Ok(());
        };
        let root = self
            .nodes
            .get(root_index)
            .ok_or_else(|| internal("invalid symmetric reused root index"))?;
        if !same_board(root.board, expected) {
            return Ok(());
        }

        let nodes = self.compact_subtree(root_index)?;
        let carried_root_visits = nodes[0].total_visits();
        let carried_nodes = nodes.len();
        let mut referenced_graphs = HashSet::new();
        let mut referenced_candidates = HashSet::new();
        for node in &nodes {
            referenced_graphs.extend(node.board.graphs);
            referenced_candidates.extend(node.candidates.iter().copied());
            referenced_graphs.extend(node.edges.iter().filter_map(|edge| edge.after_graph));
        }

        // Preserve the promoted root's statistics. The next run snapshots
        // these counts as its baseline: allocation uses fresh-visit deltas,
        // while Q estimates and the policy target use the carried aggregate.

        let mut created_graphs = Vec::new();
        for graph in self.created_graphs.drain(..) {
            if referenced_graphs.contains(&graph) {
                created_graphs.push(graph);
            } else {
                self.releasable.graphs.push(graph);
            }
        }
        let mut created_candidates = Vec::new();
        for candidate in self.created_candidates.drain(..) {
            if referenced_candidates.contains(&candidate) {
                created_candidates.push(candidate);
            } else {
                self.releasable.candidates.push(candidate);
            }
        }
        self.reused = Some(ReusedTree {
            nodes,
            created_graphs,
            created_candidates,
            carried_nodes,
            carried_root_visits,
        });
        Ok(())
    }

    fn compact_subtree(&self, root_index: usize) -> EngineResult<Vec<Node<G, C>>> {
        let mut remap = vec![None; self.nodes.len()];
        let mut old_indices = Vec::new();
        let mut stack = vec![root_index];
        while let Some(index) = stack.pop() {
            let node = self
                .nodes
                .get(index)
                .ok_or_else(|| internal("invalid symmetric subtree node index"))?;
            if remap[index].is_some() {
                continue;
            }
            remap[index] = Some(old_indices.len());
            old_indices.push(index);
            for edge in node.edges.iter().rev() {
                if let Some(Branch {
                    target: BranchTarget::Node(child),
                    ..
                }) = edge.branch
                {
                    stack.push(child);
                }
            }
            if let Some(Branch {
                target: BranchTarget::Node(child),
                ..
            }) = node.pass
            {
                stack.push(child);
            }
        }

        let mut nodes = Vec::with_capacity(old_indices.len());
        for old_index in old_indices {
            let mut node = self.nodes[old_index].clone();
            for edge in &mut node.edges {
                if let Some(branch) = &mut edge.branch {
                    remap_branch(branch, &remap)?;
                }
            }
            if let Some(branch) = &mut node.pass {
                remap_branch(branch, &remap)?;
            }
            nodes.push(node);
        }
        Ok(nodes)
    }

    fn transfer_selected_graph(&mut self, selected: G) -> EngineResult<()> {
        if let Some(reused) = &mut self.reused
            && let Some(index) = reused
                .created_graphs
                .iter()
                .position(|graph| *graph == selected)
        {
            reused.created_graphs.swap_remove(index);
            return Ok(());
        }
        let index = self
            .created_graphs
            .iter()
            .position(|graph| *graph == selected)
            .ok_or_else(|| internal("selected symmetric graph is not owned"))?;
        self.created_graphs.swap_remove(index);
        Ok(())
    }

    fn finish_root(
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
                portable_contexts: self.portable_contexts,
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
            budget_fraction: super::super::schedule::budget_fraction(self.config.max_steps, step),
            budget_step: 1.0 / self.config.max_steps.max(1) as f32,
        }
    }

    fn board_position(&self, board: Board<G>, player: GumbelPlayer) -> EvalPositionContext {
        let mut position = self.position(board.rewrites[player.index()]);
        if !board.active(self.config.max_steps, player) {
            // Preserve the true rewrite count for the length tiebreak while
            // exposing retirement independently in the budget-step sign.
            position.budget_step = -position.budget_step.abs();
        }
        position
    }

    fn stopped_board(&self, node_index: usize) -> Board<G> {
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

    fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }
}

fn candidate_ref(context: ReplayGraphContext, hash: CandidateHash) -> PortableSearchActionRef {
    PortableSearchActionRef::candidate(PortableCandidateRef::new(context, hash))
}

fn terminal_value<G>(
    board: &Board<G>,
    p1: &MeasureResult<G>,
    p2: &MeasureResult<G>,
) -> EngineResult<f32> {
    let p1_reward = measured_reward(p1)?;
    let p2_reward = measured_reward(p2)?;
    let p1_value = if p1_reward > p2_reward {
        1.0
    } else if p1_reward < p2_reward {
        -1.0
    } else if board.rewrites[0] < board.rewrites[1] {
        1.0
    } else if board.rewrites[0] > board.rewrites[1] {
        -1.0
    } else {
        0.0
    };
    Ok(if board.player == GumbelPlayer::One {
        p1_value
    } else {
        -p1_value
    })
}

fn measured_reward<G>(measure: &MeasureResult<G>) -> EngineResult<f32> {
    if !measure.measured || !measure.valid {
        return Err(internal("invalid symmetric terminal measure"));
    }
    measure
        .scalar_reward
        .filter(|reward| reward.is_finite())
        .ok_or_else(|| internal("symmetric terminal has no finite reward"))
}

#[derive(Clone, Copy)]
struct Board<G> {
    graphs: [G; 2],
    contexts: [Option<ReplayGraphContext>; 2],
    rewrites: [usize; 2],
    inactive: [bool; 2],
    stopped: [bool; 2],
    player: GumbelPlayer,
}

impl<G: Copy> Board<G> {
    fn current_graph(self) -> G {
        self.graphs[self.player.index()]
    }

    fn active(self, max_steps: usize, player: GumbelPlayer) -> bool {
        !self.inactive[player.index()] && self.rewrites[player.index()] < max_steps
    }

    fn terminal(self, max_steps: usize) -> bool {
        !self.active(max_steps, GumbelPlayer::One) && !self.active(max_steps, GumbelPlayer::Two)
    }
}

#[derive(Clone, Copy)]
struct RootCandidate {
    hash: CandidateHash,
    summary: SearchCandidateSummary,
}

#[derive(Clone, Copy)]
struct CandidateEntry<C> {
    handle: C,
    hash: CandidateHash,
    summary: SearchCandidateSummary,
}

struct Expansion<C> {
    context: ReplayGraphContext,
    candidates: Vec<CandidateEntry<C>>,
    eval_actions: Vec<EvalAction>,
}

#[derive(Clone)]
struct Node<G, C> {
    board: Board<G>,
    context: ReplayGraphContext,
    candidates: Vec<C>,
    candidate_metadata: Vec<RootCandidate>,
    logits: Vec<f32>,
    priors: Vec<f32>,
    value: f32,
    model_version: gz_engine::ModelVersion,
    masked: Vec<bool>,
    edges: Vec<ActionEdge<G>>,
    pass: Option<Branch>,
}

struct ReusedTree<G, C> {
    nodes: Vec<Node<G, C>>,
    created_graphs: Vec<G>,
    created_candidates: Vec<C>,
    carried_nodes: usize,
    carried_root_visits: u32,
}

impl<G, C> Node<G, C> {
    fn action_count(&self) -> usize {
        self.edges.len()
    }

    fn is_stop(&self, action: usize) -> bool {
        action == self.candidates.len() && self.action_count() > self.candidates.len()
    }

    fn action_refs(&self, candidates: &[RootCandidate]) -> Vec<PortableSearchActionRef> {
        let mut actions = candidates
            .iter()
            .map(|candidate| candidate_ref(self.context, candidate.hash))
            .collect::<Vec<_>>();
        if self.action_count() > self.candidates.len() {
            actions.push(PortableSearchActionRef::stop(self.context));
        }
        actions
    }

    fn total_visits(&self) -> u32 {
        self.edges.iter().map(|edge| edge.visits).sum()
    }

    fn max_visits(&self) -> u32 {
        self.edges.iter().map(|edge| edge.visits).max().unwrap_or(0)
    }

    fn search_value(&self) -> f32 {
        let visits = self.total_visits();
        if visits == 0 {
            self.value
        } else {
            self.edges.iter().map(|edge| edge.value_sum).sum::<f32>() / visits as f32
        }
    }
}

#[derive(Clone)]
struct ActionEdge<G> {
    branch: Option<Branch>,
    after_graph: Option<G>,
    after_context: Option<ReplayGraphContext>,
    visits: u32,
    value_sum: f32,
}

impl<G> Default for ActionEdge<G> {
    fn default() -> Self {
        Self {
            branch: None,
            after_graph: None,
            after_context: None,
            visits: 0,
            value_sum: 0.0,
        }
    }
}

impl<G> ActionEdge<G> {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Copy)]
struct Branch {
    target: BranchTarget,
    flip: bool,
}

#[derive(Clone, Copy)]
enum BranchTarget {
    Node(usize),
    Terminal(f32),
}

fn same_board<G: Eq>(left: Board<G>, right: Board<G>) -> bool {
    left.graphs == right.graphs
        && left.contexts == right.contexts
        && left.rewrites == right.rewrites
        && left.inactive == right.inactive
        && left.stopped == right.stopped
        && left.player == right.player
}

fn remap_branch(branch: &mut Branch, remap: &[Option<usize>]) -> EngineResult<()> {
    let BranchTarget::Node(old_index) = branch.target else {
        return Ok(());
    };
    let new_index = remap
        .get(old_index)
        .copied()
        .flatten()
        .ok_or_else(|| internal("symmetric subtree child was not retained"))?;
    branch.target = BranchTarget::Node(new_index);
    Ok(())
}

struct Run {
    base_scores: Vec<f32>,
    considered: Vec<usize>,
    schedule: Vec<u32>,
    root_visit_baseline: Vec<u32>,
    schedule_index: usize,
    simulations: usize,
    descent: Option<Descent>,
}

struct WaveRun<G, C> {
    run: Run,
    active: Vec<WaveSimulation<G, C>>,
    root_apply_preflight: bool,
}

impl<G, C> WaveRun<G, C> {
    fn new(run: Run) -> Self {
        Self {
            run,
            active: Vec::new(),
            root_apply_preflight: false,
        }
    }

    fn root_apply_preflight_ready(&self) -> bool {
        self.root_apply_preflight
            && !self.active.is_empty()
            && self.active.iter().all(|simulation| {
                matches!(simulation.state, WaveSimulationState::RootApplyReady(_))
            })
    }

    fn has_pending(&self, token: WorkToken) -> bool {
        self.active
            .iter()
            .any(|simulation| simulation.state.pending_token() == Some(token))
    }

    fn any_pending(&self) -> bool {
        self.active
            .iter()
            .any(|simulation| simulation.state.pending_token().is_some())
    }
}

struct WaveSimulation<G, C> {
    state: WaveSimulationState<G, C>,
}

struct RootApplyReady<G, C> {
    descent: Descent,
    node: usize,
    action: usize,
    applied: ApplyResult<G, C>,
}

#[allow(clippy::large_enum_variant)]
enum WaveSimulationState<G, C> {
    Running(Descent),
    Resolve {
        descent: Descent,
        board: Board<G>,
        attach: Attach,
    },
    ApplyPending {
        token: WorkToken,
        descent: Descent,
        node: usize,
        action: usize,
    },
    RootApplyReady(RootApplyReady<G, C>),
    ExpandPending {
        token: WorkToken,
        descent: Descent,
        board: Board<G>,
        attach: Attach,
    },
    EvalReady {
        descent: Descent,
        board: Board<G>,
        attach: Attach,
        expansion: Expansion<C>,
    },
    EvalPending {
        token: WorkToken,
        descent: Descent,
        board: Board<G>,
        attach: Attach,
        expansion: Expansion<C>,
        request: EvalRequest,
    },
    EvalComplete {
        descent: Descent,
        board: Board<G>,
        attach: Attach,
        expansion: Expansion<C>,
        request: EvalRequest,
        output: EvalOutput,
    },
    MeasureP1Pending {
        token: WorkToken,
        descent: Descent,
        board: Board<G>,
        attach: Attach,
    },
    MeasureP2Ready {
        descent: Descent,
        board: Board<G>,
        attach: Attach,
        p1_measure: MeasureResult<G>,
    },
    MeasureP2Pending {
        token: WorkToken,
        descent: Descent,
        board: Board<G>,
        attach: Attach,
        p1_measure: MeasureResult<G>,
    },
    Outcome {
        descent: Descent,
        value: f32,
    },
    Vacant,
}

impl<G, C> WaveSimulationState<G, C> {
    fn pending_token(&self) -> Option<WorkToken> {
        match self {
            Self::ApplyPending { token, .. }
            | Self::ExpandPending { token, .. }
            | Self::EvalPending { token, .. }
            | Self::MeasureP1Pending { token, .. }
            | Self::MeasureP2Pending { token, .. } => Some(*token),
            _ => None,
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum WaveAdvance<G, C> {
    Work(SearchWork<G, C>),
    Progressed,
    Waiting,
}

fn wave_target_state<G, C>(
    mut descent: Descent,
    target: BranchTarget,
) -> WaveSimulationState<G, C> {
    match target {
        BranchTarget::Node(node) => {
            descent.node = node;
            WaveSimulationState::Running(descent)
        }
        BranchTarget::Terminal(value) => WaveSimulationState::Outcome { descent, value },
    }
}

struct Descent {
    node: usize,
    path: Vec<PathStep>,
    seen: [HashSet<ReplayGraphContext>; 2],
    forced: Option<usize>,
}

#[derive(Clone, Copy)]
enum PathStep {
    Decision {
        node: usize,
        action: usize,
        flip: bool,
    },
    Transform {
        flip: bool,
    },
}

#[derive(Clone, Copy)]
struct Attach {
    slot: AttachSlot,
    turns: u8,
}

#[derive(Clone, Copy)]
enum AttachSlot {
    Action { node: usize, action: usize },
    Pass { node: usize },
}

#[allow(clippy::large_enum_variant)]
enum State<G, C> {
    Resolve {
        board: Board<G>,
        attach: Option<Attach>,
        run: Option<Run>,
    },
    Eval {
        board: Board<G>,
        attach: Option<Attach>,
        run: Option<Run>,
        expansion: Expansion<C>,
    },
    Running(Run),
    WaveRunning(WaveRun<G, C>),
    MeasureP1 {
        board: Board<G>,
        attach: Option<Attach>,
        run: Run,
    },
    MeasureP2 {
        board: Board<G>,
        attach: Option<Attach>,
        run: Run,
        p1_measure: MeasureResult<G>,
    },
    Work(SearchWork<G, C>),
    DoneResult(SymmetricRootResult<G, C>),
    Done,
}

#[allow(clippy::large_enum_variant)]
enum Pending<G, C> {
    Expand {
        token: WorkToken,
        board: Board<G>,
        attach: Option<Attach>,
        run: Option<Run>,
    },
    Eval {
        token: WorkToken,
        board: Board<G>,
        attach: Option<Attach>,
        run: Option<Run>,
        expansion: Expansion<C>,
        request: EvalRequest,
    },
    Apply {
        token: WorkToken,
        run: Run,
        node: usize,
        action: usize,
    },
    MeasureP1 {
        token: WorkToken,
        board: Board<G>,
        attach: Option<Attach>,
        run: Run,
    },
    MeasureP2 {
        token: WorkToken,
        board: Board<G>,
        attach: Option<Attach>,
        run: Run,
        p1_measure: MeasureResult<G>,
    },
}

impl<G, C> Pending<G, C> {
    fn token(&self) -> WorkToken {
        match self {
            Self::Expand { token, .. }
            | Self::Eval { token, .. }
            | Self::Apply { token, .. }
            | Self::MeasureP1 { token, .. }
            | Self::MeasureP2 { token, .. } => *token,
        }
    }
}
