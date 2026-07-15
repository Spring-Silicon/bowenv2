use super::super::schedule::{
    GumbelRng, considered_actions, considered_visit_sequence, overlap_noise_scale, root_seed,
    sample_root_gumbels, softmax,
};
use super::super::{
    GumbelHandleBatch, GumbelMcts, GumbelMctsConfig, GumbelPlayer, GumbelRootStats, GumbelStep,
};
use crate::support::{internal, score, step_ref};
use crate::work::{
    ApplyWork, EngineIdentity, EvalModel, EvalOpponentWork, EvalWork, ExpandResult, ExpandWork,
    MeasureWork, SearchPoll, SearchWork, SearchWorkResult, WorkToken,
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

const CHANCE_SALT: u64 = 0x7361_6d70_5f74_7265;

pub struct SampledTreeRootResult<G, C> {
    pub step: GumbelStep<G, C>,
    pub selected_after: G,
    pub selected_after_context: ReplayGraphContext,
    pub selected_stop: bool,
    pub stats: GumbelRootStats,
}

pub struct SampledTreeRootTask<G, C> {
    config: GumbelMctsConfig,
    reference_mask_stop: bool,
    identity: EngineIdentity,
    learner_player: GumbelPlayer,
    root_step: usize,
    noise_seed: u64,
    visited: HashSet<ReplayGraphContext>,
    learner_nodes: Vec<LearnerNode<G, C>>,
    chance_nodes: Vec<ChanceNode<G>>,
    chance_policies: Vec<ChancePolicy<C>>,
    graph_refs: Vec<Option<G>>,
    candidate_batches: Vec<Option<Vec<C>>>,
    releasable: GumbelHandleBatch<G, C>,
    root_candidates: Vec<RootCandidateEntry>,
    root_context: Option<ReplayGraphContext>,
    next_token: u64,
    eval_count: usize,
    portable_contexts: usize,
    pending: Option<Pending<G, C>>,
    state: State<G, C>,
}

impl<G, C> SampledTreeRootTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        learner_player: GumbelPlayer,
        learner: G,
        opponent: G,
        opponent_context: Option<ReplayGraphContext>,
        learner_step: usize,
        opponent_step: usize,
        opponent_stopped: bool,
        noise_seed: u64,
        visited: HashSet<ReplayGraphContext>,
    ) -> Self {
        assert!(
            !search.config().tree_reuse,
            "sampled-tree does not support tree reuse"
        );
        let config = search.config();
        Self {
            config,
            reference_mask_stop: search.reference_mask_stop(),
            identity,
            learner_player,
            root_step: learner_step,
            noise_seed,
            visited,
            learner_nodes: Vec::new(),
            chance_nodes: Vec::new(),
            chance_policies: Vec::new(),
            graph_refs: Vec::new(),
            candidate_batches: Vec::new(),
            releasable: GumbelHandleBatch::default(),
            root_candidates: Vec::new(),
            root_context: None,
            next_token: 0,
            eval_count: 0,
            portable_contexts: 0,
            pending: None,
            state: State::ExpandLearner {
                graph: learner,
                opponent,
                opponent_context,
                learner_step,
                opponent_step,
                opponent_stopped,
                depth: 0,
                attach: None,
                run: None,
            },
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, SampledTreeRootResult<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }

        loop {
            let state = std::mem::replace(&mut self.state, State::Done);
            match state {
                State::ExpandLearner {
                    graph,
                    opponent,
                    opponent_context,
                    learner_step,
                    opponent_step,
                    opponent_stopped,
                    depth,
                    attach,
                    run,
                } => {
                    let token = self.next_token();
                    self.pending = Some(Pending::ExpandLearner {
                        token,
                        graph,
                        opponent,
                        opponent_context,
                        learner_step,
                        opponent_step,
                        opponent_stopped,
                        depth,
                        attach,
                        run,
                    });
                    return Ok(SearchPoll::Work(SearchWork::Expand(ExpandWork {
                        token,
                        graph,
                        options: self.config.candidate_options,
                    })));
                }
                State::EvalLearner { expansion, run } => {
                    return self.poll_learner_eval(expansion, run);
                }
                State::ExpandChance { chance, run } => {
                    let graph = self.chance_nodes[chance].opponent;
                    let token = self.next_token();
                    self.pending = Some(Pending::ExpandChance { token, chance, run });
                    return Ok(SearchPoll::Work(SearchWork::Expand(ExpandWork {
                        token,
                        graph,
                        options: self.config.candidate_options,
                    })));
                }
                State::EvalChance {
                    chance,
                    expansion,
                    run,
                } => return self.poll_chance_eval(chance, expansion, run),
                State::Running(mut run) => {
                    if run.descent.is_none() && !self.start_descent(&mut run) {
                        self.state = State::Done;
                        return Ok(SearchPoll::Done(self.finish_root(run)?));
                    }
                    self.continue_descent(run)?;
                }
                State::MeasureLearner { terminal, run } => {
                    let token = self.next_token();
                    self.pending = Some(Pending::MeasureLearner {
                        token,
                        terminal,
                        run,
                    });
                    return Ok(SearchPoll::Work(SearchWork::Measure(MeasureWork {
                        token,
                        graph: terminal.learner,
                        options: self.config.measure_options,
                    })));
                }
                State::MeasureOpponent {
                    terminal,
                    learner_measure,
                    run,
                } => {
                    let token = self.next_token();
                    self.pending = Some(Pending::MeasureOpponent {
                        token,
                        terminal,
                        learner_measure,
                        run,
                    });
                    return Ok(SearchPoll::Work(SearchWork::Measure(MeasureWork {
                        token,
                        graph: terminal.opponent,
                        options: self.config.measure_options,
                    })));
                }
                State::Work { work, pending } => {
                    self.pending = Some(*pending);
                    return Ok(SearchPoll::Work(*work));
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

        match (pending, result) {
            (
                Pending::ExpandLearner {
                    graph,
                    opponent,
                    opponent_context,
                    learner_step,
                    opponent_step,
                    opponent_stopped,
                    depth,
                    attach,
                    run,
                    ..
                },
                SearchWorkResult::Expand(result),
            ) => self.resume_expand_learner(
                LearnerExpansionInput {
                    graph,
                    opponent,
                    opponent_context,
                    learner_step,
                    opponent_step,
                    opponent_stopped,
                    depth,
                    attach,
                },
                run,
                result,
            ),
            (
                Pending::EvalLearner {
                    expansion,
                    request,
                    run,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => self.resume_eval_learner(expansion, *request, run, output),
            (Pending::ApplyLearner { run, action, .. }, SearchWorkResult::Apply(applied)) => {
                self.resume_apply_learner(run, action, applied)
            }
            (Pending::ExpandChance { chance, run, .. }, SearchWorkResult::Expand(result)) => {
                self.resume_expand_chance(chance, run, result)
            }
            (
                Pending::EvalChance {
                    chance,
                    expansion,
                    request,
                    run,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => self.resume_eval_chance(chance, expansion, *request, run, output),
            (
                Pending::ApplyChance {
                    chance,
                    action,
                    run,
                    ..
                },
                SearchWorkResult::Apply(applied),
            ) => self.resume_apply_chance(chance, action, run, applied),
            (Pending::MeasureLearner { terminal, run, .. }, SearchWorkResult::Measure(measure)) => {
                self.state = State::MeasureOpponent {
                    terminal,
                    learner_measure: measure,
                    run,
                };
                Ok(())
            }
            (
                Pending::MeasureOpponent {
                    terminal,
                    learner_measure,
                    mut run,
                    ..
                },
                SearchWorkResult::Measure(opponent_measure),
            ) => {
                let learner = score(&learner_measure)
                    .ok_or_else(|| internal("invalid sampled-tree learner measure"))?;
                let opponent = score(&opponent_measure)
                    .ok_or_else(|| internal("invalid sampled-tree opponent measure"))?;
                let value = terminal_value(self.learner_player, learner, opponent);
                self.set_branch(terminal.slot, Branch::Terminal(value))?;
                self.backup_and_complete(&mut run, value)?;
                self.state = State::Running(run);
                Ok(())
            }
            (pending, _) => {
                self.pending = Some(pending);
                Err(internal("mismatched work result"))
            }
        }
    }

    pub fn take_releasable(&mut self) -> GumbelHandleBatch<G, C> {
        std::mem::take(&mut self.releasable)
    }

    pub fn take_all_handles(&mut self) -> GumbelHandleBatch<G, C> {
        self.release_all_owned();
        self.take_releasable()
    }

    fn poll_learner_eval(
        &mut self,
        expansion: LearnerExpansion<G, C>,
        run: Option<Run<G, C>>,
    ) -> EngineResult<SearchPoll<G, C, SampledTreeRootResult<G, C>>> {
        let request = EvalRequest::with_position(
            expansion.context,
            expansion.eval_actions.clone(),
            self.learner_position(expansion.depth),
        )
        .map_err(|_| internal("invalid sampled-tree learner eval request"))?;
        let token = self.next_token();
        let work = EvalWork {
            token,
            graph: expansion.graph,
            candidates: expansion
                .candidates
                .iter()
                .map(|entry| entry.candidate)
                .collect(),
            request: request.clone(),
            measure_options: self.config.measure_options,
            model: EvalModel::Current,
            opponent: Some(Box::new(EvalOpponentWork {
                graph: expansion.opponent,
                position: self.actor_position(expansion.opponent_step),
            })),
        };
        self.pending = Some(Pending::EvalLearner {
            token,
            expansion,
            request: Box::new(request),
            run,
        });
        Ok(SearchPoll::Work(SearchWork::Eval(work)))
    }

    fn poll_chance_eval(
        &mut self,
        chance: usize,
        expansion: ChanceExpansion<C>,
        run: Run<G, C>,
    ) -> EngineResult<SearchPoll<G, C, SampledTreeRootResult<G, C>>> {
        let opponent = self.chance_nodes[chance].opponent;
        let opponent_context = self.chance_nodes[chance].opponent_context;
        let opponent_step = self.chance_nodes[chance].opponent_step;
        let request = EvalRequest::with_position(
            opponent_context,
            expansion.eval_actions.clone(),
            self.actor_position(opponent_step),
        )
        .map_err(|_| internal("invalid sampled-tree incumbent eval request"))?;
        let token = self.next_token();
        let work = EvalWork {
            token,
            graph: opponent,
            candidates: expansion
                .candidates
                .iter()
                .map(|entry| entry.candidate)
                .collect(),
            request: request.clone(),
            measure_options: self.config.measure_options,
            model: EvalModel::Incumbent,
            opponent: None,
        };
        self.pending = Some(Pending::EvalChance {
            token,
            chance,
            expansion,
            request: Box::new(request),
            run,
        });
        Ok(SearchPoll::Work(SearchWork::Eval(work)))
    }

    fn resume_expand_learner(
        &mut self,
        input: LearnerExpansionInput<G>,
        run: Option<Run<G, C>>,
        result: ExpandResult<C>,
    ) -> EngineResult<()> {
        let candidate_batch = self.register_candidate_batch(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.candidate)
                .collect(),
        );
        let context = self.identity.context(result.graph_hash);
        self.portable_contexts += 1;
        let opponent_context = input
            .opponent_context
            .or_else(|| (input.opponent == input.graph).then_some(context))
            .ok_or_else(|| internal("missing sampled-tree opponent context"))?;
        let (candidates, eval_actions) = candidate_entries(context, result);
        self.state = State::EvalLearner {
            expansion: LearnerExpansion {
                graph: input.graph,
                context,
                opponent: input.opponent,
                opponent_context,
                learner_step: input.learner_step,
                opponent_step: input.opponent_step,
                opponent_stopped: input.opponent_stopped,
                depth: input.depth,
                attach: input.attach,
                candidates,
                candidate_batch,
                eval_actions,
            },
            run,
        };
        Ok(())
    }

    fn resume_eval_learner(
        &mut self,
        expansion: LearnerExpansion<G, C>,
        request: EvalRequest,
        mut run: Option<Run<G, C>>,
        output: EvalOutput,
    ) -> EngineResult<()> {
        output
            .validate_for(&request)
            .map_err(eval_error_to_engine_error)?;
        let node_index = self.learner_nodes.len();
        let mut logits = output.policy_logits;
        let stop = expansion.candidates.len();
        if self.config.mask_stop && !expansion.candidates.is_empty() {
            logits[stop] = f32::NEG_INFINITY;
        }
        let priors = softmax(&logits);
        if node_index == 0 {
            self.root_candidates = expansion
                .candidates
                .iter()
                .map(|entry| RootCandidateEntry {
                    candidate_hash: entry.candidate_hash,
                    summary: entry.summary,
                })
                .collect();
        }
        let candidates = expansion
            .candidates
            .into_iter()
            .map(|entry| entry.candidate)
            .collect();
        self.learner_nodes.push(LearnerNode {
            live: true,
            graph: expansion.graph,
            context: expansion.context,
            opponent: expansion.opponent,
            opponent_context: expansion.opponent_context,
            learner_step: expansion.learner_step,
            opponent_step: expansion.opponent_step,
            opponent_stopped: expansion.opponent_stopped,
            candidates,
            candidate_batch: Some(expansion.candidate_batch),
            logits,
            priors,
            value: output.value,
            model_version: output.model_version,
            incumbent_policy: None,
            edge_by_action: vec![u32::MAX; request.actions.len()],
            edges: Vec::new(),
        });
        self.eval_count += 1;

        if let Some(attach) = expansion.attach {
            self.set_branch(attach, Branch::Learner(node_index))?;
        } else {
            self.root_context = Some(expansion.context);
        }

        if let Some(mut run) = run.take() {
            let value = self.learner_nodes[node_index].value;
            self.backup_and_complete(&mut run, value)?;
            self.state = State::Running(run);
        } else {
            self.state = State::Running(self.start_run()?);
        }
        Ok(())
    }

    fn resume_apply_learner(
        &mut self,
        mut run: Run<G, C>,
        action: usize,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<()> {
        let graph_ref = self.register_graph(applied.after);
        let descent = run
            .descent
            .as_mut()
            .ok_or_else(|| internal("missing sampled-tree descent"))?;
        if applied.rejected.is_some() {
            self.release_graph_ref(graph_ref)?;
            self.mask_learner_action(descent.node_index, action);
            self.state = State::Running(run);
            return Ok(());
        }
        let context = self.identity.context(applied.after_hash);
        self.portable_contexts += 1;
        if self.config.no_backtrack
            && (descent.seen.contains(&context) || self.visited.contains(&context))
        {
            self.release_graph_ref(graph_ref)?;
            self.mask_learner_action(descent.node_index, action);
            self.state = State::Running(run);
            return Ok(());
        }
        descent.path.push(LearnerEdge {
            node_index: descent.node_index,
            action,
        });
        descent.seen.insert(context);
        self.install_chance(run, action, applied.after, context, false, Some(graph_ref))
    }

    fn resume_expand_chance(
        &mut self,
        chance: usize,
        run: Run<G, C>,
        result: ExpandResult<C>,
    ) -> EngineResult<()> {
        let candidate_batch = self.register_candidate_batch(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.candidate)
                .collect(),
        );
        let context = self.identity.context(result.graph_hash);
        self.portable_contexts += 1;
        if context != self.chance_nodes[chance].opponent_context {
            return Err(internal("sampled-tree chance context mismatch"));
        }
        let (candidates, eval_actions) = candidate_entries(context, result);
        self.state = State::EvalChance {
            chance,
            expansion: ChanceExpansion {
                candidates,
                candidate_batch,
                eval_actions,
            },
            run,
        };
        Ok(())
    }

    fn resume_eval_chance(
        &mut self,
        chance: usize,
        expansion: ChanceExpansion<C>,
        request: EvalRequest,
        run: Run<G, C>,
        output: EvalOutput,
    ) -> EngineResult<()> {
        output
            .validate_for(&request)
            .map_err(eval_error_to_engine_error)?;
        let mut logits = output.policy_logits;
        let stop = expansion.candidates.len();
        if self.reference_mask_stop && !expansion.candidates.is_empty() {
            logits[stop] = f32::NEG_INFINITY;
        }
        let priors = softmax(&logits);
        let policy = self.chance_policies.len();
        self.chance_policies.push(ChancePolicy {
            live: true,
            candidates: expansion
                .candidates
                .into_iter()
                .map(|entry| entry.candidate)
                .collect(),
            priors,
            candidate_batch: Some(expansion.candidate_batch),
        });
        let owner = self.chance_nodes[chance].policy_owner;
        if let Some(owner) = owner {
            let incumbent_policy = &mut self.learner_nodes[owner].incumbent_policy;
            if incumbent_policy.is_some() {
                return Err(internal("sampled-tree incumbent policy already installed"));
            }
            *incumbent_policy = Some(policy);
        }
        self.chance_nodes[chance].policy = Some(policy);
        self.eval_count += 1;
        self.continue_chance(run, chance)
    }

    fn resume_apply_chance(
        &mut self,
        chance: usize,
        action: usize,
        run: Run<G, C>,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<()> {
        let graph_ref = self.register_graph(applied.after);
        if applied.rejected.is_some() {
            self.release_graph_ref(graph_ref)?;
            self.mask_chance_action(chance, action)?;
            return self.continue_chance(run, chance);
        }
        let context = self.identity.context(applied.after_hash);
        self.portable_contexts += 1;
        let node = &self.chance_nodes[chance];
        let next = NextPair {
            learner: node.learner,
            learner_context: node.learner_context,
            learner_step: node.learner_step,
            learner_stopped: node.learner_stopped,
            opponent: applied.after,
            opponent_context: context,
            opponent_step: node.opponent_step + 1,
            opponent_stopped: false,
        };
        self.advance_after_chance(
            run,
            BranchSlot::Action(chance, action, Some(graph_ref)),
            next,
        )
    }

    fn start_run(&self) -> EngineResult<Run<G, C>> {
        let root = self
            .learner_nodes
            .first()
            .ok_or_else(|| internal("missing sampled-tree root"))?;
        let mut rng = GumbelRng::new(root_seed(
            self.config.seed ^ self.noise_seed,
            self.root_step as u32,
        ));
        let scale = if self.config.gumbel_noise_overlap >= 0.0 {
            overlap_noise_scale(
                &root.logits,
                self.config.max_considered_actions.get(),
                self.config.gumbel_noise_overlap,
                self.config.gumbel_scale,
            )
        } else {
            self.config.gumbel_scale
        };
        let gumbels = sample_root_gumbels(root.logits.len(), scale, &mut rng);
        let base_scores = root
            .logits
            .iter()
            .zip(gumbels)
            .map(|(logit, noise)| logit + noise)
            .collect::<Vec<_>>();
        let considered = considered_actions(&base_scores, self.config.max_considered_actions.get());
        let schedule = considered_visit_sequence(considered.len(), self.config.simulations.get());
        Ok(Run {
            base_scores,
            considered,
            schedule,
            schedule_index: 0,
            simulations: 0,
            chance_rng: GumbelRng::new(root_seed(
                self.config.seed ^ self.noise_seed ^ CHANCE_SALT,
                self.root_step as u32,
            )),
            descent: None,
            _marker: std::marker::PhantomData,
        })
    }

    fn start_descent(&self, run: &mut Run<G, C>) -> bool {
        if run.schedule_index >= run.schedule.len() {
            return false;
        }
        let root_scores = self.root_scores(0, &run.base_scores);
        let target = run.schedule[run.schedule_index];
        let action = run
            .considered
            .iter()
            .copied()
            .filter(|&action| {
                self.learner_nodes[0].logits[action].is_finite()
                    && self.learner_nodes[0].visits(action) == target
            })
            .max_by(|&left, &right| {
                root_scores[left]
                    .total_cmp(&root_scores[right])
                    .then_with(|| right.cmp(&left))
            });
        let Some(action) = action.or_else(|| {
            run.considered
                .iter()
                .copied()
                .filter(|&action| self.learner_nodes[0].logits[action].is_finite())
                .max_by(|&left, &right| {
                    root_scores[left]
                        .total_cmp(&root_scores[right])
                        .then_with(|| right.cmp(&left))
                })
        }) else {
            return false;
        };
        run.descent = Some(Descent {
            node_index: 0,
            depth: 0,
            path: Vec::new(),
            seen: HashSet::from([self.learner_nodes[0].context]),
            forced: Some(action),
        });
        true
    }

    fn continue_descent(&mut self, mut run: Run<G, C>) -> EngineResult<()> {
        let descent = run
            .descent
            .as_mut()
            .ok_or_else(|| internal("missing sampled-tree descent"))?;
        let action = descent
            .forced
            .take()
            .unwrap_or_else(|| self.select_nonroot(descent.node_index));
        let node_index = descent.node_index;
        if !self.learner_nodes[node_index].logits[action].is_finite() {
            self.state = State::Running(run);
            return Ok(());
        }
        if let Some(chance) = self.learner_nodes[node_index].chance(action) {
            descent.path.push(LearnerEdge { node_index, action });
            return self.continue_chance(run, chance);
        }
        if self.learner_nodes[node_index].is_stop(action) {
            descent.path.push(LearnerEdge { node_index, action });
            let graph = self.learner_nodes[node_index].graph;
            let context = self.learner_nodes[node_index].context;
            return self.install_chance(run, action, graph, context, true, None);
        }
        let token = self.next_token();
        let graph = self.learner_nodes[node_index].graph;
        let candidate = self.learner_nodes[node_index].candidates[action];
        self.pending = Some(Pending::ApplyLearner { token, run, action });
        let pending = self.pending.take().expect("pending was just installed");
        self.state = State::Work {
            work: Box::new(SearchWork::Apply(ApplyWork {
                token,
                graph,
                candidate,
            })),
            pending: Box::new(pending),
        };
        Ok(())
    }

    fn install_chance(
        &mut self,
        run: Run<G, C>,
        action: usize,
        learner: G,
        learner_context: ReplayGraphContext,
        learner_stopped: bool,
        learner_graph_ref: Option<usize>,
    ) -> EngineResult<()> {
        let descent = run
            .descent
            .as_ref()
            .ok_or_else(|| internal("missing sampled-tree descent"))?;
        let parent = &self.learner_nodes[descent.node_index];
        let chance = self.chance_nodes.len();
        self.chance_nodes.push(ChanceNode {
            live: true,
            learner,
            learner_context,
            learner_step: parent.learner_step + 1,
            learner_stopped,
            opponent: parent.opponent,
            opponent_context: parent.opponent_context,
            opponent_step: parent.opponent_step,
            opponent_stopped: parent.opponent_stopped,
            learner_graph_ref,
            policy_owner: Some(descent.node_index),
            policy: parent.incumbent_policy,
            branches: Vec::new(),
            pass: None,
        });
        self.learner_nodes[descent.node_index].set_chance(action, chance)?;
        self.continue_chance(run, chance)
    }

    fn continue_chance(&mut self, mut run: Run<G, C>, chance: usize) -> EngineResult<()> {
        let node = &self.chance_nodes[chance];
        if node.opponent_stopped || node.opponent_step >= self.config.max_steps {
            if let Some(branch) = node.pass {
                return self.follow_branch(run, branch);
            }
            let next = NextPair {
                learner: node.learner,
                learner_context: node.learner_context,
                learner_step: node.learner_step,
                learner_stopped: node.learner_stopped,
                opponent: node.opponent,
                opponent_context: node.opponent_context,
                opponent_step: node.opponent_step,
                opponent_stopped: true,
            };
            return self.advance_after_chance(run, BranchSlot::Pass(chance), next);
        }
        if node.policy.is_none() {
            self.state = State::ExpandChance { chance, run };
            return Ok(());
        }
        let action = self.sample_chance_action(chance, &mut run.chance_rng)?;
        if let Some(branch) = self.chance_nodes[chance].branch(action) {
            return self.follow_branch(run, branch);
        }
        let policy = self.chance_nodes[chance]
            .policy
            .expect("chance policy checked");
        let stop = self.chance_policies[policy].candidates.len();
        if action == stop {
            let node = &self.chance_nodes[chance];
            let next = NextPair {
                learner: node.learner,
                learner_context: node.learner_context,
                learner_step: node.learner_step,
                learner_stopped: node.learner_stopped,
                opponent: node.opponent,
                opponent_context: node.opponent_context,
                opponent_step: node.opponent_step + 1,
                opponent_stopped: true,
            };
            return self.advance_after_chance(run, BranchSlot::Action(chance, action, None), next);
        }
        let token = self.next_token();
        let candidate = self.chance_policies[policy].candidates[action];
        let graph = self.chance_nodes[chance].opponent;
        self.pending = Some(Pending::ApplyChance {
            token,
            chance,
            action,
            run,
        });
        let pending = self.pending.take().expect("pending was just installed");
        self.state = State::Work {
            work: Box::new(SearchWork::Apply(ApplyWork {
                token,
                graph,
                candidate,
            })),
            pending: Box::new(pending),
        };
        Ok(())
    }

    fn advance_after_chance(
        &mut self,
        run: Run<G, C>,
        slot: BranchSlot,
        next: NextPair<G>,
    ) -> EngineResult<()> {
        let learner_active = !next.learner_stopped && next.learner_step < self.config.max_steps;
        let opponent_active = !next.opponent_stopped && next.opponent_step < self.config.max_steps;
        if learner_active {
            let depth = run
                .descent
                .as_ref()
                .ok_or_else(|| internal("missing sampled-tree descent"))?
                .depth
                + 1;
            self.state = State::ExpandLearner {
                graph: next.learner,
                opponent: next.opponent,
                opponent_context: Some(next.opponent_context),
                learner_step: next.learner_step,
                opponent_step: next.opponent_step,
                opponent_stopped: !opponent_active,
                depth,
                attach: Some(slot),
                run: Some(run),
            };
            return Ok(());
        }
        if opponent_active {
            let chance = self.chance_nodes.len();
            self.chance_nodes.push(ChanceNode {
                live: true,
                learner: next.learner,
                learner_context: next.learner_context,
                learner_step: next.learner_step,
                learner_stopped: true,
                opponent: next.opponent,
                opponent_context: next.opponent_context,
                opponent_step: next.opponent_step,
                opponent_stopped: false,
                learner_graph_ref: None,
                policy_owner: None,
                policy: None,
                branches: Vec::new(),
                pass: None,
            });
            self.set_branch(slot, Branch::Chance(chance))?;
            return self.continue_chance(run, chance);
        }
        self.state = State::MeasureLearner {
            terminal: TerminalPair {
                slot,
                learner: next.learner,
                opponent: next.opponent,
            },
            run,
        };
        Ok(())
    }

    fn follow_branch(&mut self, mut run: Run<G, C>, branch: Branch) -> EngineResult<()> {
        match branch {
            Branch::Learner(node) => {
                let descent = run
                    .descent
                    .as_mut()
                    .ok_or_else(|| internal("missing sampled-tree descent"))?;
                descent.node_index = node;
                descent.depth += 1;
                self.state = State::Running(run);
            }
            Branch::Chance(chance) => return self.continue_chance(run, chance),
            Branch::Terminal(value) => {
                self.backup_and_complete(&mut run, value)?;
                self.state = State::Running(run);
            }
        }
        Ok(())
    }

    fn set_branch(&mut self, slot: BranchSlot, branch: Branch) -> EngineResult<()> {
        match slot {
            BranchSlot::Pass(chance) => {
                let target = &mut self.chance_nodes[chance].pass;
                if target.is_some() {
                    return Err(internal("sampled-tree pass branch already set"));
                }
                *target = Some(branch);
            }
            BranchSlot::Action(chance, action, graph_ref) => {
                self.chance_nodes[chance].set_branch(action, branch, graph_ref)?;
            }
        }
        Ok(())
    }

    fn backup_and_complete(&mut self, run: &mut Run<G, C>, value: f32) -> EngineResult<()> {
        let path = run
            .descent
            .as_ref()
            .ok_or_else(|| internal("missing sampled-tree descent"))?
            .path
            .clone();
        for edge in path {
            let node = &mut self.learner_nodes[edge.node_index];
            let stats = node.edge_mut(edge.action)?;
            stats.visits += 1;
            stats.value_sum += value;
        }
        run.simulations += 1;
        run.schedule_index += 1;
        run.descent = None;
        self.prune_eliminated_root_branches(run)?;
        Ok(())
    }

    fn prune_eliminated_root_branches(&mut self, run: &Run<G, C>) -> EngineResult<()> {
        let Some(&next_target) = run.schedule.get(run.schedule_index) else {
            return Ok(());
        };
        let eliminated = run
            .considered
            .iter()
            .copied()
            .filter(|&action| self.learner_nodes[0].visits(action) < next_target)
            .collect::<Vec<_>>();
        for action in eliminated {
            if let Some(chance) = self.learner_nodes[0].take_chance(action)? {
                self.prune_chance(chance)?;
            }
        }
        Ok(())
    }

    fn prune_branch(&mut self, branch: Branch) -> EngineResult<()> {
        match branch {
            Branch::Learner(node) => self.prune_learner(node),
            Branch::Chance(chance) => self.prune_chance(chance),
            Branch::Terminal(_) => Ok(()),
        }
    }

    fn prune_learner(&mut self, node: usize) -> EngineResult<()> {
        let (candidate_batch, incumbent_policy, chances) = {
            let node = self
                .learner_nodes
                .get_mut(node)
                .ok_or_else(|| internal("invalid sampled-tree learner node"))?;
            if !node.live {
                return Err(internal("sampled-tree learner node already pruned"));
            }
            node.live = false;
            let chances = node
                .edges
                .iter_mut()
                .filter_map(|edge| edge.chance.take())
                .collect::<Vec<_>>();
            node.edge_by_action = Vec::new();
            node.edges = Vec::new();
            node.candidates = Vec::new();
            node.logits = Vec::new();
            node.priors = Vec::new();
            (
                node.candidate_batch.take(),
                node.incumbent_policy.take(),
                chances,
            )
        };
        for chance in chances {
            self.prune_chance(chance)?;
        }
        if let Some(policy) = incumbent_policy {
            self.prune_policy(policy)?;
        }
        if let Some(candidate_batch) = candidate_batch {
            self.release_candidate_batch(candidate_batch)?;
        }
        Ok(())
    }

    fn prune_chance(&mut self, chance: usize) -> EngineResult<()> {
        let (learner_graph_ref, owned_policy, branches, pass) = {
            let node = self
                .chance_nodes
                .get_mut(chance)
                .ok_or_else(|| internal("invalid sampled-tree chance node"))?;
            if !node.live {
                return Err(internal("sampled-tree chance node already pruned"));
            }
            node.live = false;
            let owned_policy = node.policy_owner.is_none().then_some(node.policy).flatten();
            (
                node.learner_graph_ref.take(),
                owned_policy,
                std::mem::take(&mut node.branches),
                node.pass.take(),
            )
        };
        for edge in branches {
            self.prune_branch(edge.branch)?;
            if let Some(graph_ref) = edge.graph_ref {
                self.release_graph_ref(graph_ref)?;
            }
        }
        if let Some(pass) = pass {
            self.prune_branch(pass)?;
        }
        if let Some(policy) = owned_policy {
            self.prune_policy(policy)?;
        }
        if let Some(graph_ref) = learner_graph_ref {
            self.release_graph_ref(graph_ref)?;
        }
        Ok(())
    }

    fn prune_policy(&mut self, policy: usize) -> EngineResult<()> {
        let candidate_batch = {
            let policy = self
                .chance_policies
                .get_mut(policy)
                .ok_or_else(|| internal("invalid sampled-tree chance policy"))?;
            if !policy.live {
                return Err(internal("sampled-tree chance policy already pruned"));
            }
            policy.live = false;
            policy.candidates = Vec::new();
            policy.priors = Vec::new();
            policy.candidate_batch.take()
        };
        if let Some(candidate_batch) = candidate_batch {
            self.release_candidate_batch(candidate_batch)?;
        }
        Ok(())
    }

    fn mask_learner_action(&mut self, node: usize, action: usize) {
        let node = &mut self.learner_nodes[node];
        node.logits[action] = f32::NEG_INFINITY;
        node.priors[action] = 0.0;
        if node.logits.iter().all(|logit| !logit.is_finite()) {
            let stop = node.candidates.len();
            node.logits[stop] = 0.0;
            node.priors[stop] = 1.0;
        }
    }

    fn mask_chance_action(&mut self, chance: usize, action: usize) -> EngineResult<()> {
        let policy = self.chance_nodes[chance]
            .policy
            .ok_or_else(|| internal("missing sampled-tree chance policy"))?;
        let policy = &mut self.chance_policies[policy];
        let Some(prior) = policy.priors.get_mut(action) else {
            return Err(internal("invalid sampled-tree chance action"));
        };
        *prior = 0.0;
        let mass = policy.priors.iter().sum::<f32>();
        if mass <= 0.0 {
            let stop = policy.candidates.len();
            policy.priors[stop] = 1.0;
        } else {
            for prior in &mut policy.priors {
                *prior /= mass;
            }
        }
        Ok(())
    }

    fn sample_chance_action(&self, chance: usize, rng: &mut GumbelRng) -> EngineResult<usize> {
        let policy = self.chance_nodes[chance]
            .policy
            .ok_or_else(|| internal("missing sampled-tree chance policy"))?;
        let policy = &self.chance_policies[policy];
        let mut threshold = rng.unit();
        for (action, prior) in policy.priors.iter().copied().enumerate() {
            if threshold <= prior {
                return Ok(action);
            }
            threshold -= prior;
        }
        policy
            .priors
            .iter()
            .rposition(|prior| *prior > 0.0)
            .ok_or_else(|| internal("sampled-tree chance has no legal action"))
    }

    fn select_nonroot(&self, node_index: usize) -> usize {
        let node = &self.learner_nodes[node_index];
        let policy = self.improved_policy(node_index);
        let total = node.total_visits() as f32;
        policy
            .iter()
            .enumerate()
            .filter(|(action, _)| node.logits[*action].is_finite())
            .max_by(|(left, left_policy), (right, right_policy)| {
                let left_score = **left_policy - node.visits(*left) as f32 / (1.0 + total);
                let right_score = **right_policy - node.visits(*right) as f32 / (1.0 + total);
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right.cmp(left))
            })
            .map(|(action, _)| action)
            .unwrap_or(node.candidates.len())
    }

    fn completed_q(&self, node_index: usize) -> Vec<f32> {
        let node = &self.learner_nodes[node_index];
        let visits = node.total_visits();
        let mixed = if visits == 0 {
            node.value
        } else {
            let mut mass = 0.0;
            let mut weighted = 0.0;
            for edge in &node.edges {
                if edge.visits > 0 {
                    let prior = node.priors[edge.action];
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
        (0..node.logits.len())
            .map(|action| {
                let edge = node.edge(action);
                if edge.is_some_and(|edge| edge.visits > 0) {
                    edge.expect("visited edge checked").q()
                } else {
                    mixed
                }
            })
            .collect()
    }

    fn improved_policy(&self, node_index: usize) -> Vec<f32> {
        let node = &self.learner_nodes[node_index];
        let max_visits = node.max_visits() as f32;
        let scale = (self.config.c_visit + max_visits) * self.config.c_scale;
        let scores = node
            .logits
            .iter()
            .zip(self.completed_q(node_index))
            .map(|(logit, q)| logit + scale * q)
            .collect::<Vec<_>>();
        softmax(&scores)
    }

    fn root_scores(&self, node_index: usize, base: &[f32]) -> Vec<f32> {
        let node = &self.learner_nodes[node_index];
        let max_visits = node.max_visits() as f32;
        let scale = (self.config.c_visit + max_visits) * self.config.c_scale;
        base.iter()
            .zip(&node.logits)
            .zip(self.completed_q(node_index))
            .map(|((base, logit), q)| {
                if logit.is_finite() {
                    base + scale * q
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect()
    }

    fn finish_root(&mut self, run: Run<G, C>) -> EngineResult<SampledTreeRootResult<G, C>> {
        let node = self
            .learner_nodes
            .first()
            .ok_or_else(|| internal("missing sampled-tree root"))?;
        let scores = self.root_scores(0, &run.base_scores);
        let selectable = run
            .considered
            .iter()
            .copied()
            .filter(|action| node.logits[*action].is_finite())
            .collect::<Vec<_>>();
        let selectable = if selectable.is_empty() {
            node.logits
                .iter()
                .enumerate()
                .filter_map(|(action, logit)| logit.is_finite().then_some(action))
                .collect::<Vec<_>>()
        } else {
            selectable
        };
        let selected = selectable
            .iter()
            .copied()
            .max_by(|&left, &right| {
                node.visits(left)
                    .cmp(&node.visits(right))
                    .then_with(|| scores[left].total_cmp(&scores[right]))
                    .then_with(|| right.cmp(&left))
            })
            .ok_or_else(|| internal("sampled-tree root has no selectable action"))?;
        let selected_stop = node.is_stop(selected);
        let chance = node
            .chance(selected)
            .ok_or_else(|| internal("sampled-tree selected action was not simulated"))?;
        let selected_after = self.chance_nodes[chance].learner;
        let selected_after_context = self.chance_nodes[chance].learner_context;
        let selected_graph_ref = self.chance_nodes[chance].learner_graph_ref;
        let action = node.search_action(selected)?;
        let action_ref = root_action_ref(node.context, &self.root_candidates, selected)?;
        let root_value = node.value;
        let root_search_value = search_value(node);
        let root_q_max = node
            .edges
            .iter()
            .filter_map(|edge| (edge.visits > 0).then_some(edge.q()))
            .reduce(f32::max)
            .unwrap_or(root_value);
        let result = SampledTreeRootResult {
            step: GumbelStep {
                before: node.graph,
                after: selected_after,
                action,
                step_ref: step_ref(node.context, action_ref, selected_after_context)?,
                selected_action: action_ref,
                selected_candidate: self
                    .root_candidates
                    .get(selected)
                    .map(|entry| entry.summary),
                engine_candidate_count: node.candidates.len(),
                action_count: node.logits.len(),
                selected_rank: selected,
                legal_actions: root_action_refs(node.context, &self.root_candidates),
                policy_target: self.improved_policy(0),
                considered_action_indices: run.considered,
                root_value,
                root_search_value,
                root_q_max,
                model_version: node.model_version,
            },
            selected_after,
            selected_after_context,
            selected_stop,
            stats: GumbelRootStats {
                simulations: run.simulations,
                expanded_nodes: self.learner_nodes.len() + self.chance_nodes.len(),
                eval_count: self.eval_count,
                portable_contexts: self.portable_contexts,
                carried_nodes: 0,
                carried_root_visits: 0,
            },
        };
        if !selected_stop {
            let graph_ref = selected_graph_ref
                .ok_or_else(|| internal("sampled-tree selected graph is not owned"))?;
            let owned = self.take_graph_ref(graph_ref)?;
            if owned != selected_after {
                self.releasable.graphs.push(owned);
                return Err(internal("sampled-tree selected graph reference mismatch"));
            }
        }
        self.release_all_owned();
        Ok(result)
    }

    fn learner_position(&self, depth: usize) -> EvalPositionContext {
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
            root_step: self.root_step as u32,
            leaf_depth: depth as u32,
            budget_fraction: super::super::schedule::budget_fraction(
                self.config.max_steps,
                self.root_step,
            ),
            budget_step: 1.0 / self.config.max_steps.max(1) as f32,
            opponent: None,
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
            budget_fraction: super::super::schedule::budget_fraction(self.config.max_steps, step),
            budget_step: 1.0 / self.config.max_steps.max(1) as f32,
            opponent: None,
        }
    }

    fn register_graph(&mut self, graph: G) -> usize {
        let index = self.graph_refs.len();
        self.graph_refs.push(Some(graph));
        index
    }

    fn take_graph_ref(&mut self, index: usize) -> EngineResult<G> {
        self.graph_refs
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(|| internal("missing sampled-tree graph reference"))
    }

    fn release_graph_ref(&mut self, index: usize) -> EngineResult<()> {
        let graph = self.take_graph_ref(index)?;
        self.releasable.graphs.push(graph);
        Ok(())
    }

    fn register_candidate_batch(&mut self, candidates: Vec<C>) -> usize {
        let index = self.candidate_batches.len();
        self.candidate_batches.push(Some(candidates));
        index
    }

    fn release_candidate_batch(&mut self, index: usize) -> EngineResult<()> {
        let mut candidates = self
            .candidate_batches
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(|| internal("missing sampled-tree candidate batch"))?;
        self.releasable.candidates.append(&mut candidates);
        Ok(())
    }

    fn release_all_owned(&mut self) {
        for graph in &mut self.graph_refs {
            if let Some(graph) = graph.take() {
                self.releasable.graphs.push(graph);
            }
        }
        for candidates in &mut self.candidate_batches {
            if let Some(mut candidates) = candidates.take() {
                self.releasable.candidates.append(&mut candidates);
            }
        }
    }

    fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }
}

fn candidate_entries<C: Copy>(
    context: ReplayGraphContext,
    result: ExpandResult<C>,
) -> (Vec<CandidateEntry<C>>, Vec<EvalAction>) {
    let mut candidates = Vec::with_capacity(result.candidates.len());
    let mut actions = Vec::with_capacity(result.candidates.len() + 1);
    for candidate in result.candidates {
        let candidate_ref = PortableCandidateRef::new(context, candidate.candidate_hash);
        actions.push(EvalAction::candidate(
            candidate_ref,
            candidate.kind,
            candidate.tags,
            candidate.static_prior,
        ));
        candidates.push(CandidateEntry {
            candidate: candidate.candidate,
            candidate_hash: candidate.candidate_hash,
            summary: SearchCandidateSummary {
                kind: candidate.kind,
                tags: candidate.tags,
                static_prior: candidate.static_prior,
            },
        });
    }
    actions.push(EvalAction::stop(context));
    (candidates, actions)
}

fn terminal_value(player: GumbelPlayer, learner: f32, opponent: f32) -> f32 {
    if learner > opponent {
        1.0
    } else if learner < opponent {
        -1.0
    } else if player == GumbelPlayer::One {
        1.0
    } else {
        -1.0
    }
}

fn search_value<G, C>(node: &LearnerNode<G, C>) -> f32 {
    let mut visits = 0;
    let mut value = 0.0;
    for edge in &node.edges {
        if edge.visits > 0 {
            visits += edge.visits;
            value += edge.value_sum;
        }
    }
    if visits == 0 {
        node.value
    } else {
        value / visits as f32
    }
}

#[derive(Clone, Copy)]
struct CandidateEntry<C> {
    candidate: C,
    candidate_hash: CandidateHash,
    summary: SearchCandidateSummary,
}

#[derive(Clone, Copy)]
struct RootCandidateEntry {
    candidate_hash: CandidateHash,
    summary: SearchCandidateSummary,
}

struct LearnerNode<G, C> {
    live: bool,
    graph: G,
    context: ReplayGraphContext,
    opponent: G,
    opponent_context: ReplayGraphContext,
    learner_step: usize,
    opponent_step: usize,
    opponent_stopped: bool,
    candidates: Vec<C>,
    candidate_batch: Option<usize>,
    logits: Vec<f32>,
    priors: Vec<f32>,
    value: f32,
    model_version: gz_engine::ModelVersion,
    incumbent_policy: Option<usize>,
    edge_by_action: Vec<u32>,
    edges: Vec<LearnerActionEdge>,
}

impl<G: Copy, C: Copy> LearnerNode<G, C> {
    fn is_stop(&self, action: usize) -> bool {
        action == self.candidates.len()
    }

    fn search_action(&self, action: usize) -> EngineResult<SearchAction<C>> {
        if self.is_stop(action) {
            Ok(SearchAction::Stop)
        } else {
            self.candidates
                .get(action)
                .copied()
                .map(SearchAction::Candidate)
                .ok_or_else(|| internal("invalid sampled-tree action"))
        }
    }

    fn edge(&self, action: usize) -> Option<&LearnerActionEdge> {
        let index = *self.edge_by_action.get(action)?;
        (index != u32::MAX).then(|| &self.edges[index as usize])
    }

    fn edge_mut(&mut self, action: usize) -> EngineResult<&mut LearnerActionEdge> {
        let Some(index) = self.edge_by_action.get(action).copied() else {
            return Err(internal("invalid sampled-tree learner action"));
        };
        if index == u32::MAX {
            return Err(internal("missing sampled-tree learner edge"));
        }
        Ok(&mut self.edges[index as usize])
    }

    fn chance(&self, action: usize) -> Option<usize> {
        self.edge(action).and_then(|edge| edge.chance)
    }

    fn take_chance(&mut self, action: usize) -> EngineResult<Option<usize>> {
        let Some(index) = self.edge_by_action.get(action).copied() else {
            return Err(internal("invalid sampled-tree learner action"));
        };
        if index == u32::MAX {
            return Ok(None);
        }
        Ok(self.edges[index as usize].chance.take())
    }

    fn set_chance(&mut self, action: usize, chance: usize) -> EngineResult<()> {
        let Some(index) = self.edge_by_action.get_mut(action) else {
            return Err(internal("invalid sampled-tree learner action"));
        };
        if *index == u32::MAX {
            *index = u32::try_from(self.edges.len())
                .map_err(|_| internal("sampled-tree learner edge overflow"))?;
            self.edges.push(LearnerActionEdge {
                action,
                chance: Some(chance),
                visits: 0,
                value_sum: 0.0,
            });
            return Ok(());
        }
        let edge = &mut self.edges[*index as usize];
        if edge.chance.is_some() {
            return Err(internal("sampled-tree learner chance already installed"));
        }
        edge.chance = Some(chance);
        Ok(())
    }

    fn visits(&self, action: usize) -> u32 {
        self.edge(action).map_or(0, |edge| edge.visits)
    }

    fn total_visits(&self) -> u32 {
        self.edges.iter().map(|edge| edge.visits).sum()
    }

    fn max_visits(&self) -> u32 {
        self.edges.iter().map(|edge| edge.visits).max().unwrap_or(0)
    }
}

struct LearnerActionEdge {
    action: usize,
    chance: Option<usize>,
    visits: u32,
    value_sum: f32,
}

impl LearnerActionEdge {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

fn root_action_ref(
    context: ReplayGraphContext,
    candidates: &[RootCandidateEntry],
    action: usize,
) -> EngineResult<PortableSearchActionRef> {
    if action == candidates.len() {
        Ok(PortableSearchActionRef::stop(context))
    } else {
        candidates
            .get(action)
            .map(|entry| {
                PortableSearchActionRef::candidate(PortableCandidateRef::new(
                    context,
                    entry.candidate_hash,
                ))
            })
            .ok_or_else(|| internal("invalid sampled-tree root action"))
    }
}

fn root_action_refs(
    context: ReplayGraphContext,
    candidates: &[RootCandidateEntry],
) -> Vec<PortableSearchActionRef> {
    let mut out = candidates
        .iter()
        .map(|entry| {
            PortableSearchActionRef::candidate(PortableCandidateRef::new(
                context,
                entry.candidate_hash,
            ))
        })
        .collect::<Vec<_>>();
    out.push(PortableSearchActionRef::stop(context));
    out
}

struct ChanceNode<G> {
    live: bool,
    learner: G,
    learner_context: ReplayGraphContext,
    learner_step: usize,
    learner_stopped: bool,
    opponent: G,
    opponent_context: ReplayGraphContext,
    opponent_step: usize,
    opponent_stopped: bool,
    learner_graph_ref: Option<usize>,
    policy_owner: Option<usize>,
    policy: Option<usize>,
    branches: Vec<ChanceBranch>,
    pass: Option<Branch>,
}

impl<G> ChanceNode<G> {
    fn branch(&self, action: usize) -> Option<Branch> {
        self.branches
            .iter()
            .find(|edge| edge.action == action)
            .map(|edge| edge.branch)
    }

    fn set_branch(
        &mut self,
        action: usize,
        branch: Branch,
        graph_ref: Option<usize>,
    ) -> EngineResult<()> {
        if self.branch(action).is_some() {
            return Err(internal("sampled-tree action branch already set"));
        }
        self.branches.push(ChanceBranch {
            action,
            branch,
            graph_ref,
        });
        Ok(())
    }
}

struct ChanceBranch {
    action: usize,
    branch: Branch,
    graph_ref: Option<usize>,
}

struct ChancePolicy<C> {
    live: bool,
    candidates: Vec<C>,
    priors: Vec<f32>,
    candidate_batch: Option<usize>,
}

struct LearnerExpansionInput<G> {
    graph: G,
    opponent: G,
    opponent_context: Option<ReplayGraphContext>,
    learner_step: usize,
    opponent_step: usize,
    opponent_stopped: bool,
    depth: usize,
    attach: Option<BranchSlot>,
}

struct LearnerExpansion<G, C> {
    graph: G,
    context: ReplayGraphContext,
    opponent: G,
    opponent_context: ReplayGraphContext,
    learner_step: usize,
    opponent_step: usize,
    opponent_stopped: bool,
    depth: usize,
    attach: Option<BranchSlot>,
    candidates: Vec<CandidateEntry<C>>,
    candidate_batch: usize,
    eval_actions: Vec<EvalAction>,
}

struct ChanceExpansion<C> {
    candidates: Vec<CandidateEntry<C>>,
    candidate_batch: usize,
    eval_actions: Vec<EvalAction>,
}

struct Run<G, C> {
    base_scores: Vec<f32>,
    considered: Vec<usize>,
    schedule: Vec<u32>,
    schedule_index: usize,
    simulations: usize,
    chance_rng: GumbelRng,
    descent: Option<Descent>,
    _marker: std::marker::PhantomData<(G, C)>,
}

struct Descent {
    node_index: usize,
    depth: usize,
    path: Vec<LearnerEdge>,
    seen: HashSet<ReplayGraphContext>,
    forced: Option<usize>,
}

#[derive(Clone, Copy)]
struct LearnerEdge {
    node_index: usize,
    action: usize,
}

#[derive(Clone, Copy)]
enum Branch {
    Learner(usize),
    Chance(usize),
    Terminal(f32),
}

#[derive(Clone, Copy)]
enum BranchSlot {
    Pass(usize),
    Action(usize, usize, Option<usize>),
}

#[derive(Clone, Copy)]
struct NextPair<G> {
    learner: G,
    learner_context: ReplayGraphContext,
    learner_step: usize,
    learner_stopped: bool,
    opponent: G,
    opponent_context: ReplayGraphContext,
    opponent_step: usize,
    opponent_stopped: bool,
}

#[derive(Clone, Copy)]
struct TerminalPair<G> {
    slot: BranchSlot,
    learner: G,
    opponent: G,
}

#[allow(clippy::large_enum_variant)]
enum State<G, C> {
    ExpandLearner {
        graph: G,
        opponent: G,
        opponent_context: Option<ReplayGraphContext>,
        learner_step: usize,
        opponent_step: usize,
        opponent_stopped: bool,
        depth: usize,
        attach: Option<BranchSlot>,
        run: Option<Run<G, C>>,
    },
    EvalLearner {
        expansion: LearnerExpansion<G, C>,
        run: Option<Run<G, C>>,
    },
    ExpandChance {
        chance: usize,
        run: Run<G, C>,
    },
    EvalChance {
        chance: usize,
        expansion: ChanceExpansion<C>,
        run: Run<G, C>,
    },
    Running(Run<G, C>),
    MeasureLearner {
        terminal: TerminalPair<G>,
        run: Run<G, C>,
    },
    MeasureOpponent {
        terminal: TerminalPair<G>,
        learner_measure: MeasureResult<G>,
        run: Run<G, C>,
    },
    Work {
        work: Box<SearchWork<G, C>>,
        pending: Box<Pending<G, C>>,
    },
    Done,
}

#[allow(clippy::large_enum_variant)]
enum Pending<G, C> {
    ExpandLearner {
        token: WorkToken,
        graph: G,
        opponent: G,
        opponent_context: Option<ReplayGraphContext>,
        learner_step: usize,
        opponent_step: usize,
        opponent_stopped: bool,
        depth: usize,
        attach: Option<BranchSlot>,
        run: Option<Run<G, C>>,
    },
    EvalLearner {
        token: WorkToken,
        expansion: LearnerExpansion<G, C>,
        request: Box<EvalRequest>,
        run: Option<Run<G, C>>,
    },
    ApplyLearner {
        token: WorkToken,
        run: Run<G, C>,
        action: usize,
    },
    ExpandChance {
        token: WorkToken,
        chance: usize,
        run: Run<G, C>,
    },
    EvalChance {
        token: WorkToken,
        chance: usize,
        expansion: ChanceExpansion<C>,
        request: Box<EvalRequest>,
        run: Run<G, C>,
    },
    ApplyChance {
        token: WorkToken,
        chance: usize,
        action: usize,
        run: Run<G, C>,
    },
    MeasureLearner {
        token: WorkToken,
        terminal: TerminalPair<G>,
        run: Run<G, C>,
    },
    MeasureOpponent {
        token: WorkToken,
        terminal: TerminalPair<G>,
        learner_measure: MeasureResult<G>,
        run: Run<G, C>,
    },
}

impl<G, C> Pending<G, C> {
    fn token(&self) -> WorkToken {
        match self {
            Self::ExpandLearner { token, .. }
            | Self::EvalLearner { token, .. }
            | Self::ApplyLearner { token, .. }
            | Self::ExpandChance { token, .. }
            | Self::EvalChance { token, .. }
            | Self::ApplyChance { token, .. }
            | Self::MeasureLearner { token, .. }
            | Self::MeasureOpponent { token, .. } => *token,
        }
    }
}
