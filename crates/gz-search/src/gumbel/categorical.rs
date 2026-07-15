use super::schedule::{GumbelRng, budget_fraction, root_seed, sample_root_gumbels};
use super::{
    GumbelEpisode, GumbelEpisodeContext, GumbelHandleBatch, GumbelMcts, GumbelMctsConfig,
    GumbelRootStats, GumbelStep, GumbelStopReason,
};
use crate::support::{internal, step_ref};
use crate::work::{
    ApplyWork, EngineIdentity, EvalWork, ExpandResult, ExpandWork, MeasureWork, SearchPoll,
    SearchWork, SearchWorkResult, WorkToken,
};
use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    ApplyResult, EngineResult, PortableCandidateRef, PortableSearchActionRef, ReplayGraphContext,
    SearchConfigHash,
};
use gz_eval::{
    EvalAction, EvalOutput, EvalPositionContext, EvalRequest, eval_error_to_engine_error,
};
use std::collections::HashSet;
use std::hash::Hash;

/// Direct categorical policy rollout used by sampled-trajectory opponents.
/// It evaluates each played root once, ranks actions by logit + unit Gumbel,
/// and tries that fixed ranking until an action passes apply/history masks.
pub struct CategoricalPolicyEpisodeTask<G, C> {
    config: GumbelMctsConfig,
    search_config_hash: SearchConfigHash,
    identity: EngineIdentity,
    root: G,
    context: GumbelEpisodeContext,
    current: G,
    current_context: Option<ReplayGraphContext>,
    root_context: Option<ReplayGraphContext>,
    visited: HashSet<ReplayGraphContext>,
    steps: Vec<GumbelStep<G, C>>,
    root_stats: Vec<GumbelRootStats>,
    created_graphs: Vec<G>,
    created_candidates: Vec<C>,
    releasable: GumbelHandleBatch<G, C>,
    step_index: usize,
    next_token: u64,
    pending: Option<Pending<G, C>>,
    state: State<G, C>,
}

impl<G, C> CategoricalPolicyEpisodeTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    #[must_use]
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelEpisodeContext,
    ) -> Self {
        let config = search.config();
        Self {
            config,
            search_config_hash: crate::categorical_policy_config_hash(
                config.max_steps,
                config.seed,
                config.mask_stop,
                config.no_backtrack,
                config.candidate_options,
                config.measure_options,
            ),
            identity,
            root,
            context,
            current: root,
            current_context: None,
            root_context: None,
            visited: HashSet::new(),
            steps: Vec::new(),
            root_stats: Vec::new(),
            created_graphs: Vec::new(),
            created_candidates: Vec::new(),
            releasable: GumbelHandleBatch::default(),
            step_index: 0,
            next_token: 0,
            pending: None,
            state: State::Start,
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelEpisode<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }

        loop {
            let state = std::mem::replace(&mut self.state, State::Done);
            match state {
                State::Start => {
                    self.state = if self.config.max_steps == 0 {
                        State::Measure(GumbelStopReason::MaxSteps)
                    } else {
                        State::Expand
                    };
                }
                State::Expand => {
                    let token = self.next_token();
                    self.pending = Some(Pending::Expand { token });
                    return Ok(SearchPoll::Work(SearchWork::Expand(ExpandWork {
                        token,
                        graph: self.current,
                        options: self.config.candidate_options,
                    })));
                }
                State::Eval(root) => {
                    let request = EvalRequest::with_position(
                        root.context,
                        root.eval_actions.clone(),
                        self.position(),
                    )
                    .map_err(|_| internal("invalid categorical eval request"))?;
                    let token = self.next_token();
                    let work = EvalWork {
                        token,
                        graph: self.current,
                        candidates: root
                            .candidates
                            .iter()
                            .map(|entry| entry.candidate)
                            .collect(),
                        request: request.clone(),
                        measure_options: self.config.measure_options,
                        model: crate::work::EvalModel::Episode,
                        opponent: None,
                    };
                    self.pending = Some(Pending::Eval {
                        token,
                        root,
                        request: Box::new(request),
                    });
                    return Ok(SearchPoll::Work(SearchWork::Eval(work)));
                }
                State::Choose(mut choice) => {
                    let Some(action) = choice.ranking.get(choice.cursor).copied() else {
                        self.select_stop(choice.root, choice.output)?;
                        continue;
                    };
                    choice.cursor += 1;
                    let stop = choice.root.candidates.len();
                    if action == stop {
                        self.select_stop(choice.root, choice.output)?;
                        continue;
                    }
                    let token = self.next_token();
                    let candidate = choice.root.candidates[action].candidate;
                    self.pending = Some(Pending::Apply {
                        token,
                        choice,
                        action,
                    });
                    return Ok(SearchPoll::Work(SearchWork::Apply(ApplyWork {
                        token,
                        graph: self.current,
                        candidate,
                    })));
                }
                State::Measure(stop_reason) => {
                    let token = self.next_token();
                    self.pending = Some(Pending::Measure { token, stop_reason });
                    return Ok(SearchPoll::Work(SearchWork::Measure(MeasureWork {
                        token,
                        graph: self.current,
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

        match (pending, result) {
            (Pending::Expand { .. }, SearchWorkResult::Expand(expanded)) => {
                self.resume_expand(expanded)
            }
            (Pending::Eval { root, request, .. }, SearchWorkResult::Eval(output)) => {
                self.resume_eval(root, *request, output)
            }
            (Pending::Apply { choice, action, .. }, SearchWorkResult::Apply(applied)) => {
                self.resume_apply(choice, action, applied)
            }
            (Pending::Measure { stop_reason, .. }, SearchWorkResult::Measure(measure)) => {
                let final_context = self
                    .current_context
                    .unwrap_or_else(|| self.identity.context(measure.graph_hash));
                let root_context = self.root_context.unwrap_or(final_context);
                self.state = State::DoneResult(Box::new(GumbelEpisode {
                    root: self.root,
                    final_graph: self.current,
                    root_context,
                    final_context,
                    steps: std::mem::take(&mut self.steps),
                    root_stats: std::mem::take(&mut self.root_stats),
                    created_graphs: std::mem::take(&mut self.created_graphs),
                    created_candidates: std::mem::take(&mut self.created_candidates),
                    final_measure: measure,
                    stop_reason,
                    search_config_hash: self.search_config_hash,
                    competitive: None,
                }));
                Ok(())
            }
            (pending, _) => {
                self.pending = Some(pending);
                Err(internal("mismatched work result"))
            }
        }
    }

    #[must_use]
    pub const fn step_index(&self) -> usize {
        self.step_index
    }

    pub fn take_releasable(&mut self) -> GumbelHandleBatch<G, C> {
        std::mem::take(&mut self.releasable)
    }

    pub fn track_owned_root(&mut self) {
        self.created_graphs.push(self.root);
    }

    pub fn take_all_handles(&mut self) -> GumbelHandleBatch<G, C> {
        let mut handles = self.take_releasable();
        handles.graphs.append(&mut self.created_graphs);
        handles.candidates.append(&mut self.created_candidates);
        handles
    }

    fn resume_expand(&mut self, expanded: ExpandResult<C>) -> EngineResult<()> {
        self.created_candidates.extend(
            expanded
                .candidates
                .iter()
                .map(|candidate| candidate.candidate),
        );
        let context = self.identity.context(expanded.graph_hash);
        if let Some(expected) = self.current_context
            && expected != context
        {
            return Err(internal("expand graph hash mismatch"));
        }
        self.current_context = Some(context);
        self.root_context.get_or_insert(context);

        let mut candidates = Vec::with_capacity(expanded.candidates.len());
        let mut eval_actions = Vec::with_capacity(expanded.candidates.len() + 1);
        for candidate in expanded.candidates {
            let action_ref = PortableSearchActionRef::candidate(PortableCandidateRef::new(
                context,
                candidate.candidate_hash,
            ));
            eval_actions.push(EvalAction::candidate(
                PortableCandidateRef::new(context, candidate.candidate_hash),
                candidate.kind,
                candidate.tags,
                candidate.static_prior,
            ));
            candidates.push(CandidateEntry {
                candidate: candidate.candidate,
                action_ref,
                summary: SearchCandidateSummary {
                    kind: candidate.kind,
                    tags: candidate.tags,
                    static_prior: candidate.static_prior,
                },
            });
        }
        eval_actions.push(EvalAction::stop(context));
        self.state = State::Eval(RootData {
            context,
            candidates,
            eval_actions,
        });
        Ok(())
    }

    fn resume_eval(
        &mut self,
        root: RootData<C>,
        request: EvalRequest,
        output: EvalOutput,
    ) -> EngineResult<()> {
        output
            .validate_for(&request)
            .map_err(eval_error_to_engine_error)?;
        let mut rng = GumbelRng::new(root_seed(
            self.config.seed ^ self.context.noise_seed,
            self.step_index as u32,
        ));
        let gumbels = sample_root_gumbels(output.policy_logits.len(), 1.0, &mut rng);
        let stop = root.candidates.len();
        let mut ranking = (0..output.policy_logits.len()).collect::<Vec<_>>();
        ranking.sort_by(|&left, &right| {
            let left_score = if self.config.mask_stop && !root.candidates.is_empty() && left == stop
            {
                f32::NEG_INFINITY
            } else {
                output.policy_logits[left] + gumbels[left]
            };
            let right_score =
                if self.config.mask_stop && !root.candidates.is_empty() && right == stop {
                    f32::NEG_INFINITY
                } else {
                    output.policy_logits[right] + gumbels[right]
                };
            right_score
                .total_cmp(&left_score)
                .then_with(|| left.cmp(&right))
        });
        if self.config.mask_stop && !root.candidates.is_empty() {
            ranking.retain(|&action| action != stop);
        }
        self.state = State::Choose(Box::new(Choice {
            root,
            output,
            ranking,
            cursor: 0,
        }));
        Ok(())
    }

    fn resume_apply(
        &mut self,
        choice: Box<Choice<C>>,
        action: usize,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<()> {
        let after_context = self.identity.context(applied.after_hash);
        if applied.rejected.is_some()
            || (self.config.no_backtrack
                && (Some(after_context) == self.current_context
                    || self.visited.contains(&after_context)))
        {
            self.releasable.graphs.push(applied.after);
            self.state = State::Choose(choice);
            return Ok(());
        }

        self.created_graphs.push(applied.after);
        let before = self.current;
        let before_context = self
            .current_context
            .ok_or_else(|| internal("missing categorical root context"))?;
        let entry = choice.root.candidates[action];
        self.push_step(
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
        if self.config.no_backtrack {
            self.visited.insert(before_context);
        }
        self.current = applied.after;
        self.current_context = Some(after_context);
        self.step_index += 1;
        self.state = if self.step_index >= self.config.max_steps {
            State::Measure(GumbelStopReason::MaxSteps)
        } else {
            State::Expand
        };
        Ok(())
    }

    fn select_stop(&mut self, root: RootData<C>, output: EvalOutput) -> EngineResult<()> {
        let context = self
            .current_context
            .ok_or_else(|| internal("missing categorical root context"))?;
        let stop = root.candidates.len();
        let choice = Choice {
            root,
            output,
            ranking: Vec::new(),
            cursor: 0,
        };
        self.push_step(
            self.current,
            self.current,
            context,
            context,
            SearchAction::Stop,
            PortableSearchActionRef::stop(context),
            None,
            stop,
            &choice,
        )?;
        self.step_index += 1;
        self.state = State::Measure(GumbelStopReason::SelectedStop);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn push_step(
        &mut self,
        before: G,
        after: G,
        before_context: ReplayGraphContext,
        after_context: ReplayGraphContext,
        action: SearchAction<C>,
        action_ref: PortableSearchActionRef,
        summary: Option<SearchCandidateSummary>,
        selected: usize,
        choice: &Choice<C>,
    ) -> EngineResult<()> {
        let action_count = choice.root.candidates.len() + 1;
        let mut legal_actions = choice
            .root
            .candidates
            .iter()
            .map(|entry| entry.action_ref)
            .collect::<Vec<_>>();
        legal_actions.push(PortableSearchActionRef::stop(before_context));
        let mut policy_target = vec![0.0; action_count];
        policy_target[selected] = 1.0;
        self.steps.push(GumbelStep {
            before,
            after,
            action,
            step_ref: step_ref(before_context, action_ref, after_context)?,
            selected_action: action_ref,
            selected_candidate: summary,
            engine_candidate_count: choice.root.candidates.len(),
            action_count,
            selected_rank: selected,
            legal_actions,
            policy_target,
            considered_action_indices: vec![selected],
            root_value: choice.output.value,
            root_search_value: choice.output.value,
            root_q_max: choice.output.value,
            model_version: choice.output.model_version,
        });
        self.root_stats.push(GumbelRootStats {
            simulations: 0,
            expanded_nodes: 1,
            eval_count: 1,
            portable_contexts: 1,
            carried_nodes: 0,
            carried_root_visits: 0,
        });
        Ok(())
    }

    fn position(&self) -> EvalPositionContext {
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
            root_step: self.step_index as u32,
            leaf_depth: 0,
            budget_fraction: budget_fraction(self.config.max_steps, self.step_index),
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

#[derive(Clone, Copy)]
struct CandidateEntry<C> {
    candidate: C,
    action_ref: PortableSearchActionRef,
    summary: SearchCandidateSummary,
}

struct RootData<C> {
    context: ReplayGraphContext,
    candidates: Vec<CandidateEntry<C>>,
    eval_actions: Vec<EvalAction>,
}

struct Choice<C> {
    root: RootData<C>,
    output: EvalOutput,
    ranking: Vec<usize>,
    cursor: usize,
}

enum State<G, C> {
    Start,
    Expand,
    Eval(RootData<C>),
    Choose(Box<Choice<C>>),
    Measure(GumbelStopReason),
    DoneResult(Box<GumbelEpisode<G, C>>),
    Done,
}

enum Pending<G, C> {
    Expand {
        token: WorkToken,
    },
    Eval {
        token: WorkToken,
        root: RootData<C>,
        request: Box<EvalRequest>,
    },
    Apply {
        token: WorkToken,
        choice: Box<Choice<C>>,
        action: usize,
    },
    Measure {
        token: WorkToken,
        stop_reason: GumbelStopReason,
    },
    #[allow(dead_code)]
    Marker(std::marker::PhantomData<G>),
}

impl<G, C> Pending<G, C> {
    const fn token(&self) -> WorkToken {
        match self {
            Self::Expand { token }
            | Self::Eval { token, .. }
            | Self::Apply { token, .. }
            | Self::Measure { token, .. } => *token,
            Self::Marker(_) => unreachable!(),
        }
    }
}
