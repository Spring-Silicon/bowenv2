use super::super::strategy::{MctsStrategy, MctsStrategyState};
use super::super::tree::{MctsEdge, MctsNode, MctsTree};
use super::super::types::{
    MctsConfig, MctsHandleBatch, MctsRootResult, MctsRootStats, MctsSearchContext,
};
use super::state::{
    DescentPoll, DescentState, NodeExpansion, PendingRootWork, RootTaskState, RunState,
};
use crate::SearchCandidateSummary;
use crate::support::internal;
use crate::work::{
    ApplyWork, EngineIdentity, EvalModel, EvalWork, ExpandResult, ExpandWork, SearchPoll,
    SearchWork, SearchWorkResult, WorkToken,
};
use gz_engine::{ApplyResult, EngineResult, PortableCandidateRef, ReplayGraphContext};
use gz_eval::{EvalAction, EvalRequest, eval_error_to_engine_error};
use std::collections::HashSet;

pub(crate) struct MctsRootTask<G, C, S>
where
    S: MctsStrategyState,
{
    config: MctsConfig,
    strategy: S,
    identity: EngineIdentity,
    root: G,
    pub(super) context: MctsSearchContext,
    root_context: Option<ReplayGraphContext>,
    visited: HashSet<ReplayGraphContext>,
    tree: MctsTree<G, C>,
    next_token: u64,
    pending: Option<PendingRootWork<G, C, S::RootState>>,
    state: RootTaskState<G, C, S::RootState>,
}

impl<G, C, S> MctsRootTask<G, C, S>
where
    G: Copy,
    C: Copy,
    S: MctsStrategy<G, C>,
{
    pub(crate) fn new(
        config: MctsConfig,
        strategy: S,
        identity: EngineIdentity,
        root: G,
        context: MctsSearchContext,
    ) -> Self {
        Self {
            config,
            strategy,
            identity,
            root,
            context,
            root_context: None,
            visited: HashSet::new(),
            tree: MctsTree::new(config),
            next_token: 0,
            pending: None,
            state: RootTaskState::EmitNodeExpand {
                graph: root,
                expected_context: None,
                depth: 0,
                run: None,
            },
        }
    }

    pub(crate) const fn root_context(&self) -> Option<ReplayGraphContext> {
        self.root_context
    }

    pub(crate) fn set_visited(&mut self, visited: HashSet<ReplayGraphContext>) {
        self.visited = visited;
    }

    pub(crate) fn reused_child_task(
        &self,
        action: usize,
        root: G,
        expected_context: ReplayGraphContext,
        context: MctsSearchContext,
    ) -> EngineResult<Option<ReusedMctsRootTask<G, C, S>>> {
        let Some(child_index) = self.tree.nodes[0].children[action] else {
            return Ok(None);
        };
        let (tree, root_context, handles) = self.tree.compact_subtree(child_index)?;
        if root_context != expected_context {
            return Err(internal("reused root context mismatch"));
        }

        let mut task = Self {
            config: self.config,
            strategy: self.strategy,
            identity: self.identity,
            root,
            context,
            root_context: Some(root_context),
            visited: HashSet::new(),
            tree,
            next_token: 0,
            pending: None,
            state: RootTaskState::Done,
        };
        let run = task.start_run_state();
        task.state = RootTaskState::Running(run);
        Ok(Some(ReusedMctsRootTask { task, handles }))
    }

    pub(crate) fn poll(&mut self) -> EngineResult<SearchPoll<G, C, MctsRootResult<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }

        let state = std::mem::replace(&mut self.state, RootTaskState::Done);
        match state {
            RootTaskState::EmitNodeExpand {
                graph,
                expected_context,
                depth,
                run,
            } => {
                let token = self.next_token();
                self.pending = Some(PendingRootWork::ExpandNode {
                    token,
                    graph,
                    expected_context,
                    depth,
                    run,
                });
                Ok(SearchPoll::Work(SearchWork::Expand(ExpandWork {
                    token,
                    graph,
                    options: self.config.candidate_options,
                })))
            }
            RootTaskState::EmitNodeEval { expansion, run } => self.poll_node_eval(expansion, run),
            RootTaskState::Running(run) => self.poll_running(run),
            RootTaskState::Done => Err(internal("poll after done")),
        }
    }

    pub(crate) fn resume(
        &mut self,
        token: WorkToken,
        result: SearchWorkResult<G, C>,
    ) -> EngineResult<()> {
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
                PendingRootWork::ExpandNode {
                    graph,
                    expected_context,
                    depth,
                    run,
                    ..
                },
                SearchWorkResult::Expand(result),
            ) => self.resume_expand(graph, expected_context, depth, run, result),
            (
                PendingRootWork::EvalNode {
                    expansion,
                    run,
                    request,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => self.resume_node_eval(expansion, run, &request, output),
            (PendingRootWork::Apply { run, action, .. }, SearchWorkResult::Apply(result)) => {
                self.resume_apply(run, action, result)
            }
            (
                PendingRootWork::StopEval {
                    mut run, request, ..
                },
                SearchWorkResult::Eval(output),
            ) => {
                output
                    .validate_for(&request)
                    .map_err(eval_error_to_engine_error)?;
                self.tree.eval_count += 1;
                let path = run
                    .descent
                    .as_ref()
                    .ok_or_else(|| internal("missing descent"))?
                    .path
                    .clone();
                self.tree.backup(&path, output.value);
                run.complete_simulation();
                self.state = RootTaskState::Running(run);
                Ok(())
            }
            (pending, _) => {
                self.pending = Some(pending);
                Err(internal("mismatched work result"))
            }
        }
    }

    fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }

    fn start_run_state(&self) -> RunState<S::RootState> {
        RunState {
            strategy: self.strategy.start_root(&self.tree, self.context),
            simulations: 0,
            descent: None,
        }
    }

    fn poll_node_eval(
        &mut self,
        expansion: NodeExpansion<G, C>,
        run: Option<RunState<S::RootState>>,
    ) -> EngineResult<SearchPoll<G, C, MctsRootResult<G, C>>> {
        let request = EvalRequest::with_position(
            expansion.context,
            expansion.eval_actions.clone(),
            self.context.position(expansion.depth),
        )
        .map_err(|_| internal("invalid mcts eval request"))?;
        let token = self.next_token();
        let work = EvalWork {
            token,
            graph: expansion.graph,
            candidates: expansion.candidates.clone(),
            request: request.clone(),
            measure_options: self.config.measure_options,
            model: EvalModel::Episode,
            opponent: None,
        };
        self.pending = Some(PendingRootWork::EvalNode {
            token,
            expansion,
            run,
            request,
        });
        Ok(SearchPoll::Work(SearchWork::Eval(work)))
    }

    fn poll_running(
        &mut self,
        mut run: RunState<S::RootState>,
    ) -> EngineResult<SearchPoll<G, C, MctsRootResult<G, C>>> {
        loop {
            if run.descent.is_none() && !self.start_descent(&mut run) {
                return self.finish_root(run);
            }
            match self.poll_descent(run)? {
                DescentPoll::Continue(next) => run = next,
                DescentPoll::Work(work, pending) => {
                    self.pending = Some(pending);
                    return Ok(SearchPoll::Work(work));
                }
            }
        }
    }

    fn start_descent(&self, run: &mut RunState<S::RootState>) -> bool {
        let Some(action) =
            self.strategy
                .select_root(&mut run.strategy, &self.tree, run.simulations)
        else {
            return false;
        };
        run.descent = Some(DescentState {
            node_index: 0,
            depth: 0,
            path: Vec::new(),
            seen: HashSet::from([self.tree.nodes[0].context]),
            forced: Some(action),
        });
        true
    }

    fn poll_descent(
        &mut self,
        mut run: RunState<S::RootState>,
    ) -> EngineResult<DescentPoll<G, C, S::RootState>> {
        let mut descent = run
            .descent
            .take()
            .ok_or_else(|| internal("missing descent"))?;
        let action = match descent.forced.take() {
            Some(action) => action,
            None => self
                .strategy
                .select_nonroot(&run.strategy, &self.tree, descent.node_index),
        };

        if self.tree.nodes[descent.node_index].is_stop(action) {
            descent.path.push(MctsEdge {
                node_index: descent.node_index,
                action,
            });
            if let Some(request) = self.stop_eval_request(descent.node_index, descent.depth)? {
                run.descent = Some(descent);
                let token = self.next_token();
                let node = &self.tree.nodes[run.descent.as_ref().unwrap().node_index];
                let work = EvalWork {
                    token,
                    graph: node.graph,
                    candidates: node.candidates.clone(),
                    request: request.clone(),
                    measure_options: self.config.measure_options,
                    model: EvalModel::Episode,
                    opponent: None,
                };
                let pending = PendingRootWork::StopEval {
                    token,
                    run,
                    request,
                };
                return Ok(DescentPoll::Work(SearchWork::Eval(work), pending));
            }

            let value = self.tree.nodes[descent.node_index].value;
            self.tree.backup(&descent.path, value);
            run.descent = Some(descent);
            run.complete_simulation();
            return Ok(DescentPoll::Continue(run));
        }

        let graph = self.tree.nodes[descent.node_index].graph;
        let candidate = self.tree.nodes[descent.node_index].candidates[action];
        run.descent = Some(descent);
        let token = self.next_token();
        let work = ApplyWork {
            token,
            graph,
            candidate,
        };
        let pending = PendingRootWork::Apply { token, run, action };
        Ok(DescentPoll::Work(SearchWork::Apply(work), pending))
    }

    fn resume_expand(
        &mut self,
        graph: G,
        expected_context: Option<ReplayGraphContext>,
        depth: usize,
        run: Option<RunState<S::RootState>>,
        result: ExpandResult<C>,
    ) -> EngineResult<()> {
        let context = self.identity.context(result.graph_hash);
        self.tree.portable_contexts += 1;
        if let Some(expected) = expected_context
            && expected != context
        {
            return Err(internal("expand graph hash mismatch"));
        }
        if run.is_none() {
            self.root_context = Some(context);
        }

        let mut candidates = Vec::with_capacity(result.candidates.len());
        let mut eval_actions = Vec::with_capacity(result.candidates.len() + 1);
        let mut candidate_hashes = Vec::with_capacity(result.candidates.len());
        let mut summaries = Vec::with_capacity(result.candidates.len() + 1);
        for expanded in result.candidates {
            candidates.push(expanded.candidate);
            let candidate_ref = PortableCandidateRef::new(context, expanded.candidate_hash);
            candidate_hashes.push(expanded.candidate_hash);
            eval_actions.push(EvalAction::candidate(
                candidate_ref,
                expanded.kind,
                expanded.tags,
                expanded.static_prior,
            ));
            summaries.push(Some(SearchCandidateSummary {
                kind: expanded.kind,
                tags: expanded.tags,
                static_prior: expanded.static_prior,
            }));
        }
        eval_actions.push(EvalAction::stop(context));
        summaries.push(None);
        self.state = RootTaskState::EmitNodeEval {
            expansion: NodeExpansion {
                graph,
                context,
                depth,
                candidates,
                eval_actions,
                candidate_hashes,
                summaries,
            },
            run,
        };
        Ok(())
    }

    fn resume_node_eval(
        &mut self,
        expansion: NodeExpansion<G, C>,
        mut run: Option<RunState<S::RootState>>,
        request: &EvalRequest,
        output: gz_eval::EvalOutput,
    ) -> EngineResult<()> {
        output
            .validate_for(request)
            .map_err(eval_error_to_engine_error)?;
        let node_index = self.finalize_node(expansion, output);
        if let Some(mut run) = run.take() {
            let value = self.tree.nodes[node_index].value;
            let descent = run
                .descent
                .as_ref()
                .ok_or_else(|| internal("missing descent"))?;
            let edge = *descent
                .path
                .last()
                .ok_or_else(|| internal("missing edge"))?;
            self.tree.nodes[edge.node_index].children[edge.action] = Some(node_index);
            self.tree.backup(&descent.path, value);
            run.complete_simulation();
            self.state = RootTaskState::Running(run);
        } else {
            self.state = RootTaskState::Running(self.start_run_state());
        }
        Ok(())
    }

    fn resume_apply(
        &mut self,
        mut run: RunState<S::RootState>,
        action: usize,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<()> {
        let mut descent = run
            .descent
            .take()
            .ok_or_else(|| internal("missing descent"))?;
        if applied.rejected.is_some() {
            self.tree.mask_action(descent.node_index, action);
            run.descent = Some(descent);
            self.state = RootTaskState::Running(run);
            return Ok(());
        }

        let child_context = self.identity.context(applied.after_hash);
        self.tree.portable_contexts += 1;
        if self.config.no_backtrack
            && (self.root_context == Some(child_context) || self.visited.contains(&child_context))
        {
            self.tree.mask_action(descent.node_index, action);
            run.descent = Some(descent);
            self.state = RootTaskState::Running(run);
            return Ok(());
        }

        descent.path.push(MctsEdge {
            node_index: descent.node_index,
            action,
        });
        if let Some(child) = self.tree.nodes[descent.node_index].children[action] {
            if !descent.seen.insert(self.tree.nodes[child].context) {
                let value = self.tree.nodes[child].value;
                self.tree.backup(&descent.path, value);
                run.descent = Some(descent);
                run.complete_simulation();
                self.state = RootTaskState::Running(run);
                return Ok(());
            }
            descent.node_index = child;
            descent.depth += 1;
            run.descent = Some(descent);
            self.state = RootTaskState::Running(run);
            return Ok(());
        }

        let depth = descent.depth + 1;
        run.descent = Some(descent);
        self.state = RootTaskState::EmitNodeExpand {
            graph: applied.after,
            expected_context: Some(child_context),
            depth,
            run: Some(run),
        };
        Ok(())
    }

    fn stop_eval_request(
        &self,
        node_index: usize,
        depth: usize,
    ) -> EngineResult<Option<EvalRequest>> {
        let Some(opponent) = self.context.opponent else {
            return Ok(None);
        };
        let Some(last) = opponent.row_count.checked_sub(1) else {
            return Ok(None);
        };
        let effective_depth = depth.max(last.saturating_sub(self.context.root_step) as usize);
        if effective_depth == depth {
            return Ok(None);
        }
        let node = &self.tree.nodes[node_index];
        EvalRequest::with_position(
            node.context,
            node.eval_actions.clone(),
            self.context.position(effective_depth),
        )
        .map(Some)
        .map_err(|_| internal("invalid mcts stop eval request"))
    }

    fn finalize_node(
        &mut self,
        expansion: NodeExpansion<G, C>,
        output: gz_eval::EvalOutput,
    ) -> usize {
        let eval_actions = if self.context.opponent.is_some() {
            expansion.eval_actions
        } else {
            Vec::new()
        };
        let node = MctsNode::new(
            expansion.graph,
            expansion.context,
            expansion.candidates,
            eval_actions,
            expansion.candidate_hashes,
            expansion.summaries,
            output,
            self.config.mask_stop,
        );
        self.tree.eval_count += 1;
        self.tree.nodes.push(node);
        self.tree.nodes.len() - 1
    }

    fn finish_root(
        &mut self,
        run: RunState<S::RootState>,
    ) -> EngineResult<SearchPoll<G, C, MctsRootResult<G, C>>> {
        let root_index = 0;
        let root_node = &self.tree.nodes[root_index];
        let strategy_result = self
            .strategy
            .finish_root(run.strategy, &self.tree, self.context);
        let selected = strategy_result.selected;
        let selected_after = self.tree.selected_after(root_index, selected)?;
        let selected_after_context = self.tree.selected_after_context(root_index, selected)?;
        let selected_action = root_node.search_action(selected)?;
        let root_context = self
            .root_context
            .ok_or_else(|| internal("missing root context"))?;

        self.state = RootTaskState::Done;
        Ok(SearchPoll::Done(MctsRootResult {
            root: self.root,
            root_context,
            selected_after,
            selected_after_context,
            selected_action,
            selected_action_ref: root_node.action_ref(selected)?,
            selected_candidate: root_node.summaries[selected],
            selected_action_index: selected,
            engine_candidate_count: root_node.candidates.len(),
            action_count: root_node.action_count(),
            legal_actions: root_node.action_refs(),
            considered_action_indices: strategy_result.considered_action_indices,
            policy_target: strategy_result.policy_target,
            root_value: root_node.value,
            root_search_value: strategy_result.root_search_value,
            root_q_max: strategy_result.root_q_max,
            model_version: root_node.model_version,
            stats: MctsRootStats {
                simulations: run.simulations,
                expanded_nodes: self.tree.nodes.len(),
                eval_count: self.tree.eval_count,
                portable_contexts: self.tree.portable_contexts,
                carried_nodes: self.tree.carried_nodes,
                carried_root_visits: self.tree.carried_root_visits,
            },
        }))
    }
}

pub(crate) struct ReusedMctsRootTask<G, C, S>
where
    S: MctsStrategyState,
{
    pub(crate) task: MctsRootTask<G, C, S>,
    pub(crate) handles: MctsHandleBatch<G, C>,
}
