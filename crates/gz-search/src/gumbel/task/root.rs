use super::super::strategy::GumbelStrategy;
use super::super::{GumbelMcts, GumbelRootResult, GumbelSearchContext, GumbelValueMode};
use crate::mcts::task::MctsRootTask;
use crate::mcts::types::{MctsConfig, MctsRootResult, MctsSearchContext};
use crate::work::{EngineIdentity, SearchPoll, SearchWorkResult, WorkToken};
use gz_engine::{EngineResult, ReplayGraphContext};

pub struct GumbelRootTask<G, C> {
    inner: MctsRootTask<G, C, GumbelStrategy>,
}

impl<G, C> GumbelRootTask<G, C>
where
    G: Copy,
    C: Copy,
{
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelSearchContext,
    ) -> Self {
        Self {
            inner: MctsRootTask::new(
                common_config(search),
                GumbelStrategy::new(search.config),
                identity,
                root,
                common_context(context),
            ),
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelRootResult<G, C>>> {
        match self.inner.poll()? {
            SearchPoll::Work(work) => Ok(SearchPoll::Work(work)),
            SearchPoll::Blocked => Ok(SearchPoll::Blocked),
            SearchPoll::Done(result) => Ok(SearchPoll::Done(gumbel_result(result))),
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        self.inner.resume(token, result)
    }

    #[must_use]
    pub const fn root_context(&self) -> Option<ReplayGraphContext> {
        self.inner.root_context()
    }
}

pub(crate) fn common_config(search: &GumbelMcts) -> MctsConfig {
    MctsConfig {
        max_steps: search.config.max_steps,
        simulations: search.config.simulations,
        seed: search.config.seed,
        temperature_moves: search.config.temperature_moves,
        tree_reuse: search.config.tree_reuse,
        export_position: search.config.export_position,
        mask_stop: search.config.mask_stop,
        no_backtrack: search.config.no_backtrack,
        predicted_horizon: search.config.value_mode == GumbelValueMode::SingleVanilla,
        candidate_options: search.config.candidate_options,
        measure_options: search.config.measure_options,
    }
}

pub(crate) fn common_context(context: GumbelSearchContext) -> MctsSearchContext {
    MctsSearchContext {
        root_step: context.root_step,
        budget_fraction: context.budget_fraction,
        budget_step: context.budget_step,
        selection_temperature: context.selection_temperature,
        opponent: context.opponent,
        noise_seed: context.noise_seed,
        export_position: context.export_position,
    }
}

pub(crate) fn gumbel_result<G, C>(result: MctsRootResult<G, C>) -> GumbelRootResult<G, C> {
    GumbelRootResult {
        root: result.root,
        root_context: result.root_context,
        selected_after: result.selected_after,
        selected_after_context: result.selected_after_context,
        selected_action: result.selected_action,
        selected_action_ref: result.selected_action_ref,
        selected_candidate: result.selected_candidate,
        selected_action_index: result.selected_action_index,
        engine_candidate_count: result.engine_candidate_count,
        action_count: result.action_count,
        legal_actions: result.legal_actions,
        considered_action_indices: result.considered_action_indices,
        policy_target: result.policy_target,
        root_value: result.root_value,
        root_search_value: result.root_search_value,
        root_q_max: result.root_q_max,
        model_version: result.model_version,
        stats: super::super::GumbelRootStats {
            simulations: result.stats.simulations,
            expanded_nodes: result.stats.expanded_nodes,
            eval_count: result.stats.eval_count,
            portable_contexts: result.stats.portable_contexts,
            carried_nodes: result.stats.carried_nodes,
            carried_root_visits: result.stats.carried_root_visits,
        },
    }
}
