use super::tree::MctsTree;
use super::types::MctsSearchContext;

pub(crate) struct StrategyRootResult {
    pub(crate) selected: usize,
    pub(crate) considered_action_indices: Vec<usize>,
    pub(crate) policy_target: Vec<f32>,
    pub(crate) root_search_value: f32,
    pub(crate) root_q_max: f32,
}

pub(crate) trait MctsStrategyState: Copy {
    type RootState;
}

pub(crate) trait MctsStrategy<G, C>: MctsStrategyState {
    fn start_root(&self, tree: &MctsTree<G, C>, context: MctsSearchContext) -> Self::RootState;

    fn select_root(
        &self,
        state: &mut Self::RootState,
        tree: &MctsTree<G, C>,
        simulations: usize,
    ) -> Option<usize>;

    fn select_nonroot(
        &self,
        state: &Self::RootState,
        tree: &MctsTree<G, C>,
        node_index: usize,
    ) -> usize;

    fn finish_root(
        &self,
        state: Self::RootState,
        tree: &MctsTree<G, C>,
        context: MctsSearchContext,
    ) -> StrategyRootResult;
}
