use super::PuctMctsConfig;
use crate::mcts::math::{MctsRng, root_seed, sample_visit_action};
use crate::mcts::strategy::{MctsStrategy, MctsStrategyState, StrategyRootResult};
use crate::mcts::tree::{MctsNode, MctsTree};
use crate::mcts::types::MctsSearchContext;

#[derive(Clone, Copy)]
pub(crate) struct PuctStrategy {
    config: PuctMctsConfig,
}

impl PuctStrategy {
    pub(crate) const fn new(config: PuctMctsConfig) -> Self {
        Self { config }
    }
}

pub(crate) struct PuctRootState {
    baseline_visits: Vec<u32>,
    rng: MctsRng,
}

impl MctsStrategyState for PuctStrategy {
    type RootState = PuctRootState;
}

impl<G, C> MctsStrategy<G, C> for PuctStrategy
where
    G: Copy,
    C: Copy,
{
    fn start_root(&self, tree: &MctsTree<G, C>, context: MctsSearchContext) -> Self::RootState {
        Self::RootState {
            baseline_visits: tree.nodes[0].visits.clone(),
            rng: MctsRng::new(root_seed(
                self.config.seed ^ context.noise_seed,
                context.root_step,
            )),
        }
    }

    fn select_root(
        &self,
        _state: &mut Self::RootState,
        tree: &MctsTree<G, C>,
        simulations: usize,
    ) -> Option<usize> {
        (simulations < self.config.simulations.get())
            .then(|| select_action(&tree.nodes[0], self.config.c_puct))
    }

    fn select_nonroot(
        &self,
        _state: &mut Self::RootState,
        tree: &MctsTree<G, C>,
        node_index: usize,
    ) -> usize {
        select_action(&tree.nodes[node_index], self.config.c_puct)
    }

    fn finish_root(
        &self,
        mut state: Self::RootState,
        tree: &MctsTree<G, C>,
        context: MctsSearchContext,
    ) -> StrategyRootResult {
        let node = &tree.nodes[0];
        let selectable = node.unmasked_actions().collect::<Vec<_>>();
        let fallback = deterministic_election(node, &state.baseline_visits, &selectable);
        let selected = sample_visit_action(
            &mut state.rng,
            &node.visits,
            &state.baseline_visits,
            &selectable,
            context.selection_temperature,
            fallback,
        );
        let policy_target = policy_target(node, &state.baseline_visits);

        StrategyRootResult {
            selected,
            considered_action_indices: selectable,
            root_search_value: search_value(node, &state.baseline_visits),
            root_q_max: root_q_max(node),
            policy_target,
        }
    }
}

fn select_action<G, C>(node: &MctsNode<G, C>, c_puct: f32) -> usize
where
    C: Copy,
{
    let total_visits = node.visits.iter().copied().sum::<u32>() as f32;
    let parent_scale = total_visits.max(1.0).sqrt();
    node.unmasked_actions()
        .max_by(|left, right| {
            let left_score = score(node, *left, c_puct, parent_scale);
            let right_score = score(node, *right, c_puct, parent_scale);
            left_score
                .total_cmp(&right_score)
                .then_with(|| right.cmp(left))
        })
        .expect("STOP guarantees an unmasked action")
}

fn score<G, C>(node: &MctsNode<G, C>, action: usize, c_puct: f32, parent_scale: f32) -> f32 {
    node.q[action]
        + c_puct * node.priors[action] * parent_scale / (1.0 + node.visits[action] as f32)
}

fn deterministic_election<G, C>(
    node: &MctsNode<G, C>,
    baseline_visits: &[u32],
    selectable: &[usize],
) -> usize {
    selectable
        .iter()
        .copied()
        .max_by(|left, right| {
            let left_delta = node.visits[*left].saturating_sub(baseline_visits[*left]);
            let right_delta = node.visits[*right].saturating_sub(baseline_visits[*right]);
            left_delta
                .cmp(&right_delta)
                .then_with(|| node.q[*left].total_cmp(&node.q[*right]))
                .then_with(|| node.priors[*left].total_cmp(&node.priors[*right]))
                .then_with(|| right.cmp(left))
        })
        .expect("STOP guarantees an unmasked action")
}

fn policy_target<G, C>(node: &MctsNode<G, C>, baseline_visits: &[u32]) -> Vec<f32>
where
    C: Copy,
{
    let deltas = node
        .visits
        .iter()
        .zip(baseline_visits)
        .zip(&node.masked)
        .map(|((visits, baseline), masked)| {
            if *masked {
                0
            } else {
                visits.saturating_sub(*baseline)
            }
        })
        .collect::<Vec<_>>();
    let total = deltas.iter().copied().sum::<u32>();
    if total == 0 {
        let mut target = vec![0.0; node.action_count()];
        target[node.candidates.len()] = 1.0;
        return target;
    }
    deltas
        .into_iter()
        .map(|visits| visits as f32 / total as f32)
        .collect()
}

fn search_value<G, C>(node: &MctsNode<G, C>, baseline_visits: &[u32]) -> f32 {
    let mut visits = 0;
    let mut value = 0.0;
    for (index, q) in node.q.iter().copied().enumerate() {
        let delta = node.visits[index].saturating_sub(baseline_visits[index]);
        if delta > 0 {
            visits += delta;
            value += delta as f32 * q;
        }
    }
    if visits == 0 {
        node.value
    } else {
        value / visits as f32
    }
}

fn root_q_max<G, C>(node: &MctsNode<G, C>) -> f32 {
    node.visits
        .iter()
        .zip(&node.q)
        .zip(&node.masked)
        .filter_map(|((visits, q), masked)| (*visits > 0 && !*masked).then_some(*q))
        .reduce(f32::max)
        .unwrap_or(node.value)
}
