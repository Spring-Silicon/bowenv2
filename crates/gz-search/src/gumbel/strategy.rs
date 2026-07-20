use super::GumbelMctsConfig;
use super::schedule::{
    GumbelRng, considered_actions, considered_visit_sequence, overlap_noise_scale, root_seed,
    sample_count_action, sample_root_gumbels,
};
use crate::mcts::math::masked_softmax;
use crate::mcts::strategy::{MctsStrategy, MctsStrategyState, StrategyRootResult};
use crate::mcts::tree::{MctsNode, MctsTree};
use crate::mcts::types::MctsSearchContext;

#[derive(Clone, Copy)]
pub(crate) struct GumbelStrategy {
    config: GumbelMctsConfig,
}

impl GumbelStrategy {
    pub(crate) const fn new(config: GumbelMctsConfig) -> Self {
        Self { config }
    }
}

pub(crate) struct GumbelRootState {
    base_scores: Vec<f32>,
    considered: Vec<usize>,
    baseline_visits: Vec<u32>,
    schedule: Vec<u32>,
    rng: GumbelRng,
}

impl MctsStrategyState for GumbelStrategy {
    type RootState = GumbelRootState;
}

impl<G, C> MctsStrategy<G, C> for GumbelStrategy
where
    G: Copy,
    C: Copy,
{
    fn start_root(&self, tree: &MctsTree<G, C>, context: MctsSearchContext) -> Self::RootState {
        let root = &tree.nodes[0];
        let mut rng = GumbelRng::new(root_seed(
            self.config.seed ^ context.noise_seed,
            context.root_step,
        ));
        let mut base_scores = masked_logits(root);
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
        let gumbels = sample_root_gumbels(root.action_count(), scale, &mut rng);
        for (score, gumbel) in base_scores.iter_mut().zip(gumbels) {
            *score += gumbel;
        }
        let considered = considered_actions(&base_scores, self.config.max_considered_actions.get());
        let schedule = considered_visit_sequence(considered.len(), self.config.simulations.get());

        GumbelRootState {
            base_scores,
            considered,
            baseline_visits: root.visits.clone(),
            schedule,
            rng,
        }
    }

    fn select_root(
        &self,
        state: &mut Self::RootState,
        tree: &MctsTree<G, C>,
        simulations: usize,
    ) -> Option<usize> {
        let target_visits = *state.schedule.get(simulations)?;
        let node = &tree.nodes[0];
        let scores = root_scores(node, &state.base_scores, self.config);
        state
            .considered
            .iter()
            .copied()
            .filter(|action| {
                !node.masked[*action]
                    && node.visits[*action].saturating_sub(state.baseline_visits[*action])
                        == target_visits
            })
            .max_by(|left, right| {
                scores[*left]
                    .total_cmp(&scores[*right])
                    .then_with(|| right.cmp(left))
            })
    }

    fn select_nonroot(
        &self,
        _state: &mut Self::RootState,
        tree: &MctsTree<G, C>,
        node_index: usize,
    ) -> usize {
        let node = &tree.nodes[node_index];
        let policy = improved_policy(node, self.config);
        let total_visits = node.visits.iter().copied().sum::<u32>() as f32;
        node.unmasked_actions()
            .max_by(|left, right| {
                let left_score = policy[*left] - node.visits[*left] as f32 / (1.0 + total_visits);
                let right_score =
                    policy[*right] - node.visits[*right] as f32 / (1.0 + total_visits);
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right.cmp(left))
            })
            .expect("STOP guarantees an unmasked action")
    }

    fn finish_root(
        &self,
        mut state: Self::RootState,
        tree: &MctsTree<G, C>,
        context: MctsSearchContext,
    ) -> StrategyRootResult {
        let node = &tree.nodes[0];
        let scores = root_scores(node, &state.base_scores, self.config);
        let mut selectable = state
            .considered
            .iter()
            .copied()
            .filter(|action| !node.masked[*action])
            .collect::<Vec<_>>();
        if selectable.is_empty() {
            selectable = node.unmasked_actions().collect();
        }

        let fallback = if self.config.tree_reuse {
            best_score_action(&scores, &selectable)
        } else {
            best_count_action(&node.visits, &selectable, &scores)
        };
        let selected = sample_count_action(
            &mut state.rng,
            &node.visits,
            &state.baseline_visits,
            &selectable,
            context.selection_temperature,
            fallback,
        );

        StrategyRootResult {
            selected,
            considered_action_indices: state.considered,
            policy_target: improved_policy(node, self.config),
            root_search_value: search_value(node),
            root_q_max: root_q_max(node),
        }
    }
}

fn masked_logits<G, C>(node: &MctsNode<G, C>) -> Vec<f32> {
    node.logits
        .iter()
        .zip(&node.masked)
        .map(
            |(logit, masked)| {
                if *masked { f32::NEG_INFINITY } else { *logit }
            },
        )
        .collect()
}

fn root_scores<G, C>(
    node: &MctsNode<G, C>,
    base_scores: &[f32],
    config: GumbelMctsConfig,
) -> Vec<f32> {
    let completed_q = completed_q(node);
    let max_visits = node.visits.iter().copied().max().unwrap_or(0) as f32;
    let scale = (config.c_visit + max_visits) * config.c_scale;
    base_scores
        .iter()
        .zip(completed_q)
        .zip(&node.masked)
        .map(|((score, q), masked)| {
            if *masked {
                f32::NEG_INFINITY
            } else {
                score + scale * q
            }
        })
        .collect()
}

fn improved_policy<G, C>(node: &MctsNode<G, C>, config: GumbelMctsConfig) -> Vec<f32> {
    let completed_q = completed_q(node);
    let max_visits = node.visits.iter().copied().max().unwrap_or(0) as f32;
    let scale = (config.c_visit + max_visits) * config.c_scale;
    let scores = node
        .logits
        .iter()
        .zip(completed_q)
        .map(|(logit, q)| logit + scale * q)
        .collect::<Vec<_>>();
    masked_softmax(&scores, &node.masked)
}

fn completed_q<G, C>(node: &MctsNode<G, C>) -> Vec<f32> {
    let mixed = mixed_value(node);
    node.visits
        .iter()
        .zip(&node.q)
        .map(|(visits, q)| if *visits > 0 { *q } else { mixed })
        .collect()
}

fn mixed_value<G, C>(node: &MctsNode<G, C>) -> f32 {
    let visits = node.visits.iter().copied().sum::<u32>();
    if visits == 0 {
        return node.value;
    }
    let mut prior_mass = 0.0;
    let mut weighted = 0.0;
    for ((visits, prior), q) in node.visits.iter().zip(&node.priors).zip(&node.q) {
        if *visits == 0 {
            continue;
        }
        prior_mass += prior;
        weighted += prior * q;
    }
    if prior_mass <= 0.0 {
        node.value
    } else {
        (node.value + visits as f32 * weighted / prior_mass) / (1.0 + visits as f32)
    }
}

fn search_value<G, C>(node: &MctsNode<G, C>) -> f32 {
    let mut visits = 0;
    let mut value = 0.0;
    for (count, q) in node.visits.iter().zip(&node.q) {
        if *count > 0 {
            visits += *count;
            value += *count as f32 * *q;
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
        .filter_map(|(visits, q)| (*visits > 0).then_some(*q))
        .reduce(f32::max)
        .unwrap_or(node.value)
}

fn best_score_action(scores: &[f32], considered: &[usize]) -> usize {
    considered
        .iter()
        .copied()
        .max_by(|left, right| {
            scores[*left]
                .total_cmp(&scores[*right])
                .then_with(|| right.cmp(left))
        })
        .expect("selectable actions is non-empty")
}

fn best_count_action(visits: &[u32], considered: &[usize], scores: &[f32]) -> usize {
    considered
        .iter()
        .copied()
        .max_by(|left, right| {
            visits[*left]
                .cmp(&visits[*right])
                .then_with(|| scores[*left].total_cmp(&scores[*right]))
                .then_with(|| right.cmp(left))
        })
        .expect("selectable actions is non-empty")
}
