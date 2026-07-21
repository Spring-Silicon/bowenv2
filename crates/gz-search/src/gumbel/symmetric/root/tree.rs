use super::super::super::schedule::softmax;
use super::{Descent, PathStep, SymmetricSelfplayRootTask};
use std::hash::Hash;

impl<G, C> SymmetricSelfplayRootTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub(super) fn backup_descent(&mut self, descent: &Descent, mut value: f32) {
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

    pub(super) fn mask_action(&mut self, node: usize, action: usize) {
        self.nodes[node].masked[action] = true;
        self.nodes[node].priors[action] = 0.0;
    }

    pub(super) fn select_nonroot(&self, node_index: usize) -> Option<usize> {
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

    pub(super) fn completed_q(&self, node_index: usize) -> Vec<f32> {
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

    pub(super) fn improved_policy(&self, node_index: usize) -> Vec<f32> {
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

    pub(super) fn root_scores(&self, node_index: usize, base: &[f32]) -> Vec<f32> {
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
}
