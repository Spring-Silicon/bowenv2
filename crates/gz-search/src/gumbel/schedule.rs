use super::tree::Node;

pub fn considered_visit_sequence(max_considered: usize, simulations: usize) -> Vec<u32> {
    if max_considered <= 1 {
        return (0..simulations as u32).collect();
    }

    let log2max = (max_considered as f64).log2().ceil() as usize;
    let mut sequence = Vec::with_capacity(simulations);
    let mut visits = vec![0_u32; max_considered];
    let mut considered = max_considered;

    while sequence.len() < simulations {
        let extra = (simulations / (log2max * considered)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..considered]);
            for visit in &mut visits[..considered] {
                *visit += 1;
            }
        }
        considered = (considered / 2).max(2);
    }

    sequence.truncate(simulations);
    sequence
}

pub(super) fn considered_actions(base_scores: &[f32], max_considered: usize) -> Vec<usize> {
    let mut actions = (0..base_scores.len()).collect::<Vec<_>>();
    actions.sort_by(|&left, &right| {
        base_scores[right]
            .total_cmp(&base_scores[left])
            .then_with(|| left.cmp(&right))
    });
    actions.truncate(max_considered.min(actions.len()));
    actions
}

pub(super) fn best_eligible<G, C>(
    node: &Node<G, C>,
    considered: &[usize],
    target_visits: u32,
    scores: &[f32],
    tree_reuse: bool,
) -> Option<usize> {
    considered
        .iter()
        .copied()
        .filter(|&action| {
            node.logits[action].is_finite()
                && if tree_reuse {
                    node.visits[action] <= target_visits
                } else {
                    node.visits[action] == target_visits
                }
        })
        .max_by(|&left, &right| {
            scores[left]
                .total_cmp(&scores[right])
                .then_with(|| right.cmp(&left))
        })
}

pub(super) fn selectable_root_actions<G, C>(node: &Node<G, C>, considered: &[usize]) -> Vec<usize> {
    let mut actions = considered
        .iter()
        .copied()
        .filter(|&action| node.logits[action].is_finite())
        .collect::<Vec<_>>();

    if actions.is_empty() {
        actions.extend(
            node.logits
                .iter()
                .enumerate()
                .filter_map(|(action, logit)| logit.is_finite().then_some(action)),
        );
    }

    actions
}

pub(super) fn best_count_action<G, C>(
    node: &Node<G, C>,
    considered: &[usize],
    scores: &[f32],
) -> usize {
    considered
        .iter()
        .copied()
        .max_by(|&left, &right| {
            node.visits[left]
                .cmp(&node.visits[right])
                .then_with(|| scores[left].total_cmp(&scores[right]))
                .then_with(|| right.cmp(&left))
        })
        .expect("considered actions is non-empty")
}

pub(super) fn completed_q<G, C>(node: &Node<G, C>) -> Vec<f32> {
    let mixed = mixed_value(node);
    node.visits
        .iter()
        .zip(&node.q)
        .map(|(visits, q)| if *visits > 0 { *q } else { mixed })
        .collect()
}

pub(super) fn mixed_value<G, C>(node: &Node<G, C>) -> f32 {
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
        return node.value;
    }

    (node.value + visits as f32 * weighted / prior_mass) / (1.0 + visits as f32)
}

pub(super) fn search_value<G, C>(node: &Node<G, C>) -> f32 {
    let mut visits = 0;
    let mut value = 0.0;

    for (count, q) in node.visits.iter().zip(&node.q) {
        if *count == 0 {
            continue;
        }
        visits += *count;
        value += *count as f32 * *q;
    }

    if visits == 0 {
        node.value
    } else {
        value / visits as f32
    }
}

pub(super) fn root_q_max<G, C>(node: &Node<G, C>) -> f32 {
    node.visits
        .iter()
        .zip(&node.q)
        .filter_map(|(visits, q)| (*visits > 0).then_some(*q))
        .reduce(f32::max)
        .unwrap_or(node.value)
}

pub(super) fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f32::max);
    let Some(max) = max else {
        return vec![1.0 / values.len() as f32; values.len()];
    };

    let mut out = Vec::with_capacity(values.len());
    let mut total = 0.0;

    for value in values {
        let next = if value.is_finite() {
            (*value - max).exp()
        } else {
            0.0
        };
        total += next;
        out.push(next);
    }

    if total <= 0.0 || !total.is_finite() {
        let legal = values.iter().filter(|value| value.is_finite()).count();
        let uniform = 1.0 / legal.max(1) as f32;
        for (out, value) in out.iter_mut().zip(values) {
            *out = if value.is_finite() { uniform } else { 0.0 };
        }
        return out;
    }

    for value in &mut out {
        *value /= total;
    }

    out
}

pub(super) fn sample_root_gumbels(count: usize, scale: f32, rng: &mut GumbelRng) -> Vec<f32> {
    if scale == 0.0 {
        return vec![0.0; count];
    }

    (0..count)
        .map(|_| scale * -(-rng.unit().ln()).ln())
        .collect()
}

pub(super) fn sample_count_action(
    rng: &mut GumbelRng,
    visits: &[u32],
    temperature: f32,
    fallback: usize,
) -> usize {
    if temperature <= 0.0 {
        return fallback;
    }

    let inv_temp = 1.0 / temperature;
    let mut total = 0.0;
    let mut weights = Vec::with_capacity(visits.len());

    for visits in visits {
        let weight = if *visits == 0 {
            0.0
        } else {
            (*visits as f32).powf(inv_temp)
        };
        total += weight;
        weights.push(weight);
    }

    if total <= 0.0 || !total.is_finite() {
        return fallback;
    }

    let mut threshold = rng.unit() * total;
    for (index, weight) in weights.into_iter().enumerate() {
        if threshold <= weight {
            return index;
        }
        threshold -= weight;
    }

    fallback
}

pub(super) fn budget_fraction(max_steps: usize, step: usize) -> f32 {
    if max_steps == 0 {
        1.0
    } else {
        max_steps.saturating_sub(step) as f32 / max_steps as f32
    }
}

pub(super) fn root_seed(seed: u64, root_step: u32) -> u64 {
    seed ^ 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(u64::from(root_step) + 1)
}

pub(super) struct GumbelRng {
    state: u64,
}

impl GumbelRng {
    const STEP: u64 = 0x9e37_79b9_7f4a_7c15;

    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn unit(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        let unit = (value as f32 + 0.5) / (1_u32 << 24) as f32;
        unit.clamp(1.0e-7, 1.0 - 1.0e-7)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::STEP);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
