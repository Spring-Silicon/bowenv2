pub(crate) fn softmax(values: &[f32]) -> Vec<f32> {
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

pub(crate) fn masked_softmax(values: &[f32], masked: &[bool]) -> Vec<f32> {
    let scores = values
        .iter()
        .zip(masked)
        .map(
            |(value, masked)| {
                if *masked { f32::NEG_INFINITY } else { *value }
            },
        )
        .collect::<Vec<_>>();
    softmax(&scores)
}

pub(crate) fn sample_visit_action(
    rng: &mut MctsRng,
    visits: &[u32],
    baseline_visits: &[u32],
    allowed: &[usize],
    temperature: f32,
    fallback: usize,
) -> usize {
    if temperature <= 0.0 {
        return fallback;
    }

    let inv_temp = 1.0 / temperature;
    let mut total = 0.0;
    let mut weights = vec![0.0; visits.len()];
    for &action in allowed {
        let count = visits[action].saturating_sub(baseline_visits[action]);
        let weight = if count == 0 {
            0.0
        } else {
            (count as f32).powf(inv_temp)
        };
        total += weight;
        weights[action] = weight;
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

pub(crate) use crate::sampling::{SearchRng as MctsRng, root_seed};
pub(crate) use crate::support::budget_fraction;
