pub(super) use crate::mcts::math::{
    MctsRng as GumbelRng, budget_fraction, root_seed, sample_visit_action as sample_count_action,
    softmax,
};

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

/// The Gumbel scale at which a noisy root argmax lands in the prior's
/// top-m actions with probability `overlap + 0.05` (whittlezero's
/// gumbel_noise_overlap). argmax(logits + s*Gumbel) distributes as
/// softmax(logits/s), so the top-m mass is monotone decreasing in s and
/// an 18-step bisection over [1e-3, 64] pins the target. Masked actions
/// (-inf logits) are excluded; when m covers every legal action the base
/// scale is returned unchanged.
pub(super) fn overlap_noise_scale(
    logits: &[f32],
    considered: usize,
    overlap: f32,
    base_scale: f32,
) -> f32 {
    let mut legal: Vec<f32> = logits.iter().copied().filter(|l| l.is_finite()).collect();
    if legal.len() <= considered {
        return base_scale;
    }
    legal.sort_unstable_by(|a, b| b.total_cmp(a));

    let floor = considered as f32 / legal.len() as f32 + 1e-6;
    let target = (overlap + 0.05).clamp(floor, 0.999_999);
    let (mut lo, mut hi) = (1e-3_f32, 64.0_f32);
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        if top_mass(&legal, considered, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn top_mass(sorted_desc: &[f32], m: usize, scale: f32) -> f32 {
    let max = sorted_desc[0];
    let mut top = 0.0;
    let mut total = 0.0;
    for (index, logit) in sorted_desc.iter().enumerate() {
        let weight = ((logit - max) / scale).exp();
        total += weight;
        if index < m {
            top += weight;
        }
    }
    top / total
}

pub(super) fn sample_root_gumbels(count: usize, scale: f32, rng: &mut GumbelRng) -> Vec<f32> {
    if scale == 0.0 {
        return vec![0.0; count];
    }

    (0..count)
        .map(|_| scale * -(-rng.unit().ln()).ln())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{overlap_noise_scale, top_mass};

    #[test]
    fn overlap_scale_shrinks_for_flat_priors_and_grows_for_sharp_ones() {
        // Flat priors: top-m mass is m/n at every scale, below the target,
        // so the bisection settles at the minimum -- no noise can help.
        let flat = vec![0.0; 100];
        assert!(overlap_noise_scale(&flat, 8, 0.5, 1.0) < 0.01);

        // One dominant logit: the solution of (1 + 7x)/(1 + 99x) = 0.55
        // with x = exp(-20/s) gives s ~ 4.3.
        let mut sharp = vec![0.0; 100];
        sharp[0] = 20.0;
        let scale = overlap_noise_scale(&sharp, 8, 0.5, 1.0);
        assert!((2.0..8.0).contains(&scale), "scale {scale}");

        let mut sorted = sharp.clone();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let mass = top_mass(&sorted, 8, scale);
        assert!((mass - 0.55).abs() < 0.01, "mass {mass}");
    }

    #[test]
    fn overlap_scale_keeps_base_when_everything_is_considered() {
        assert_eq!(overlap_noise_scale(&[1.0, 2.0], 8, 0.5, 0.7), 0.7);
        // Masked (-inf) actions do not count toward the legal set.
        let mut masked = vec![f32::NEG_INFINITY; 10];
        masked[0] = 1.0;
        masked[1] = 0.0;
        assert_eq!(overlap_noise_scale(&masked, 8, 0.5, 0.7), 0.7);
    }
}
