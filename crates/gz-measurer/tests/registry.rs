use gz_engine::{ModelVersion, ReplayGraphContext, SearchConfigHash};
use gz_measurer::{
    ArenaGateRegistry, PolicyModel, ReferenceRegistry, ReferenceStep, RolloutOutcome,
};

#[test]
fn registry_claims_each_version_once_and_retries_unmeasured() {
    let registry = ReferenceRegistry::new();
    let version = ModelVersion::from_bytes([1; 16]);

    assert!(registry.claim_challenge(Some(version)));
    assert!(!registry.claim_challenge(Some(version)));

    assert_eq!(registry.finish_challenge(None), None);
    assert!(registry.claim_challenge(Some(version)));

    let accepted = registry
        .finish_challenge(Some(outcome(version, 2.0)))
        .expect("measured challenge emits gate event");
    assert!(accepted.accepted);
    assert_eq!(accepted.version, version);
    assert_eq!(registry.current().unwrap().ref_id, 1);

    assert!(!registry.claim_challenge(Some(version)));
    let next = ModelVersion::from_bytes([2; 16]);
    assert!(registry.claim_challenge(Some(next)));
    let rejected = registry
        .finish_challenge(Some(outcome(next, 1.0)))
        .expect("measured rejection emits gate event");
    assert!(!rejected.accepted);
    assert_eq!(registry.current().unwrap().version, version);
    assert!(!registry.claim_challenge(Some(version)));
    assert!(!registry.claim_challenge(Some(next)));
}

#[test]
fn gamma_samples_latest_rejected_challenger_without_replacing_best() {
    let registry = gz_measurer::ReferenceRegistry::with_gamma(0.999_999, 17);
    let first = ModelVersion::from_bytes([1; 16]);
    let second = ModelVersion::from_bytes([2; 16]);

    assert!(registry.claim_challenge(Some(first)));
    registry.finish_challenge(Some(outcome(first, 2.0)));
    assert!(registry.claim_challenge(Some(second)));
    registry.finish_challenge(Some(outcome(second, 1.0)));

    let best = registry.current().expect("accepted challenger is best");
    let latest = registry.latest().expect("measured rejection is latest");
    assert_eq!(best.ref_id, 1);
    assert_eq!(best.version, first);
    assert_eq!(latest.ref_id, 2);
    assert_eq!(latest.version, second);
    assert_eq!(registry.sampled().unwrap().ref_id, latest.ref_id);
}

#[test]
fn gamma_zero_is_best_only() {
    let first = ModelVersion::from_bytes([1; 16]);
    let second = ModelVersion::from_bytes([2; 16]);
    let registry = gz_measurer::ReferenceRegistry::with_gamma(0.0, 99);

    assert!(registry.claim_challenge(Some(first)));
    registry.finish_challenge(Some(outcome(first, 2.0)));
    assert!(registry.claim_challenge(Some(second)));
    registry.finish_challenge(Some(outcome(second, 1.0)));

    for _ in 0..32 {
        assert_eq!(registry.sampled().unwrap().ref_id, 1);
    }
    assert_eq!(registry.current().unwrap().ref_id, 1);
    assert_eq!(registry.latest().unwrap().ref_id, 2);
}

#[test]
fn gamma_seeded_sampling_is_repeatable_and_near_configured_rate() {
    let first = ModelVersion::from_bytes([1; 16]);
    let second = ModelVersion::from_bytes([2; 16]);
    let left = gz_measurer::ReferenceRegistry::with_gamma(0.2, 99);
    let right = gz_measurer::ReferenceRegistry::with_gamma(0.2, 99);

    for registry in [&left, &right] {
        assert!(registry.claim_challenge(Some(first)));
        registry.finish_challenge(Some(outcome(first, 2.0)));
        assert!(registry.claim_challenge(Some(second)));
        registry.finish_challenge(Some(outcome(second, 1.0)));
    }

    let left_picks = (0..1_000)
        .map(|_| left.sampled().unwrap().ref_id)
        .collect::<Vec<_>>();
    let right_picks = (0..1_000)
        .map(|_| right.sampled().unwrap().ref_id)
        .collect::<Vec<_>>();
    assert_eq!(left_picks, right_picks);
    let latest_picks = left_picks.iter().filter(|&&ref_id| ref_id == 2).count();
    assert!(
        (150..=250).contains(&latest_picks),
        "latest picked {latest_picks}/1000"
    );
}

#[test]
fn accepted_best_keeps_its_snapshot_identity_after_rejections() {
    let registry = gz_measurer::ReferenceRegistry::with_gamma(0.999_999, 3);
    let first = ModelVersion::from_bytes([1; 16]);
    let rejected = ModelVersion::from_bytes([2; 16]);
    let accepted = ModelVersion::from_bytes([3; 16]);

    assert!(registry.claim_challenge(Some(first)));
    registry.finish_challenge(Some(outcome(first, 1.0)));
    assert!(registry.claim_challenge(Some(rejected)));
    registry.finish_challenge(Some(outcome(rejected, 0.0)));
    assert!(registry.claim_challenge(Some(accepted)));
    registry.finish_challenge(Some(outcome(accepted, 2.0)));

    assert_eq!(registry.current().unwrap().ref_id, 3);
    assert_eq!(registry.latest().unwrap().ref_id, 3);
}

#[test]
fn trajectory_pool_claims_are_bounded_and_sampled_per_reference() {
    let version = ModelVersion::from_bytes([1; 16]);
    let registry = ReferenceRegistry::with_gamma_and_trajectory_pool(0.0, 29, 2);
    assert!(registry.claim_challenge(Some(version)));
    registry.finish_challenge(Some(outcome(version, 3.0)));
    assert!(!registry.admission_ready());

    assert_eq!(registry.claim_sample(Some(version)), Some(version));
    assert_eq!(registry.claim_sample(Some(version)), Some(version));
    assert_eq!(registry.claim_sample(Some(version)), None);
    assert!(registry.finish_sample(version, Some(outcome(version, 2.0))));
    assert!(registry.finish_sample(version, Some(outcome(version, 1.0))));
    assert_eq!(registry.trajectory_pool_len(), 2);
    assert!(registry.admission_ready());
    assert_eq!(registry.current().unwrap().ref_id, 1);

    let picks = (0..64)
        .map(|_| registry.sampled().unwrap().ref_id)
        .collect::<Vec<_>>();
    assert!(picks.iter().all(|ref_id| matches!(ref_id, 2 | 3)));
    assert!(picks.contains(&2));
    assert!(picks.contains(&3));
}

#[test]
fn trajectory_pool_retries_failed_or_wrong_version_samples() {
    let version = ModelVersion::from_bytes([1; 16]);
    let other = ModelVersion::from_bytes([2; 16]);
    let registry = ReferenceRegistry::with_gamma_and_trajectory_pool(0.0, 7, 1);
    assert!(registry.claim_challenge(Some(version)));
    registry.finish_challenge(Some(outcome(version, 3.0)));

    assert_eq!(registry.claim_sample(Some(version)), Some(version));
    assert!(!registry.finish_sample(version, Some(outcome(other, 2.0))));
    assert_eq!(registry.trajectory_pool_len(), 0);
    assert_eq!(registry.claim_sample(Some(version)), Some(version));
    assert!(!registry.finish_sample(version, None));
    assert_eq!(registry.claim_sample(Some(version)), Some(version));
}

#[test]
fn trajectory_pool_refills_incumbent_after_current_model_advances() {
    let incumbent = ModelVersion::from_bytes([1; 16]);
    let current = ModelVersion::from_bytes([2; 16]);
    let registry = ReferenceRegistry::with_gamma_and_trajectory_pool(0.0, 7, 1);
    assert!(registry.claim_challenge(Some(incumbent)));
    registry.finish_challenge(Some(outcome(incumbent, 3.0)));

    assert_eq!(registry.claim_sample(Some(current)), Some(incumbent));
    assert!(registry.finish_sample(incumbent, Some(outcome(incumbent, 2.0))));
    assert!(registry.admission_ready());
}

#[test]
fn versioned_challenge_retries_when_rollout_used_another_checkpoint() {
    let expected = ModelVersion::from_bytes([1; 16]);
    let actual = ModelVersion::from_bytes([2; 16]);
    let registry = ReferenceRegistry::new();

    assert!(registry.claim_challenge(Some(expected)));
    assert_eq!(registry.finish_challenge(Some(outcome(actual, 1.0))), None);
    assert!(registry.claim_challenge(Some(expected)));
    assert!(registry.current().is_none());
}

#[test]
fn arena_gate_partitions_fixed_roots_and_uses_strict_summed_margin() {
    let best = ModelVersion::from_bytes([1; 16]);
    let rejected = ModelVersion::from_bytes([2; 16]);
    let accepted = ModelVersion::from_bytes([3; 16]);
    let registry = ArenaGateRegistry::new(4, 0.0, 9);
    assert!(registry.initialize(best, best, best));

    let lane_zero = registry.claim_arena(0, 2).unwrap();
    let lane_one = registry.claim_arena(1, 2).unwrap();
    assert_eq!(
        (lane_zero.index, lane_zero.model),
        (0, PolicyModel::Incumbent)
    );
    assert_eq!(
        (lane_one.index, lane_one.model),
        (1, PolicyModel::Incumbent)
    );
    registry.finish_arena(lane_zero, Some(best), Some(0.1), 3);
    registry.finish_arena(lane_one, Some(best), Some(0.2), 4);
    for expected in [2, 3] {
        let claim = registry.claim_arena(expected % 2, 2).unwrap();
        assert_eq!(claim.index, expected);
        registry.finish_arena(claim, Some(best), Some(0.3 + expected as f32 / 10.0), 5);
    }
    assert!(registry.admission_ready());

    registry.observe_current(rejected);
    registry.observe_challenger(rejected);
    let mut event = None;
    for score in [0.2, 0.1, 0.5, 0.6] {
        let claim = registry.claim_arena(0, 1).unwrap();
        assert_eq!(claim.model, PolicyModel::Challenger);
        event = registry
            .finish_arena(claim, Some(rejected), Some(score), 6)
            .or(event);
    }
    let event = event.unwrap();
    assert!(!event.accepted);
    assert_eq!(event.margin_sum, 0.0);
    assert_eq!(registry.incumbent_version(), Some(best));

    registry.observe_current(accepted);
    registry.observe_challenger(accepted);
    let mut event = None;
    for score in [0.2, 0.3, 0.5, 0.6] {
        let claim = registry.claim_arena(0, 1).unwrap();
        event = registry
            .finish_arena(claim, Some(accepted), Some(score), 7)
            .or(event);
    }
    let event = event.unwrap();
    assert!(event.accepted);
    assert!(event.margin_sum > 0.0);
    assert_eq!(event.steps, 28);
    assert_eq!(registry.incumbent_version(), Some(accepted));
}

#[test]
fn arena_gate_preserves_active_challenger_and_queues_latest() {
    let best = ModelVersion::from_bytes([1; 16]);
    let stale = ModelVersion::from_bytes([2; 16]);
    let skipped = ModelVersion::from_bytes([3; 16]);
    let latest = ModelVersion::from_bytes([4; 16]);
    let registry = ArenaGateRegistry::new(1, 0.0, 4);
    assert!(registry.initialize(best, stale, stale));
    assert!(registry.admission_ready());
    assert_eq!(registry.claim_episode().unwrap().version, best);

    let baseline = registry.claim_arena(0, 1).unwrap();
    registry.finish_arena(baseline, Some(stale), Some(1.0), 1);
    assert!(registry.admission_ready());
    let baseline = registry.claim_arena(0, 1).unwrap();
    registry.finish_arena(baseline, Some(best), Some(1.0), 1);
    assert!(registry.admission_ready());

    let stale_claim = registry.claim_arena(0, 1).unwrap();
    assert_eq!(stale_claim.version, stale);
    registry.observe_current(skipped);
    registry.observe_challenger(skipped);
    registry.observe_current(latest);
    registry.observe_challenger(latest);
    let event = registry
        .finish_arena(stale_claim, Some(stale), Some(2.0), 1)
        .unwrap();
    assert_eq!(event.version, stale);
    assert!(event.accepted);
    let latest_claim = registry.claim_arena(0, 1).unwrap();
    assert_eq!(latest_claim.version, latest);
    assert_eq!(latest_claim.model, PolicyModel::Challenger);
}

#[test]
fn arena_episode_claims_mix_current_without_changing_incumbent() {
    let best = ModelVersion::from_bytes([1; 16]);
    let current = ModelVersion::from_bytes([2; 16]);
    let registry = ArenaGateRegistry::new(1, 0.999_999, 17);
    assert!(registry.initialize(best, current, current));
    let baseline = registry.claim_arena(0, 1).unwrap();
    registry.finish_arena(baseline, Some(best), Some(1.0), 1);

    for _ in 0..32 {
        let claim = registry.claim_episode().unwrap();
        assert_eq!(claim.model, PolicyModel::Current);
        assert_eq!(claim.version, current);
    }
    assert_eq!(registry.incumbent_version(), Some(best));
}

fn outcome(version: ModelVersion, final_reward: f32) -> RolloutOutcome {
    RolloutOutcome {
        final_reward,
        final_graph: context(7),
        steps: vec![ReferenceStep {
            context: context(1),
            features: None,
        }],
        search_config_hash: SearchConfigHash::from_bytes([9; 32]),
        model_version: Some(version),
    }
}

fn context(seed: u8) -> ReplayGraphContext {
    ReplayGraphContext::new(
        gz_engine::PortableGraphId::new(
            gz_engine::GraphHash::from_bytes([seed; 32]),
            gz_engine::EngineId::from_bytes([1; 16]),
            gz_engine::EngineVersion::from_bytes([2; 16]),
        ),
        gz_engine::ActionSetHash::from_bytes([3; 32]),
    )
}
