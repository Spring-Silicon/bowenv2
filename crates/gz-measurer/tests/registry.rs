use gz_engine::{ModelVersion, ReplayGraphContext, SearchConfigHash};
use gz_measurer::{ReferenceRegistry, ReferenceStep, RolloutOutcome};

#[test]
fn registry_claims_one_challenge_per_version_and_retries_unmeasured() {
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
