use gz_engine::{
    ActionSetHash, CandidateHash, CandidateKindId, CandidateTags, EngineId, EngineVersion,
    GraphHash, PortableCandidateRef, PortableGraphId, ReplayGraphContext,
};
use gz_eval::{
    EvalAction, EvalError, EvalRequest, Evaluator, RandomValueEvaluator, RandomValueEvaluatorConfig,
};

fn context(byte: u8) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(
            GraphHash::from_bytes([byte; 32]),
            EngineId::from_bytes([1; 16]),
            EngineVersion::from_bytes([2; 16]),
        ),
        ActionSetHash::from_bytes([3; 32]),
    )
}

fn request(byte: u8) -> EvalRequest {
    let context = context(byte);
    EvalRequest::new(
        context,
        vec![
            EvalAction::candidate(
                PortableCandidateRef::new(context, CandidateHash::from_bytes([byte; 32])),
                CandidateKindId::new(byte.into()),
                CandidateTags::EMPTY,
                0.0,
            ),
            EvalAction::stop(context),
        ],
    )
    .unwrap()
}

fn evaluator(seed: u64) -> RandomValueEvaluator {
    RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed,
        value_min: -2.0,
        value_max: 3.0,
    })
    .unwrap()
}

#[test]
fn random_evaluator_returns_one_output_per_request() {
    let requests = vec![request(1), request(2)];
    let mut outputs = Vec::new();

    evaluator(7)
        .evaluate_batch(&requests, &mut outputs)
        .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].policy_logits.len(), 2);
    assert_eq!(outputs[1].policy_logits.len(), 2);
}

#[test]
fn random_evaluator_policy_logits_are_uniform_zeroes() {
    let output = evaluator(7).evaluate_one(&request(1)).unwrap();

    assert_eq!(output.policy_logits, vec![0.0, 0.0]);
}

#[test]
fn random_evaluator_value_is_within_configured_range() {
    let output = evaluator(7).evaluate_one(&request(1)).unwrap();

    assert!((-2.0..=3.0).contains(&output.value));
}

#[test]
fn random_evaluator_is_deterministic_for_the_same_graph_and_config() {
    let request = request(1);
    let left = evaluator(7).evaluate_one(&request).unwrap();
    let right = evaluator(7).evaluate_one(&request).unwrap();

    assert_eq!(left, right);
}

#[test]
fn random_evaluator_output_is_independent_of_batch_order() {
    let request_a = request(1);
    let request_b = request(2);

    let mut outputs = Vec::new();
    evaluator(7)
        .evaluate_batch(&[request_a.clone(), request_b.clone()], &mut outputs)
        .unwrap();
    let first_a = outputs[0].clone();
    let first_b = outputs[1].clone();

    evaluator(7)
        .evaluate_batch(&[request_b, request_a], &mut outputs)
        .unwrap();

    assert_eq!(outputs[0], first_b);
    assert_eq!(outputs[1], first_a);
}

#[test]
fn random_evaluator_model_version_changes_when_config_changes() {
    let base = evaluator(7).model_version();
    let seed = evaluator(8).model_version();
    let range = RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: 7,
        value_min: -2.0,
        value_max: 4.0,
    })
    .unwrap()
    .model_version();

    assert_ne!(base, seed);
    assert_ne!(base, range);
}

#[test]
fn random_evaluator_rejects_invalid_value_range() {
    let error = RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: 0,
        value_min: 1.0,
        value_max: -1.0,
    })
    .unwrap_err();

    assert_eq!(
        error,
        EvalError::InvalidValueRange {
            value_min: 1.0,
            value_max: -1.0
        }
    );
}
