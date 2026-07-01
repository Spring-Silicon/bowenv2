use gz_engine::{
    ActionSetHash, CandidateHash, CandidateKindId, CandidateTags, EngineId, EngineVersion,
    GraphHash, PortableCandidateRef, PortableGraphId, ReplayGraphContext,
};
use gz_eval::{EvalAction, EvalActionMetadata, EvalError, EvalOutput, EvalRequest};

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

fn candidate(context: ReplayGraphContext, byte: u8) -> EvalAction {
    EvalAction::candidate(
        PortableCandidateRef::new(context, CandidateHash::from_bytes([byte; 32])),
        CandidateKindId::new(byte.into()),
        CandidateTags::EMPTY,
        0.0,
    )
}

#[test]
fn request_validation_accepts_candidates_followed_by_stop() {
    let context = context(1);
    let request = EvalRequest::new(
        context,
        vec![
            candidate(context, 10),
            candidate(context, 11),
            EvalAction::stop(context),
        ],
    )
    .unwrap();

    assert_eq!(request.action_count(), 3);
}

#[test]
fn request_validation_rejects_empty_actions() {
    let error = EvalRequest::new(context(1), Vec::new()).unwrap_err();

    assert_eq!(error, EvalError::EmptyActions);
}

#[test]
fn request_validation_rejects_missing_stop() {
    let context = context(1);
    let error = EvalRequest::new(context, vec![candidate(context, 10)]).unwrap_err();

    assert_eq!(error, EvalError::MissingStop);
}

#[test]
fn request_validation_rejects_duplicate_stop() {
    let context = context(1);
    let error = EvalRequest::new(
        context,
        vec![EvalAction::stop(context), EvalAction::stop(context)],
    )
    .unwrap_err();

    assert_eq!(
        error,
        EvalError::DuplicateStop {
            first: 0,
            second: 1
        }
    );
}

#[test]
fn request_validation_rejects_stop_before_last_action() {
    let context = context(1);
    let error = EvalRequest::new(
        context,
        vec![EvalAction::stop(context), candidate(context, 10)],
    )
    .unwrap_err();

    assert_eq!(error, EvalError::StopNotLast { index: 0, last: 1 });
}

#[test]
fn request_validation_rejects_action_context_mismatch() {
    let expected = context(1);
    let actual = context(2);
    let error = EvalRequest::new(
        expected,
        vec![candidate(actual, 10), EvalAction::stop(expected)],
    )
    .unwrap_err();

    assert_eq!(
        error,
        EvalError::ActionContextMismatch {
            expected: Box::new(expected),
            actual: Box::new(actual)
        }
    );
}

#[test]
fn request_validation_rejects_action_kind_mismatch() {
    let context = context(1);
    let error = EvalRequest::new(
        context,
        vec![
            EvalAction {
                action_ref: candidate(context, 10).action_ref,
                metadata: EvalActionMetadata::Stop,
            },
            EvalAction::stop(context),
        ],
    )
    .unwrap_err();

    assert_eq!(error, EvalError::ActionKindMismatch { action_index: 0 });
}

#[test]
fn request_validation_rejects_non_finite_static_prior() {
    let context = context(1);
    let error = EvalRequest::new(
        context,
        vec![
            EvalAction::candidate(
                PortableCandidateRef::new(context, CandidateHash::from_bytes([10; 32])),
                CandidateKindId::new(10),
                CandidateTags::EMPTY,
                f32::NAN,
            ),
            EvalAction::stop(context),
        ],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        EvalError::NonFiniteStaticPrior {
            action_index: 0,
            ..
        }
    ));
}

#[test]
fn output_validation_rejects_wrong_policy_length() {
    let context = context(1);
    let request = EvalRequest::new(context, vec![EvalAction::stop(context)]).unwrap();
    let output = EvalOutput {
        model_version: gz_engine::ModelVersion::from_bytes([1; 16]),
        policy_logits: vec![0.0, 0.0],
        value: 0.0,
    };

    assert_eq!(
        output.validate_for(&request).unwrap_err(),
        EvalError::PolicyLenMismatch {
            expected: 1,
            actual: 2
        }
    );
}

#[test]
fn output_validation_rejects_non_finite_policy_logit() {
    let context = context(1);
    let request = EvalRequest::new(context, vec![EvalAction::stop(context)]).unwrap();
    let output = EvalOutput {
        model_version: gz_engine::ModelVersion::from_bytes([1; 16]),
        policy_logits: vec![f32::INFINITY],
        value: 0.0,
    };

    assert!(matches!(
        output.validate_for(&request).unwrap_err(),
        EvalError::NonFinitePolicyLogit { index: 0, .. }
    ));
}

#[test]
fn output_validation_rejects_non_finite_value() {
    let context = context(1);
    let request = EvalRequest::new(context, vec![EvalAction::stop(context)]).unwrap();
    let output = EvalOutput {
        model_version: gz_engine::ModelVersion::from_bytes([1; 16]),
        policy_logits: vec![0.0],
        value: f32::NEG_INFINITY,
    };

    assert!(matches!(
        output.validate_for(&request).unwrap_err(),
        EvalError::NonFiniteValue { .. }
    ));
}
