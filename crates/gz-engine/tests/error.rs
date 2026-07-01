use gz_engine::{
    CandidateHash, EngineError, ErrorCode, ErrorMessage, ErrorMessageTooLong, GraphHash,
    OperationKind,
};

fn graph_hash(byte: u8) -> GraphHash {
    GraphHash::from_bytes([byte; 32])
}

fn candidate_hash(byte: u8) -> CandidateHash {
    CandidateHash::from_bytes([byte; 32])
}

fn message(value: &str) -> ErrorMessage {
    ErrorMessage::new(value).unwrap()
}

#[test]
fn error_message_rejects_overlong_messages() {
    let too_long = "x".repeat(ErrorMessage::MAX_LEN + 1);

    assert_eq!(
        ErrorMessage::new(too_long).unwrap_err(),
        ErrorMessageTooLong {
            max: ErrorMessage::MAX_LEN,
            actual: ErrorMessage::MAX_LEN + 1,
        }
    );
}

#[test]
fn engine_error_preserves_stale_candidate_context() {
    let error = EngineError::StaleCandidate {
        expected_graph_hash: graph_hash(1),
        actual_graph_hash: graph_hash(2),
        candidate_hash: candidate_hash(3),
    };

    assert_eq!(
        error,
        EngineError::StaleCandidate {
            expected_graph_hash: graph_hash(1),
            actual_graph_hash: graph_hash(2),
            candidate_hash: candidate_hash(3),
        }
    );
}

#[test]
fn display_includes_operation_and_code_where_relevant() {
    assert_eq!(
        EngineError::Timeout {
            operation: OperationKind::Measure,
            limit_ms: 500,
        }
        .to_string(),
        "measure timed out after 500 ms"
    );

    assert_eq!(
        EngineError::Internal {
            code: ErrorCode::new(9),
            message: message("bad state"),
        }
        .to_string(),
        "internal engine error: code 9: bad state"
    );
}

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;
    use serde_test::{Token, assert_tokens};

    #[test]
    fn serde_roundtrip_error_code_message_and_operation() {
        assert_tokens(&ErrorCode::new(7), &[Token::U32(7)]);
        assert_tokens(&message("bad input"), &[Token::Str("bad input")]);
        assert_tokens(
            &OperationKind::Apply,
            &[Token::UnitVariant {
                name: "OperationKind",
                variant: "Apply",
            }],
        );
    }

    #[test]
    fn serde_roundtrip_engine_error() {
        assert_tokens(
            &EngineError::Timeout {
                operation: OperationKind::Measure,
                limit_ms: 100,
            },
            &[
                Token::StructVariant {
                    name: "EngineError",
                    variant: "Timeout",
                    len: 2,
                },
                Token::Str("operation"),
                Token::UnitVariant {
                    name: "OperationKind",
                    variant: "Measure",
                },
                Token::Str("limit_ms"),
                Token::U64(100),
                Token::StructVariantEnd,
            ],
        );
    }
}
