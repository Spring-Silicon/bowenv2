use gz_engine::ReplayGraphContext;
use std::fmt;

pub type EvalResult<T> = Result<T, EvalError>;

#[derive(Clone, Debug, PartialEq)]
pub enum EvalError {
    EmptyActions,
    MissingStop,
    DuplicateStop {
        first: usize,
        second: usize,
    },
    StopNotLast {
        index: usize,
        last: usize,
    },
    ActionContextMismatch {
        expected: Box<ReplayGraphContext>,
        actual: Box<ReplayGraphContext>,
    },
    ActionKindMismatch {
        action_index: usize,
    },
    NonFiniteStaticPrior {
        action_index: usize,
        static_prior: f32,
    },
    OutputCountMismatch {
        expected: usize,
        actual: usize,
    },
    PolicyLenMismatch {
        expected: usize,
        actual: usize,
    },
    NonFinitePolicyLogit {
        index: usize,
        value: f32,
    },
    NonFiniteValue {
        value: f32,
    },
    InvalidValueRange {
        value_min: f32,
        value_max: f32,
    },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyActions => f.write_str("eval request must contain at least one action"),
            Self::MissingStop => f.write_str("eval request must contain a STOP action"),
            Self::DuplicateStop { first, second } => {
                write!(
                    f,
                    "eval request has duplicate STOP actions at {first} and {second}"
                )
            }
            Self::StopNotLast { index, last } => {
                write!(f, "STOP action must be last, got index {index} of {last}")
            }
            Self::ActionContextMismatch { .. } => {
                f.write_str("eval action context does not match request context")
            }
            Self::ActionKindMismatch { action_index } => {
                write!(
                    f,
                    "eval action {action_index} metadata does not match action ref"
                )
            }
            Self::NonFiniteStaticPrior {
                action_index,
                static_prior,
            } => write!(
                f,
                "eval action {action_index} has non-finite static_prior {static_prior}"
            ),
            Self::OutputCountMismatch { expected, actual } => {
                write!(f, "expected {expected} eval outputs, got {actual}")
            }
            Self::PolicyLenMismatch { expected, actual } => {
                write!(f, "expected {expected} policy logits, got {actual}")
            }
            Self::NonFinitePolicyLogit { index, value } => {
                write!(f, "policy logit {index} is non-finite: {value}")
            }
            Self::NonFiniteValue { value } => {
                write!(f, "eval value is non-finite: {value}")
            }
            Self::InvalidValueRange {
                value_min,
                value_max,
            } => write!(
                f,
                "invalid random evaluator value range [{value_min}, {value_max}]"
            ),
        }
    }
}

impl std::error::Error for EvalError {}
