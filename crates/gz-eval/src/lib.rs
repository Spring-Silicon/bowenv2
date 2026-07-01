#![forbid(unsafe_code)]

//! Policy/value evaluation boundary for GraphZero search.

mod error;
mod random;
mod types;

pub use error::{EvalError, EvalResult};
pub use random::{RandomValueEvaluator, RandomValueEvaluatorConfig};
pub use types::{
    EvalAction, EvalActionMetadata, EvalOutput, EvalRequest, Evaluator, validate_outputs,
};
