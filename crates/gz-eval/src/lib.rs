#![forbid(unsafe_code)]

//! Policy/value evaluation boundary for GraphZero search.

mod cache;
mod error;
mod random;
mod types;

pub use cache::NnEvalCache;
pub use error::{EvalError, EvalResult};
pub use random::{RandomValueEvaluator, RandomValueEvaluatorConfig};
pub use types::{
    EngineEvalRequest, EngineEvaluator, EvalAction, EvalActionMetadata, EvalOpponentContext,
    EvalOutput, EvalPositionContext, EvalRequest, Evaluator, eval_error_to_engine_error,
    validate_outputs,
};
