#![forbid(unsafe_code)]

//! Whittle-specific evaluators for GraphZero search.

use gz_engine::{EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine, ModelVersion};
use gz_engine_whittle::{WhittleEngine, WhittleGraphId};
use gz_eval::{EngineEvalRequest, EngineEvaluator, EvalOutput, eval_error_to_engine_error};

#[derive(Clone, Copy, Debug, Default)]
pub struct WhittleMeasureEvaluator;

impl WhittleMeasureEvaluator {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    #[must_use]
    pub const fn model_version(self) -> ModelVersion {
        ModelVersion::from_bytes(*b"gz-whittle-meas1")
    }
}

impl EngineEvaluator<WhittleEngine> for WhittleMeasureEvaluator {
    fn evaluate(
        &mut self,
        engine: &mut WhittleEngine,
        input: EngineEvalRequest<'_, WhittleEngine>,
    ) -> EngineResult<EvalOutput> {
        input
            .request
            .validate_ref()
            .map_err(eval_error_to_engine_error)?;

        if input.request.action_count() != input.candidates.len() + 1 {
            return Err(internal("candidate/action count mismatch"));
        }

        let before = measure_reward(engine, input.graph, input.measure_options)?;
        let mut policy_logits = Vec::with_capacity(input.request.action_count());
        let mut created_graphs = Vec::new();

        let result = (|| {
            for candidate in input.candidates.iter().copied() {
                let applied = engine.apply(input.graph, candidate)?;
                created_graphs.push(applied.after);
                let after = measure_reward(engine, applied.after, input.measure_options)?;
                policy_logits.push(logit_for_delta(before, after));
            }

            policy_logits.push(0.5);

            Ok(EvalOutput {
                model_version: self.model_version(),
                policy_logits,
                value: before,
            })
        })();

        let release = engine.release(&created_graphs, &[]);
        match result {
            Ok(output) => {
                release?;
                Ok(output)
            }
            Err(error) => {
                let _ = release;
                Err(error)
            }
        }
    }
}

fn measure_reward(
    engine: &mut WhittleEngine,
    graph: WhittleGraphId,
    options: gz_engine::MeasureOptions,
) -> EngineResult<f32> {
    let reward = engine
        .measure(graph, options)?
        .scalar_reward
        .ok_or_else(|| internal("Whittle measure returned no scalar reward"))?;

    if reward.is_finite() {
        Ok(reward)
    } else {
        Err(internal("Whittle measure returned non-finite reward"))
    }
}

fn logit_for_delta(before: f32, after: f32) -> f32 {
    if after > before {
        1.0
    } else if after < before {
        0.0
    } else {
        0.5
    }
}

fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(2),
        message: ErrorMessage::new(message).expect("static error message is valid"),
    }
}
