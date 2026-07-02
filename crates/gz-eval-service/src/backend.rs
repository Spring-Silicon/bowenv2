use crate::{STUB_MODEL_VERSION, ServiceError, ServiceResult, stub_row_outputs};
use gz_engine::ModelVersion;
use gz_features::{FeatureBatchView, RowOutput};

pub trait FeatureEvalBackend {
    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackendOutputs {
    pub model_version: ModelVersion,
    pub rows: Vec<RowOutput>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StubBackend;

impl FeatureEvalBackend for StubBackend {
    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        let view = FeatureBatchView::parse(batch_bytes)
            .map_err(|error| ServiceError::protocol(error.to_string()))?;
        validate_action_counts(&view, action_counts)?;
        Ok(BackendOutputs {
            model_version: STUB_MODEL_VERSION,
            rows: stub_row_outputs(&view),
        })
    }
}

pub(crate) fn validate_action_counts(
    view: &FeatureBatchView,
    action_counts: &[u32],
) -> ServiceResult<()> {
    let row_count = view.row_count as usize;
    if action_counts.len() != row_count {
        return Err(ServiceError::protocol("action count length mismatch"));
    }
    for (index, (&expected, &actual)) in action_counts
        .iter()
        .zip(view.action_count.iter())
        .enumerate()
    {
        if expected != actual {
            return Err(ServiceError::protocol(format!(
                "action count mismatch at row {index}"
            )));
        }
        if expected > view.max_actions {
            return Err(ServiceError::protocol("action count exceeds max_actions"));
        }
    }
    Ok(())
}
