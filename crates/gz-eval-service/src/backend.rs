use crate::{STUB_MODEL_VERSION, ServiceError, ServiceResult, stub_row_outputs};
use gz_engine::ModelVersion;
use gz_features::{FeatureBatchView, RowOutput};
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelGeneration {
    /// Nonzero evaluator-connection-local generation identifier.
    pub id: u64,
    /// Immutable identity of the model weights in this generation.
    pub version: ModelVersion,
}

impl ModelGeneration {
    #[must_use]
    pub const fn initial(version: ModelVersion) -> Self {
        Self { id: 1, version }
    }
}

pub trait FeatureEvalBackend {
    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs>;

    /// Generation that a newly admitted episode should lease.
    fn model_generation(&self) -> ModelGeneration;

    /// Fixed batch capacity negotiated with a remote backend. In-process
    /// backends return None and use the orchestrator's configured capacity.
    fn batch_capacity(&self) -> Option<NonZeroUsize> {
        None
    }

    /// Evaluator-capacity work represented by a completed request. In-process
    /// backends scale with the real row count by default; fixed-capacity remote
    /// backends override this because a partial request occupies a full batch.
    fn capacity_work(&self, actual_rows: usize, _max_batch: usize) -> usize {
        actual_rows
    }

    /// Submits a batch without waiting for its outputs; pair with
    /// `receive`. Backends that cannot overlap compute simply evaluate
    /// here (the default), so callers may pipeline unconditionally.
    /// Submitted batches must be received in FIFO order.
    fn submit(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<PendingBatch> {
        Ok(PendingBatch::Ready(self.eval(batch_bytes, action_counts)?))
    }

    /// Submits against an exact leased generation. A backend must fail rather
    /// than silently serve another version when the generation is unavailable.
    fn submit_for_model(
        &mut self,
        model: ModelGeneration,
        batch_bytes: &[u8],
        action_counts: &[u32],
    ) -> ServiceResult<PendingBatch> {
        let outputs = self.eval(batch_bytes, action_counts)?;
        if outputs.model_version != model.version {
            return Err(ServiceError::backend(
                1,
                "evaluator served the wrong model version",
            ));
        }
        Ok(PendingBatch::Ready(outputs))
    }

    /// Releases a non-active generation after its final episode and in-flight
    /// request are gone. Static backends need no release work.
    fn release_model_generation(&mut self, _model: ModelGeneration) -> ServiceResult<()> {
        Ok(())
    }

    fn receive(&mut self, pending: PendingBatch) -> ServiceResult<BackendOutputs> {
        match pending {
            PendingBatch::Ready(outputs) => Ok(outputs),
            PendingBatch::InFlight { .. } => Err(ServiceError::protocol(
                "backend cannot receive in-flight batches",
            )),
        }
    }
}

/// A submitted batch awaiting `receive`. `Ready` is the non-pipelining
/// default (outputs computed at submit); `InFlight` is a batch on the
/// wire of a pipelining backend.
#[derive(Clone, Debug, PartialEq)]
pub enum PendingBatch {
    Ready(BackendOutputs),
    InFlight {
        batch_id: u64,
        action_counts: Vec<u32>,
        model: ModelGeneration,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackendOutputs {
    pub model_version: ModelVersion,
    pub active_generation: ModelGeneration,
    pub rows: Vec<RowOutput>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StubBackend;

impl FeatureEvalBackend for StubBackend {
    fn model_generation(&self) -> ModelGeneration {
        ModelGeneration::initial(STUB_MODEL_VERSION)
    }

    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        let view = FeatureBatchView::parse(batch_bytes)
            .map_err(|error| ServiceError::protocol(error.to_string()))?;
        validate_action_counts(&view, action_counts)?;
        Ok(BackendOutputs {
            model_version: STUB_MODEL_VERSION,
            active_generation: self.model_generation(),
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
