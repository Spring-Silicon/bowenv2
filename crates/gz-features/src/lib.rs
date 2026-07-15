#![forbid(unsafe_code)]

//! Feature schema, row validation, and fixed-layout batch encoding.

mod codec;
mod collator;
mod error;
mod row;
mod schema;
mod wire;

pub use codec::{
    RowTargets, TrainingTargetsView, decode_feature_row, decode_feature_schema_config,
    encode_feature_row, encode_feature_schema_config, encode_training_targets,
    validate_feature_row_header,
};
pub use collator::{
    FeatureBatchView, FeatureCollator, OpponentBatchRef, RowOutput, decode_outputs,
    validate_batch_action_counts,
};
pub use error::{FeatureError, FeatureResult};
pub use row::{ActionFeature, FeatureEdge, FeatureRow, OpponentStateFeatures, PositionFeatures};
pub use schema::{
    BATCH_ENCODING_VERSION, ENCODING_VERSION, FeatureSchema, FeatureSchemaConfig,
    FeatureSchemaHash, STOP_ACTION_KIND_TOKEN,
};
pub use wire::{bf16_bits_to_f32, f32_to_bf16_bits};

use gz_engine::GraphEngine;

pub trait FeatureExtractor<E: GraphEngine>: Send {
    fn schema(&self) -> &FeatureSchema;

    fn extract(
        &mut self,
        engine: &E,
        graph: E::Graph,
        candidates: &[E::Candidate],
        position: PositionFeatures,
    ) -> FeatureResult<FeatureRow>;
}
