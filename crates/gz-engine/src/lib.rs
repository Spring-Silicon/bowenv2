#![forbid(unsafe_code)]

//! Engine traits and engine-boundary types for GraphZero.

pub mod contract;
pub mod error;
pub mod hash;
pub mod measure;
pub mod metadata;
pub mod options;
pub mod refs;
pub mod traits;

pub use contract::{ContractError, EngineContractFixture, run_engine_contract};
pub use error::{
    EngineError, EngineResult, ErrorCode, ErrorMessage, ErrorMessageTooLong, OperationKind,
};
pub use hash::{
    ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash, HexParseError,
    MeasureConfigHash, ModelVersion, SearchConfigHash,
};
pub use measure::{
    LatencyStats, MeasureFailure, MeasureMetadata, MeasureResult, MeasureSummary,
    MeasurementValidationError,
};
pub use metadata::{
    ApplyJob, ApplyMetrics, ApplyResult, ApplyValidationError, CandidateInfo, CandidateInfoError,
    CandidateKindId, CandidateMetadata, CandidateTags, GraphArtifact, GraphArtifactFormat,
    RewriteRejection, SubjectId,
};
pub use options::{CandidateOptions, MeasureOptions, MeasureOptionsError};
pub use refs::{
    PortableCandidateRef, PortableGraphId, PortableSearchActionRef, ReplayGraphContext,
    SearchStepRef, SearchStepRefError,
};
pub use traits::{BatchGraphEngine, EngineReplayResolver, GraphEngine};
