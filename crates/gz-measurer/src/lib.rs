#![forbid(unsafe_code)]

//! Central reference snapshots and replay projection.

mod project;
mod registry;
mod service;

pub use project::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasurerError, ProjectedReference,
    ProjectionMode, episode_reward, project_episode, sign_target,
};
pub use registry::{
    GateEvent, PendingChallenge, ReferenceRegistry, ReferenceSnapshot, ReferenceStep,
    RolloutOutcome,
};
pub use service::{
    MeasureLedgerSnapshot, MeasuredEpisode, MeasurerAdmission, MeasurerAdmissionStatus,
    MeasurerLaneSummary, MeasurerRunSummary, MeasurerStats, ReplayMeasurer,
};
