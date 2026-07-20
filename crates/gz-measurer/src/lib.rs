#![forbid(unsafe_code)]

//! Central reference snapshots and replay projection.

mod project;
mod registry;
mod service;

pub use project::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasurerError, ProjectedReference,
    ProjectionMode, ValueTargetConfig, episode_reward, horizon_value_targets, outcome_target,
    project_episode, project_episode_with_value_target, sign_target,
};
pub use registry::{
    ArenaGateEvent, ArenaGateRegistry, ArenaRolloutClaim, EpisodeRolloutClaim, GateEvent,
    PendingChallenge, PolicyModel, ReferenceRegistry, ReferenceSnapshot, ReferenceStep,
    RolloutOutcome,
};
pub use service::{
    MeasureLedgerSnapshot, MeasuredCompetitiveGame, MeasuredEpisode, MeasuredSymmetricGame,
    MeasurerAdmission, MeasurerAdmissionStatus, MeasurerLaneSummary, MeasurerRunSummary,
    MeasurerStats, ReplayMeasurer,
};
