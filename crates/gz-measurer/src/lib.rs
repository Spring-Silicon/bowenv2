#![forbid(unsafe_code)]

//! Measured symmetric selfplay projection into replay rows.

mod project;
mod service;

pub use project::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasurerError, episode_reward,
    horizon_value_targets, project_episode,
};
pub use service::{
    MeasureLedgerSnapshot, MeasuredSymmetricGame, MeasurerAdmission, MeasurerAdmissionStatus,
    MeasurerLaneSummary, MeasurerRunSummary, MeasurerStats, ReplayMeasurer,
};
