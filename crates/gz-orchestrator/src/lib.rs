#![forbid(unsafe_code)]

//! Execution drivers for GraphZero search workers.

mod bench;
mod ids;
mod serial;

pub use bench::{
    SelfplayBenchConfig, SelfplayBenchReport, SelfplayEpisodeStats, SelfplayRunStats,
    run_selfplay_benchmark, run_serial_selfplay_benchmark,
};
pub use ids::{EpisodeId, WorkerId};
pub use serial::{SerialEpisode, SerialGumbelOrchestrator};
