#![forbid(unsafe_code)]

//! Whittle boolean rewrite engine adapter for GraphZero.

mod engine;
pub mod features;
mod graph;
mod rules;

pub use engine::{
    ArenaOccupancy, CandidateStorageStats, GeneratedWhittleGraph, HashVolumeCounters,
    WhittleContractFixture, WhittleEngine, WhittleEngineConfig, WhittleGeneratorConfigError,
    WhittleGraphGenerator, WhittleGraphGeneratorConfig, WhittleMeasureMode, WhittleRng,
    WhittleRoot,
};
pub use features::{WhittleFeatureExtractor, WhittleFeatureExtractorConfig};
pub use graph::{NO_NODE, OpCode, WhittleCandidateId, WhittleGraph, WhittleGraphId};
pub use rules::{RULE_COUNT, RuleId, rule_name};
