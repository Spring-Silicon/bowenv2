#![forbid(unsafe_code)]

//! Whittle boolean rewrite engine adapter for GraphZero.

mod engine;
mod graph;
mod rules;

pub use engine::{
    GeneratedWhittleGraph, WhittleContractFixture, WhittleEngine, WhittleEngineConfig,
    WhittleGeneratorConfigError, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
    WhittleMeasureMode, WhittleRng, WhittleRoot,
};
pub use graph::{NO_NODE, OpCode, WhittleCandidateId, WhittleGraph, WhittleGraphId};
pub use rules::{RuleId, rule_name};
