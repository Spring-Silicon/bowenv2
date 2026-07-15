#![forbid(unsafe_code)]

//! Search kernels and search result records for GraphZero.

mod beam;
mod episode;
mod greedy;
mod gumbel;
mod hash;
mod random;
mod scratch;
mod support;
mod work;

pub use beam::{
    BeamEntrySummary, BeamEpisode, BeamLayer, BeamSearch, BeamSearchConfig, BeamStopReason,
};
pub use episode::{SearchAction, SearchCandidateSummary, SearchEpisode, SearchStep};
pub use greedy::{GreedyEpisode, GreedySearch, GreedySearchConfig, GreedyStopReason};
pub use gumbel::{
    CategoricalPolicyEpisodeTask, GumbelCompetitiveTrace, GumbelEpisode, GumbelEpisodeContext,
    GumbelEpisodeTask, GumbelHandleBatch, GumbelMcts, GumbelMctsConfig, GumbelOpponentContext,
    GumbelPlayer, GumbelRootResult, GumbelRootStats, GumbelRootTask, GumbelSearchContext,
    GumbelStep, GumbelStopReason, SampledTreeEpisodeTask, SampledTreeRootTask,
    considered_visit_sequence,
};
pub use hash::{
    beam_search_config_hash, categorical_policy_config_hash, greedy_search_config_hash,
    gumbel_search_config_hash, random_search_config_hash, reducing_uniform_distill_config_hash,
    sampled_tree_search_config_hash,
};
pub use random::{RandomEpisode, RandomSearch, RandomSearchConfig, RandomStopReason};
pub use work::{
    ApplyWork, EngineIdentity, EvalModel, EvalOpponentWork, EvalWork, ExpandResult, ExpandWork,
    ExpandedCandidate, MeasureWork, SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
