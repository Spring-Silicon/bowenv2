#![forbid(unsafe_code)]

//! Search kernels and search result records for GraphZero.

mod beam;
mod episode;
mod greedy;
mod gumbel;
mod hash;
mod mcts;
mod policy_rollout;
mod puct;
mod random;
mod sampling;
mod scratch;
mod support;
mod work;

pub use beam::{
    BeamEntrySummary, BeamEpisode, BeamLayer, BeamSearch, BeamSearchConfig, BeamStopReason,
};
pub use episode::{
    SearchAction, SearchCandidateSummary, SearchEpisode, SearchHandleBatch, SearchStep,
};
pub use greedy::{GreedyEpisode, GreedySearch, GreedySearchConfig, GreedyStopReason};
pub use gumbel::{
    GumbelCompetitiveTrace, GumbelEpisode, GumbelEpisodeContext, GumbelEpisodeTask,
    GumbelHandleBatch, GumbelMcts, GumbelMctsConfig, GumbelOpponentContext, GumbelPlayer,
    GumbelRootResult, GumbelRootStats, GumbelRootTask, GumbelSearchContext, GumbelStep,
    GumbelStopReason, GumbelValueMode, SampledTreeEpisodeTask, SampledTreeRootTask,
    SymmetricActorTrace, SymmetricEpisode, SymmetricRootAction, SymmetricRootResult,
    SymmetricSelfplayEpisodeTask, SymmetricSelfplayRootTask, considered_visit_sequence,
};
pub use hash::{
    beam_search_config_hash, greedy_search_config_hash, gumbel_search_config_hash,
    policy_rollout_config_hash, puct_search_config_hash, random_search_config_hash,
    reducing_uniform_distill_config_hash, sampled_tree_search_config_hash,
    symmetric_selfplay_search_config_hash,
};
pub use mcts::{MctsHandleBatch, MctsOpponentContext};
pub use policy_rollout::{
    PolicyRollout, PolicyRolloutConfig, PolicyRolloutContext, PolicyRolloutEpisode,
    PolicyRolloutEpisodeTask, PolicyRolloutHandleBatch, PolicyRolloutRootStats, PolicyRolloutStep,
    PolicyRolloutStopReason,
};
pub use puct::{
    PuctEpisode, PuctEpisodeContext, PuctEpisodeTask, PuctHandleBatch, PuctMcts, PuctMctsConfig,
    PuctOpponentContext, PuctRootResult, PuctRootStats, PuctRootTask, PuctSearchContext, PuctStep,
    PuctStopReason,
};
pub use random::{RandomEpisode, RandomSearch, RandomSearchConfig, RandomStopReason};
pub use work::{
    ApplyWork, EngineIdentity, EvalModel, EvalOpponentWork, EvalWork, ExpandResult, ExpandWork,
    ExpandedCandidate, MeasureWork, SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
