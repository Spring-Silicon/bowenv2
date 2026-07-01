#![forbid(unsafe_code)]

//! Search kernels and search result records for GraphZero.

mod beam;
mod episode;
mod greedy;
mod hash;
mod random;
mod scratch;
mod support;

pub use beam::{
    BeamEntrySummary, BeamEpisode, BeamLayer, BeamSearch, BeamSearchConfig, BeamStopReason,
};
pub use episode::{SearchAction, SearchCandidateSummary, SearchEpisode, SearchStep};
pub use greedy::{GreedyEpisode, GreedySearch, GreedySearchConfig, GreedyStopReason};
pub use hash::{beam_search_config_hash, greedy_search_config_hash, random_search_config_hash};
pub use random::{RandomEpisode, RandomSearch, RandomSearchConfig, RandomStopReason};
