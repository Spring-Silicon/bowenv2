use gz_engine::{CandidateOptions, MeasureOptions};
use std::num::NonZeroUsize;

pub type PuctHandleBatch<G, C> = crate::mcts::types::MctsHandleBatch<G, C>;
pub type PuctEpisode<G, C> = crate::mcts::types::MctsEpisode<G, C>;
pub type PuctEpisodeContext = crate::mcts::types::MctsEpisodeContext;
pub type PuctRootResult<G, C> = crate::mcts::types::MctsRootResult<G, C>;
pub type PuctRootStats = crate::mcts::types::MctsRootStats;
pub type PuctStep<G, C> = crate::mcts::types::MctsStep<G, C>;
pub type PuctStopReason = crate::mcts::types::MctsStopReason;
pub type PuctSearchContext = crate::mcts::types::MctsSearchContext;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuctMctsConfig {
    pub max_steps: usize,
    pub simulations: NonZeroUsize,
    pub c_puct: f32,
    pub seed: u64,
    pub temperature_moves: usize,
    pub tree_reuse: bool,
    pub export_position: bool,
    pub mask_stop: bool,
    pub no_backtrack: bool,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}
