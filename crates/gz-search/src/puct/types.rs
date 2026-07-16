use gz_engine::{
    CandidateOptions, MeasureOptions, MeasureResult, ReplayGraphContext, SearchConfigHash,
};
use std::num::NonZeroUsize;

pub type PuctHandleBatch<G, C> = crate::mcts::types::MctsHandleBatch<G, C>;
pub type PuctOpponentContext = crate::mcts::types::MctsOpponentContext;
pub type PuctRootResult<G, C> = crate::mcts::types::MctsRootResult<G, C>;
pub type PuctRootStats = crate::mcts::types::MctsRootStats;
pub type PuctStep<G, C> = crate::mcts::types::MctsStep<G, C>;
pub type PuctStopReason = crate::mcts::types::MctsStopReason;

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

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PuctEpisodeContext {
    pub opponent: Option<PuctOpponentContext>,
    pub noise_seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuctSearchContext {
    pub root_step: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
    pub selection_temperature: f32,
    pub opponent: Option<PuctOpponentContext>,
    pub noise_seed: u64,
    pub export_position: bool,
}

impl Default for PuctSearchContext {
    fn default() -> Self {
        Self {
            root_step: 0,
            budget_fraction: 1.0,
            budget_step: 0.0,
            selection_temperature: 0.0,
            opponent: None,
            noise_seed: 0,
            export_position: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PuctEpisode<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<PuctStep<G, C>>,
    pub root_stats: Vec<PuctRootStats>,
    pub created_graphs: Vec<G>,
    pub created_candidates: Vec<C>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: PuctStopReason,
    pub search_config_hash: SearchConfigHash,
}
