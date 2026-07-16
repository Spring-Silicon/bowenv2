use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    CandidateOptions, MeasureOptions, MeasureResult, ModelVersion, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash, SearchStepRef,
};
use gz_eval::{EvalOpponentContext, EvalPositionContext};
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MctsConfig {
    pub(crate) max_steps: usize,
    pub(crate) simulations: NonZeroUsize,
    pub(crate) seed: u64,
    pub(crate) temperature_moves: usize,
    pub(crate) tree_reuse: bool,
    pub(crate) export_position: bool,
    pub(crate) mask_stop: bool,
    pub(crate) no_backtrack: bool,
    pub(crate) candidate_options: CandidateOptions,
    pub(crate) measure_options: MeasureOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(crate) struct MctsEpisodeContext {
    pub(crate) opponent: Option<MctsOpponentContext>,
    pub(crate) noise_seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MctsSearchContext {
    pub(crate) root_step: u32,
    pub(crate) budget_fraction: f32,
    pub(crate) budget_step: f32,
    pub(crate) selection_temperature: f32,
    pub(crate) opponent: Option<MctsOpponentContext>,
    pub(crate) noise_seed: u64,
    pub(crate) export_position: bool,
}

impl MctsSearchContext {
    pub(crate) fn position(self, leaf_depth: usize) -> EvalPositionContext {
        let opponent = self
            .opponent
            .map(|opponent| opponent.aligned_to(u64::from(self.root_step) + leaf_depth as u64));
        if !self.export_position {
            return EvalPositionContext {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 0.0,
                budget_step: 0.0,
                opponent,
            };
        }

        EvalPositionContext {
            root_step: self.root_step,
            leaf_depth: leaf_depth as u32,
            budget_fraction: self.budget_fraction,
            budget_step: self.budget_step,
            opponent,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsOpponentContext {
    pub trajectory_id: u64,
    pub row_count: u32,
    pub final_reward: f32,
}

impl MctsOpponentContext {
    #[must_use]
    pub fn aligned_to(self, step: u64) -> EvalOpponentContext {
        EvalOpponentContext {
            trajectory_id: self.trajectory_id,
            row_count: self.row_count,
            final_reward: self.final_reward,
            row: step.min(u64::from(self.row_count.saturating_sub(1))) as u32,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MctsHandleBatch<G, C> {
    pub graphs: Vec<G>,
    pub candidates: Vec<C>,
}

impl<G, C> Default for MctsHandleBatch<G, C> {
    fn default() -> Self {
        Self {
            graphs: Vec::new(),
            candidates: Vec::new(),
        }
    }
}

impl<G, C> MctsHandleBatch<G, C> {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty() && self.candidates.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsRootResult<G, C> {
    pub root: G,
    pub root_context: ReplayGraphContext,
    pub selected_after: G,
    pub selected_after_context: ReplayGraphContext,
    pub selected_action: SearchAction<C>,
    pub selected_action_ref: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub selected_action_index: usize,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub considered_action_indices: Vec<usize>,
    pub policy_target: Vec<f32>,
    pub root_value: f32,
    pub root_search_value: f32,
    pub root_q_max: f32,
    pub model_version: ModelVersion,
    pub stats: MctsRootStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MctsRootStats {
    pub simulations: usize,
    pub expanded_nodes: usize,
    pub eval_count: usize,
    pub portable_contexts: usize,
    pub carried_nodes: usize,
    pub carried_root_visits: u32,
}

#[derive(Clone, Debug)]
pub struct MctsStep<G, C> {
    pub before: G,
    pub after: G,
    pub action: SearchAction<C>,
    pub step_ref: SearchStepRef,
    pub selected_action: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub selected_rank: usize,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<f32>,
    pub considered_action_indices: Vec<usize>,
    pub root_value: f32,
    pub root_search_value: f32,
    pub root_q_max: f32,
    pub model_version: ModelVersion,
}

impl<G, C> PartialEq for MctsStep<G, C> {
    fn eq(&self, other: &Self) -> bool {
        self.step_ref == other.step_ref
            && self.selected_action == other.selected_action
            && self.selected_candidate == other.selected_candidate
            && self.engine_candidate_count == other.engine_candidate_count
            && self.action_count == other.action_count
            && self.selected_rank == other.selected_rank
            && self.legal_actions == other.legal_actions
            && self.policy_target == other.policy_target
            && self.considered_action_indices == other.considered_action_indices
            && self.root_value == other.root_value
            && self.root_search_value == other.root_search_value
            && self.root_q_max == other.root_q_max
            && self.model_version == other.model_version
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MctsStopReason {
    MaxSteps,
    SelectedStop,
}

pub(crate) struct MctsEpisode<G, C> {
    pub(crate) root: G,
    pub(crate) final_graph: G,
    pub(crate) root_context: ReplayGraphContext,
    pub(crate) final_context: ReplayGraphContext,
    pub(crate) steps: Vec<MctsStep<G, C>>,
    pub(crate) root_stats: Vec<MctsRootStats>,
    pub(crate) created_graphs: Vec<G>,
    pub(crate) created_candidates: Vec<C>,
    pub(crate) final_measure: MeasureResult<G>,
    pub(crate) stop_reason: MctsStopReason,
    pub(crate) search_config_hash: SearchConfigHash,
}
