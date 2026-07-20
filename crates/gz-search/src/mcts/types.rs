use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    CandidateOptions, MeasureOptions, MeasureResult, ModelVersion, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash, SearchStepRef,
};
use gz_eval::EvalPositionContext;
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
    pub(crate) predicted_horizon: bool,
    pub(crate) candidate_options: CandidateOptions,
    pub(crate) measure_options: MeasureOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MctsEpisodeContext {
    pub noise_seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsSearchContext {
    pub root_step: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
    pub selection_temperature: f32,
    pub noise_seed: u64,
    pub export_position: bool,
}

impl Default for MctsSearchContext {
    fn default() -> Self {
        Self {
            root_step: 0,
            budget_fraction: 1.0,
            budget_step: 0.0,
            selection_temperature: 0.0,
            noise_seed: 0,
            export_position: true,
        }
    }
}

impl MctsSearchContext {
    pub(crate) fn position(self, leaf_depth: usize) -> EvalPositionContext {
        if !self.export_position {
            return EvalPositionContext {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 0.0,
                budget_step: 0.0,
            };
        }

        EvalPositionContext {
            root_step: self.root_step,
            leaf_depth: leaf_depth as u32,
            budget_fraction: self.budget_fraction,
            budget_step: self.budget_step,
        }
    }
}

pub type MctsHandleBatch<G, C> = crate::SearchHandleBatch<G, C>;

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

#[derive(Clone, Debug)]
pub struct MctsEpisode<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<MctsStep<G, C>>,
    pub root_stats: Vec<MctsRootStats>,
    pub created_graphs: Vec<G>,
    pub created_candidates: Vec<C>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: MctsStopReason,
    pub search_config_hash: SearchConfigHash,
}
