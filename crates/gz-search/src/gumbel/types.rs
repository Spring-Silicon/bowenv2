use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    CandidateOptions, MeasureOptions, MeasureResult, ModelVersion, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash, SearchStepRef,
};
use std::num::NonZeroUsize;

pub type GumbelHandleBatch<G, C> = crate::mcts::types::MctsHandleBatch<G, C>;
pub type GumbelOpponentContext = crate::mcts::types::MctsOpponentContext;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GumbelValueMode {
    #[default]
    Competitive,
    SingleVanilla,
    SymmetricSelfplay,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelMctsConfig {
    pub max_steps: usize,
    pub simulations: NonZeroUsize,
    pub max_considered_actions: NonZeroUsize,
    pub seed: u64,
    pub gumbel_scale: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    pub temperature_moves: usize,
    /// Auto-temper the root Gumbel noise (whittlezero's overlap): when
    /// non-negative, per-root bisection replaces gumbel_scale with the
    /// scale at which a noisy argmax lands in the prior's top-m actions
    /// with probability overlap + 0.05 (the noisy argmax distributes as
    /// softmax(logits/scale)). Sharp policies get more noise, flat ones
    /// less; negative disables. Part of the search config hash.
    pub gumbel_noise_overlap: f32,
    /// Shift the selected child subtree into the next root. This carries
    /// cached graph/candidate/eval bodies and the subtree's visit/Q ledgers;
    /// the next root still receives a fresh simulation budget, counted
    /// relative to the carried visit baseline. Symmetric selfplay follows the
    /// same carry-all root-statistics contract.
    pub tree_reuse: bool,
    /// Export real position features (root_step, leaf_depth, budget) to
    /// evals and feature rows. Off zeroes the exported values so the
    /// model conditions on graph + opponent alone (and eval-cache keys
    /// collide across steps/depths). The search itself always uses the
    /// real values internally (noise seeding, budgets); deliberately not
    /// part of the search config hash.
    pub export_position: bool,
    /// Mask STOP out of node priors/logits wherever a rewrite exists
    /// (STOP-only nodes keep it). Policy rollouts preserve the caller's
    /// setting. Part of the search config hash.
    pub mask_stop: bool,
    /// Mask any action whose applied child is the current root or a
    /// prior root of this episode (whittlezero's no_backtrack): the
    /// search must find genuinely new states, and a root where every
    /// rewrite revisits history collapses the policy target onto STOP.
    /// Within-simulation cycles are already handled by the descent seen
    /// set. Part of the search config hash.
    pub no_backtrack: bool,
    /// Controls how completed Q values are interpreted by Gumbel search.
    /// Single Vanilla normalizes them with fresh per-root running bounds and
    /// treats the search horizon as a value-predicted terminal boundary.
    pub value_mode: GumbelValueMode,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct GumbelEpisodeContext {
    pub opponent: Option<GumbelOpponentContext>,
    /// Mixed into the root Gumbel RNG so episodes sharing a root explore
    /// differently. Zero (the default) preserves the historical seeding;
    /// drivers derive it from the episode id.
    pub noise_seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelSearchContext {
    pub root_step: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
    pub selection_temperature: f32,
    pub opponent: Option<GumbelOpponentContext>,
    pub noise_seed: u64,
    /// See [`GumbelMctsConfig::export_position`]; consulted only when
    /// exporting eval position contexts, never for search internals.
    pub export_position: bool,
}

impl Default for GumbelSearchContext {
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

#[derive(Clone, Debug, PartialEq)]
pub struct GumbelRootResult<G, C> {
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
    pub stats: GumbelRootStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GumbelRootStats {
    pub simulations: usize,
    pub expanded_nodes: usize,
    pub eval_count: usize,
    pub portable_contexts: usize,
    pub carried_nodes: usize,
    pub carried_root_visits: u32,
}

#[derive(Clone, Debug)]
pub struct GumbelEpisode<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<GumbelStep<G, C>>,
    pub root_stats: Vec<GumbelRootStats>,
    pub created_graphs: Vec<G>,
    pub created_candidates: Vec<C>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: GumbelStopReason,
    pub search_config_hash: SearchConfigHash,
    pub competitive: Option<Box<GumbelCompetitiveTrace<G, C>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GumbelPlayer {
    One,
    Two,
}

impl GumbelPlayer {
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::One => Self::Two,
            Self::Two => Self::One,
        }
    }

    #[must_use]
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::One => 0,
            Self::Two => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GumbelCompetitiveTrace<G, C> {
    pub learner_player: GumbelPlayer,
    pub opponent_root: G,
    pub opponent_final_graph: G,
    pub opponent_root_context: ReplayGraphContext,
    pub opponent_final_context: ReplayGraphContext,
    pub opponent_steps: Vec<GumbelStep<G, C>>,
    pub opponent_final_measure: MeasureResult<G>,
    pub opponent_stop_reason: GumbelStopReason,
}

impl<G: PartialEq, C: PartialEq> PartialEq for GumbelEpisode<G, C> {
    fn eq(&self, other: &Self) -> bool {
        self.root_context == other.root_context
            && self.final_context == other.final_context
            && self.steps == other.steps
            && self.root_stats == other.root_stats
            && measure_result_eq(&self.final_measure, &other.final_measure)
            && self.stop_reason == other.stop_reason
            && self.search_config_hash == other.search_config_hash
            && competitive_trace_eq(self.competitive.as_deref(), other.competitive.as_deref())
    }
}

fn competitive_trace_eq<G, C>(
    left: Option<&GumbelCompetitiveTrace<G, C>>,
    right: Option<&GumbelCompetitiveTrace<G, C>>,
) -> bool
where
    G: PartialEq,
    C: PartialEq,
{
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.learner_player == right.learner_player
                && left.opponent_root == right.opponent_root
                && left.opponent_final_graph == right.opponent_final_graph
                && left.opponent_root_context == right.opponent_root_context
                && left.opponent_final_context == right.opponent_final_context
                && left.opponent_steps == right.opponent_steps
                && measure_result_eq(&left.opponent_final_measure, &right.opponent_final_measure)
                && left.opponent_stop_reason == right.opponent_stop_reason
        }
        _ => false,
    }
}

fn measure_result_eq<G>(left: &MeasureResult<G>, right: &MeasureResult<G>) -> bool {
    left.graph_hash == right.graph_hash
        && left.config_hash == right.config_hash
        && left.measured == right.measured
        && left.valid == right.valid
        && left.latency == right.latency
        && left.scalar_reward == right.scalar_reward
        && left.failure == right.failure
        && left.metadata == right.metadata
}

#[derive(Clone, Debug)]
pub struct GumbelStep<G, C> {
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

impl<G, C> PartialEq for GumbelStep<G, C> {
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
pub enum GumbelStopReason {
    MaxSteps,
    SelectedStop,
}
