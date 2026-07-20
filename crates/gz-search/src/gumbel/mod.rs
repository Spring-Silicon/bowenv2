mod sampled_tree;
mod schedule;
mod strategy;
mod symmetric;
mod task;
mod types;

pub use sampled_tree::{SampledTreeEpisodeTask, SampledTreeRootTask};
pub use schedule::considered_visit_sequence;
pub use symmetric::{
    SymmetricActorTrace, SymmetricEpisode, SymmetricRootAction, SymmetricRootResult,
    SymmetricSelfplayEpisodeTask, SymmetricSelfplayRootTask,
};
pub use task::{GumbelEpisodeTask, GumbelRootTask};
pub use types::{
    GumbelCompetitiveTrace, GumbelEpisode, GumbelEpisodeContext, GumbelHandleBatch,
    GumbelMctsConfig, GumbelOpponentContext, GumbelPlayer, GumbelRootResult, GumbelRootStats,
    GumbelSearchContext, GumbelStep, GumbelStopReason, GumbelValueMode,
};

use crate::gumbel_search_config_hash;
use crate::mcts::driver::{run_episode, run_root};
use crate::mcts::math::budget_fraction;
use crate::mcts::task::{MctsEpisodeTask, MctsRootTask};
use crate::mcts::types::{MctsEpisodeContext, MctsRootResult};
use crate::work::EngineIdentity;
use gz_engine::{EngineResult, GraphEngine, SearchConfigHash};
use gz_eval::EngineEvaluator;
use std::num::NonZeroUsize;

pub struct GumbelMcts {
    config: GumbelMctsConfig,
    search_config_hash: SearchConfigHash,
    policy_rollout_mask_stop: Option<bool>,
    symmetric_wave_batching: bool,
}

impl GumbelMcts {
    #[must_use]
    pub fn new(config: GumbelMctsConfig) -> Self {
        assert!(config.gumbel_scale.is_finite() && config.gumbel_scale >= 0.0);
        assert!(config.gumbel_noise_overlap.is_finite() && config.gumbel_noise_overlap < 1.0);
        assert!(config.c_visit.is_finite() && config.c_visit >= 0.0);
        assert!(config.c_scale.is_finite() && config.c_scale >= 0.0);
        assert!(
            config.value_mode != GumbelValueMode::SingleVanilla || !config.tree_reuse,
            "single-vanilla search requires a fresh tree per move"
        );
        let search_config_hash = gumbel_search_config_hash(
            config.max_steps,
            config.simulations.get(),
            config.max_considered_actions.get(),
            config.seed,
            config.gumbel_scale,
            config.gumbel_noise_overlap,
            config.c_visit,
            config.c_scale,
            config.temperature_moves,
            config.tree_reuse,
            config.mask_stop,
            config.no_backtrack,
            config.candidate_options,
            config.measure_options,
            config.value_mode,
        );

        Self {
            config,
            search_config_hash,
            policy_rollout_mask_stop: None,
            symmetric_wave_batching: false,
        }
    }

    /// Overrides STOP masking only for the derived greedy policy rollout.
    /// Absent an override, the rollout inherits the learner setting.
    #[must_use]
    pub const fn with_policy_rollout_mask_stop(mut self, mask_stop: bool) -> Self {
        self.policy_rollout_mask_stop = Some(mask_stop);
        self
    }

    /// Executes independent symmetric root branches concurrently without
    /// changing the search configuration or replay identity.
    #[must_use]
    pub const fn with_symmetric_wave_batching(mut self, enabled: bool) -> Self {
        self.symmetric_wave_batching = enabled;
        self
    }

    #[must_use]
    pub const fn symmetric_wave_batching(&self) -> bool {
        self.symmetric_wave_batching
    }

    #[must_use]
    pub const fn config(&self) -> GumbelMctsConfig {
        self.config
    }

    /// The opponent-rollout search derived from this one: a single
    /// simulation over a single considered action with no noise -- a
    /// greedy argmax-policy rollout at temperature 0, preserving the
    /// caller's STOP masking policy. Step budget and engine options carry
    /// over unchanged.
    #[must_use]
    pub fn policy_rollout(&self) -> Self {
        Self::new(GumbelMctsConfig {
            simulations: NonZeroUsize::MIN,
            max_considered_actions: NonZeroUsize::MIN,
            gumbel_scale: 0.0,
            gumbel_noise_overlap: -1.0,
            temperature_moves: 0,
            tree_reuse: false,
            mask_stop: match self.policy_rollout_mask_stop {
                Some(mask_stop) => mask_stop,
                None => self.config.mask_stop,
            },
            // The reference is a plain greedy rollout (whittlezero's
            // policy_rollout has no revisit masking either).
            no_backtrack: false,
            ..self.config
        })
    }

    /// A categorical policy rollout for trajectory-pool references. Gumbel
    /// top-1 with unit scale samples exactly from softmax(policy logits) at
    /// each root; no tree search or overlap tempering is involved.
    #[must_use]
    pub fn policy_sample_rollout(&self) -> Self {
        Self::new(GumbelMctsConfig {
            simulations: NonZeroUsize::MIN,
            max_considered_actions: NonZeroUsize::MIN,
            gumbel_scale: 1.0,
            gumbel_noise_overlap: -1.0,
            temperature_moves: 0,
            tree_reuse: false,
            mask_stop: match self.policy_rollout_mask_stop {
                Some(mask_stop) => mask_stop,
                None => self.config.mask_stop,
            },
            no_backtrack: false,
            ..self.config
        })
    }

    #[must_use]
    pub(crate) const fn reference_mask_stop(&self) -> bool {
        match self.policy_rollout_mask_stop {
            Some(mask_stop) => mask_stop,
            None => self.config.mask_stop,
        }
    }

    #[must_use]
    pub const fn search_config_hash(&self) -> SearchConfigHash {
        self.search_config_hash
    }

    #[must_use]
    pub fn root_budget(&self, step: usize) -> (f32, f32) {
        let budget_step = if self.config.max_steps == 0 {
            0.0
        } else {
            1.0 / self.config.max_steps as f32
        };

        (budget_fraction(self.config.max_steps, step), budget_step)
    }

    pub fn run_from_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
    ) -> EngineResult<GumbelEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let root = engine.root();
        self.run(engine, evaluator, root, GumbelEpisodeContext::default())
    }

    pub fn run<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let task = MctsEpisodeTask::new(
            task::common_config(self),
            strategy::GumbelStrategy::new(self.config),
            self.search_config_hash,
            EngineIdentity::from_engine(engine),
            root,
            MctsEpisodeContext {
                opponent: context.opponent,
                noise_seed: context.noise_seed,
            },
        );
        run_episode(engine, evaluator, task).map(task::gumbel_episode)
    }

    pub fn search_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: GumbelSearchContext,
    ) -> EngineResult<GumbelRootResult<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let task = MctsRootTask::new(
            task::common_config(self),
            strategy::GumbelStrategy::new(self.config),
            EngineIdentity::from_engine(engine),
            root,
            task::common_context(context),
        );
        run_root(engine, evaluator, task)
            .map(|result: MctsRootResult<_, _>| task::gumbel_result(result))
    }
}
