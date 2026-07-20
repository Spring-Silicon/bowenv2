mod schedule;
mod strategy;
mod symmetric;
mod task;
mod types;

pub use schedule::considered_visit_sequence;
pub use symmetric::{
    SymmetricActorTrace, SymmetricEpisode, SymmetricRootAction, SymmetricRootResult,
    SymmetricSelfplayEpisodeTask, SymmetricSelfplayRootTask,
};
pub use task::{GumbelEpisodeTask, GumbelRootTask};
pub use types::{
    GumbelEpisode, GumbelEpisodeContext, GumbelHandleBatch, GumbelMctsConfig, GumbelPlayer,
    GumbelRootResult, GumbelRootStats, GumbelSearchContext, GumbelStep, GumbelStopReason,
    GumbelValueMode,
};

use crate::gumbel_search_config_hash;
use crate::mcts::driver::{run_episode, run_root};
use crate::mcts::math::budget_fraction;
use crate::mcts::task::{MctsEpisodeTask, MctsRootTask};
use crate::mcts::types::{MctsEpisodeContext, MctsRootResult};
use crate::work::EngineIdentity;
use gz_engine::{EngineResult, GraphEngine, SearchConfigHash};
use gz_eval::EngineEvaluator;

pub struct GumbelMcts {
    config: GumbelMctsConfig,
    search_config_hash: SearchConfigHash,
    symmetric_wave_batching: bool,
}

impl GumbelMcts {
    #[must_use]
    pub fn new(config: GumbelMctsConfig) -> Self {
        assert!(config.gumbel_scale.is_finite() && config.gumbel_scale >= 0.0);
        assert!(config.gumbel_noise_overlap.is_finite() && config.gumbel_noise_overlap < 1.0);
        assert!(config.c_visit.is_finite() && config.c_visit >= 0.0);
        assert!(config.c_scale.is_finite() && config.c_scale >= 0.0);
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
            symmetric_wave_batching: false,
        }
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
