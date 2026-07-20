mod strategy;
mod task;
mod types;

pub use task::{PuctEpisodeTask, PuctRootTask};
pub use types::{
    PuctEpisode, PuctEpisodeContext, PuctHandleBatch, PuctMctsConfig, PuctRootResult,
    PuctRootStats, PuctSearchContext, PuctStep, PuctStopReason,
};

use crate::mcts::driver::{run_episode, run_root};
use crate::mcts::math::budget_fraction;
use crate::mcts::task::{MctsEpisodeTask, MctsRootTask};
use crate::puct_search_config_hash;
use crate::work::EngineIdentity;
use gz_engine::{EngineResult, GraphEngine, SearchConfigHash};
use gz_eval::EngineEvaluator;

pub struct PuctMcts {
    pub(crate) config: PuctMctsConfig,
    pub(crate) search_config_hash: SearchConfigHash,
}

impl PuctMcts {
    #[must_use]
    pub fn new(config: PuctMctsConfig) -> Self {
        assert!(config.c_puct.is_finite() && config.c_puct >= 0.0);
        let search_config_hash = puct_search_config_hash(
            config.max_steps,
            config.simulations.get(),
            config.c_puct,
            config.seed,
            config.temperature_moves,
            config.tree_reuse,
            config.mask_stop,
            config.no_backtrack,
            config.candidate_options,
            config.measure_options,
        );
        Self {
            config,
            search_config_hash,
        }
    }

    #[must_use]
    pub const fn config(&self) -> PuctMctsConfig {
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
    ) -> EngineResult<PuctEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let root = engine.root();
        self.run(engine, evaluator, root, PuctEpisodeContext::default())
    }

    pub fn run<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: PuctEpisodeContext,
    ) -> EngineResult<PuctEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let task = MctsEpisodeTask::new(
            task::common_config(self),
            strategy::PuctStrategy::new(self.config),
            self.search_config_hash,
            EngineIdentity::from_engine(engine),
            root,
            context,
        );
        run_episode(engine, evaluator, task)
    }

    pub fn search_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: PuctSearchContext,
    ) -> EngineResult<PuctRootResult<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let task = MctsRootTask::new(
            task::common_config(self),
            strategy::PuctStrategy::new(self.config),
            EngineIdentity::from_engine(engine),
            root,
            context,
        );
        run_root(engine, evaluator, task)
    }
}
