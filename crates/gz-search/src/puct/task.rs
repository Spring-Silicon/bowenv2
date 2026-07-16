use super::strategy::PuctStrategy;
use super::{
    PuctEpisode, PuctEpisodeContext, PuctHandleBatch, PuctMcts, PuctRootResult, PuctSearchContext,
};
use crate::mcts::task::{MctsEpisodeTask, MctsRootTask};
use crate::mcts::types::{MctsConfig, MctsEpisode, MctsEpisodeContext, MctsSearchContext};
use crate::work::{EngineIdentity, SearchPoll, SearchWorkResult, WorkToken};
use gz_engine::{EngineResult, ReplayGraphContext};
use std::hash::Hash;

pub struct PuctRootTask<G, C> {
    inner: MctsRootTask<G, C, PuctStrategy>,
}

impl<G, C> PuctRootTask<G, C>
where
    G: Copy,
    C: Copy,
{
    pub fn new(
        search: &PuctMcts,
        identity: EngineIdentity,
        root: G,
        context: PuctSearchContext,
    ) -> Self {
        Self {
            inner: MctsRootTask::new(
                common_config(search),
                PuctStrategy::new(search.config),
                identity,
                root,
                common_search_context(context),
            ),
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, PuctRootResult<G, C>>> {
        self.inner.poll()
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        self.inner.resume(token, result)
    }

    #[must_use]
    pub const fn root_context(&self) -> Option<ReplayGraphContext> {
        self.inner.root_context()
    }
}

pub struct PuctEpisodeTask<G, C> {
    inner: MctsEpisodeTask<G, C, PuctStrategy>,
}

impl<G, C> PuctEpisodeTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub fn new(
        search: &PuctMcts,
        identity: EngineIdentity,
        root: G,
        context: PuctEpisodeContext,
    ) -> Self {
        Self {
            inner: MctsEpisodeTask::new(
                common_config(search),
                PuctStrategy::new(search.config),
                search.search_config_hash,
                identity,
                root,
                common_episode_context(context),
            ),
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, PuctEpisode<G, C>>> {
        match self.inner.poll()? {
            SearchPoll::Work(work) => Ok(SearchPoll::Work(work)),
            SearchPoll::Blocked => Ok(SearchPoll::Blocked),
            SearchPoll::Done(episode) => Ok(SearchPoll::Done(puct_episode(episode))),
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        self.inner.resume(token, result)
    }

    #[must_use]
    pub fn step_index(&self) -> usize {
        self.inner.step_index()
    }

    pub fn take_releasable(&mut self) -> PuctHandleBatch<G, C> {
        self.inner.take_releasable()
    }

    pub fn track_owned_root(&mut self) {
        self.inner.track_owned_root();
    }

    pub fn take_all_handles(&mut self) -> PuctHandleBatch<G, C> {
        self.inner.take_all_handles()
    }
}

pub(crate) fn common_config(search: &PuctMcts) -> MctsConfig {
    MctsConfig {
        max_steps: search.config.max_steps,
        simulations: search.config.simulations,
        seed: search.config.seed,
        temperature_moves: search.config.temperature_moves,
        tree_reuse: search.config.tree_reuse,
        export_position: search.config.export_position,
        mask_stop: search.config.mask_stop,
        no_backtrack: search.config.no_backtrack,
        candidate_options: search.config.candidate_options,
        measure_options: search.config.measure_options,
    }
}

pub(crate) fn common_search_context(context: PuctSearchContext) -> MctsSearchContext {
    MctsSearchContext {
        root_step: context.root_step,
        budget_fraction: context.budget_fraction,
        budget_step: context.budget_step,
        selection_temperature: context.selection_temperature,
        opponent: context.opponent,
        noise_seed: context.noise_seed,
        export_position: context.export_position,
    }
}

pub(crate) fn common_episode_context(context: PuctEpisodeContext) -> MctsEpisodeContext {
    MctsEpisodeContext {
        opponent: context.opponent,
        noise_seed: context.noise_seed,
    }
}

pub(crate) fn puct_episode<G, C>(episode: MctsEpisode<G, C>) -> PuctEpisode<G, C> {
    PuctEpisode {
        root: episode.root,
        final_graph: episode.final_graph,
        root_context: episode.root_context,
        final_context: episode.final_context,
        steps: episode.steps,
        root_stats: episode.root_stats,
        created_graphs: episode.created_graphs,
        created_candidates: episode.created_candidates,
        final_measure: episode.final_measure,
        stop_reason: episode.stop_reason,
        search_config_hash: episode.search_config_hash,
    }
}
