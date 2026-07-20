use super::super::strategy::GumbelStrategy;
use super::super::{
    GumbelEpisode, GumbelEpisodeContext, GumbelHandleBatch, GumbelMcts, GumbelRootStats,
    GumbelStep, GumbelStopReason,
};
use super::root::common_config;
use crate::mcts::task::MctsEpisodeTask;
use crate::mcts::types::{MctsEpisode, MctsEpisodeContext, MctsStep, MctsStopReason};
use crate::work::{EngineIdentity, SearchPoll, SearchWorkResult, WorkToken};
use gz_engine::EngineResult;
use std::hash::Hash;

pub struct GumbelEpisodeTask<G, C> {
    inner: MctsEpisodeTask<G, C, GumbelStrategy>,
}

impl<G, C> GumbelEpisodeTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelEpisodeContext,
    ) -> Self {
        Self {
            inner: MctsEpisodeTask::new(
                common_config(search),
                GumbelStrategy::new(search.config),
                search.search_config_hash,
                identity,
                root,
                MctsEpisodeContext {
                    noise_seed: context.noise_seed,
                },
            ),
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelEpisode<G, C>>> {
        match self.inner.poll()? {
            SearchPoll::Work(work) => Ok(SearchPoll::Work(work)),
            SearchPoll::Blocked => Ok(SearchPoll::Blocked),
            SearchPoll::Done(episode) => Ok(SearchPoll::Done(gumbel_episode(episode))),
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        self.inner.resume(token, result)
    }

    #[must_use]
    pub fn step_index(&self) -> usize {
        self.inner.step_index()
    }

    pub fn take_releasable(&mut self) -> GumbelHandleBatch<G, C> {
        self.inner.take_releasable()
    }

    pub fn track_owned_root(&mut self) {
        self.inner.track_owned_root();
    }

    pub fn take_all_handles(&mut self) -> GumbelHandleBatch<G, C> {
        self.inner.take_all_handles()
    }
}

pub(crate) fn gumbel_episode<G, C>(episode: MctsEpisode<G, C>) -> GumbelEpisode<G, C> {
    GumbelEpisode {
        root: episode.root,
        final_graph: episode.final_graph,
        root_context: episode.root_context,
        final_context: episode.final_context,
        steps: episode.steps.into_iter().map(gumbel_step).collect(),
        root_stats: episode
            .root_stats
            .into_iter()
            .map(|stats| GumbelRootStats {
                simulations: stats.simulations,
                expanded_nodes: stats.expanded_nodes,
                eval_count: stats.eval_count,
                portable_contexts: stats.portable_contexts,
                carried_nodes: stats.carried_nodes,
                carried_root_visits: stats.carried_root_visits,
            })
            .collect(),
        created_graphs: episode.created_graphs,
        created_candidates: episode.created_candidates,
        final_measure: episode.final_measure,
        stop_reason: match episode.stop_reason {
            MctsStopReason::MaxSteps => GumbelStopReason::MaxSteps,
            MctsStopReason::SelectedStop => GumbelStopReason::SelectedStop,
        },
        search_config_hash: episode.search_config_hash,
    }
}

fn gumbel_step<G, C>(step: MctsStep<G, C>) -> GumbelStep<G, C> {
    GumbelStep {
        before: step.before,
        after: step.after,
        action: step.action,
        step_ref: step.step_ref,
        selected_action: step.selected_action,
        selected_candidate: step.selected_candidate,
        engine_candidate_count: step.engine_candidate_count,
        action_count: step.action_count,
        selected_rank: step.selected_rank,
        legal_actions: step.legal_actions,
        policy_target: step.policy_target,
        considered_action_indices: step.considered_action_indices,
        root_value: step.root_value,
        root_search_value: step.root_search_value,
        root_q_max: step.root_q_max,
        model_version: step.model_version,
    }
}
