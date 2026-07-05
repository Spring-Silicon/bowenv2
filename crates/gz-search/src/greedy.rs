use crate::scratch::GreedyScratch;
use crate::support::{candidate_info, graph_context, graph_context_from_hash, score, step_ref};
use crate::{
    SearchAction, SearchCandidateSummary, SearchEpisode, SearchStep, greedy_search_config_hash,
};
use gz_engine::{
    CandidateHash, CandidateOptions, EngineResult, GraphEngine, MeasureOptions, MeasureResult,
    MeasureSummary, PortableCandidateRef, PortableSearchActionRef, ReplayGraphContext,
    SearchConfigHash,
};

pub type GreedyEpisode<G, C> = SearchEpisode<G, C, GreedyStopReason>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GreedySearchConfig {
    pub max_steps: usize,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

pub struct GreedySearch {
    config: GreedySearchConfig,
    search_config_hash: SearchConfigHash,
}

impl GreedySearch {
    #[must_use]
    pub fn new(config: GreedySearchConfig) -> Self {
        let search_config_hash = greedy_search_config_hash(
            config.max_steps,
            config.candidate_options,
            config.measure_options,
        );

        Self {
            config,
            search_config_hash,
        }
    }

    #[must_use]
    pub const fn config(&self) -> GreedySearchConfig {
        self.config
    }

    #[must_use]
    pub const fn search_config_hash(&self) -> SearchConfigHash {
        self.search_config_hash
    }

    pub fn run_from_root<E: GraphEngine>(
        &self,
        engine: &mut E,
    ) -> EngineResult<GreedyEpisode<E::Graph, E::Candidate>> {
        self.run(engine, engine.root())
    }

    pub fn run<E: GraphEngine>(
        &self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<GreedyEpisode<E::Graph, E::Candidate>> {
        let root_context = graph_context(engine, root)?;
        let mut current = root;
        let mut current_context = root_context;
        let mut current_measure = None;
        let mut steps = Vec::new();
        let mut created_graphs = Vec::new();
        let mut created_candidates = Vec::new();
        let mut scratch = GreedyScratch::default();

        for _ in 0..self.config.max_steps {
            let measure = match current_measure.take() {
                Some(cached_measure) => cached_measure,
                None => engine.measure(current, self.config.measure_options)?,
            };

            let Some(current_reward) = score(&measure) else {
                return Ok(SearchEpisode {
                    root,
                    final_graph: current,
                    root_context,
                    final_context: current_context,
                    steps,
                    created_graphs,
                    created_candidates,
                    final_measure: measure,
                    stop_reason: GreedyStopReason::UnscoredCurrentGraph,
                    search_config_hash: self.search_config_hash,
                });
            };

            engine.candidates(
                current,
                self.config.candidate_options,
                &mut scratch.candidates,
            )?;
            created_candidates.extend(scratch.candidates.iter().copied());

            let mut best = None;

            for (selected_rank, candidate) in scratch.candidates.iter().copied().enumerate() {
                let info = candidate_info(engine, current, candidate)?;
                let candidate_ref = PortableCandidateRef::new(current_context, info.candidate_hash);
                let action_ref = PortableSearchActionRef::candidate(candidate_ref);

                let applied = engine.apply(current, candidate)?;
                created_graphs.push(applied.after);
                if applied.rejected.is_some() {
                    continue;
                }

                let measure = engine.measure(applied.after, self.config.measure_options)?;
                let Some(reward) = score(&measure) else {
                    continue;
                };

                if reward <= current_reward {
                    continue;
                }

                let after_context = graph_context_from_hash(engine, applied.after_hash);
                let candidate_hash = info.candidate_hash;
                let static_prior = info.static_prior;

                let next = CandidateScore {
                    candidate,
                    action_ref,
                    after: applied.after,
                    after_context,
                    measure,
                    selected_rank,
                    candidate_hash,
                    summary: SearchCandidateSummary {
                        kind: info.kind,
                        tags: info.tags,
                        static_prior,
                    },
                    reward,
                    static_prior,
                };

                if is_better_candidate(best.as_ref(), &next) {
                    best = Some(next);
                }
            }

            let engine_candidate_count = scratch.candidates.len();
            let stop_ref = PortableSearchActionRef::stop(current_context);
            let stop_rank = engine_candidate_count;
            let action_count = engine_candidate_count + 1;

            if let Some(selected) = best {
                let step_ref =
                    step_ref(current_context, selected.action_ref, selected.after_context)?;
                let selected_measure = MeasureSummary::from(&selected.measure);

                steps.push(SearchStep {
                    before: current,
                    after: selected.after,
                    action: SearchAction::Candidate(selected.candidate),
                    step_ref,
                    selected_action: selected.action_ref,
                    selected_candidate: Some(selected.summary),
                    selected_measure,
                    engine_candidate_count,
                    action_count,
                    selected_rank: selected.selected_rank,
                });

                current = selected.after;
                current_context = selected.after_context;
                current_measure = Some(selected.measure);
            } else {
                let step_ref = step_ref(current_context, stop_ref, current_context)?;
                let selected_measure = MeasureSummary::from(&measure);

                steps.push(SearchStep {
                    before: current,
                    after: current,
                    action: SearchAction::Stop,
                    step_ref,
                    selected_action: stop_ref,
                    selected_candidate: None,
                    selected_measure,
                    engine_candidate_count,
                    action_count,
                    selected_rank: stop_rank,
                });

                return Ok(SearchEpisode {
                    root,
                    final_graph: current,
                    root_context,
                    final_context: current_context,
                    steps,
                    created_graphs,
                    created_candidates,
                    final_measure: measure,
                    stop_reason: GreedyStopReason::SelectedStop,
                    search_config_hash: self.search_config_hash,
                });
            }
        }

        let final_measure = match current_measure {
            Some(cached_measure) => cached_measure,
            None => engine.measure(current, self.config.measure_options)?,
        };

        Ok(SearchEpisode {
            root,
            final_graph: current,
            root_context,
            final_context: current_context,
            steps,
            created_graphs,
            created_candidates,
            final_measure,
            stop_reason: GreedyStopReason::MaxSteps,
            search_config_hash: self.search_config_hash,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GreedyStopReason {
    MaxSteps,
    SelectedStop,
    UnscoredCurrentGraph,
}

struct CandidateScore<G, C> {
    candidate: C,
    action_ref: PortableSearchActionRef,
    after: G,
    after_context: ReplayGraphContext,
    measure: MeasureResult<G>,
    selected_rank: usize,
    candidate_hash: CandidateHash,
    summary: SearchCandidateSummary,
    reward: f32,
    static_prior: f32,
}

fn is_better_candidate<G, C>(
    current: Option<&CandidateScore<G, C>>,
    next: &CandidateScore<G, C>,
) -> bool {
    let Some(current) = current else {
        return true;
    };

    next.reward > current.reward
        || (next.reward == current.reward
            && (next.static_prior > current.static_prior
                || (next.static_prior == current.static_prior
                    && next.candidate_hash < current.candidate_hash)))
}
