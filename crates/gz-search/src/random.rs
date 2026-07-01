use crate::support::{candidate_info, graph_context, graph_context_from_hash, score, step_ref};
use crate::{
    SearchAction, SearchCandidateSummary, SearchEpisode, SearchStep, random_search_config_hash,
};
use gz_engine::{
    CandidateOptions, EngineResult, GraphEngine, MeasureOptions, MeasureSummary,
    PortableCandidateRef, PortableSearchActionRef, SearchConfigHash,
};

pub type RandomEpisode<G, C> = SearchEpisode<G, C, RandomStopReason>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RandomSearchConfig {
    pub max_steps: usize,
    pub seed: u64,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

pub struct RandomSearch {
    config: RandomSearchConfig,
    search_config_hash: SearchConfigHash,
}

impl RandomSearch {
    #[must_use]
    pub fn new(config: RandomSearchConfig) -> Self {
        let search_config_hash = random_search_config_hash(
            config.max_steps,
            config.seed,
            config.candidate_options,
            config.measure_options,
        );

        Self {
            config,
            search_config_hash,
        }
    }

    #[must_use]
    pub const fn config(&self) -> RandomSearchConfig {
        self.config
    }

    #[must_use]
    pub const fn search_config_hash(&self) -> SearchConfigHash {
        self.search_config_hash
    }

    pub fn run_from_root<E: GraphEngine>(
        &self,
        engine: &mut E,
    ) -> EngineResult<RandomEpisode<E::Graph, E::Candidate>> {
        self.run(engine, engine.root())
    }

    pub fn run<E: GraphEngine>(
        &self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<RandomEpisode<E::Graph, E::Candidate>> {
        let root_context = graph_context(engine, root)?;
        let mut current = root;
        let mut current_context = root_context;
        let mut current_measure = None;
        let mut steps = Vec::new();
        let mut candidates = Vec::new();
        let mut rng = RandomRng::new(self.config.seed);

        for _ in 0..self.config.max_steps {
            let measure = match current_measure.take() {
                Some(cached_measure) => cached_measure,
                None => engine.measure(current, self.config.measure_options)?,
            };

            if score(&measure).is_none() {
                return Ok(SearchEpisode {
                    root,
                    final_graph: current,
                    root_context,
                    final_context: current_context,
                    steps,
                    final_measure: measure,
                    stop_reason: RandomStopReason::UnscoredCurrentGraph,
                    search_config_hash: self.search_config_hash,
                });
            }

            engine.candidates(current, self.config.candidate_options, &mut candidates)?;

            let engine_candidate_count = candidates.len();
            let stop_ref = PortableSearchActionRef::stop(current_context);
            let stop_rank = engine_candidate_count;
            let action_count = engine_candidate_count + 1;
            let selected_rank = rng.index(action_count);

            if selected_rank == stop_rank {
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
                    selected_rank,
                });

                return Ok(SearchEpisode {
                    root,
                    final_graph: current,
                    root_context,
                    final_context: current_context,
                    steps,
                    final_measure: measure,
                    stop_reason: RandomStopReason::SelectedStop,
                    search_config_hash: self.search_config_hash,
                });
            }

            let candidate = candidates[selected_rank];
            let info = candidate_info(engine, current, candidate)?;
            let action_ref = PortableSearchActionRef::candidate(PortableCandidateRef::new(
                current_context,
                info.candidate_hash,
            ));
            let applied = engine.apply(current, candidate)?;

            if applied.rejected.is_some() {
                return Ok(SearchEpisode {
                    root,
                    final_graph: current,
                    root_context,
                    final_context: current_context,
                    steps,
                    final_measure: measure,
                    stop_reason: RandomStopReason::RejectedSelectedCandidate,
                    search_config_hash: self.search_config_hash,
                });
            }

            let selected_measure = engine.measure(applied.after, self.config.measure_options)?;
            let after_context = graph_context_from_hash(engine, applied.after_hash);
            let step_ref = step_ref(current_context, action_ref, after_context)?;

            steps.push(SearchStep {
                before: current,
                after: applied.after,
                action: SearchAction::Candidate(candidate),
                step_ref,
                selected_action: action_ref,
                selected_candidate: Some(SearchCandidateSummary {
                    kind: info.kind,
                    tags: info.tags,
                    static_prior: info.static_prior,
                }),
                selected_measure: MeasureSummary::from(&selected_measure),
                engine_candidate_count,
                action_count,
                selected_rank,
            });

            current = applied.after;
            current_context = after_context;

            if score(&selected_measure).is_none() {
                return Ok(SearchEpisode {
                    root,
                    final_graph: current,
                    root_context,
                    final_context: current_context,
                    steps,
                    final_measure: selected_measure,
                    stop_reason: RandomStopReason::UnscoredSelectedGraph,
                    search_config_hash: self.search_config_hash,
                });
            }

            current_measure = Some(selected_measure);
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
            final_measure,
            stop_reason: RandomStopReason::MaxSteps,
            search_config_hash: self.search_config_hash,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RandomStopReason {
    MaxSteps,
    SelectedStop,
    UnscoredCurrentGraph,
    RejectedSelectedCandidate,
    UnscoredSelectedGraph,
}

struct RandomRng {
    state: u64,
}

impl RandomRng {
    const STEP: u64 = 0x9e37_79b9_7f4a_7c15;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn index(&mut self, upper: usize) -> usize {
        let upper = upper as u64;
        let threshold = upper.wrapping_neg() % upper;

        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % upper) as usize;
            }
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::STEP);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
