use gz_engine::{
    CandidateKindId, CandidateTags, MeasureResult, MeasureSummary, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash, SearchStepRef,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SearchAction<C> {
    Candidate(C),
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchCandidateSummary {
    pub kind: CandidateKindId,
    pub tags: CandidateTags,
    pub static_prior: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchEpisode<G, C, S> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<SearchStep<G, C>>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: S,
    pub search_config_hash: SearchConfigHash,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchStep<G, C> {
    pub before: G,
    pub after: G,
    pub action: SearchAction<C>,
    pub step_ref: SearchStepRef,
    pub selected_action: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub selected_measure: MeasureSummary,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub selected_rank: usize,
}
