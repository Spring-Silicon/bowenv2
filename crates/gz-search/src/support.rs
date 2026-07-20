use gz_engine::{
    CandidateInfo, EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine, GraphHash,
    MeasureResult, PortableGraphId, PortableSearchActionRef, ReplayGraphContext, SearchStepRef,
};

pub(crate) fn candidate_info<E: GraphEngine>(
    engine: &E,
    graph: E::Graph,
    candidate: E::Candidate,
) -> EngineResult<CandidateInfo> {
    engine
        .candidate_info(graph, candidate)?
        .validate()
        .map_err(|_| internal("invalid candidate info"))
}

pub(crate) fn graph_context<E: GraphEngine>(
    engine: &E,
    graph: E::Graph,
) -> EngineResult<ReplayGraphContext> {
    Ok(graph_context_from_hash(engine, engine.hash(graph)?))
}

pub(crate) fn graph_context_from_hash<E: GraphEngine>(
    engine: &E,
    graph_hash: GraphHash,
) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(graph_hash, engine.engine_id(), engine.engine_version()),
        engine.action_set_hash(),
    )
}

pub(crate) fn score<G>(measure: &MeasureResult<G>) -> Option<f32> {
    if !measure.measured || !measure.valid {
        return None;
    }

    match measure.scalar_reward {
        Some(reward) if reward.is_finite() => Some(reward),
        _ => None,
    }
}

pub(crate) fn budget_fraction(max_steps: usize, step: usize) -> f32 {
    if max_steps == 0 {
        1.0
    } else {
        max_steps.saturating_sub(step) as f32 / max_steps as f32
    }
}

pub(crate) fn step_ref(
    before: ReplayGraphContext,
    action: PortableSearchActionRef,
    after: ReplayGraphContext,
) -> EngineResult<SearchStepRef> {
    SearchStepRef::new(before, action, after).map_err(|_| internal("invalid search step ref"))
}

pub(crate) fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(1),
        message: ErrorMessage::new(message).expect("internal search error message is short"),
    }
}
