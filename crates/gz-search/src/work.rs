use gz_engine::{
    ActionSetHash, ApplyResult, CandidateHash, CandidateKindId, CandidateOptions, CandidateTags,
    EngineId, EngineVersion, GraphEngine, GraphHash, MeasureOptions, MeasureResult,
    PortableGraphId, ReplayGraphContext,
};
use gz_eval::{EvalOutput, EvalRequest};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WorkToken(u64);

impl WorkToken {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SearchPoll<G, C, R> {
    Work(SearchWork<G, C>),
    Blocked,
    Done(R),
}

impl<G, C, R> SearchPoll<G, C, R> {
    #[must_use]
    pub fn map_done<T>(self, map: impl FnOnce(R) -> T) -> SearchPoll<G, C, T> {
        match self {
            Self::Work(work) => SearchPoll::Work(work),
            Self::Blocked => SearchPoll::Blocked,
            Self::Done(result) => SearchPoll::Done(map(result)),
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
// Boxing Eval would allocate on every search evaluation; worker slots hold one
// work item at a time, so the larger enum is the intentional hot-path tradeoff.
#[allow(clippy::large_enum_variant)]
pub enum SearchWork<G, C> {
    Expand(ExpandWork<G>),
    Apply(ApplyWork<G, C>),
    Measure(MeasureWork<G>),
    Eval(EvalWork<G, C>),
}

impl<G, C> SearchWork<G, C> {
    #[must_use]
    pub const fn token(&self) -> WorkToken {
        match self {
            Self::Expand(work) => work.token,
            Self::Apply(work) => work.token,
            Self::Measure(work) => work.token,
            Self::Eval(work) => work.token,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExpandWork<G> {
    pub token: WorkToken,
    pub graph: G,
    pub options: CandidateOptions,
}

#[derive(Clone, Copy, Debug)]
pub struct ApplyWork<G, C> {
    pub token: WorkToken,
    pub graph: G,
    pub candidate: C,
}

#[derive(Clone, Copy, Debug)]
pub struct MeasureWork<G> {
    pub token: WorkToken,
    pub graph: G,
    pub options: MeasureOptions,
}

#[derive(Clone, Debug)]
pub struct EvalWork<G, C> {
    pub token: WorkToken,
    pub graph: G,
    pub candidates: Vec<C>,
    pub request: EvalRequest,
    pub measure_options: MeasureOptions,
    pub opponent: Option<Box<EvalOpponentWork<G>>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalOpponentWork<G> {
    pub graph: G,
    pub position: gz_eval::EvalPositionContext,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum SearchWorkResult<G, C> {
    Expand(ExpandResult<C>),
    Apply(ApplyResult<G, C>),
    Measure(MeasureResult<G>),
    Eval(EvalOutput),
}

#[derive(Clone, Debug)]
pub struct ExpandResult<C> {
    pub graph_hash: GraphHash,
    pub candidates: Vec<ExpandedCandidate<C>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ExpandedCandidate<C> {
    pub candidate: C,
    pub candidate_hash: CandidateHash,
    pub kind: CandidateKindId,
    pub tags: CandidateTags,
    pub static_prior: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EngineIdentity {
    pub engine_id: EngineId,
    pub engine_version: EngineVersion,
    pub action_set_hash: ActionSetHash,
}

impl EngineIdentity {
    #[must_use]
    pub fn from_engine<E: GraphEngine>(engine: &E) -> Self {
        Self {
            engine_id: engine.engine_id(),
            engine_version: engine.engine_version(),
            action_set_hash: engine.action_set_hash(),
        }
    }

    #[must_use]
    pub fn context(&self, graph_hash: GraphHash) -> ReplayGraphContext {
        ReplayGraphContext::new(
            PortableGraphId::new(graph_hash, self.engine_id, self.engine_version),
            self.action_set_hash,
        )
    }
}
