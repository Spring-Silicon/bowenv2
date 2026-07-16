use gz_engine::{
    ActionSetHash, ApplyMetrics, ApplyResult, CandidateHash, CandidateInfo, CandidateKindId,
    CandidateMetadata, CandidateOptions, CandidateTags, EngineId, EngineResult, EngineVersion,
    GraphArtifact, GraphArtifactFormat, GraphEngine, GraphHash, MeasureConfigHash, MeasureMetadata,
    MeasureOptions, MeasureResult, PortableCandidateRef, PortableGraphId, PortableSearchActionRef,
    ReplayGraphContext, RewriteRejection,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct ApplyCase {
    after: u8,
    rejected: bool,
}

#[derive(Clone, Copy)]
struct MeasureCase {
    measured: bool,
    valid: bool,
    reward: Option<f32>,
}

pub(crate) struct TestEngine {
    candidates: BTreeMap<u8, Vec<u8>>,
    applies: BTreeMap<(u8, u8), ApplyCase>,
    measures: BTreeMap<u8, MeasureCase>,
    priors: BTreeMap<u8, f32>,
    pub(crate) apply_calls: Vec<(u8, u8)>,
    pub(crate) measure_calls: Vec<u8>,
    pub(crate) released_graphs: Vec<u8>,
    pub(crate) released_candidates: Vec<u8>,
}

impl TestEngine {
    pub(crate) fn new() -> Self {
        Self {
            candidates: BTreeMap::new(),
            applies: BTreeMap::new(),
            measures: BTreeMap::new(),
            priors: BTreeMap::new(),
            apply_calls: Vec::new(),
            measure_calls: Vec::new(),
            released_graphs: Vec::new(),
            released_candidates: Vec::new(),
        }
    }

    pub(crate) fn candidates(mut self, graph: u8, candidates: impl Into<Vec<u8>>) -> Self {
        self.candidates.insert(graph, candidates.into());
        self
    }

    #[allow(dead_code)]
    pub(crate) fn apply(mut self, graph: u8, candidate: u8, after: u8) -> Self {
        self.applies.insert(
            (graph, candidate),
            ApplyCase {
                after,
                rejected: false,
            },
        );
        self
    }

    #[allow(dead_code)]
    pub(crate) fn rejected(mut self, graph: u8, candidate: u8) -> Self {
        self.applies.insert(
            (graph, candidate),
            ApplyCase {
                after: graph,
                rejected: true,
            },
        );
        self
    }

    pub(crate) fn reward(mut self, graph: u8, reward: f32) -> Self {
        self.measures.insert(
            graph,
            MeasureCase {
                measured: true,
                valid: true,
                reward: Some(reward),
            },
        );
        self
    }

    #[allow(dead_code)]
    pub(crate) fn unscored(mut self, graph: u8) -> Self {
        self.measures.insert(
            graph,
            MeasureCase {
                measured: true,
                valid: false,
                reward: Some(0.0),
            },
        );
        self
    }

    #[allow(dead_code)]
    pub(crate) fn prior(mut self, candidate: u8, prior: f32) -> Self {
        self.priors.insert(candidate, prior);
        self
    }

    pub(crate) fn graph_hash(graph: u8) -> GraphHash {
        GraphHash::from_bytes([graph; 32])
    }

    pub(crate) fn candidate_hash(graph: u8, candidate: u8) -> CandidateHash {
        let mut bytes = [0; 32];
        bytes[0] = graph;
        bytes[1] = candidate;
        CandidateHash::from_bytes(bytes)
    }
}

impl GraphEngine for TestEngine {
    type Graph = u8;
    type Candidate = u8;

    fn engine_id(&self) -> EngineId {
        EngineId::from_bytes([1; 16])
    }

    fn engine_version(&self) -> EngineVersion {
        EngineVersion::from_bytes([2; 16])
    }

    fn action_set_hash(&self) -> ActionSetHash {
        ActionSetHash::from_bytes([3; 32])
    }

    fn root(&self) -> Self::Graph {
        0
    }

    fn hash(&self, graph: Self::Graph) -> EngineResult<GraphHash> {
        Ok(Self::graph_hash(graph))
    }

    fn candidates(
        &mut self,
        graph: Self::Graph,
        options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()> {
        out.clear();
        out.extend(self.candidates.get(&graph).into_iter().flatten().copied());

        if let Some(max_candidates) = options.max_candidates {
            out.truncate(max_candidates);
        }

        Ok(())
    }

    fn candidate_info(
        &self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<CandidateInfo> {
        Ok(CandidateInfo {
            candidate_hash: Self::candidate_hash(graph, candidate),
            graph_hash: Self::graph_hash(graph),
            action_set_hash: self.action_set_hash(),
            kind: CandidateKindId::new(candidate.into()),
            display_name: format!("candidate-{candidate}"),
            static_prior: self.priors.get(&candidate).copied().unwrap_or(0.0),
            tags: CandidateTags::EMPTY,
            subjects: Vec::new(),
            metadata: CandidateMetadata::default(),
        })
    }

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>> {
        self.apply_calls.push((graph, candidate));

        let case = self
            .applies
            .get(&(graph, candidate))
            .copied()
            .unwrap_or(ApplyCase {
                after: candidate,
                rejected: false,
            });

        Ok(ApplyResult {
            before: graph,
            after: case.after,
            before_hash: Self::graph_hash(graph),
            after_hash: Self::graph_hash(case.after),
            candidate,
            candidate_hash: Self::candidate_hash(graph, candidate),
            changed: graph != case.after,
            rejected: case.rejected.then(|| RewriteRejection {
                code: gz_engine::ErrorCode::new(1),
                message: gz_engine::ErrorMessage::new("rejected").unwrap(),
            }),
            metrics: ApplyMetrics::default(),
        })
    }

    fn measure(
        &mut self,
        graph: Self::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<Self::Graph>> {
        self.measure_calls.push(graph);
        let case = self.measures.get(&graph).copied().unwrap_or(MeasureCase {
            measured: true,
            valid: true,
            reward: Some(f32::from(graph)),
        });

        Ok(MeasureResult {
            graph,
            graph_hash: Self::graph_hash(graph),
            config_hash: options.config_hash,
            measured: case.measured,
            valid: case.valid,
            latency: None,
            scalar_reward: case.reward,
            failure: None,
            metadata: MeasureMetadata::default(),
        })
    }

    fn release(
        &mut self,
        graphs: &[Self::Graph],
        candidates: &[Self::Candidate],
    ) -> EngineResult<()> {
        self.released_graphs.extend_from_slice(graphs);
        self.released_candidates.extend_from_slice(candidates);
        Ok(())
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact> {
        Ok(GraphArtifact {
            graph_hash: Self::graph_hash(graph),
            format: GraphArtifactFormat::Text,
            bytes: vec![graph],
        })
    }
}

pub(crate) fn measure_options() -> MeasureOptions {
    MeasureOptions::new(MeasureConfigHash::from_bytes([9; 32]), 1, None, true).unwrap()
}

#[allow(dead_code)]
pub(crate) fn context(graph: u8) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(
            TestEngine::graph_hash(graph),
            EngineId::from_bytes([1; 16]),
            EngineVersion::from_bytes([2; 16]),
        ),
        ActionSetHash::from_bytes([3; 32]),
    )
}

#[allow(dead_code)]
pub(crate) fn candidate_ref(graph: u8, candidate: u8) -> PortableSearchActionRef {
    PortableSearchActionRef::candidate(PortableCandidateRef::new(
        context(graph),
        TestEngine::candidate_hash(graph, candidate),
    ))
}

#[allow(dead_code)]
pub(crate) fn stop_ref(graph: u8) -> PortableSearchActionRef {
    PortableSearchActionRef::stop(context(graph))
}
