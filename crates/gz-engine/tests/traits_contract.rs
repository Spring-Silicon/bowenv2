use gz_engine::{
    ActionSetHash, ApplyJob, ApplyMetrics, ApplyResult, BatchGraphEngine, CandidateHash,
    CandidateInfo, CandidateKindId, CandidateMetadata, CandidateOptions, CandidateTags,
    ContractError, EngineContractFixture, EngineError, EngineId, EngineResult, EngineVersion,
    ErrorCode, ErrorMessage, GraphArtifact, GraphArtifactFormat, GraphEngine, GraphHash,
    LatencyStats, MeasureConfigHash, MeasureMetadata, MeasureOptions, MeasureResult,
    MeasureSummary, SubjectId, run_engine_contract,
};

#[derive(Clone)]
struct TinyEngine {
    fail_graph: u8,
}

impl TinyEngine {
    fn graph_hash(graph: u8) -> GraphHash {
        GraphHash::from_bytes([graph; 32])
    }

    fn candidate_hash(graph: u8, candidate: u8) -> CandidateHash {
        let mut bytes = [0; 32];
        bytes[0] = graph;
        bytes[1] = candidate;
        CandidateHash::from_bytes(bytes)
    }

    fn measure_options() -> MeasureOptions {
        MeasureOptions::new(MeasureConfigHash::from_bytes([9; 32]), 1, None, true).unwrap()
    }
}

impl GraphEngine for TinyEngine {
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
        if graph == self.fail_graph {
            Err(EngineError::UnknownGraph {
                graph_hash: Some(Self::graph_hash(graph)),
            })
        } else {
            Ok(Self::graph_hash(graph))
        }
    }

    fn candidates(
        &mut self,
        graph: Self::Graph,
        options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()> {
        self.hash(graph)?;
        out.clear();
        out.extend([graph + 1, graph + 2]);
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
        self.hash(graph)?;
        if candidate == 99 || candidate == 251 {
            return Err(EngineError::UnknownCandidate {
                candidate_hash: Some(Self::candidate_hash(graph, candidate)),
            });
        }

        Ok(CandidateInfo {
            candidate_hash: Self::candidate_hash(graph, candidate),
            graph_hash: Self::graph_hash(graph),
            action_set_hash: self.action_set_hash(),
            kind: CandidateKindId::new(1),
            display_name: format!("candidate-{candidate}"),
            static_prior: 0.0,
            tags: CandidateTags::EMPTY,
            subjects: vec![SubjectId::new(candidate.into())],
            metadata: CandidateMetadata::default(),
        })
    }

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>> {
        let before_hash = self.hash(graph)?;
        let info = self.candidate_info(graph, candidate)?;
        let after = candidate;
        Ok(ApplyResult {
            before: graph,
            after,
            before_hash,
            after_hash: Self::graph_hash(after),
            candidate,
            candidate_hash: info.candidate_hash,
            changed: graph != after,
            rejected: None,
            metrics: ApplyMetrics::default(),
        })
    }

    fn measure(
        &mut self,
        graph: Self::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<Self::Graph>> {
        Ok(MeasureResult {
            graph,
            graph_hash: self.hash(graph)?,
            config_hash: options.config_hash,
            measured: true,
            valid: true,
            latency: Some(LatencyStats::from_samples(vec![1.0]).unwrap()),
            scalar_reward: Some(f32::from(graph)),
            failure: None,
            metadata: MeasureMetadata::default(),
        })
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact> {
        Ok(GraphArtifact {
            graph_hash: self.hash(graph)?,
            format: GraphArtifactFormat::Text,
            bytes: vec![graph],
        })
    }
}

impl BatchGraphEngine for TinyEngine {}

#[test]
fn default_candidates_batch_matches_ordered_single_calls() {
    let mut engine = TinyEngine { fail_graph: 255 };
    let graphs = [0, 1];
    let batch = engine.candidates_batch(&graphs, CandidateOptions::default());

    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].as_ref().unwrap(), &vec![1, 2]);
    assert_eq!(batch[1].as_ref().unwrap(), &vec![2, 3]);
}

#[test]
fn default_apply_batch_matches_ordered_single_calls() {
    let mut engine = TinyEngine { fail_graph: 255 };
    let jobs = [ApplyJob::new(0, 1), ApplyJob::new(1, 2)];
    let batch = engine.apply_batch(&jobs);

    assert_eq!(batch.len(), 2);
    assert_eq!(
        batch[0].as_ref().unwrap().after_hash,
        TinyEngine::graph_hash(1)
    );
    assert_eq!(
        batch[1].as_ref().unwrap().after_hash,
        TinyEngine::graph_hash(2)
    );
}

#[test]
fn default_measure_batch_matches_ordered_single_calls() {
    let mut engine = TinyEngine { fail_graph: 255 };
    let batch = engine.measure_batch(&[0, 1], TinyEngine::measure_options());
    let summaries: Vec<_> = batch
        .into_iter()
        .map(|result| MeasureSummary::from(&result.unwrap()).graph_hash)
        .collect();

    assert_eq!(
        summaries,
        vec![TinyEngine::graph_hash(0), TinyEngine::graph_hash(1)]
    );
}

#[test]
fn default_batch_failure_does_not_stop_later_rows() {
    let mut engine = TinyEngine { fail_graph: 1 };
    let batch = engine.candidates_batch(&[1, 2], CandidateOptions::default());

    assert!(matches!(batch[0], Err(EngineError::UnknownGraph { .. })));
    assert_eq!(batch[1].as_ref().unwrap(), &vec![3, 4]);
}

#[test]
fn candidates_implementation_clears_stale_output() {
    let mut engine = TinyEngine { fail_graph: 255 };
    let mut out = vec![99];

    engine
        .candidates(0, CandidateOptions::default(), &mut out)
        .unwrap();

    assert_eq!(out, vec![1, 2]);
}

struct Fixture;

impl EngineContractFixture for Fixture {
    type Engine = TinyEngine;

    fn make_engine(&self) -> Self::Engine {
        TinyEngine { fail_graph: 250 }
    }

    fn measure_options(&self) -> MeasureOptions {
        TinyEngine::measure_options()
    }

    fn known_path(&self) -> Vec<<Self::Engine as GraphEngine>::Candidate> {
        vec![1, 2]
    }

    fn unknown_graph(&self) -> Option<<Self::Engine as GraphEngine>::Graph> {
        Some(250)
    }

    fn unknown_candidate(&self) -> Option<<Self::Engine as GraphEngine>::Candidate> {
        Some(251)
    }
}

#[test]
fn contract_harness_accepts_valid_engine_fixture() {
    run_engine_contract(&Fixture).unwrap();
}

#[test]
fn contract_error_display_delegates() {
    let error = ContractError::AssertionFailed("failed");
    assert_eq!(error.to_string(), "failed");

    let error = ContractError::Engine(EngineError::Internal {
        code: ErrorCode::new(1),
        message: ErrorMessage::new("bad").unwrap(),
    });
    assert!(error.to_string().contains("internal engine error"));
}
