use gz_engine::{
    ActionSetHash, ApplyMetrics, ApplyValidationError, CandidateHash, CandidateInfo,
    CandidateInfoError, CandidateKindId, CandidateMetadata, CandidateTags, ErrorCode, ErrorMessage,
    GraphHash, LatencyStats, MeasureConfigHash, MeasureFailure, MeasureMetadata, MeasureResult,
    MeasureSummary, MeasurementValidationError, SubjectId,
};

fn graph_hash(byte: u8) -> GraphHash {
    GraphHash::from_bytes([byte; 32])
}

fn candidate_hash(byte: u8) -> CandidateHash {
    CandidateHash::from_bytes([byte; 32])
}

fn config_hash(byte: u8) -> MeasureConfigHash {
    MeasureConfigHash::from_bytes([byte; 32])
}

fn candidate_info(static_prior: f32) -> CandidateInfo {
    CandidateInfo {
        candidate_hash: candidate_hash(1),
        graph_hash: graph_hash(2),
        action_set_hash: ActionSetHash::from_bytes([3; 32]),
        kind: CandidateKindId::new(4),
        display_name: "candidate".to_owned(),
        static_prior,
        tags: CandidateTags::EMPTY,
        subjects: vec![SubjectId::new(5)],
        metadata: CandidateMetadata { bytes: vec![6, 7] },
    }
}

#[test]
fn candidate_info_rejects_nan_static_prior() {
    assert!(matches!(
        candidate_info(f32::NAN).validate().unwrap_err(),
        CandidateInfoError::NonFiniteStaticPrior { .. }
    ));
}

#[test]
fn apply_metrics_reject_invalid_elapsed_ms() {
    assert!(matches!(
        ApplyMetrics::new(Some(f32::INFINITY), None).unwrap_err(),
        ApplyValidationError::NonFiniteElapsedMs { .. }
    ));
    assert_eq!(
        ApplyMetrics::new(Some(-1.0), None).unwrap_err(),
        ApplyValidationError::NegativeElapsedMs { elapsed_ms: -1.0 }
    );
}

#[test]
fn latency_stats_reject_invalid_values() {
    assert!(matches!(
        LatencyStats::new(f32::NAN, 1.0, 1.0, Vec::new()).unwrap_err(),
        MeasurementValidationError::NonFiniteLatency {
            field: "mean_ms",
            ..
        }
    ));
    assert_eq!(
        LatencyStats::new(1.0, -1.0, 1.0, Vec::new()).unwrap_err(),
        MeasurementValidationError::NegativeLatency {
            field: "median_ms",
            value: -1.0,
        }
    );
    assert!(matches!(
        LatencyStats::new(1.0, 1.0, 1.0, vec![f32::INFINITY]).unwrap_err(),
        MeasurementValidationError::InvalidLatencySample { index: 0, .. }
    ));
}

#[test]
fn latency_stats_from_samples_computes_summary() {
    let stats = LatencyStats::from_samples(vec![4.0, 1.0, 2.0, 3.0]).unwrap();

    assert_eq!(stats.mean_ms, 2.5);
    assert_eq!(stats.median_ms, 2.0);
    assert_eq!(stats.p95_ms, 4.0);
    assert_eq!(stats.samples_ms, vec![4.0, 1.0, 2.0, 3.0]);
}

#[test]
fn measure_result_rejects_nan_scalar_reward() {
    let result = MeasureResult {
        graph: 1u32,
        graph_hash: graph_hash(1),
        config_hash: config_hash(2),
        measured: true,
        valid: false,
        latency: None,
        scalar_reward: Some(f32::NAN),
        failure: None,
        metadata: MeasureMetadata::default(),
    };

    assert!(matches!(
        result.validate().unwrap_err(),
        MeasurementValidationError::NonFiniteScalarReward { .. }
    ));
}

#[test]
fn measure_summary_drops_engine_local_graph_handle() {
    let result = MeasureResult {
        graph: "local-handle",
        graph_hash: graph_hash(1),
        config_hash: config_hash(2),
        measured: true,
        valid: true,
        latency: Some(LatencyStats::from_samples(vec![1.0]).unwrap()),
        scalar_reward: Some(3.0),
        failure: Some(MeasureFailure {
            code: ErrorCode::new(5),
            message: ErrorMessage::new("failure").unwrap(),
        }),
        metadata: MeasureMetadata { bytes: vec![8, 9] },
    };

    let summary = MeasureSummary::from(&result);

    assert_eq!(summary.graph_hash, graph_hash(1));
    assert_eq!(summary.config_hash, config_hash(2));
    assert_eq!(summary.failure_code, Some(ErrorCode::new(5)));
    assert_eq!(summary.scalar_reward, Some(3.0));
}
