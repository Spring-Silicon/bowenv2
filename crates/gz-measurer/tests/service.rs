use gz_engine::{
    ActionSetHash, EngineId, EngineVersion, GraphHash, MeasureConfigHash, MeasureSummary,
    ModelVersion, PortableGraphId, PortableSearchActionRef, ReplayGraphContext, SearchConfigHash,
};
use gz_measurer::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasuredEpisode, MeasurerAdmissionStatus,
    MeasurerError, ProjectedReference, ProjectionMode, ReplayMeasurer,
};
use gz_replay::{ReplayReferenceKind, ReplayStore};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gz-measurer-service-test-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();

        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn require_reference_refuses_unlabeled_artifact_without_appending() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let mut measurer = ReplayMeasurer::new(&store, false);

    let admission = measurer
        .admit(MeasuredEpisode {
            lane: 0,
            episode_id: 7,
            artifact: artifact(1, 4.0, version(1)),
            reference: None,
            mode: ProjectionMode::RequireReference,
        })
        .unwrap();
    let summary = measurer.finish();

    assert_eq!(
        admission.status,
        MeasurerAdmissionStatus::Dropped {
            reason: MeasurerError::MissingReference
        }
    );
    assert_eq!(admission.learner_reward, Some(4.0));
    assert_eq!(store.counters().produced_rows, 0);
    assert_eq!(summary.episodes_appended, 0);
    assert_eq!(summary.episodes_dropped, 1);
}

#[test]
fn append_updates_replay_store_and_final_graph_ledger() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let mut measurer = ReplayMeasurer::new(&store, false);
    let reference = ProjectedReference {
        kind: ReplayReferenceKind::GatedPolicy,
        final_reward: 1.0,
        final_graph: Some(context(9)),
        ref_id: Some(3),
        search_config_hash: Some(SearchConfigHash::from_bytes([8; 32])),
        model_version: Some(version(1)),
        step_count: 1,
    };

    for episode_id in 0..3 {
        let final_seed = if episode_id == 2 { 2 } else { 1 };
        let admission = measurer
            .admit(MeasuredEpisode {
                lane: episode_id as usize % 2,
                episode_id,
                artifact: artifact(
                    final_seed,
                    4.0 + episode_id as f32,
                    version(episode_id as u8),
                ),
                reference: Some(reference.clone()),
                mode: ProjectionMode::RequireReference,
            })
            .unwrap();
        assert!(matches!(
            admission.status,
            MeasurerAdmissionStatus::Appended { row_count: 1 }
        ));
    }

    let summary = measurer.finish();

    assert_eq!(store.counters().produced_rows, 3);
    assert_eq!(summary.episodes_appended, 3);
    assert_eq!(summary.episodes_dropped, 0);
    assert_eq!(summary.replay_rows, 3);
    assert_eq!(summary.measure_ledger.finals, 3);
    assert_eq!(summary.measure_ledger.distinct_finals, 2);
    assert_eq!(summary.measure_ledger.repeat_finals, 1);
    assert_eq!(summary.measure_ledger.distinct_by_version.len(), 2);
    assert_eq!(summary.lanes[0].episodes_appended, 2);
    assert_eq!(summary.lanes[1].episodes_appended, 1);
}

fn artifact(final_seed: u8, reward: f32, model_version: ModelVersion) -> CompletedEpisodeArtifact {
    let state = context(final_seed);
    CompletedEpisodeArtifact {
        root: state,
        final_graph: state,
        final_measure: MeasureSummary {
            graph_hash: state.graph.graph_hash,
            config_hash: MeasureConfigHash::from_bytes([6; 32]),
            measured: true,
            valid: true,
            latency: None,
            scalar_reward: Some(reward),
            failure_code: None,
        },
        stop_selected: true,
        search_config_hash: SearchConfigHash::from_bytes([7; 32]),
        steps: vec![CompletedEpisodeStep {
            before: state,
            after: state,
            selected_action: PortableSearchActionRef::stop(state),
            legal_actions: vec![PortableSearchActionRef::stop(state)],
            policy_target: vec![1.0],
            model_version: Some(model_version),
        }],
        feature_rows: None,
    }
}

fn version(seed: u8) -> ModelVersion {
    ModelVersion::from_bytes([seed; 16])
}

fn context(seed: u8) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(
            GraphHash::from_bytes([seed; 32]),
            EngineId::from_bytes([1; 16]),
            EngineVersion::from_bytes([2; 16]),
        ),
        ActionSetHash::from_bytes([3; 32]),
    )
}
