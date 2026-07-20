use gz_engine::{
    ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash, MeasureConfigHash,
    MeasureSummary, ModelVersion, PortableCandidateRef, PortableGraphId, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash,
};
use gz_measurer::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasuredCompetitiveGame, MeasuredEpisode,
    MeasuredSymmetricGame, MeasurerAdmissionStatus, MeasurerError, ProjectedReference,
    ProjectionMode, ReplayMeasurer, ValueTargetConfig, horizon_value_targets,
};
use gz_replay::{ReplayEpisodeId, ReplayReferenceKind, ReplayStore};
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
            root_reward: -10.0,
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
                root_reward: -10.0,
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

#[test]
fn competitive_tie_labels_p1_as_winner_and_appends_one_game() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let mut measurer = ReplayMeasurer::new(&store, false);
    let mut p2 = artifact(2, 4.0, version(2));
    p2.steps[0].policy_target[0] = 0.0;

    let admission = measurer
        .admit_competitive(MeasuredCompetitiveGame {
            lane: 0,
            game_id: 9,
            learner_is_p1: false,
            root_reward: -10.0,
            p1_artifact: artifact(1, 4.0, version(1)),
            p1_reference: reference(4.0, 2),
            p2_artifact: p2,
            p2_reference: reference(4.0, 1),
        })
        .unwrap();
    let summary = measurer.finish();

    assert_eq!(admission.learner_reward, Some(4.0));
    assert_eq!(
        admission.status,
        MeasurerAdmissionStatus::Appended { row_count: 2 }
    );
    assert_eq!(store.episode_counters(), (1, 1));
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(0))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(-1.0)
    );
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(1))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(1.0)
    );
    assert_eq!(summary.episodes_appended, 1);
    assert_eq!(summary.replay_rows, 2);
    assert_eq!(summary.measure_ledger.finals, 2);
}

#[test]
fn competitive_length_tiebreak_labels_shorter_p2_as_winner() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let mut measurer = ReplayMeasurer::new(&store, true);
    let mut p1 = artifact(1, 4.0, version(1));
    p1.steps.push(p1.steps[0].clone());

    measurer
        .admit_competitive(MeasuredCompetitiveGame {
            lane: 0,
            game_id: 9,
            learner_is_p1: true,
            root_reward: -10.0,
            p1_artifact: p1,
            p1_reference: reference(4.0, 2),
            p2_artifact: artifact(2, 4.0, version(2)),
            p2_reference: reference(4.0, 1),
        })
        .unwrap();
    let _ = measurer.finish();

    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(0))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(-1.0)
    );
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(1))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(1.0)
    );
}

#[test]
fn competitive_value_sign_accuracy_uses_learner_role_and_step_40_boundary() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let mut measurer = ReplayMeasurer::new(&store, false);
    let mut learner = artifact(2, 4.0, version(2));
    let template = learner.steps[0].clone();
    learner.steps = (0..42)
        .map(|step| {
            let mut artifact = template.clone();
            artifact.root_value = Some(match step {
                0..30 | 40 => -0.5,
                _ => 0.5,
            });
            artifact
        })
        .collect();

    measurer
        .admit_competitive(MeasuredCompetitiveGame {
            lane: 0,
            game_id: 10,
            learner_is_p1: false,
            root_reward: -10.0,
            p1_artifact: artifact(1, 5.0, version(1)),
            p1_reference: reference(4.0, 2),
            p2_artifact: learner,
            p2_reference: reference(5.0, 1),
        })
        .unwrap();

    let (early, late) = store.value_sign_accuracy_emas();
    assert!((early.unwrap() - 0.75).abs() < 1.0e-9);
    assert!((late.unwrap() - 0.5).abs() < 1.0e-9);
}

#[test]
fn sampled_tree_replay_keeps_learner_rows_independent_of_opponent_length() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    store
        .ensure_data_mode(gz_replay::ReplayDataMode::SampledTree)
        .unwrap();
    let mut measurer = ReplayMeasurer::new(&store, false);
    let mut opponent = artifact(2, 5.0, version(2));
    opponent.steps = vec![opponent.steps[0].clone(); 96];

    let admission = measurer
        .admit_competitive(MeasuredCompetitiveGame {
            lane: 0,
            game_id: 11,
            learner_is_p1: true,
            root_reward: -10.0,
            p1_artifact: artifact(1, 4.0, version(1)),
            p1_reference: reference(5.0, 2),
            p2_artifact: opponent,
            p2_reference: reference(4.0, 1),
        })
        .unwrap();

    assert_eq!(
        admission.status,
        MeasurerAdmissionStatus::Appended { row_count: 1 }
    );
    assert_eq!(store.counters().produced_rows, 1);
    assert_eq!(store.counters().produced_policy_rows, 1);
    assert_eq!(store.counters().produced_value_rows, 1);
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(0))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(-1.0)
    );
    assert!(store.episode(ReplayEpisodeId::new(1)).unwrap().is_none());
}

#[test]
fn graded_targets_match_root_normalized_whittle_margin() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    store
        .ensure_data_mode(gz_replay::ReplayDataMode::graded(false, 0.1).unwrap())
        .unwrap();
    let mut measurer =
        ReplayMeasurer::with_value_target(&store, false, ValueTargetConfig::graded(0.1));

    measurer
        .admit(MeasuredEpisode {
            lane: 0,
            episode_id: 7,
            artifact: artifact(1, -80.0, version(1)),
            root_reward: -100.0,
            reference: Some(reference(-90.0, 2)),
            mode: ProjectionMode::RequireReference,
        })
        .unwrap();
    let _ = measurer.finish();

    let target = store
        .episode(ReplayEpisodeId::new(0))
        .unwrap()
        .unwrap()
        .outcome
        .value_target
        .unwrap();
    assert!((target - 1.0_f32.tanh()).abs() < 1.0e-6);
}

#[test]
fn single_vanilla_projects_the_measured_reward_without_a_reference() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    store
        .ensure_data_mode(gz_replay::ReplayDataMode::SingleVanilla)
        .unwrap();
    let mut measurer =
        ReplayMeasurer::with_value_target(&store, false, ValueTargetConfig::SingleVanilla);

    let admission = measurer
        .admit(MeasuredEpisode {
            lane: 0,
            episode_id: 7,
            artifact: artifact(1, -80.0, version(1)),
            root_reward: 0.0,
            reference: None,
            mode: ProjectionMode::AllowUnlabeled,
        })
        .unwrap();

    assert_eq!(
        admission.status,
        MeasurerAdmissionStatus::Appended { row_count: 1 }
    );
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(0))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(-80.0)
    );
    assert_eq!(store.win_rate_ema(), None);
}

#[test]
fn single_vanilla_rejects_reference_projection() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    store
        .ensure_data_mode(gz_replay::ReplayDataMode::SingleVanilla)
        .unwrap();
    let mut measurer =
        ReplayMeasurer::with_value_target(&store, false, ValueTargetConfig::SingleVanilla);

    let admission = measurer
        .admit(MeasuredEpisode {
            lane: 0,
            episode_id: 7,
            artifact: artifact(1, -80.0, version(1)),
            root_reward: 0.0,
            reference: Some(reference(-90.0, 2)),
            mode: ProjectionMode::AllowUnlabeled,
        })
        .unwrap();

    assert_eq!(
        admission.status,
        MeasurerAdmissionStatus::Dropped {
            reason: MeasurerError::UnexpectedReference
        }
    );
    assert_eq!(store.counters().produced_rows, 0);
}

#[test]
fn graded_competitive_tie_is_neutral() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    store
        .ensure_data_mode(gz_replay::ReplayDataMode::graded(true, 0.1).unwrap())
        .unwrap();
    let mut measurer =
        ReplayMeasurer::with_value_target(&store, false, ValueTargetConfig::graded(0.1));

    measurer
        .admit_competitive(MeasuredCompetitiveGame {
            lane: 0,
            game_id: 9,
            learner_is_p1: true,
            root_reward: -100.0,
            p1_artifact: artifact(1, -80.0, version(1)),
            p1_reference: reference(-80.0, 2),
            p2_artifact: artifact(2, -80.0, version(2)),
            p2_reference: reference(-80.0, 1),
        })
        .unwrap();
    let _ = measurer.finish();

    assert_eq!(store.counters().produced_rows, 1);
    assert_eq!(store.episode_counters(), (1, 1));
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(0))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(0.0)
    );
    assert!(store.episode(ReplayEpisodeId::new(1)).unwrap().is_none());
}

#[test]
fn symmetric_game_uses_reward_then_rewrite_count_and_preserves_draws() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    store
        .ensure_data_mode(gz_replay::ReplayDataMode::SymmetricSelfplay)
        .unwrap();
    let mut measurer = ReplayMeasurer::new(&store, true);

    let admission = measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 0,
            game_id: 9,
            p1_artifact: symmetric_artifact(1, 4.0, 2, version(1)),
            p2_artifact: symmetric_artifact(2, 4.0, 1, version(2)),
        })
        .unwrap();
    assert_eq!(admission.learner_reward, None);
    assert_eq!(
        admission.status,
        MeasurerAdmissionStatus::Appended { row_count: 3 }
    );
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(0))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(-1.0)
    );
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(1))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(1.0)
    );

    measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 0,
            game_id: 10,
            p1_artifact: symmetric_artifact(3, 5.0, 1, version(3)),
            p2_artifact: symmetric_artifact(4, 5.0, 1, version(4)),
        })
        .unwrap();
    let summary = measurer.finish();
    assert_eq!(summary.measure_ledger.finals, 4);

    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(2))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(0.0)
    );
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(3))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(0.0)
    );
    assert_eq!(store.episode_counters(), (2, 0));
}

#[test]
fn symmetric_stop_row_does_not_count_as_a_rewrite_tiebreak_step() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    store
        .ensure_data_mode(gz_replay::ReplayDataMode::SymmetricSelfplayStop)
        .unwrap();
    let mut measurer = ReplayMeasurer::new(&store, true);

    measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 0,
            game_id: 11,
            p1_artifact: stop_enabled_symmetric_artifact(5, 4.0, 1, true, version(1)),
            p2_artifact: stop_enabled_symmetric_artifact(6, 4.0, 2, false, version(2)),
        })
        .unwrap();

    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(0))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(1.0)
    );
    assert_eq!(
        store
            .episode(ReplayEpisodeId::new(1))
            .unwrap()
            .unwrap()
            .outcome
            .value_target,
        Some(-1.0)
    );
    let metrics = store.symmetric_selfplay_metrics().unwrap();
    assert_eq!(metrics.p1_episode_len_ema, 1.0);
    assert_eq!(metrics.p2_episode_len_ema, 2.0);
}

#[test]
fn horizon_targets_follow_v8_v32_recurrence_and_negate_with_perspective() {
    let mut artifact = symmetric_artifact(7, -4.0, 3, version(1));
    let root_values = [0.0, 0.5, -0.25];
    for (step, value) in artifact.steps.iter_mut().zip(root_values) {
        step.root_search_value = Some(value);
    }

    let targets = horizon_value_targets(&artifact, 1.0).unwrap();
    for (head, lambda) in [8.0_f32 / 9.0, 32.0 / 33.0].into_iter().enumerate() {
        let expected_2 = (1.0 - lambda) * root_values[2] + lambda;
        let expected_1 = (1.0 - lambda) * root_values[1] + lambda * expected_2;
        let expected_0 = (1.0 - lambda) * root_values[0] + lambda * expected_1;
        for (actual, expected) in [targets[0][head], targets[1][head], targets[2][head]]
            .into_iter()
            .zip([expected_0, expected_1, expected_2])
        {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    for step in &mut artifact.steps {
        step.root_search_value = step.root_search_value.map(|value| -value);
    }
    let mirrored = horizon_value_targets(&artifact, -1.0).unwrap();
    for (target, mirrored) in targets.iter().zip(mirrored) {
        assert!((target[0] + mirrored[0]).abs() < 1.0e-6);
        assert!((target[1] + mirrored[1]).abs() < 1.0e-6);
    }
}

#[test]
fn horizon_targets_reject_missing_or_out_of_range_search_values() {
    let mut artifact = symmetric_artifact(8, -4.0, 1, version(1));
    artifact.steps[0].root_search_value = None;
    assert_eq!(
        horizon_value_targets(&artifact, 1.0),
        Err(MeasurerError::InvalidRootSearchValue)
    );

    artifact.steps[0].root_search_value = Some(1.01);
    assert_eq!(
        horizon_value_targets(&artifact, 1.0),
        Err(MeasurerError::InvalidRootSearchValue)
    );

    artifact.steps[0].root_search_value = Some(0.0);
    assert_eq!(
        horizon_value_targets(&artifact, f32::NAN),
        Err(MeasurerError::InvalidRootSearchValue)
    );
}

#[test]
fn horizon_targets_handle_empty_single_and_constant_traces() {
    let mut empty = symmetric_artifact(9, -4.0, 0, version(1));
    assert!(horizon_value_targets(&empty, -0.5).unwrap().is_empty());

    let action = PortableSearchActionRef::candidate(PortableCandidateRef::new(
        context(9),
        CandidateHash::from_bytes([9; 32]),
    ));
    empty.steps.push(CompletedEpisodeStep {
        before: context(9),
        after: context(10),
        selected_action: action,
        legal_actions: vec![action],
        policy_target: vec![1.0],
        root_value: Some(0.5),
        root_search_value: Some(0.5),
        model_version: Some(version(1)),
    });
    let single = horizon_value_targets(&empty, -0.5).unwrap()[0];
    assert!((single[0] - ((1.0 / 9.0) * 0.5 + (8.0 / 9.0) * -0.5)).abs() < 1.0e-6);
    assert!((single[1] - ((1.0 / 33.0) * 0.5 + (32.0 / 33.0) * -0.5)).abs() < 1.0e-6);

    let mut constant = symmetric_artifact(10, -4.0, 4, version(1));
    for step in &mut constant.steps {
        step.root_search_value = Some(0.25);
    }
    assert!(
        horizon_value_targets(&constant, 0.25)
            .unwrap()
            .iter()
            .flatten()
            .all(|value| (*value - 0.25).abs() < 1.0e-6)
    );
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
            root_value: None,
            root_search_value: None,
            model_version: Some(model_version),
        }],
        feature_rows: None,
    }
}

fn symmetric_artifact(
    final_seed: u8,
    reward: f32,
    step_count: usize,
    model_version: ModelVersion,
) -> CompletedEpisodeArtifact {
    let state = context(final_seed);
    let action = PortableSearchActionRef::candidate(PortableCandidateRef::new(
        state,
        CandidateHash::from_bytes([final_seed; 32]),
    ));
    let step = CompletedEpisodeStep {
        before: state,
        after: state,
        selected_action: action,
        legal_actions: vec![action],
        policy_target: vec![1.0],
        root_value: None,
        root_search_value: Some(0.25),
        model_version: Some(model_version),
    };
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
        stop_selected: false,
        search_config_hash: SearchConfigHash::from_bytes([7; 32]),
        steps: vec![step; step_count],
        feature_rows: None,
    }
}

fn stop_enabled_symmetric_artifact(
    final_seed: u8,
    reward: f32,
    rewrites: usize,
    stopped: bool,
    model_version: ModelVersion,
) -> CompletedEpisodeArtifact {
    let mut artifact = symmetric_artifact(final_seed, reward, rewrites, model_version);
    let state = artifact.final_graph;
    for step in &mut artifact.steps {
        step.legal_actions
            .push(PortableSearchActionRef::stop(state));
        step.policy_target.push(0.0);
    }
    if stopped {
        artifact.stop_selected = true;
        artifact.steps.push(CompletedEpisodeStep {
            before: state,
            after: state,
            selected_action: PortableSearchActionRef::stop(state),
            legal_actions: vec![PortableSearchActionRef::stop(state)],
            policy_target: vec![1.0],
            root_value: None,
            root_search_value: Some(0.25),
            model_version: Some(model_version),
        });
    }
    artifact
}

fn version(seed: u8) -> ModelVersion {
    ModelVersion::from_bytes([seed; 16])
}

fn reference(reward: f32, final_seed: u8) -> ProjectedReference {
    ProjectedReference {
        kind: ReplayReferenceKind::GatedPolicy,
        final_reward: reward,
        final_graph: Some(context(final_seed)),
        ref_id: None,
        search_config_hash: Some(SearchConfigHash::from_bytes([7; 32])),
        model_version: None,
        step_count: 1,
    }
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
