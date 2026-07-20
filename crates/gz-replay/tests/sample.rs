mod common;

use common::episode_with_rows;
use gz_replay::{ReplayError, ReplayStore, SampleConfig, SampleKind};
use std::num::{NonZeroU64, NonZeroUsize};

fn sample_config(batch: usize, window_rows: u64, seed: u64) -> SampleConfig {
    SampleConfig {
        batch: NonZeroUsize::new(batch).unwrap(),
        window_rows: NonZeroU64::new(window_rows).unwrap(),
        seed,
    }
}

#[test]
fn sampling_empty_store_returns_empty() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();

    assert_eq!(
        store.sample_rows(sample_config(1, 1, 0)).unwrap_err(),
        ReplayError::Empty
    );
}

#[test]
fn sampling_is_deterministic_for_fixed_seed_and_contents() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_rows(5);
    store.append_episode(&record, &rows).unwrap();

    let left = store.sample_rows(sample_config(8, 5, 99)).unwrap();
    let right = store.sample_rows(sample_config(8, 5, 99)).unwrap();

    assert_eq!(left, right);
}

#[test]
fn sampling_respects_window_rows() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_rows(4);
    store.append_episode(&record, &rows).unwrap();

    let sample = store.sample_rows(sample_config(20, 1, 7)).unwrap();

    assert!(sample.iter().all(|(_, row)| row.step_index == 3));
}

#[test]
fn sampling_returns_episode_id_with_rows() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_rows(1);
    let id = store.append_episode(&record, &rows).unwrap();

    let sample = store.sample_rows(sample_config(1, 1, 0)).unwrap();

    assert_eq!(sample, vec![(id, rows[0].clone())]);
}

#[test]
fn policy_and_value_streams_filter_competitive_rows() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (primary_record, primary_rows) = episode_with_rows(2);
    let (mut secondary_record, mut secondary_rows) = episode_with_rows(2);
    secondary_record.final_measure = common::measure(Some(4.0), true, true);
    secondary_record.outcome.learner_reward = 4.0;
    secondary_record.outcome.value_target = Some(-1.0);
    secondary_record.outcome.reference.as_mut().unwrap().reward = 5.0;
    for row in &mut secondary_rows {
        row.final_measure = secondary_record.final_measure.clone();
        row.reward_target = Some(4.0);
        row.value_target = Some(-1.0);
        row.policy_target.fill(0.0);
    }
    let (primary_id, secondary_id) = store
        .append_episode_pair(
            (&primary_record, &primary_rows),
            (&secondary_record, &secondary_rows),
        )
        .unwrap();
    assert_eq!(store.counters().produced_rows, 4);
    assert_eq!(store.counters().produced_policy_rows, 2);

    let policy = store
        .sample_rows_kind(sample_config(16, 4, 7), SampleKind::Policy)
        .unwrap();
    assert!(policy.iter().all(|(id, row)| {
        *id == primary_id && row.policy_target.iter().any(|target| *target > 0.0)
    }));
    let recent_policy = store
        .sample_rows_kind(sample_config(16, 2, 7), SampleKind::Policy)
        .unwrap();
    assert!(recent_policy.iter().all(|(id, _)| *id == primary_id));
    let value = store
        .sample_rows_kind(sample_config(32, 4, 11), SampleKind::Value)
        .unwrap();
    assert!(value.iter().all(|(_, row)| row.value_target.is_some()));
    assert!(value.iter().any(|(id, _)| *id == secondary_id));
}
