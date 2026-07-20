mod common;

use common::episode_with_rows;
use gz_replay::{ReplayError, ReplayStore, SampleConfig};
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
