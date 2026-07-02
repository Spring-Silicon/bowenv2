use gz_cli::selfplay::{ReferenceMode, SelfplayConfig, run};
use gz_replay::ReplayStore;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("gz-cli-selfplay-test-{}-{id}", std::process::id()));
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
fn selfplay_run_writes_replay_rows() {
    let dir = TestDir::new();
    let summary = run(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 4,
        lanes: 2,
        workers_per_lane: 2,
        reference: ReferenceMode::Root,
        seed: 3,
        max_steps: 2,
        simulations: 2,
        max_batch: 4,
    })
    .unwrap();
    let store = ReplayStore::open(dir.path()).unwrap();
    let counters = store.counters();

    assert_eq!(summary.counters, counters);
    assert_eq!(summary.rows_produced, counters.produced_rows);
    assert_eq!(summary.episodes_appended + summary.episodes_dropped, 4);
    assert!(summary.rows_produced > 0);
}
