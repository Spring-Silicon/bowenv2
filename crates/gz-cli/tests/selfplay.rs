use gz_cli::selfplay::{EvaluatorMode, ReplayInitConfig, SelfplayConfig, init_replay, run};
use gz_replay::{ReplayDataMode, ReplayEpisodeId, ReplayStore};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("gz-cli-selfplay-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn replay_init_persists_the_feature_schema() {
    let dir = TestDir::new();
    let summary = init_replay(ReplayInitConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        max_candidates: 255,
    })
    .unwrap();

    assert_eq!(summary.max_actions, 256);
    let store = ReplayStore::open(dir.path()).unwrap();
    let schema = store.feature_schema().unwrap().unwrap();
    assert_eq!(schema.max_actions, 256);
}

#[test]
fn stub_selfplay_appends_both_symmetric_perspectives() {
    let dir = TestDir::new();
    let summary = run(short_config(dir.path())).unwrap();

    assert_eq!(summary.evaluator, EvaluatorMode::Stub);
    assert_eq!(summary.episodes_appended, 2);
    assert_eq!(summary.episodes_dropped, 0);
    assert_eq!(summary.wins + summary.losses + summary.ties, 4);
    assert_eq!(summary.wins, summary.losses);
    assert_eq!(summary.rows_produced, summary.replay_rows);
    assert!(summary.rows_produced > 0);

    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(
        store.data_mode().unwrap(),
        ReplayDataMode::SymmetricSelfplay
    );
    assert!(store.feature_schema().unwrap().is_some());
    for pair in [[0, 1], [2, 3]] {
        let left = store
            .episode(ReplayEpisodeId::new(pair[0]))
            .unwrap()
            .unwrap();
        let right = store
            .episode(ReplayEpisodeId::new(pair[1]))
            .unwrap()
            .unwrap();
        assert_eq!(
            left.outcome.value_target,
            right.outcome.value_target.map(|v| -v)
        );
        assert!(left.outcome.value_target.is_some());
        assert!(right.outcome.value_target.is_some());
    }
}

#[test]
fn stop_enabled_selfplay_uses_the_stop_replay_contract() {
    let dir = TestDir::new();
    let mut config = short_config(dir.path());
    config.episodes = 1;
    config.mask_stop = false;

    run(config).unwrap();

    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(
        store.data_mode().unwrap(),
        ReplayDataMode::SymmetricSelfplayStop
    );
    for id in 0..2 {
        let episode = store.episode(ReplayEpisodeId::new(id)).unwrap().unwrap();
        assert!(episode.outcome.value_target.is_some());
    }
}

#[test]
fn validation_rejects_incoherent_runtime_settings() {
    let dir = TestDir::new();
    let mut config = short_config(dir.path());
    config.length_tiebreak = false;
    assert!(config.validate().unwrap_err().contains("length-tiebreak"));

    let mut config = short_config(dir.path());
    config.mask_stop = false;
    config.position_features = false;
    assert!(config.validate().unwrap_err().contains("position-features"));

    let mut config = short_config(dir.path());
    config.episodes = 0;
    assert!(config.validate().unwrap_err().contains("serve-socket"));

    let mut config = short_config(dir.path());
    config.eval_processes = 2;
    assert!(config.validate().unwrap_err().contains("cannot exceed"));
}

#[test]
fn torch_evaluator_args_select_checkpoint_and_device() {
    let dir = TestDir::new();
    let mut config = short_config(dir.path());
    config.evaluator = EvaluatorMode::Torch;
    config.checkpoint_dir = Some(PathBuf::from("/checkpoints"));
    config.eval_device = Some("cuda:1".to_owned());
    config.eval_poll_interval = Some(0.25);
    config.validate().unwrap();

    let args = config.evaluator_extra_args();
    assert!(args.windows(2).any(|pair| pair == ["--backend", "torch"]));
    assert!(args.windows(2).any(|pair| pair == ["--device", "cuda:1"]));
    assert!(
        args.windows(2)
            .any(|pair| pair == ["--poll-interval", "0.25"])
    );
    assert!(!args.iter().any(|arg| arg.starts_with("--require-")));
}

fn short_config(path: &Path) -> SelfplayConfig {
    SelfplayConfig {
        replay_dir: Some(path.to_path_buf()),
        episodes: 2,
        lanes: 1,
        workers_per_lane: 1,
        seed: 42,
        max_steps: 2,
        simulations: 2,
        max_considered: 2,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        tree_reuse: false,
        max_candidates: 255,
        max_batch: 2,
        evaluator: EvaluatorMode::Stub,
        mask_stop: true,
        length_tiebreak: true,
        no_backtrack: true,
        ..SelfplayConfig::default()
    }
}
