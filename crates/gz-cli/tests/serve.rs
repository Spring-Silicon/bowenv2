use gz_cli::selfplay::{EvaluatorMode, ReferenceMode, SelfplayConfig, run as run_selfplay};
use gz_cli::serve::{ReplayServeConfig, SAMPLE_PROTOCOL_VERSION, run_one};
use gz_features::{ENCODING_VERSION, FeatureBatchView, TrainingTargetsView};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("gz-cli-serve-test-{}-{id}", std::process::id()));
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
fn replay_serve_returns_feature_batch_and_targets() {
    let dir = TestDir::new();
    let summary = run_selfplay(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 2,
        lanes: 1,
        workers_per_lane: 2,
        reference: ReferenceMode::Root,
        seed: 5,
        max_steps: 2,
        simulations: 2,
        max_batch: 2,
        evaluator: EvaluatorMode::Stub,
        python_dir: None,
    })
    .unwrap();
    let socket = dir.path().join("sample.sock");
    let server_config = ReplayServeConfig {
        replay_dir: dir.path().to_path_buf(),
        socket: socket.clone(),
        max_batch: 2,
    };
    let server = std::thread::spawn(move || run_one(server_config));
    let mut stream = connect_retry(&socket);

    let mut hello = Vec::new();
    hello.extend_from_slice(&SAMPLE_PROTOCOL_VERSION.to_le_bytes());
    hello.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
    write_frame(&mut stream, 1, &[&hello]);
    let (frame_type, ack) = read_frame(&mut stream);
    assert_eq!(frame_type, 2);
    assert_eq!(
        u32::from_le_bytes(ack[0..4].try_into().unwrap()),
        SAMPLE_PROTOCOL_VERSION
    );
    assert_eq!(u32::from_le_bytes(ack[36..40].try_into().unwrap()), 2);
    assert_eq!(
        u64::from_le_bytes(ack[40..48].try_into().unwrap()),
        summary.rows_produced
    );

    let mut sample = Vec::new();
    sample.extend_from_slice(&1u32.to_le_bytes());
    sample.extend_from_slice(&summary.rows_produced.to_le_bytes());
    sample.extend_from_slice(&123u64.to_le_bytes());
    write_frame(&mut stream, 3, &[&sample]);
    let (frame_type, result) = read_frame(&mut stream);
    assert_eq!(frame_type, 4);
    let gzfb_len = u32::from_le_bytes(result[0..4].try_into().unwrap()) as usize;
    let gzfb = &result[4..4 + gzfb_len];
    let gzft = &result[4 + gzfb_len..];
    let batch = FeatureBatchView::parse(gzfb).unwrap();
    let targets = TrainingTargetsView::parse(gzft).unwrap();

    assert_eq!(batch.batch_capacity, 2);
    assert_eq!(batch.row_count, 1);
    assert_eq!(targets.capacity, 2);
    assert_eq!(targets.row_count, 1);
    assert_eq!(targets.max_actions, batch.max_actions);
    assert_eq!(
        targets.policy.len(),
        (targets.capacity * targets.max_actions) as usize
    );

    drop(stream);
    server.join().unwrap().unwrap();
}

#[test]
fn replay_serve_rejects_featureless_store() {
    let dir = TestDir::new();
    run_selfplay(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 1,
        lanes: 1,
        workers_per_lane: 1,
        reference: ReferenceMode::Root,
        seed: 7,
        max_steps: 1,
        simulations: 1,
        max_batch: 1,
        evaluator: EvaluatorMode::Random,
        python_dir: None,
    })
    .unwrap();

    let error = run_one(ReplayServeConfig {
        replay_dir: dir.path().to_path_buf(),
        socket: dir.path().join("sample.sock"),
        max_batch: 1,
    })
    .unwrap_err();

    assert!(error.contains("store was not produced by featurized selfplay"));
}

fn connect_retry(path: &Path) -> UnixStream {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match UnixStream::connect(path) {
            Ok(stream) => return stream,
            Err(error) if Instant::now() < deadline => {
                assert!(
                    matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                    ),
                    "{error}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("{error}"),
        }
    }
}

fn write_frame(stream: &mut UnixStream, frame_type: u8, parts: &[&[u8]]) {
    let body_len = 1 + parts.iter().map(|part| part.len()).sum::<usize>();
    stream.write_all(&(body_len as u32).to_le_bytes()).unwrap();
    stream.write_all(&[frame_type]).unwrap();
    for part in parts {
        stream.write_all(part).unwrap();
    }
}

fn read_frame(stream: &mut UnixStream) -> (u8, Vec<u8>) {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).unwrap();
    let body_len = u32::from_le_bytes(len) as usize;
    let mut body = vec![0; body_len];
    stream.read_exact(&mut body).unwrap();
    (body[0], body[1..].to_vec())
}
