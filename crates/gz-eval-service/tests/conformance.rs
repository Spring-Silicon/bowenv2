mod common;

use common::{assert_outputs_equal_bits, collate, hello as make_hello, row, schema as make_schema};
use gz_eval_service::{
    ERROR_CAPACITY, ERROR_ENCODING, ERROR_PROTOCOL, ERROR_SCHEMA, EvaluatorProcess,
    EvaluatorProcessConfig, FeatureEvalBackend, STUB_MODEL_VERSION, ServiceError, StubBackend,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[test]
fn python_process_matches_stub_backend_bit_for_bit() {
    require_numpy();
    let schema = make_schema("python-equivalence", 6);
    let rows = vec![row(4, 3), row(1, 1), row(5, 4)];
    let (batch, action_counts) = collate(schema.clone(), 5, &rows);
    let expected = StubBackend.eval(&batch, &action_counts).unwrap();
    let (mut process, mut backend) = spawn_backend(&make_hello(&schema, 5));

    backend.ping().unwrap();
    assert_eq!(backend.model_version(), STUB_MODEL_VERSION);
    let actual = backend.eval(&batch, &action_counts).unwrap();

    assert_eq!(actual.model_version, STUB_MODEL_VERSION);
    assert_outputs_equal_bits(&actual.rows, &expected.rows);
    drop(backend);
    assert_child_exits(&mut process);
}

#[test]
fn python_process_enforces_adopted_schema_and_capacity() {
    require_numpy();
    let schema = make_schema("python-schema-a", 4);
    let other_schema = make_schema("python-schema-b", 4);
    let rows = vec![row(3, 2)];
    let (batch, action_counts) = collate(schema.clone(), 2, &rows);
    let (other_batch, other_action_counts) = collate(other_schema, 2, &rows);
    let (mut process, mut backend) = spawn_backend(&make_hello(&schema, 2));

    backend.eval(&batch, &action_counts).unwrap();
    assert!(matches!(
        backend.eval(&other_batch, &other_action_counts),
        Err(ServiceError::Backend {
            code: ERROR_SCHEMA,
            ..
        })
    ));
    drop(backend);
    assert_child_exits(&mut process);

    let (capacity_batch, capacity_counts) = collate(schema.clone(), 3, &rows);
    let (mut process, mut backend) = spawn_backend(&make_hello(&schema, 2));
    assert!(matches!(
        backend.eval(&capacity_batch, &capacity_counts),
        Err(ServiceError::Backend {
            code: ERROR_CAPACITY,
            ..
        })
    ));
    drop(backend);
    assert_child_exits(&mut process);
}

#[test]
fn python_process_rejects_bad_handshake_versions() {
    require_numpy();
    let schema = make_schema("python-handshake", 4);

    let mut hello = make_hello(&schema, 2);
    hello.protocol_version += 1;
    let (mut process, error) = spawn_bad_handshake(&hello);
    assert!(
        matches!(error, ServiceError::Handshake(message) if message.contains(&ERROR_PROTOCOL.to_string()))
    );
    assert_child_exits(&mut process);

    let mut hello = make_hello(&schema, 2);
    hello.encoding_version += 1;
    let (mut process, error) = spawn_bad_handshake(&hello);
    assert!(
        matches!(error, ServiceError::Handshake(message) if message.contains(&ERROR_ENCODING.to_string()))
    );
    assert_child_exits(&mut process);
}

fn spawn_backend(
    hello: &gz_eval_service::Hello,
) -> (EvaluatorProcess, gz_eval_service::ProcessBackend) {
    let mut process = EvaluatorProcess::spawn(process_config()).unwrap_or_else(|error| {
        panic!("failed to spawn Python evaluator: {error}; requires python3 + numpy")
    });
    let backend = process.connect(hello).unwrap_or_else(|error| {
        panic!("failed to connect Python evaluator: {error}; requires python3 + numpy")
    });
    (process, backend)
}

fn spawn_bad_handshake(hello: &gz_eval_service::Hello) -> (EvaluatorProcess, ServiceError) {
    let mut process = EvaluatorProcess::spawn(process_config()).unwrap_or_else(|error| {
        panic!("failed to spawn Python evaluator: {error}; requires python3 + numpy")
    });
    let error = match process.connect(hello) {
        Ok(_) => panic!("bad handshake unexpectedly succeeded"),
        Err(error) => error,
    };
    (process, error)
}

fn process_config() -> EvaluatorProcessConfig {
    EvaluatorProcessConfig {
        working_dir: python_dir(),
        socket_path: common::temp_socket("python"),
        ready_timeout: Duration::from_secs(10),
        io_timeout: Duration::from_secs(10),
        ..EvaluatorProcessConfig::default()
    }
}

fn python_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python")
}

fn assert_child_exits(process: &mut EvaluatorProcess) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match process.try_wait().unwrap() {
            Some(status) => {
                assert!(status.success(), "Python evaluator exited with {status}");
                return;
            }
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            None => panic!("Python evaluator did not exit after backend dropped"),
        }
    }
}

fn require_numpy() {
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg("import numpy")
        .status()
        .expect("failed to run python3; conformance tests require python3 + numpy");
    assert!(
        status.success(),
        "python3 -c 'import numpy' failed; conformance tests require python3 + numpy"
    );
}
