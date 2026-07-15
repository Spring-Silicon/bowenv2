mod common;

use common::{
    ScriptedServer, hello, output_payload, read_frame_type, row, schema, send_ack, send_error,
};
use gz_eval_service::{
    ERROR_PROTOCOL, EvaluatorProcess, EvaluatorProcessConfig, FRAME_EVAL_RESULT,
    FeatureEvalBackend, ServiceError, StubBackend, write_frame,
};
use std::fs;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

#[test]
fn spawn_failure_is_io_error() {
    let config = EvaluatorProcessConfig {
        python: PathBuf::from("/definitely/missing/python3"),
        socket_path: common::temp_socket("spawn-failure"),
        ready_timeout: Duration::from_millis(50),
        ..EvaluatorProcessConfig::default()
    };

    assert!(matches!(
        EvaluatorProcess::spawn(config),
        Err(ServiceError::Io(_))
    ));
}

#[test]
fn process_spawn_appends_extra_args_after_socket() {
    let dir = std::env::temp_dir().join(format!("gz-eval-service-argv-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("argv_probe.py"),
        r#"
import pathlib
import sys

path = pathlib.Path(sys.argv[sys.argv.index("--argv-out") + 1])
path.write_text("\n".join(sys.argv), encoding="utf-8")
"#,
    )
    .unwrap();
    let argv_out = dir.join("argv.txt");
    let socket = dir.join("probe.sock");
    let mut process = EvaluatorProcess::spawn(EvaluatorProcessConfig {
        module: "argv_probe".to_owned(),
        working_dir: dir.clone(),
        socket_path: socket.clone(),
        extra_args: vec![
            "--flag".to_owned(),
            "value".to_owned(),
            "--argv-out".to_owned(),
            argv_out.display().to_string(),
        ],
        ..EvaluatorProcessConfig::default()
    })
    .unwrap();

    let status = process.wait().unwrap();
    assert!(status.success());
    let argv = fs::read_to_string(&argv_out).unwrap();
    let args = argv.lines().collect::<Vec<_>>();
    let socket_index = args.iter().position(|arg| *arg == "--socket").unwrap();

    assert_eq!(args[socket_index + 1], socket.display().to_string());
    assert_eq!(args[socket_index + 2], "--flag");
    assert_eq!(args[socket_index + 3], "value");
    assert_eq!(args[socket_index + 4], "--argv-out");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scripted_server_can_drive_eval_result_decoding() {
    let schema = schema("process-scripted", 4);
    let rows = [row(3, 2)];
    let (batch, action_counts) = common::collate(schema.clone(), 2, &rows);
    let expected = StubBackend.eval(&batch, &action_counts).unwrap();
    let result_payload = output_payload(&expected.rows, 2, 4);
    let server = ScriptedServer::new("process-eval", move |mut stream| {
        assert_eq!(read_frame_type(&mut stream), gz_eval_service::FRAME_HELLO);
        send_ack(&mut stream).unwrap();
        let mut read_buf = Vec::new();
        let (frame_type, payload) =
            gz_eval_service::read_frame(&mut stream, &mut read_buf).unwrap();
        assert_eq!(frame_type, gz_eval_service::FRAME_EVAL);
        let batch_id = &payload[0..8];
        let mut parts = Vec::new();
        parts.extend_from_slice(batch_id);
        parts.extend_from_slice(&common::model_version_bytes());
        parts.extend_from_slice(&result_payload);
        let mut write_buf = Vec::new();
        write_frame(&mut stream, &mut write_buf, FRAME_EVAL_RESULT, &[&parts]).unwrap();
    });

    let stream = UnixStream::connect(&server.path).unwrap();
    let mut backend = gz_eval_service::ProcessBackend::connect_stream(
        stream,
        &hello(&schema, 2),
        Duration::from_secs(1),
    )
    .unwrap();
    assert_eq!(backend.batch_capacity().unwrap().get(), 2);
    let actual = backend.eval(&batch, &action_counts).unwrap();

    common::assert_outputs_equal_bits(&actual.rows, &expected.rows);
}

#[test]
fn scripted_server_errors_map_to_backend_error() {
    let schema = schema("process-error", 4);
    let rows = [row(3, 2)];
    let (batch, action_counts) = common::collate(schema.clone(), 2, &rows);
    let server = ScriptedServer::new("process-error", move |mut stream| {
        assert_eq!(read_frame_type(&mut stream), gz_eval_service::FRAME_HELLO);
        send_ack(&mut stream).unwrap();
        assert_eq!(read_frame_type(&mut stream), gz_eval_service::FRAME_EVAL);
        send_error(&mut stream, ERROR_PROTOCOL, "bad eval").unwrap();
    });

    let stream = UnixStream::connect(&server.path).unwrap();
    let mut backend = gz_eval_service::ProcessBackend::connect_stream(
        stream,
        &hello(&schema, 2),
        Duration::from_secs(1),
    )
    .unwrap();

    assert!(matches!(
        backend.eval(&batch, &action_counts),
        Err(ServiceError::Backend {
            code: ERROR_PROTOCOL,
            ..
        })
    ));
}
