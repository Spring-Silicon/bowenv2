use gz_engine::{ActionSetHash, EngineId, EngineVersion};
use gz_eval_service::{
    FRAME_ERROR, FRAME_EVAL, FRAME_EVAL_RESULT, FRAME_HELLO, FRAME_HELLO_ACK, FRAME_MODEL_RELEASE,
    FRAME_PING, FRAME_PONG, Hello, PROTOCOL_VERSION, ServiceError, decode_error, read_frame,
    write_frame,
};
use gz_features::{BATCH_ENCODING_VERSION, FeatureSchemaHash};
use std::io::Write;
use std::os::unix::net::UnixStream;

#[test]
fn frame_roundtrip_accepts_every_known_type() {
    for frame_type in [
        FRAME_HELLO,
        FRAME_HELLO_ACK,
        FRAME_EVAL,
        FRAME_EVAL_RESULT,
        FRAME_PING,
        FRAME_PONG,
        FRAME_ERROR,
        FRAME_MODEL_RELEASE,
    ] {
        let (mut left, mut right) = UnixStream::pair().unwrap();
        let mut write_buf = Vec::new();
        let mut read_buf = Vec::new();

        write_frame(&mut left, &mut write_buf, frame_type, &[b"abc", b"def"]).unwrap();
        let (actual_type, payload) = read_frame(&mut right, &mut read_buf).unwrap();

        assert_eq!(actual_type, frame_type);
        assert_eq!(payload, b"abcdef");
    }
}

#[test]
fn malformed_frame_headers_are_rejected() {
    let (mut left, mut right) = UnixStream::pair().unwrap();
    left.write_all(&0u32.to_le_bytes()).unwrap();
    let mut read_buf = Vec::new();
    assert!(matches!(
        read_frame(&mut right, &mut read_buf),
        Err(ServiceError::Protocol(_))
    ));

    let (mut left, mut right) = UnixStream::pair().unwrap();
    left.write_all(&(gz_eval_service::MAX_FRAME as u32 + 1).to_le_bytes())
        .unwrap();
    let mut read_buf = Vec::new();
    assert!(matches!(
        read_frame(&mut right, &mut read_buf),
        Err(ServiceError::Protocol(_))
    ));

    let (mut left, mut right) = UnixStream::pair().unwrap();
    left.write_all(&1u32.to_le_bytes()).unwrap();
    left.write_all(&[99]).unwrap();
    let mut read_buf = Vec::new();
    assert!(matches!(
        read_frame(&mut right, &mut read_buf),
        Err(ServiceError::Protocol(_))
    ));
}

#[test]
fn hello_field_order_is_pinned() {
    let hello = Hello::new(
        FeatureSchemaHash::from_bytes([0x11; 32]),
        7,
        EngineId::from_bytes([0x22; 16]),
        EngineVersion::from_bytes([0x33; 16]),
        ActionSetHash::from_bytes([0x44; 32]),
    );
    let mut bytes = Vec::new();
    hello.encode(&mut bytes);

    let mut expected = Vec::new();
    expected.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    expected.extend_from_slice(&BATCH_ENCODING_VERSION.to_le_bytes());
    expected.extend_from_slice(&[0x11; 32]);
    expected.extend_from_slice(&7u32.to_le_bytes());
    expected.extend_from_slice(&[0x22; 16]);
    expected.extend_from_slice(&[0x33; 16]);
    expected.extend_from_slice(&[0x44; 32]);

    assert_eq!(bytes, expected);
    assert_eq!(Hello::decode(&bytes).unwrap(), hello);
}

#[test]
fn error_payload_decodes_bounded_lossy_message() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&3u16.to_le_bytes());
    bytes.extend_from_slice(&[b'a', 0xff, b'b']);

    let (code, message) = decode_error(&bytes).unwrap();

    assert_eq!(code, 3);
    assert_eq!(message, "a\u{fffd}b");
}
