from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path

import numpy as np

from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash
from gz.codec import BatchView
from gz.evaluator import StubBackend, serve
from gz.model.stub import STUB_MODEL_VERSION, stub
from gz.proto import (
    BATCH_ENCODING_VERSION,
    ERROR_CAPACITY,
    ERROR_ENCODING,
    ERROR_MALFORMED,
    ERROR_PROTOCOL,
    ERROR_SCHEMA,
    FRAME_ERROR,
    FRAME_EVAL,
    FRAME_EVAL_RESULT,
    FRAME_HELLO,
    FRAME_HELLO_ACK,
    FRAME_PING,
    FRAME_PONG,
    Hello,
    PROTOCOL_VERSION,
    decode_error,
    read_frame,
    write_frame,
)
from python.tests.test_codec import SCHEMA_HASH, make_batch


def test_server_eval_and_ping(tmp_path: Path) -> None:
    batch = make_batch(attr_dim=1)
    batch_view = BatchView.parse(batch)
    client, thread = start_client(tmp_path, SCHEMA_HASH, 2)
    try:
        write_frame(client, FRAME_PING, struct.pack("<Q", 99))
        frame_type, payload = read_frame(client, bytearray())
        assert frame_type == FRAME_PONG
        assert struct.unpack_from("<Q", payload, 0)[0] == 99
        del payload

        write_frame(client, FRAME_EVAL, struct.pack("<Q", 11), batch)
        frame_type, payload = read_frame(client, bytearray())
        assert frame_type == FRAME_EVAL_RESULT
        assert struct.unpack_from("<Q", payload, 0)[0] == 11
        assert bytes(payload[8:24]) == bytes(STUB_MODEL_VERSION)

        values, logits = stub(batch_view)
        expected = expected_output_bytes(values, logits, batch_view.row_count)
        assert bytes(payload[24:]) == expected
    finally:
        client.close()
        thread.join(timeout=1)


def test_server_rejects_bad_protocol(tmp_path: Path) -> None:
    client, thread = raw_client(tmp_path)
    try:
        bad_hello = make_hello(protocol_version=PROTOCOL_VERSION + 1)
        write_frame(client, FRAME_HELLO, bad_hello.encode())
        assert_error(client, ERROR_PROTOCOL)
    finally:
        client.close()
        thread.join(timeout=1)


def test_server_rejects_bad_encoding(tmp_path: Path) -> None:
    client, thread = raw_client(tmp_path)
    try:
        bad_hello = make_hello(encoding_version=BATCH_ENCODING_VERSION + 1)
        write_frame(client, FRAME_HELLO, bad_hello.encode())
        assert_error(client, ERROR_ENCODING)
    finally:
        client.close()
        thread.join(timeout=1)


def test_server_rejects_changed_schema_and_capacity(tmp_path: Path) -> None:
    client, thread = start_client(tmp_path, SCHEMA_HASH, 2)
    try:
        write_frame(client, FRAME_EVAL, struct.pack("<Q", 1), make_batch(attr_dim=1, schema_hash=b"x" * 32))
        assert_error(client, ERROR_SCHEMA)
    finally:
        client.close()
        thread.join(timeout=1)

    client, thread = start_client(tmp_path, SCHEMA_HASH, 2)
    try:
        write_frame(client, FRAME_EVAL, struct.pack("<Q", 1), make_batch(attr_dim=1, capacity=3))
        assert_error(client, ERROR_CAPACITY)
    finally:
        client.close()
        thread.join(timeout=1)


def test_server_rejects_malformed_ping(tmp_path: Path) -> None:
    client, thread = start_client(tmp_path, SCHEMA_HASH, 2)
    try:
        write_frame(client, FRAME_PING)
        assert_error(client, ERROR_MALFORMED)
    finally:
        client.close()
        thread.join(timeout=1)


def start_client(
    tmp_path: Path,
    schema_hash: bytes,
    capacity: int,
) -> tuple[socket.socket, threading.Thread]:
    client, thread = raw_client(tmp_path)
    write_frame(client, FRAME_HELLO, make_hello(schema_hash=schema_hash, batch_capacity=capacity).encode())
    frame_type, payload = read_frame(client, bytearray())
    assert frame_type == FRAME_HELLO_ACK
    assert struct.unpack_from("<I", payload, 0)[0] == PROTOCOL_VERSION
    del payload
    return client, thread


def raw_client(tmp_path: Path) -> tuple[socket.socket, threading.Thread]:
    socket_path = tmp_path / "eval.sock"
    ready = threading.Event()
    thread = threading.Thread(
        target=serve,
        args=(socket_path, StubBackend()),
        kwargs={"ready_event": ready},
        daemon=True,
    )
    thread.start()
    assert ready.wait(timeout=1)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))
    return client, thread


def make_hello(
    protocol_version: int = PROTOCOL_VERSION,
    encoding_version: int = BATCH_ENCODING_VERSION,
    schema_hash: bytes = SCHEMA_HASH,
    batch_capacity: int = 2,
) -> Hello:
    return Hello(
        protocol_version=protocol_version,
        encoding_version=encoding_version,
        feature_schema_hash=FeatureSchemaHash.from_bytes(schema_hash),
        batch_capacity=batch_capacity,
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
    )


def assert_error(client: socket.socket, code: int) -> None:
    frame_type, payload = read_frame(client, bytearray())
    assert frame_type == FRAME_ERROR
    actual, _message = decode_error(payload)
    assert actual == code


def expected_output_bytes(values: np.ndarray, logits: np.ndarray, row_count: int) -> bytes:
    out = bytearray()
    out.extend(b"GZFO")
    out.extend(struct.pack("<III", BATCH_ENCODING_VERSION, row_count, logits.shape[1]))
    out.extend(values.astype("<f4", copy=False).tobytes())
    out.extend(logits.astype("<f4", copy=False).tobytes())
    return bytes(out)
