from __future__ import annotations

import os
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from gz.codec import BatchView
from gz.codec.batch import EncodingError
from gz.common.tags import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash
from gz.evaluator.backends import StubBackend
from gz.model.stub import STUB_MODEL_VERSION
from gz.proto import (
    ENCODING_VERSION,
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
    HelloAck,
    PROTOCOL_VERSION,
    ProtocolError,
    encode_error,
    read_frame,
    write_frame,
)


@dataclass(frozen=True, slots=True)
class _ConnectionState:
    feature_schema_hash: FeatureSchemaHash
    batch_capacity: int
    engine_id: EngineId
    engine_version: EngineVersion
    action_set_hash: ActionSetHash


def serve(socket_path: str | Path, backend: StubBackend, *, ready_event: Event | None = None) -> None:
    path = Path(socket_path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        listener.listen(1)
        if ready_event is not None:
            ready_event.set()
        conn, _ = listener.accept()
        with conn:
            _serve_connection(conn, backend)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _serve_connection(conn: socket.socket, backend: StubBackend) -> None:
    buf = bytearray()
    try:
        state = _handshake(conn, buf)
        while True:
            frame_type, payload = read_frame(conn, buf)
            try:
                if frame_type == FRAME_PING:
                    _handle_ping(conn, payload)
                elif frame_type == FRAME_EVAL:
                    _handle_eval(conn, backend, state, payload)
                else:
                    raise ProtocolError(ERROR_PROTOCOL, "unexpected frame type")
            finally:
                del payload
    except ProtocolError as error:
        _send_error(conn, error.code, error.message)
    except EncodingError as error:
        _send_error(conn, ERROR_ENCODING, str(error))


def _handshake(conn: socket.socket, buf: bytearray) -> _ConnectionState:
    frame_type, payload = read_frame(conn, buf)
    if frame_type != FRAME_HELLO:
        raise ProtocolError(ERROR_PROTOCOL, "expected HELLO")
    hello = Hello.decode(payload)
    if hello.protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(ERROR_PROTOCOL, "protocol version mismatch")
    if hello.encoding_version != ENCODING_VERSION:
        raise ProtocolError(ERROR_ENCODING, "encoding version mismatch")
    if hello.batch_capacity == 0:
        raise ProtocolError(ERROR_CAPACITY, "zero batch capacity")
    write_frame(
        conn,
        FRAME_HELLO_ACK,
        HelloAck(PROTOCOL_VERSION, STUB_MODEL_VERSION).encode(),
    )
    return _ConnectionState(
        feature_schema_hash=hello.feature_schema_hash,
        batch_capacity=hello.batch_capacity,
        engine_id=hello.engine_id,
        engine_version=hello.engine_version,
        action_set_hash=hello.action_set_hash,
    )


def _handle_ping(conn: socket.socket, payload: memoryview) -> None:
    if len(payload) != 8:
        raise ProtocolError(ERROR_MALFORMED, "bad PING length")
    write_frame(conn, FRAME_PONG, payload)


def _handle_eval(
    conn: socket.socket,
    backend: StubBackend,
    state: _ConnectionState,
    payload: memoryview,
) -> None:
    if len(payload) < 8:
        raise ProtocolError(ERROR_MALFORMED, "EVAL frame truncated")
    batch_id = struct.unpack_from("<Q", payload, 0)[0]
    try:
        batch = BatchView.parse(payload[8:])
    except EncodingError as error:
        raise ProtocolError(ERROR_ENCODING, str(error)) from error
    if batch.feature_schema_hash != state.feature_schema_hash:
        raise ProtocolError(ERROR_SCHEMA, "feature schema hash mismatch")
    if batch.batch_capacity != state.batch_capacity:
        raise ProtocolError(ERROR_CAPACITY, "batch capacity mismatch")
    result = backend.eval(batch)
    write_frame(
        conn,
        FRAME_EVAL_RESULT,
        struct.pack("<Q", batch_id),
        bytes(result.model_version),
        result.payload,
    )


def _send_error(conn: socket.socket, code: int, message: str) -> None:
    try:
        write_frame(conn, FRAME_ERROR, encode_error(code, message))
    except OSError:
        pass
