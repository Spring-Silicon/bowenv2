from __future__ import annotations

import socket
import struct

import pytest

from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash, ModelVersion
from gz.proto import (
    ERROR_MALFORMED,
    FRAME_HELLO,
    FRAME_PING,
    Hello,
    HelloAck,
    ProtocolError,
    decode_error,
    encode_error,
    read_frame,
    write_frame,
)


def test_frame_roundtrip_over_socketpair() -> None:
    left, right = socket.socketpair()
    try:
        write_frame(left, FRAME_PING, struct.pack("<Q", 123))
        frame_type, payload = read_frame(right, bytearray())

        assert frame_type == FRAME_PING
        assert struct.unpack_from("<Q", payload, 0)[0] == 123
    finally:
        left.close()
        right.close()


def test_truncated_frame_raises() -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("<I", 8) + b"\x05")
        left.close()

        with pytest.raises(ProtocolError) as error:
            read_frame(right, bytearray())
        assert error.value.code == ERROR_MALFORMED
    finally:
        right.close()


def test_hello_roundtrip() -> None:
    hello = Hello(
        protocol_version=1,
        encoding_version=1,
        feature_schema_hash=FeatureSchemaHash.from_bytes(b"f" * 32),
        batch_capacity=4,
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
    )
    ack = HelloAck(1, ModelVersion.from_bytes(b"m" * 16), 7)

    assert Hello.decode(memoryview(hello.encode())) == hello
    assert HelloAck.decode(memoryview(ack.encode())) == ack


def test_error_frame_codec() -> None:
    payload = encode_error(3, "schema mismatch")

    assert decode_error(memoryview(payload)) == (3, "schema mismatch")
