from __future__ import annotations

import struct
from dataclasses import dataclass

from gz.common.tags import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash, ModelVersion
from gz.proto.errors import ERROR_MALFORMED, ProtocolError

HELLO_LEN = 108
HELLO_ACK_LEN = 20


@dataclass(frozen=True, slots=True)
class Hello:
    protocol_version: int
    encoding_version: int
    feature_schema_hash: FeatureSchemaHash
    batch_capacity: int
    engine_id: EngineId
    engine_version: EngineVersion
    action_set_hash: ActionSetHash

    def encode(self) -> bytes:
        return (
            struct.pack("<II", self.protocol_version, self.encoding_version)
            + bytes(self.feature_schema_hash)
            + struct.pack("<I", self.batch_capacity)
            + bytes(self.engine_id)
            + bytes(self.engine_version)
            + bytes(self.action_set_hash)
        )

    @classmethod
    def decode(cls, buf: memoryview) -> Hello:
        if len(buf) != HELLO_LEN:
            raise ProtocolError(ERROR_MALFORMED, "bad hello length")
        protocol_version, encoding_version = struct.unpack_from("<II", buf, 0)
        return cls(
            protocol_version=protocol_version,
            encoding_version=encoding_version,
            feature_schema_hash=FeatureSchemaHash.from_bytes(buf[8:40]),
            batch_capacity=struct.unpack_from("<I", buf, 40)[0],
            engine_id=EngineId.from_bytes(buf[44:60]),
            engine_version=EngineVersion.from_bytes(buf[60:76]),
            action_set_hash=ActionSetHash.from_bytes(buf[76:108]),
        )


@dataclass(frozen=True, slots=True)
class HelloAck:
    protocol_version: int
    model_version: ModelVersion

    def encode(self) -> bytes:
        return struct.pack("<I", self.protocol_version) + bytes(self.model_version)

    @classmethod
    def decode(cls, buf: memoryview) -> HelloAck:
        if len(buf) != HELLO_ACK_LEN:
            raise ProtocolError(ERROR_MALFORMED, "bad hello ack length")
        return cls(
            protocol_version=struct.unpack_from("<I", buf, 0)[0],
            model_version=ModelVersion.from_bytes(buf[4:20]),
        )
