from __future__ import annotations

import struct
from dataclasses import dataclass

from gz.common.tags import (
    ActionSetHash,
    EngineIdentity,
    EngineId,
    EngineVersion,
    FeatureSchemaHash,
    ModelVersion,
)
from gz.proto.errors import ERROR_MALFORMED, ProtocolError

HELLO_LEN = 108
HELLO_ACK_LEN = 28


@dataclass(frozen=True, slots=True)
class Hello:
    protocol_version: int
    encoding_version: int
    feature_schema_hash: FeatureSchemaHash
    batch_capacity: int
    engine_identity: EngineIdentity

    @property
    def engine_id(self) -> EngineId:
        return self.engine_identity.engine_id

    @property
    def engine_version(self) -> EngineVersion:
        return self.engine_identity.engine_version

    @property
    def action_set_hash(self) -> ActionSetHash:
        return self.engine_identity.action_set_hash

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
        try:
            engine_identity = EngineIdentity.from_parts(
                EngineId.from_bytes(buf[44:60]),
                EngineVersion.from_bytes(buf[60:76]),
                ActionSetHash.from_bytes(buf[76:108]),
            )
        except ValueError as error:
            raise ProtocolError(ERROR_MALFORMED, str(error)) from error
        return cls(
            protocol_version=protocol_version,
            encoding_version=encoding_version,
            feature_schema_hash=FeatureSchemaHash.from_bytes(buf[8:40]),
            batch_capacity=struct.unpack_from("<I", buf, 40)[0],
            engine_identity=engine_identity,
        )


@dataclass(frozen=True, slots=True)
class HelloAck:
    protocol_version: int
    model_version: ModelVersion
    model_generation: int

    def encode(self) -> bytes:
        return (
            struct.pack("<I", self.protocol_version)
            + bytes(self.model_version)
            + struct.pack("<Q", self.model_generation)
        )

    @classmethod
    def decode(cls, buf: memoryview) -> HelloAck:
        if len(buf) != HELLO_ACK_LEN:
            raise ProtocolError(ERROR_MALFORMED, "bad hello ack length")
        return cls(
            protocol_version=struct.unpack_from("<I", buf, 0)[0],
            model_version=ModelVersion.from_bytes(buf[4:20]),
            model_generation=struct.unpack_from("<Q", buf, 20)[0],
        )
