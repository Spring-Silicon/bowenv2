from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class FixedBytes:
    value: bytes

    WIDTH: ClassVar[int] = 0

    def __post_init__(self) -> None:
        if len(self.value) != self.WIDTH:
            raise ValueError(f"{type(self).__name__} expects {self.WIDTH} bytes")

    @classmethod
    def from_bytes(cls, value: bytes | bytearray | memoryview) -> FixedBytes:
        return cls(bytes(value))

    @classmethod
    def from_hex(cls, value: str) -> FixedBytes:
        return cls(bytes.fromhex(value))

    def hex(self) -> str:
        return self.value.hex()

    def __bytes__(self) -> bytes:
        return self.value

    def __str__(self) -> str:
        return self.hex()


class EngineId(FixedBytes):
    WIDTH = 16


class EngineVersion(FixedBytes):
    WIDTH = 16


class ModelVersion(FixedBytes):
    WIDTH = 16


class ActionSetHash(FixedBytes):
    WIDTH = 32


class FeatureSchemaHash(FixedBytes):
    WIDTH = 32


@dataclass(frozen=True, slots=True)
class EngineIdentity:
    engine_id: EngineId
    engine_version: EngineVersion
    action_set_hash: ActionSetHash

    def __post_init__(self) -> None:
        self.require_specified()

    @classmethod
    def from_parts(
        cls,
        engine_id: EngineId,
        engine_version: EngineVersion,
        action_set_hash: ActionSetHash,
    ) -> EngineIdentity:
        return cls(engine_id, engine_version, action_set_hash)

    def require_specified(self) -> None:
        if (
            bytes(self.engine_id) == bytes(EngineId.WIDTH)
            and bytes(self.engine_version) == bytes(EngineVersion.WIDTH)
            and bytes(self.action_set_hash) == bytes(ActionSetHash.WIDTH)
        ):
            raise ValueError("engine identity must not be all zero")
