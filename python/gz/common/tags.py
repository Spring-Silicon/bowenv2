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
