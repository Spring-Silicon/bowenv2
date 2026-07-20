from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from gz.codec.batch import EncodingError
from gz.proto.frames import ENCODING_VERSION

TARGET_MAGIC = b"GZFT"
TARGET_HEADER_LEN = 20


@dataclass(frozen=True, slots=True)
class _Layout:
    b: int
    a: int
    policy: int
    value: int
    value_valid: int
    horizon_value: int
    horizon_value_valid: int
    reward: int
    total_len: int


@dataclass(frozen=True, slots=True)
class TargetsView:
    capacity: int
    row_count: int
    max_actions: int
    policy: np.ndarray
    value: np.ndarray
    value_valid: np.ndarray
    horizon_value: np.ndarray
    horizon_value_valid: np.ndarray
    reward: np.ndarray

    @classmethod
    def parse(cls, buf: bytes | bytearray | memoryview) -> TargetsView:
        view = memoryview(buf)
        if len(view) < TARGET_HEADER_LEN:
            raise EncodingError("target header truncated")
        if bytes(view[0:4]) != TARGET_MAGIC:
            raise EncodingError("bad target magic")
        version = _u32(view, 4)
        if version != ENCODING_VERSION:
            raise EncodingError("unsupported target version")
        capacity = _u32(view, 8)
        row_count = _u32(view, 12)
        max_actions = _u32(view, 16)
        if capacity == 0:
            raise EncodingError("zero target capacity")
        if max_actions == 0:
            raise EncodingError("zero target actions")
        if row_count > capacity:
            raise EncodingError("target row count exceeds capacity")
        layout = _layout(capacity, max_actions)
        if len(view) != layout.total_len:
            raise EncodingError("bad target length")
        return cls(
            capacity=capacity,
            row_count=row_count,
            max_actions=max_actions,
            policy=_array(view, layout.policy, "<f4", (capacity, max_actions)),
            value=_array(view, layout.value, "<f4", (capacity,)),
            value_valid=_array(view, layout.value_valid, "u1", (capacity,)),
            horizon_value=_array(
                view, layout.horizon_value, "<f4", (capacity, 2)
            ),
            horizon_value_valid=_array(
                view, layout.horizon_value_valid, "u1", (capacity,)
            ),
            reward=_array(view, layout.reward, "<f4", (capacity,)),
        )


def _layout(capacity: int, max_actions: int) -> _Layout:
    cursor = TARGET_HEADER_LEN
    policy, cursor = _section(cursor, capacity * max_actions * 4)
    value, cursor = _section(cursor, capacity * 4)
    value_valid, cursor = _section(cursor, capacity)
    horizon_value, cursor = _section(cursor, capacity * 2 * 4)
    horizon_value_valid, cursor = _section(cursor, capacity)
    reward, cursor = _section(cursor, capacity * 4)
    return _Layout(
        b=capacity,
        a=max_actions,
        policy=policy,
        value=value,
        value_valid=value_valid,
        horizon_value=horizon_value,
        horizon_value_valid=horizon_value_valid,
        reward=reward,
        total_len=_align4(cursor),
    )


def _section(cursor: int, length: int) -> tuple[int, int]:
    offset = _align4(cursor)
    return offset, offset + length


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _u32(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def _array(buf: memoryview, offset: int, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    return np.frombuffer(buf, dtype=np.dtype(dtype), count=count, offset=offset).reshape(shape)
