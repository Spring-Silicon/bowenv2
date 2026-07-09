from __future__ import annotations

import struct

import numpy as np

from gz.proto.frames import BATCH_ENCODING_VERSION

OUTPUT_MAGIC = b"GZFO"
OUTPUT_HEADER_LEN = 16


class OutputEncoder:
    def __init__(self, capacity: int, max_actions: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self.capacity = capacity
        self.max_actions = max_actions
        self._length = OUTPUT_HEADER_LEN + capacity * 4 + capacity * max_actions * 4
        self._buf = bytearray(self._length)

    def encode(
        self,
        values: np.ndarray,
        logits: np.ndarray,
        row_count: int,
        action_counts: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ) -> memoryview:
        if row_count > self.capacity:
            raise ValueError("row_count exceeds capacity")
        if logits.shape[1] != self.max_actions:
            raise ValueError("logit width mismatch")
        if values.shape[0] < row_count or logits.shape[0] < row_count:
            raise ValueError("not enough rows")
        if action_counts is None:
            return self._encode_dense(values, logits, row_count)

        counts = np.asarray(action_counts, dtype=np.int64)
        if counts.shape[0] != row_count:
            raise ValueError("action count length mismatch")
        if bool(np.any(counts < 0)) or bool(np.any(counts > self.max_actions)):
            raise ValueError("action count out of range")

        policy_floats = int(counts.sum())
        length = OUTPUT_HEADER_LEN + row_count * 4 + policy_floats * 4
        if length >= self._length:
            return self._encode_dense(values, logits, row_count)
        if len(self._buf) < length:
            raise RuntimeError("output buffer too small")
        struct.pack_into("<4sIII", self._buf, 0, OUTPUT_MAGIC, BATCH_ENCODING_VERSION, row_count, self.max_actions)
        value_view = np.frombuffer(self._buf, dtype=np.dtype("<f4"), count=row_count, offset=16)
        value_view[:] = values[:row_count]
        offset = OUTPUT_HEADER_LEN + row_count * 4
        logit_rows = np.asarray(logits[:row_count], dtype=np.dtype("<f4"), order="C")
        logit_bytes = memoryview(logit_rows).cast("B")
        row_width = self.max_actions * 4
        for row, count in enumerate(counts.tolist()):
            byte_count = count * 4
            if byte_count:
                start = row * row_width
                self._buf[offset : offset + byte_count] = logit_bytes[start : start + byte_count]
                offset += byte_count
        return memoryview(self._buf)[:length]

    def _encode_dense(self, values: np.ndarray, logits: np.ndarray, row_count: int) -> memoryview:
        struct.pack_into("<4sIII", self._buf, 0, OUTPUT_MAGIC, BATCH_ENCODING_VERSION, row_count, self.max_actions)
        value_view = np.frombuffer(self._buf, dtype=np.dtype("<f4"), count=self.capacity, offset=16)
        policy_view = np.frombuffer(
            self._buf,
            dtype=np.dtype("<f4"),
            count=self.capacity * self.max_actions,
            offset=16 + self.capacity * 4,
        ).reshape(self.capacity, self.max_actions)
        value_view.fill(0.0)
        policy_view.fill(0.0)
        value_view[:row_count] = values[:row_count]
        policy_view[:row_count, :] = logits[:row_count, :]
        return memoryview(self._buf)[: self._length]
