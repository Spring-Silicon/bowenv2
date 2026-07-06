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

    def encode(self, values: np.ndarray, logits: np.ndarray, row_count: int) -> memoryview:
        if row_count > self.capacity:
            raise ValueError("row_count exceeds capacity")
        if logits.shape[1] != self.max_actions:
            raise ValueError("logit width mismatch")
        if values.shape[0] < row_count or logits.shape[0] < row_count:
            raise ValueError("not enough rows")

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
