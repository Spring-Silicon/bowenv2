from __future__ import annotations

import numpy as np

from gz.codec import BatchView
from gz.common.tags import ModelVersion

STUB_MODEL_VERSION = ModelVersion(b"gz-stub-v1" + b"\x00" * 6)


def stub(batch: BatchView) -> tuple[np.ndarray, np.ndarray]:
    rows = batch.batch_capacity
    actions = batch.max_actions
    row_count = batch.row_count

    node_count = batch.node_count.astype(np.uint64, copy=False)
    action_count = batch.action_count.astype(np.uint64, copy=False)

    raw_values = (node_count * np.uint64(2_654_435_761) + action_count * np.uint64(40_503)) % np.uint64(4096)
    values = ((raw_values.astype(np.int64) - 2048) / 2048.0).astype(np.float32)

    action_index = np.arange(actions, dtype=np.uint64)
    raw_logits = (
        node_count[:, None]
        + np.uint64(31) * action_index[None, :]
        + np.uint64(7) * action_count[:, None]
    ) % np.uint64(64)
    logits = ((raw_logits.astype(np.int64) - 32) / 32.0).astype(np.float32)

    row_mask = np.arange(rows) < row_count
    action_mask = action_index[None, :] < action_count[:, None]
    values[~row_mask] = 0.0
    logits[~(row_mask[:, None] & action_mask)] = 0.0
    return values, logits
