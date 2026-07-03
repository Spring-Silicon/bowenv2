from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from gz.codec import BatchView, FeatureSchemaConfig, TargetsView
from gz.model.exphormer import BatchStager, GraphBatchTensors


class TrainingBatch(NamedTuple):
    features: GraphBatchTensors
    policy: object
    value: object
    value_valid: object
    reward: object
    row_count: int


class TrainingStager:
    def __init__(self, schema: FeatureSchemaConfig, capacity: int, device: str | object, pinned_staging: bool = True) -> None:
        torch = _torch()
        self.schema = schema
        self.capacity = capacity
        self.device = torch.device(device)
        self.features = BatchStager(schema, capacity, self.device, pinned_staging)
        pin = bool(pinned_staging and self.device.type == "cuda")
        self.policy = _StagedTensor((capacity, schema.max_actions), torch.float32, self.device, pin)
        self.value = _StagedTensor((capacity,), torch.float32, self.device, pin)
        self.value_valid = _StagedTensor((capacity,), torch.float32, self.device, pin)
        self.reward = _StagedTensor((capacity,), torch.float32, self.device, pin)

    def copy(self, batch: BatchView, targets: TargetsView) -> TrainingBatch:
        if batch.batch_capacity != self.capacity or targets.capacity != self.capacity:
            raise ValueError("capacity mismatch")
        if batch.row_count != targets.row_count:
            raise ValueError("row count mismatch")
        if batch.max_actions != targets.max_actions:
            raise ValueError("max action mismatch")
        self.policy.copy(targets.policy)
        self.value.copy(targets.value)
        self.value_valid.copy(targets.value_valid.astype(np.float32, copy=False))
        self.reward.copy(targets.reward)
        return TrainingBatch(
            features=self.features.copy(batch),
            policy=self.policy.device_tensor,
            value=self.value.device_tensor,
            value_valid=self.value_valid.device_tensor,
            reward=self.reward.device_tensor,
            row_count=targets.row_count,
        )


@dataclass(slots=True)
class _StagedTensor:
    cpu: object
    device_tensor: object
    non_blocking: bool

    def __init__(self, shape: tuple[int, ...], dtype: object, device: object, pin: bool) -> None:
        torch = _torch()
        self.cpu = torch.empty(shape, dtype=dtype, pin_memory=pin)
        self.device_tensor = torch.empty(shape, dtype=dtype, device=device)
        self.non_blocking = pin

    def copy(self, array: np.ndarray) -> None:
        np.copyto(self.cpu.numpy(), array, casting="unsafe")
        self.device_tensor.copy_(self.cpu, non_blocking=self.non_blocking)


def _torch():
    import torch

    return torch
