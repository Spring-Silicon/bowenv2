from __future__ import annotations

from dataclasses import dataclass

from gz.codec import BatchView, OutputEncoder
from gz.common.tags import ModelVersion
from gz.model.stub import STUB_MODEL_VERSION, stub


@dataclass(frozen=True, slots=True)
class EvalResult:
    model_version: ModelVersion
    payload: memoryview


class StubBackend:
    def __init__(self) -> None:
        self._encoder: OutputEncoder | None = None

    def eval(self, view: BatchView) -> EvalResult:
        if (
            self._encoder is None
            or self._encoder.capacity != view.batch_capacity
            or self._encoder.max_actions != view.max_actions
        ):
            self._encoder = OutputEncoder(view.batch_capacity, view.max_actions)
        values, logits = stub(view)
        return EvalResult(
            model_version=STUB_MODEL_VERSION,
            payload=self._encoder.encode(values, logits, view.row_count),
        )
