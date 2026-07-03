from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from gz.codec import BatchView, OutputEncoder
from gz.checkpoints import CheckpointSource, DirectorySource
from gz.common.tags import ModelVersion
from gz.model import build
from gz.model.exphormer import ArchConfig, BatchStager
from gz.model.stub import STUB_MODEL_VERSION, stub
from gz.proto import ERROR_CAPACITY, ERROR_SCHEMA, Hello, ProtocolError


@dataclass(frozen=True, slots=True)
class EvalResult:
    model_version: ModelVersion
    payload: memoryview


class StubBackend:
    def __init__(self) -> None:
        self._encoder: OutputEncoder | None = None

    def handshake(self, hello: Hello) -> ModelVersion:
        _ = hello
        return STUB_MODEL_VERSION

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


class TorchBackend:
    def __init__(
        self,
        source: CheckpointSource | str,
        *,
        device: str | None = None,
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
        max_batch: int = 1024,
    ) -> None:
        torch = _torch()
        self.source = DirectorySource(source) if isinstance(source, (str, Path)) else source
        self.resolved = self.source.resolve_latest()
        self.manifest = self.resolved.manifest
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.compile_model = compile_model
        self.max_batch = max_batch
        arch = ArchConfig.from_dict(self.manifest.arch_config)
        if arch.name != self.manifest.arch_name:
            raise ValueError("manifest arch name mismatch")
        model = build(self.manifest.feature_schema, arch)
        from gz.checkpoints.weights import load_state_dict

        model.load_state_dict(load_state_dict(self.resolved.weights_path))
        model.to(self.device)
        model.eval()
        self.model = model
        self.runner = torch.compile(model, fullgraph=True, mode=compile_mode) if compile_model else model
        self.stager: BatchStager | None = None
        self._encoder: OutputEncoder | None = None

    def handshake(self, hello: Hello) -> ModelVersion:
        if hello.feature_schema_hash != self.manifest.feature_schema_hash:
            raise ProtocolError(ERROR_SCHEMA, "feature schema hash mismatch")
        if hello.batch_capacity > self.max_batch:
            raise ProtocolError(ERROR_CAPACITY, "batch capacity exceeds backend maximum")
        self.stager = BatchStager(self.manifest.feature_schema, hello.batch_capacity, self.device)
        self._warm()
        return self.manifest.model_version

    def eval(self, view: BatchView) -> EvalResult:
        if self.stager is None:
            raise RuntimeError("torch backend used before handshake")
        if (
            self._encoder is None
            or self._encoder.capacity != view.batch_capacity
            or self._encoder.max_actions != view.max_actions
        ):
            self._encoder = OutputEncoder(view.batch_capacity, view.max_actions)
        tensors = self.stager.copy(view)
        values, logits = self._run(tensors)
        return EvalResult(
            model_version=self.manifest.model_version,
            payload=self._encoder.encode(
                values.detach().float().cpu().numpy(),
                logits.detach().float().cpu().numpy(),
                view.row_count,
            ),
        )

    def _warm(self) -> None:
        if self.stager is None:
            raise RuntimeError("missing stager")
        self._run(self.stager.dummy())

    def _run(self, tensors: object) -> tuple[object, object]:
        torch = _torch()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return self.runner(tensors)
            return self.runner(tensors)


def _torch():
    return importlib.import_module("torch")
