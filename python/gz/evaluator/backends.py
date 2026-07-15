from __future__ import annotations

import importlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from gz.codec import BatchView, OutputEncoder
from gz.checkpoints import CheckpointSource, DirectorySource, ResolvedCheckpoint
from gz.common.tags import ModelVersion
from gz.model import build
from gz.model.exphormer import ArchConfig, BatchStager, build_pair_serving_models
from gz.model.stub import STUB_MODEL_VERSION, stub
from gz.proto import ERROR_CAPACITY, ERROR_SCHEMA, Hello, ProtocolError

WARMUP_RUNS = 3


@dataclass(frozen=True, slots=True)
class EvalResult:
    model_version: ModelVersion
    payload: memoryview


class EvalTimingStats:
    def __init__(self, interval: int = 100) -> None:
        self.interval = interval
        self.batches = 0
        self.rows = 0
        self.payload_bytes = 0
        self.stage_s = 0.0
        self.launch_s = 0.0
        self.sync_s = 0.0
        self.encode_s = 0.0

    def record_stage(self, seconds: float) -> None:
        self.stage_s += seconds

    def record_launch(self, seconds: float) -> None:
        self.launch_s += seconds

    def record_finish(
        self,
        *,
        rows: int,
        payload_bytes: int,
        sync_s: float,
        encode_s: float,
    ) -> None:
        self.batches += 1
        self.rows += rows
        self.payload_bytes += payload_bytes
        self.sync_s += sync_s
        self.encode_s += encode_s
        if self.batches % self.interval == 0:
            denom = float(self.interval)
            print(
                "event=eval_timing"
                f" batches={self.batches}"
                f" rows={self.rows}"
                f" stage_ms={self.stage_s * 1000.0 / denom:.3f}"
                f" launch_ms={self.launch_s * 1000.0 / denom:.3f}"
                f" sync_ms={self.sync_s * 1000.0 / denom:.3f}"
                f" encode_ms={self.encode_s * 1000.0 / denom:.3f}"
                f" payload_kb={self.payload_bytes / 1024.0 / denom:.1f}",
                file=sys.stderr,
                flush=True,
            )
            self.stage_s = 0.0
            self.launch_s = 0.0
            self.sync_s = 0.0
            self.encode_s = 0.0
            self.payload_bytes = 0


# The pipelined serving contract: stage(view) copies everything it needs
# out of the request buffer (the view dies when the next frame is read),
# launch(staged) enqueues compute AND the device-to-host copy of its
# outputs (into per-slot pinned buffers, stream-ordered behind the
# replay), finish(pending) waits on the slot's event and encodes. Because
# outputs leave the CUDA-graph static buffers at launch time, up to
# HOST_SLOTS launches may be outstanding before a finish -- the server
# runs the loop at the same depth:
#   stage(N+1) -> launch(N+1) -> ... -> finish(N) -> write reply(N)
# finish(N) waits only on N's event, never on replay(N+1) queued behind
# it.


class StubBackend:
    def __init__(self) -> None:
        self._encoder: OutputEncoder | None = None

    def handshake(self, hello: Hello) -> ModelVersion:
        _ = hello
        return STUB_MODEL_VERSION

    def apply_pending_swap(self) -> None:
        return None

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
            payload=self._encoder.encode(
                values,
                logits,
                view.row_count,
                view.action_count[: view.row_count],
            ),
        )

    # The stub computes eagerly at stage; the payload is copied because the
    # encoder buffer would otherwise be clobbered by the next stage before
    # the pipelined server writes this reply.
    def stage(self, view: BatchView) -> EvalResult:
        result = self.eval(view)
        return EvalResult(model_version=result.model_version, payload=memoryview(bytes(result.payload)))

    def launch(self, staged: EvalResult) -> EvalResult:
        return staged

    def finish(self, pending: EvalResult) -> EvalResult:
        return pending


@dataclass(frozen=True, slots=True)
class _ServingSlot:
    manifest: object
    runner: object
    model_version: ModelVersion
    # Whether the model's value output is already bounded. Logit heads
    # need the serve-side tanh -- BCE trains P(win) = sigmoid(2x), so
    # E[z] = tanh(x) is the calibrated search value -- but tanh heads
    # are bounded at the model and a second tanh double-compresses.
    value_bounded: bool = False
    value_bin_centers: object = None
    opponent_runner: object = None
    opponent_dim: int = 0
    policy_only: bool = False


@dataclass(frozen=True, slots=True)
class StagedEval:
    tensors: object
    ready_event: object
    row_count: int
    action_counts: object
    opponent_trajectory_id: object
    opponent_row: object
    opponent_state_present: object
    opponent_cached: bool
    encoder: OutputEncoder


@dataclass(frozen=True, slots=True)
class _HostSlot:
    values: object
    logits: object
    event: object


@dataclass(frozen=True, slots=True)
class PendingEval:
    model_version: ModelVersion
    # CUDA path: outputs already copied to slot's pinned buffers.
    slot: _HostSlot | None
    # CPU path: raw tensors, finished synchronously.
    value_raw: object
    logits: object
    row_count: int
    action_counts: object
    encoder: OutputEncoder


PIPELINE_DEPTH = 3
HOST_SLOTS = PIPELINE_DEPTH
MAX_OPPONENT_CACHE_ROWS = 4096


class TorchBackend:
    def __init__(
        self,
        source: CheckpointSource | str,
        *,
        device: str | None = None,
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
        max_batch: int = 1024,
        poll_interval: float = 10.0,
        policy_only: bool = False,
    ) -> None:
        torch = _torch()
        self.source = DirectorySource(source) if isinstance(source, (str, Path)) else source
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.max_batch = max_batch
        self.poll_interval = poll_interval
        self.policy_only = policy_only
        self.resolved = self.source.resolve_latest()
        self._active = self._build_slot(self.resolved)
        self.manifest = self._active.manifest
        self.stager: BatchStager | None = None
        self._stagers: tuple[BatchStager, ...] = ()
        self._stage_index = 0
        self._host_slots: tuple[_HostSlot, ...] = ()
        self._transfer_stream = None
        self._slot_index = 0
        self._encoder: OutputEncoder | None = None
        self._pending: _ServingSlot | None = None
        self._pending_lock = threading.Lock()
        self._logged_rejections: set[str] = set()
        self._loader_started = False
        self._stop_polling = threading.Event()
        self._timings = EvalTimingStats()
        self._opponent_cache: dict[tuple[str, int, int], object] = {}
        self._opponent_cache_blocks: list[object] = []

    def handshake(self, hello: Hello) -> ModelVersion:
        if hello.feature_schema_hash != self._active.manifest.feature_schema_hash:
            raise ProtocolError(ERROR_SCHEMA, "feature schema hash mismatch")
        if hello.batch_capacity > self.max_batch:
            raise ProtocolError(ERROR_CAPACITY, "batch capacity exceeds backend maximum")
        torch = _torch()
        self._transfer_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.stager = BatchStager(
            self._active.manifest.feature_schema,
            hello.batch_capacity,
            self.device,
            transfer_stream=self._transfer_stream,
        )
        # Runtime staging matches the pending depth. Warmup owns a separate
        # stager because swaps can happen with old launches still queued.
        self._stagers = tuple(
            BatchStager(
                self._active.manifest.feature_schema,
                hello.batch_capacity,
                self.device,
                transfer_stream=self._transfer_stream,
            )
            for _ in range(PIPELINE_DEPTH)
        )
        self._stage_index = 0
        if self.device.type == "cuda":
            schema = self._active.manifest.feature_schema
            self._host_slots = tuple(
                _HostSlot(
                    values=torch.empty(hello.batch_capacity, dtype=torch.float32, pin_memory=True),
                    logits=torch.empty(
                        (hello.batch_capacity, schema.max_actions),
                        dtype=torch.float32,
                        pin_memory=True,
                    ),
                    event=torch.cuda.Event(),
                )
            for _ in range(HOST_SLOTS)
            )
            self._slot_index = 0
        self._clear_opponent_cache()
        self._warm_slot(self._active, self.stager, WARMUP_RUNS)
        if self.poll_interval > 0.0:
            self._start_loader()
        return self._active.model_version

    def apply_pending_swap(self) -> None:
        if self.stager is None:
            return
        with self._pending_lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return
        import time as _time

        warm_started = _time.perf_counter()
        try:
            # CUDA graph capture (reduce-overhead) only works on the serving
            # thread, so the loader publishes the slot unwarmed and warmup
            # happens here, between frames. A same-arch checkpoint hits the
            # inductor cache and warms in well under a second; a cold compile
            # pauses serving for its duration while workers park.
            self._warm_slot(pending, self.stager, WARMUP_RUNS)
        except Exception as error:
            self._log_rejection(pending.model_version.hex(), pending.model_version, error)
            return
        self._active = pending
        self.manifest = pending.manifest
        self._clear_opponent_cache()
        torch = _torch()
        allocated = (
            torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        print(
            f"event=checkpoint_swapped model_version={pending.model_version.hex()}"
            f" warm_s={_time.perf_counter() - warm_started:.2f}"
            f" gpu_alloc_mb={allocated / 1e6:.0f}",
            file=sys.stderr,
            flush=True,
        )

    def stop_polling(self) -> None:
        self._stop_polling.set()

    def eval(self, view: BatchView) -> EvalResult:
        return self.finish(self.launch(self.stage(view)))

    def stage(self, view: BatchView) -> StagedEval:
        if self.stager is None:
            raise RuntimeError("torch backend used before handshake")
        if (
            self._encoder is None
            or self._encoder.capacity != view.batch_capacity
            or self._encoder.max_actions != view.max_actions
        ):
            self._encoder = OutputEncoder(view.batch_capacity, view.max_actions)
        started = time.perf_counter()
        stager = self._stagers[self._stage_index]
        self._stage_index = (self._stage_index + 1) % len(self._stagers)
        active = self._active
        opponent_cached = self._opponent_rows_cached(
            active,
            view.opponent_trajectory_id[: view.row_count],
            view.opponent_row[: view.row_count],
            view.opponent_state_present[: view.row_count],
        )
        tensors = stager.copy(
            view,
            copy_opponent=active.opponent_runner is not None and not opponent_cached,
        )
        action_counts = view.action_count[: view.row_count].copy()
        self._timings.record_stage(time.perf_counter() - started)
        # The encoder rides with the batch: finish() runs after the NEXT
        # batch was staged, which may have re-keyed self._encoder.
        return StagedEval(
            tensors=tensors,
            ready_event=stager.ready_event,
            row_count=view.row_count,
            action_counts=action_counts,
            opponent_trajectory_id=view.opponent_trajectory_id[: view.row_count].copy(),
            opponent_row=view.opponent_row[: view.row_count].copy(),
            opponent_state_present=view.opponent_state_present[: view.row_count].copy(),
            opponent_cached=opponent_cached,
            encoder=self._encoder,
        )

    def launch(self, staged: StagedEval) -> PendingEval:
        active = self._active
        started = time.perf_counter()
        self._wait_for_stage(staged.ready_event)
        if active.policy_only:
            logits = self._run_policy_runner(active.runner, staged.tensors)
            value_raw = logits.new_zeros(logits.shape[0])
        else:
            opponent_readout = self._opponent_readout(active, staged)
            if opponent_readout is None:
                value_raw, logits = self._run_runner(active.runner, staged.tensors)
            else:
                value_raw, logits = self._run_runner(
                    active.runner,
                    staged.tensors,
                    opponent_readout,
                )
        if not self._host_slots:
            self._timings.record_launch(time.perf_counter() - started)
            return PendingEval(
                model_version=active.model_version,
                slot=None,
                value_raw=self._serve_value(active, value_raw),
                logits=logits,
                row_count=staged.row_count,
                action_counts=staged.action_counts,
                encoder=staged.encoder,
            )
        torch = _torch()
        slot = self._host_slots[self._slot_index]
        self._slot_index = (self._slot_index + 1) % len(self._host_slots)
        # Enqueued on the current stream, so these read the static
        # CUDA-graph outputs BEFORE any later replay overwrites them;
        # the event marks when the pinned copies are complete.
        with torch.inference_mode():
            rows = staged.row_count
            slot.values[:rows].copy_(self._serve_value(active, value_raw)[:rows].float(), non_blocking=True)
            slot.logits[:rows].copy_(logits[:rows].float(), non_blocking=True)
        slot.event.record()
        self._timings.record_launch(time.perf_counter() - started)
        return PendingEval(
            model_version=active.model_version,
            slot=slot,
            value_raw=None,
            logits=None,
            row_count=staged.row_count,
            action_counts=staged.action_counts,
            encoder=staged.encoder,
        )

    def finish(self, pending: PendingEval) -> EvalResult:
        if pending.slot is not None:
            sync_started = time.perf_counter()
            pending.slot.event.synchronize()
            sync_s = time.perf_counter() - sync_started
            encode_started = time.perf_counter()
            payload = pending.encoder.encode(
                pending.slot.values.numpy(),
                pending.slot.logits.numpy(),
                pending.row_count,
                pending.action_counts,
            )
            encode_s = time.perf_counter() - encode_started
            self._timings.record_finish(
                rows=pending.row_count,
                payload_bytes=len(payload),
                sync_s=sync_s,
                encode_s=encode_s,
            )
            return EvalResult(
                model_version=pending.model_version,
                payload=payload,
            )
        sync_started = time.perf_counter()
        values = pending.value_raw.detach().float().cpu().numpy()
        logits = pending.logits.detach().float().cpu().numpy()
        sync_s = time.perf_counter() - sync_started
        encode_started = time.perf_counter()
        payload = pending.encoder.encode(
            values,
            logits,
            pending.row_count,
            pending.action_counts,
        )
        encode_s = time.perf_counter() - encode_started
        self._timings.record_finish(
            rows=pending.row_count,
            payload_bytes=len(payload),
            sync_s=sync_s,
            encode_s=encode_s,
        )
        return EvalResult(
            model_version=pending.model_version,
            payload=payload,
        )

    def _serve_value(self, slot: _ServingSlot, value_raw: object) -> object:
        if slot.value_bin_centers is not None:
            torch = _torch()
            probabilities = torch.softmax(value_raw.float(), dim=-1)
            centers = slot.value_bin_centers.to(
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            return (probabilities * centers).sum(dim=-1)
        if slot.value_bounded:
            return value_raw
        torch = _torch()
        return torch.tanh(value_raw)

    def _wait_for_stage(self, ready_event: object) -> None:
        if ready_event is None:
            return
        torch = _torch()
        torch.cuda.current_stream(self.device).wait_event(ready_event)

    def _opponent_rows_cached(
        self,
        slot: _ServingSlot,
        trajectory_ids: object,
        rows: object,
        present: object,
    ) -> bool:
        if slot.opponent_runner is None:
            return False
        version = slot.model_version.hex()
        for trajectory_id, row, state_present in zip(trajectory_ids, rows, present):
            if not state_present:
                continue
            trajectory_id = int(trajectory_id)
            if trajectory_id == 0:
                return False
            if (version, trajectory_id, int(row)) not in self._opponent_cache:
                return False
        return True

    def _opponent_readout(self, slot: _ServingSlot, staged: StagedEval) -> object:
        if slot.opponent_runner is None:
            return None
        if staged.opponent_cached:
            return self._cached_opponent_readout(slot, staged)
        readout = self._run_opponent_runner(slot.opponent_runner, staged.tensors)
        self._cache_opponent_readout(slot, staged, readout)
        return readout

    def _cached_opponent_readout(self, slot: _ServingSlot, staged: StagedEval) -> object:
        torch = _torch()
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        zero = torch.zeros(slot.opponent_dim, dtype=dtype, device=self.device)
        version = slot.model_version.hex()
        rows = []
        for index in range(staged.tensors.node_count.shape[0]):
            if index >= staged.row_count or not staged.opponent_state_present[index]:
                rows.append(zero)
                continue
            key = (
                version,
                int(staged.opponent_trajectory_id[index]),
                int(staged.opponent_row[index]),
            )
            rows.append(self._opponent_cache[key])
        return torch.stack(rows)

    def _cache_opponent_readout(
        self,
        slot: _ServingSlot,
        staged: StagedEval,
        readout: object,
    ) -> None:
        version = slot.model_version.hex()
        new_rows: dict[tuple[str, int, int], int] = {}
        for index in range(staged.row_count):
            if not staged.opponent_state_present[index]:
                continue
            trajectory_id = int(staged.opponent_trajectory_id[index])
            if trajectory_id == 0:
                continue
            key = (version, trajectory_id, int(staged.opponent_row[index]))
            if key not in self._opponent_cache and key not in new_rows:
                new_rows[key] = index
        if not new_rows:
            return
        if len(new_rows) > MAX_OPPONENT_CACHE_ROWS:
            return
        if len(self._opponent_cache) + len(new_rows) > MAX_OPPONENT_CACHE_ROWS:
            self._clear_opponent_cache()
        torch = _torch()
        indexes = torch.tensor(tuple(new_rows.values()), dtype=torch.int64, device=self.device)
        block = readout.detach().index_select(0, indexes).clone()
        self._opponent_cache_blocks.append(block)
        for block_index, key in enumerate(new_rows):
            self._opponent_cache[key] = block[block_index]

    def _clear_opponent_cache(self) -> None:
        self._opponent_cache.clear()
        self._opponent_cache_blocks.clear()

    def _start_loader(self) -> None:
        if self._loader_started:
            return
        self._loader_started = True
        thread = threading.Thread(target=self._loader_loop, name="gz-evaluator-hotswap", daemon=True)
        thread.start()

    def _loader_loop(self) -> None:
        while not self._stop_polling.wait(self.poll_interval):
            try:
                self._poll_once()
            except Exception as error:
                self._log_rejection(f"loader:{type(error).__name__}:{error}", None, error)

    def _poll_once(self) -> None:
        resolved = self.source.resolve_latest()
        version = resolved.manifest.model_version
        if version.hex() in self._logged_rejections:
            return
        with self._pending_lock:
            pending_version = self._pending.model_version if self._pending is not None else None
        if version == self._active.model_version or version == pending_version:
            return
        if resolved.manifest.feature_schema_hash != self._active.manifest.feature_schema_hash:
            self._log_rejection(version.hex(), version, "feature schema hash mismatch")
            return
        try:
            slot = self._build_slot(resolved)
        except Exception as error:
            self._log_rejection(version.hex(), version, error)
            return
        with self._pending_lock:
            self._pending = slot

    def _build_slot(self, resolved: ResolvedCheckpoint) -> _ServingSlot:
        arch = ArchConfig.from_dict(resolved.manifest.arch_config)
        if arch.name != resolved.manifest.arch_name:
            raise ValueError("manifest arch name mismatch")
        model = build(resolved.manifest.feature_schema, arch)
        from gz.checkpoints.weights import load_state_dict

        model.load_state_dict(load_state_dict(resolved.weights_path))
        model.to(self.device)
        model.eval()
        value_bin_centers = (
            model.value_bin_centers
            if arch.value_head == "hl_gauss"
            else None
        )
        torch = _torch()
        opponent_runner = None
        serving_model = model.policy_logits if self.policy_only else model
        if not self.policy_only and arch.value_input == "pair":
            serving_model, opponent_model = build_pair_serving_models(model)
            opponent_runner = (
                torch.compile(opponent_model, fullgraph=True, mode=self.compile_mode)
                if self.compile_model
                else opponent_model
            )
        runner = (
            torch.compile(serving_model, fullgraph=True, mode=self.compile_mode)
            if self.compile_model
            else serving_model
        )
        return _ServingSlot(
            manifest=resolved.manifest,
            runner=runner,
            model_version=resolved.manifest.model_version,
            value_bounded=self.policy_only
            or arch.value_head == "hl_gauss"
            or arch.value_activation == "tanh",
            value_bin_centers=None if self.policy_only else value_bin_centers,
            opponent_runner=opponent_runner,
            opponent_dim=arch.dim if opponent_runner is not None else 0,
            policy_only=self.policy_only,
        )

    def _warm_slot(self, slot: _ServingSlot, stager: BatchStager, count: int) -> None:
        for _ in range(count):
            tensors = stager.dummy()
            self._wait_for_stage(stager.ready_event)
            if slot.policy_only:
                self._run_policy_runner(slot.runner, tensors)
            elif slot.opponent_runner is None:
                self._run_runner(slot.runner, tensors)
            else:
                opponent_readout = self._run_opponent_runner(slot.opponent_runner, tensors)
                self._run_runner(slot.runner, tensors, opponent_readout)
            if self.device.type == "cuda":
                _torch().cuda.current_stream(self.device).synchronize()

    def _run_runner(
        self,
        runner: object,
        tensors: object,
        opponent_readout: object = None,
    ) -> tuple[object, object]:
        torch = _torch()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if opponent_readout is None:
                        return runner(tensors)
                    return runner(tensors, opponent_readout)
            if opponent_readout is None:
                return runner(tensors)
            return runner(tensors, opponent_readout)

    def _run_opponent_runner(self, runner: object, tensors: object) -> object:
        torch = _torch()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    readout = runner(tensors)
            else:
                readout = runner(tensors)
            # Compiled CUDA-graph outputs are static replay buffers. The
            # serving graph and cache need an owned readout that survives
            # the opponent runner's next replay.
            return readout.clone()

    def _run_policy_runner(self, runner: object, tensors: object) -> object:
        torch = _torch()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return runner(tensors)
            return runner(tensors)

    def _log_rejection(self, key: str, version: ModelVersion | None, error: object) -> None:
        if key in self._logged_rejections:
            return
        self._logged_rejections.add(key)
        version_text = version.hex() if version is not None else "unknown"
        print(f"event=checkpoint_rejected model_version={version_text} error={error}", file=sys.stderr, flush=True)


def _torch():
    return importlib.import_module("torch")
