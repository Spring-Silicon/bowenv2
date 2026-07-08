from __future__ import annotations

import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from gz.checkpoints import DirectorySource, publish_checkpoint
from gz.codec import BatchView, FeatureSchemaConfig
from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash, ModelVersion
from gz.evaluator import TorchBackend, serve
from gz.model.exphormer import ArchConfig, BatchStager, build_model
from gz.proto import (
    BATCH_ENCODING_VERSION,
    ERROR_SCHEMA,
    FRAME_ERROR,
    FRAME_EVAL,
    FRAME_EVAL_RESULT,
    FRAME_HELLO,
    FRAME_HELLO_ACK,
    Hello,
    PROTOCOL_VERSION,
    decode_error,
    read_frame,
    write_frame,
)

torch = pytest.importorskip("torch")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_torch_backend_serves_checkpoint_and_rejects_wrong_schema(tmp_path: Path) -> None:
    batch = (FIXTURES / "batch_expander.gzfb").read_bytes()
    view = BatchView.parse(batch)
    manifest = publish_random_checkpoint(tmp_path, view)

    client, thread, backend = start_torch_client(tmp_path, view, manifest.feature_schema_hash)
    try:
        write_frame(client, FRAME_EVAL, struct.pack("<Q", 44), batch)
        frame_type, payload = read_frame(client, bytearray())
        first = bytes(payload)
        assert frame_type == FRAME_EVAL_RESULT
        assert struct.unpack_from("<Q", payload, 0)[0] == 44
        assert bytes(payload[8:24]) == bytes(manifest.model_version)
        assert output_shape(bytes(payload[24:])) == (view.row_count, view.max_actions)
        assert output_is_finite(bytes(payload[24:]), view.batch_capacity, view.max_actions)
        del payload

        write_frame(client, FRAME_EVAL, struct.pack("<Q", 45), batch)
        frame_type, payload = read_frame(client, bytearray())
        assert frame_type == FRAME_EVAL_RESULT
        assert bytes(payload[8:]) == first[8:]
    finally:
        backend.stop_polling()
        client.close()
        thread.join(timeout=5)

    bad_client, bad_thread, bad_backend = start_raw_torch_client(tmp_path, view)
    try:
        write_frame(bad_client, FRAME_HELLO, make_hello(view, FeatureSchemaHash.from_bytes(b"x" * 32)).encode())
        frame_type, payload = read_frame(bad_client, bytearray())
        assert frame_type == FRAME_ERROR
        code, _ = decode_error(payload)
        assert code == ERROR_SCHEMA
    finally:
        bad_backend.stop_polling()
        bad_client.close()
        bad_thread.join(timeout=5)


def test_torch_backend_hot_swaps_to_new_checkpoint(tmp_path: Path) -> None:
    batch = (FIXTURES / "batch_expander.gzfb").read_bytes()
    view = BatchView.parse(batch)
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    client, thread, backend = start_torch_client(
        tmp_path,
        view,
        first.feature_schema_hash,
        poll_interval=0.05,
        compile_model=None,
    )
    try:
        first_payload = eval_once(client, 1, batch)
        assert result_version(first_payload) == first.model_version

        second = publish_random_checkpoint(tmp_path, view, seed=12)
        second_payload = wait_for_version(client, batch, second.model_version, deadline=30.0)

        assert result_version(second_payload) == second.model_version
        assert second_payload[24:] != first_payload[24:]
    finally:
        backend.stop_polling()
        client.close()
        thread.join(timeout=5)


def test_torch_backend_rejects_bad_swap_once_and_keeps_serving(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    batch = (FIXTURES / "batch_expander.gzfb").read_bytes()
    view = BatchView.parse(batch)
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    client, thread, backend = start_torch_client(
        tmp_path,
        view,
        first.feature_schema_hash,
        poll_interval=0.05,
        compile_model=False,
    )
    try:
        assert result_version(eval_once(client, 1, batch)) == first.model_version
        bad_hash = FeatureSchemaHash.from_bytes(b"z" * 32)
        publish_random_checkpoint(tmp_path, view, seed=12, feature_schema_hash=bad_hash, schema_name="bad")

        for batch_id in range(2, 8):
            time.sleep(0.06)
            assert result_version(eval_once(client, batch_id, batch)) == first.model_version

        captured = capfd.readouterr().err
        assert captured.count("event=checkpoint_rejected") == 1
        assert "feature schema hash mismatch" in captured
    finally:
        backend.stop_polling()
        client.close()
        thread.join(timeout=5)


def test_torch_backend_ignores_broken_latest_and_keeps_serving(tmp_path: Path) -> None:
    batch = (FIXTURES / "batch_expander.gzfb").read_bytes()
    view = BatchView.parse(batch)
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    client, thread, backend = start_torch_client(
        tmp_path,
        view,
        first.feature_schema_hash,
        poll_interval=0.05,
        compile_model=False,
    )
    try:
        (tmp_path / "version_0" / "manifest.json").unlink()
        for batch_id in range(1, 5):
            time.sleep(0.06)
            assert result_version(eval_once(client, batch_id, batch)) == first.model_version
    finally:
        backend.stop_polling()
        client.close()
        thread.join(timeout=5)


def test_torch_backend_poll_interval_zero_keeps_static_model(tmp_path: Path) -> None:
    batch = (FIXTURES / "batch_expander.gzfb").read_bytes()
    view = BatchView.parse(batch)
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    client, thread, backend = start_torch_client(
        tmp_path,
        view,
        first.feature_schema_hash,
        poll_interval=0.0,
        compile_model=False,
    )
    try:
        assert result_version(eval_once(client, 1, batch)) == first.model_version
        publish_random_checkpoint(tmp_path, view, seed=12)
        time.sleep(0.2)
        assert result_version(eval_once(client, 2, batch)) == first.model_version
    finally:
        backend.stop_polling()
        client.close()
        thread.join(timeout=5)


def test_torch_backend_warms_three_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = (FIXTURES / "batch_expander.gzfb").read_bytes()
    view = BatchView.parse(batch)
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    backend = TorchBackend(
        DirectorySource(tmp_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_model=False,
        max_batch=view.batch_capacity,
        poll_interval=0.0,
    )
    calls = 0
    original = backend._run_runner

    def counted(runner: object, tensors: object) -> tuple[object, object]:
        nonlocal calls
        calls += 1
        return original(runner, tensors)

    monkeypatch.setattr(backend, "_run_runner", counted)

    assert backend.handshake(make_hello(view, first.feature_schema_hash)) == first.model_version
    assert calls == 3

    second = publish_random_checkpoint(tmp_path, view, seed=12)
    backend._poll_once()
    assert calls == 3

    backend.apply_pending_swap()
    assert calls == 6
    assert backend._active.model_version == second.model_version


def publish_random_checkpoint(
    root: Path,
    view: BatchView,
    *,
    seed: int = 11,
    feature_schema_hash: FeatureSchemaHash | None = None,
    schema_name: str = "expander-test",
    value_activation: str = "logit",
):
    schema = FeatureSchemaConfig(
        name=schema_name,
        node_vocab_size=8,
        node_attr_dim=view.dims.node_attr_dim,
        edge_type_count=3,
        action_kind_vocab_size=8,
        max_nodes=view.dims.max_nodes,
        max_edges=view.dims.max_edges,
        max_actions=view.dims.max_actions,
        max_subjects=view.dims.max_subjects,
        expander_degree=2,
        expander_seed=0,
    )
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, value_activation=value_activation)
    torch.manual_seed(seed)
    model = build_model(schema, arch)
    return publish_checkpoint(
        root,
        model.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_schema_hash or view.feature_schema_hash,
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        training_step=0,
        run_id="run",
    )


def start_torch_client(
    tmp_path: Path,
    view: BatchView,
    schema_hash: FeatureSchemaHash,
    *,
    poll_interval: float = 0.0,
    compile_model: bool | None = None,
) -> tuple[socket.socket, threading.Thread, TorchBackend]:
    client, thread, backend = start_raw_torch_client(tmp_path, view, poll_interval=poll_interval, compile_model=compile_model)
    write_frame(client, FRAME_HELLO, make_hello(view, schema_hash).encode())
    frame_type, payload = read_frame(client, bytearray())
    assert frame_type == FRAME_HELLO_ACK
    assert struct.unpack_from("<I", payload, 0)[0] == PROTOCOL_VERSION
    del payload
    return client, thread, backend


def start_raw_torch_client(
    tmp_path: Path,
    view: BatchView,
    *,
    poll_interval: float = 0.0,
    compile_model: bool | None = None,
) -> tuple[socket.socket, threading.Thread, TorchBackend]:
    socket_path = tmp_path / f"eval-{len(list(tmp_path.glob('*.sock')))}.sock"
    ready = threading.Event()
    backend = TorchBackend(
        DirectorySource(tmp_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_model=torch.cuda.is_available() if compile_model is None else compile_model,
        max_batch=view.batch_capacity,
        poll_interval=poll_interval,
    )
    thread = threading.Thread(
        target=serve,
        args=(socket_path, backend),
        kwargs={"ready_event": ready},
        daemon=True,
    )
    thread.start()
    assert ready.wait(timeout=5)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))
    return client, thread, backend


def make_hello(view: BatchView, schema_hash: FeatureSchemaHash) -> Hello:
    return Hello(
        protocol_version=PROTOCOL_VERSION,
        encoding_version=BATCH_ENCODING_VERSION,
        feature_schema_hash=schema_hash,
        batch_capacity=view.batch_capacity,
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
    )


def output_shape(payload: bytes) -> tuple[int, int]:
    assert payload[:4] == b"GZFO"
    _version, row_count, max_actions = struct.unpack_from("<III", payload, 4)
    return row_count, max_actions


def output_is_finite(payload: bytes, capacity: int, max_actions: int) -> bool:
    values = np.frombuffer(payload, dtype=np.dtype("<f4"), count=capacity, offset=16)
    logits = np.frombuffer(payload, dtype=np.dtype("<f4"), count=capacity * max_actions, offset=16 + capacity * 4)
    return bool(np.isfinite(values).all() and np.isfinite(logits).all())


def eval_once(client: socket.socket, batch_id: int, batch: bytes) -> bytes:
    write_frame(client, FRAME_EVAL, struct.pack("<Q", batch_id), batch)
    frame_type, payload = read_frame(client, bytearray())
    assert frame_type == FRAME_EVAL_RESULT
    return bytes(payload)


def wait_for_version(client: socket.socket, batch: bytes, version: ModelVersion, *, deadline: float = 5.0) -> bytes:
    deadline = time.monotonic() + deadline
    batch_id = 100
    last = b""
    while time.monotonic() < deadline:
        time.sleep(0.06)
        last = eval_once(client, batch_id, batch)
        if result_version(last) == version:
            return last
        batch_id += 1
    raise AssertionError(f"timed out waiting for model version {version}")


def result_version(payload: bytes) -> ModelVersion:
    return ModelVersion.from_bytes(payload[8:24])


def test_serve_tanh_applies_once_per_head_kind(tmp_path: Path) -> None:
    # Logit heads get the calibrating serve tanh (E[z] = tanh(x) under
    # BCE on 2x); tanh heads are already bounded and must not be
    # compressed a second time.
    from gz.evaluator.backends import TorchBackend

    view = BatchView.parse((FIXTURES / "batch_expander.gzfb").read_bytes())
    for value_activation in ("logit", "tanh"):
        root = tmp_path / value_activation
        manifest = publish_random_checkpoint(root, view, value_activation=value_activation)
        backend = TorchBackend(
            str(root),
            device="cpu",
            poll_interval=0.0,
            compile_model=False,
        )
        backend.handshake(make_hello(view, manifest.feature_schema_hash))
        staged = backend.stage(view)
        pending = backend.launch(staged)
        result = backend.finish(pending)
        values = np.frombuffer(result.payload, dtype=np.dtype("<f4"), count=view.row_count, offset=16)

        schema = FeatureSchemaConfig(
            name="expander-test",
            node_vocab_size=8,
            node_attr_dim=view.dims.node_attr_dim,
            edge_type_count=3,
            action_kind_vocab_size=8,
            max_nodes=view.dims.max_nodes,
            max_edges=view.dims.max_edges,
            max_actions=view.dims.max_actions,
            max_subjects=view.dims.max_subjects,
            expander_degree=2,
            expander_seed=0,
        )
        torch.manual_seed(11)
        reference = build_model(
            schema,
            ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, value_activation=value_activation),
        ).eval()
        stager = BatchStager(schema, view.batch_capacity, "cpu")
        with torch.inference_mode():
            raw, _ = reference(stager.copy(view))
        expected = torch.tanh(raw) if value_activation == "logit" else raw
        assert np.allclose(values, expected[: view.row_count].numpy(), atol=1e-5), value_activation
