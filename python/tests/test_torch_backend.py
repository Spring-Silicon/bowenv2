from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gz.checkpoints import DirectorySource, publish_checkpoint
from gz.codec import BatchView, FeatureSchemaConfig
from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash, ModelVersion
from gz.evaluator import TorchBackend, serve
from gz.evaluator.backends import PIPELINE_DEPTH
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
from python.tests.test_codec import _layout

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
        assert output_is_finite(bytes(payload[24:]), view.action_count[: view.row_count])
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
    batch = batch_with_opponent_refs((FIXTURES / "batch_expander.gzfb").read_bytes())
    view = BatchView.parse(batch)
    first = publish_random_checkpoint(tmp_path, view, seed=11, value_input="pair")
    client, thread, backend = start_torch_client(
        tmp_path,
        view,
        first.feature_schema_hash,
        poll_interval=0.05,
        compile_model=False,
    )
    try:
        first_payload = eval_once(client, 1, batch)
        assert result_version(first_payload) == first.model_version

        second = publish_random_checkpoint(tmp_path, view, seed=12, value_input="pair")
        second_payload = wait_for_version(client, batch, second.model_version, deadline=60.0)

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
    assert len(backend._stagers) == PIPELINE_DEPTH
    assert all(stager is not backend.stager for stager in backend._stagers)

    second = publish_random_checkpoint(tmp_path, view, seed=12)
    backend._poll_once()
    assert calls == 3

    backend.apply_pending_swap()
    assert calls == 6
    assert backend._active.model_version == second.model_version


def test_pair_backend_caches_stable_opponent_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = BatchView.parse((FIXTURES / "batch_expander.gzfb").read_bytes())
    view = replace(
        base,
        opponent_trajectory_id=np.full(base.batch_capacity, 7, dtype=np.uint64),
        opponent_row=np.arange(base.batch_capacity, dtype=np.uint32),
        opponent_state_present=np.ones(base.batch_capacity, dtype=np.uint8),
    )
    manifest = publish_random_checkpoint(tmp_path, view, value_input="pair")
    backend = TorchBackend(
        DirectorySource(tmp_path),
        device="cpu",
        compile_model=False,
        max_batch=view.batch_capacity,
        poll_interval=0.0,
    )
    opponent_calls = 0
    original_opponent = backend._run_opponent_runner

    def counted_opponent(runner: object, tensors: object) -> object:
        nonlocal opponent_calls
        opponent_calls += 1
        return original_opponent(runner, tensors)

    monkeypatch.setattr(backend, "_run_opponent_runner", counted_opponent)
    backend.handshake(make_hello(view, manifest.feature_schema_hash))
    assert opponent_calls == 3

    original_copy = BatchStager.copy
    observed_flags: list[bool] = []

    def observed_copy(
        stager: BatchStager,
        batch: BatchView,
        *,
        copy_opponent: bool = True,
    ):
        observed_flags.append(copy_opponent)
        return original_copy(stager, batch, copy_opponent=copy_opponent)

    monkeypatch.setattr(BatchStager, "copy", observed_copy)
    first = bytes(backend.eval(view).payload)
    second = bytes(backend.eval(view).payload)

    assert first == second
    assert opponent_calls == 4
    assert observed_flags == [True, False]

    uncached = replace(
        view,
        opponent_trajectory_id=np.zeros(view.batch_capacity, dtype=np.uint64),
    )
    backend.eval(uncached)
    backend.eval(uncached)
    assert opponent_calls == 6
    assert observed_flags[-2:] == [True, True]


def test_policy_only_pair_backend_matches_policy_logits_and_skips_value_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = BatchView.parse(
        batch_with_opponent_refs((FIXTURES / "batch_expander.gzfb").read_bytes())
    )
    manifest = publish_random_checkpoint(tmp_path, view, value_input="pair")
    full = TorchBackend(
        DirectorySource(tmp_path),
        device="cpu",
        compile_model=False,
        max_batch=view.batch_capacity,
        poll_interval=0.0,
    )
    policy = TorchBackend(
        DirectorySource(tmp_path),
        device="cpu",
        compile_model=False,
        max_batch=view.batch_capacity,
        poll_interval=0.0,
        policy_only=True,
    )

    full.handshake(make_hello(view, manifest.feature_schema_hash))
    monkeypatch.setattr(
        policy,
        "_run_runner",
        lambda *_args: pytest.fail("policy-only backend ran the value-serving model"),
    )
    monkeypatch.setattr(
        policy,
        "_run_opponent_runner",
        lambda *_args: pytest.fail("policy-only backend encoded the opponent graph"),
    )
    policy.handshake(make_hello(view, manifest.feature_schema_hash))

    full_payload = bytes(full.eval(view).payload)
    policy_payload = bytes(policy.eval(view).payload)
    values_offset = 16
    logits_offset = values_offset + view.row_count * 4
    policy_values = np.frombuffer(
        policy_payload,
        dtype=np.dtype("<f4"),
        count=view.row_count,
        offset=values_offset,
    )

    assert policy._active.policy_only
    assert policy._active.opponent_runner is None
    assert np.array_equal(policy_values, np.zeros(view.row_count, dtype=np.float32))
    assert policy_payload[logits_offset:] == full_payload[logits_offset:]


def test_policy_only_backend_remains_policy_only_after_hot_swap(tmp_path: Path) -> None:
    view = BatchView.parse((FIXTURES / "batch_expander.gzfb").read_bytes())
    first = publish_random_checkpoint(tmp_path, view, seed=11, value_input="pair")
    backend = TorchBackend(
        DirectorySource(tmp_path),
        device="cpu",
        compile_model=False,
        max_batch=view.batch_capacity,
        poll_interval=0.0,
        policy_only=True,
    )
    backend.handshake(make_hello(view, first.feature_schema_hash))
    first_result = backend.eval(view)

    second = publish_random_checkpoint(tmp_path, view, seed=12, value_input="pair")
    backend._poll_once()
    backend.apply_pending_swap()
    second_result = backend.eval(view)

    assert first_result.model_version == first.model_version
    assert second_result.model_version == second.model_version
    assert backend._active.policy_only
    assert backend._active.opponent_runner is None


def publish_random_checkpoint(
    root: Path,
    view: BatchView,
    *,
    seed: int = 11,
    feature_schema_hash: FeatureSchemaHash | None = None,
    schema_name: str = "expander-test",
    value_activation: str = "logit",
    value_input: str = "single",
    value_head: str = "scalar",
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
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        value_activation=value_activation,
        value_input=value_input,
        value_head=value_head,
    )
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


def output_is_finite(payload: bytes, action_counts: np.ndarray) -> bool:
    values = np.frombuffer(payload, dtype=np.dtype("<f4"), count=len(action_counts), offset=16)
    logits = np.frombuffer(
        payload,
        dtype=np.dtype("<f4"),
        count=int(action_counts.sum()),
        offset=16 + len(action_counts) * 4,
    )
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


def batch_with_opponent_refs(batch: bytes) -> bytes:
    view = BatchView.parse(batch)
    layout = _layout(
        view.batch_capacity,
        view.dims.max_nodes,
        view.dims.max_edges,
        view.dims.max_actions,
        view.dims.max_subjects,
        view.dims.node_attr_dim,
    )
    out = bytearray(batch)
    for index in range(view.row_count):
        out[layout["opponent_state_present"] + index] = 1
        struct.pack_into("<Q", out, layout["opponent_trajectory_id"] + index * 8, 7)
        struct.pack_into("<I", out, layout["opponent_row"] + index * 4, index)
    return bytes(out)


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


def test_serve_hl_gauss_decodes_expected_value_without_tanh(tmp_path: Path) -> None:
    view = BatchView.parse((FIXTURES / "batch_expander.gzfb").read_bytes())
    manifest = publish_random_checkpoint(tmp_path, view, value_head="hl_gauss")
    backend = TorchBackend(
        str(tmp_path),
        device="cpu",
        poll_interval=0.0,
        compile_model=False,
    )
    backend.handshake(make_hello(view, manifest.feature_schema_hash))
    result = backend.finish(backend.launch(backend.stage(view)))
    values = np.frombuffer(
        result.payload,
        dtype=np.dtype("<f4"),
        count=view.row_count,
        offset=16,
    )

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
        ArchConfig(
            dim=16,
            layers=1,
            heads=4,
            ffn_dim=32,
            dropout=0.0,
            value_head="hl_gauss",
        ),
    ).eval()
    with torch.inference_mode():
        logits, _ = reference(BatchStager(schema, view.batch_capacity, "cpu").copy(view))
        expected = reference.decode_value(logits)

    assert np.allclose(values, expected[: view.row_count].numpy(), atol=1e-5)
