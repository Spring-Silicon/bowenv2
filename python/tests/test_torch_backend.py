from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

from gz.checkpoints import DirectorySource, publish_checkpoint
from gz.codec import BatchView, FeatureSchemaConfig
from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash
from gz.evaluator import TorchBackend, serve
from gz.model.exphormer import ArchConfig, build_model
from gz.proto import (
    ENCODING_VERSION,
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

    client, thread = start_torch_client(tmp_path, view, manifest.feature_schema_hash)
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
        client.close()
        thread.join(timeout=5)

    bad_client, bad_thread = start_raw_torch_client(tmp_path, view)
    try:
        write_frame(bad_client, FRAME_HELLO, make_hello(view, FeatureSchemaHash.from_bytes(b"x" * 32)).encode())
        frame_type, payload = read_frame(bad_client, bytearray())
        assert frame_type == FRAME_ERROR
        code, _ = decode_error(payload)
        assert code == ERROR_SCHEMA
    finally:
        bad_client.close()
        bad_thread.join(timeout=5)


def publish_random_checkpoint(root: Path, view: BatchView):
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
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    torch.manual_seed(11)
    model = build_model(schema, arch)
    return publish_checkpoint(
        root,
        model.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=view.feature_schema_hash,
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
) -> tuple[socket.socket, threading.Thread]:
    client, thread = start_raw_torch_client(tmp_path, view)
    write_frame(client, FRAME_HELLO, make_hello(view, schema_hash).encode())
    frame_type, payload = read_frame(client, bytearray())
    assert frame_type == FRAME_HELLO_ACK
    assert struct.unpack_from("<I", payload, 0)[0] == PROTOCOL_VERSION
    del payload
    return client, thread


def start_raw_torch_client(tmp_path: Path, view: BatchView) -> tuple[socket.socket, threading.Thread]:
    socket_path = tmp_path / f"eval-{len(list(tmp_path.glob('*.sock')))}.sock"
    ready = threading.Event()
    backend = TorchBackend(
        DirectorySource(tmp_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_model=torch.cuda.is_available(),
        max_batch=view.batch_capacity,
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
    return client, thread


def make_hello(view: BatchView, schema_hash: FeatureSchemaHash) -> Hello:
    return Hello(
        protocol_version=PROTOCOL_VERSION,
        encoding_version=ENCODING_VERSION,
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
