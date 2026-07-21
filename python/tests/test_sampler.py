from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path

import pytest

from gz.codec import FeatureSchemaConfig
from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash
from gz.proto import read_frame, write_frame
from gz.trainer.sampler import SAMPLE_PROTOCOL_VERSION, SampleClient, decode_ack, step_seed
from python.tests.test_codec import SCHEMA_HASH, _layout, make_batch
from python.tests.test_targets import make_targets


def test_sample_client_handshake_and_result_owns_frame(tmp_path: Path) -> None:
    socket_path = tmp_path / "sample.sock"
    raw_batch = make_batch(attr_dim=1)
    changed_batch = bytearray(raw_batch)
    struct.pack_into("<I", changed_batch, _layout(2, 3, 2, 3, 2, 1)["node_count"], 1)
    raw_targets = make_targets()
    thread = serve_samples(
        socket_path,
        produced_rows=[2],
        responses=[(raw_batch, raw_targets), (bytes(changed_batch), raw_targets)],
    )
    client = SampleClient(socket_path, startup_timeout=1.0, backoff=0.01)
    try:
        ack = client.wait_until_ready(1)
        first = client.sample(1, 2, 99)
        first_node_count = first.batch.node_count.copy()
        second = client.sample(1, 2, 99)

        assert ack.feature_schema == schema_config()
        assert ack.feature_schema_hash == FeatureSchemaHash.from_bytes(SCHEMA_HASH)
        assert ack.engine_id == EngineId.from_bytes(b"e" * 16)
        assert ack.engine_version == EngineVersion.from_bytes(b"v" * 16)
        assert ack.action_set_hash == ActionSetHash.from_bytes(b"a" * 32)
        assert ack.value_sign_accuracy_early_ema == 0.75
        assert ack.value_sign_accuracy_late_ema == 0.25
        assert ack.symmetric_selfplay is not None
        assert ack.symmetric_selfplay.p1_win_rate_ema == pytest.approx(0.4)
        assert ack.symmetric_selfplay.p2_win_rate_ema == pytest.approx(0.35)
        assert ack.symmetric_selfplay.draw_rate_ema == pytest.approx(0.25)
        assert ack.symmetric_selfplay.seat_advantage_ema == pytest.approx(0.05)
        assert ack.symmetric_selfplay.mean_terminal_cost_ema == 61.0
        assert ack.symmetric_selfplay.game_len_ema == 161.0
        assert first.produced_rows == 2
        assert first.batch.node_count.tolist() == first_node_count.tolist()
        assert first.batch.node_count.tolist() != second.batch.node_count.tolist()
        assert first.targets.policy.tolist() == second.targets.policy.tolist()
    finally:
        client.close()
        thread.join(timeout=1)


def test_sample_client_startup_wait_reconnects_until_enough_rows(tmp_path: Path) -> None:
    socket_path = tmp_path / "sample.sock"
    thread = serve_samples(socket_path, produced_rows=[0, 4], responses=[])
    client = SampleClient(socket_path, startup_timeout=1.0, backoff=0.01)
    try:
        ack = client.wait_until_ready(4)

        assert ack.produced_rows == 4
    finally:
        client.close()
        thread.join(timeout=1)


def test_step_seed_is_deterministic_and_step_sensitive() -> None:
    assert step_seed(7, 3) == step_seed(7, 3)
    assert step_seed(7, 3) != step_seed(7, 4)
    assert step_seed(7, 3, "value-sample") != step_seed(7, 3)
    assert step_seed(7, 3, "value-sample") != step_seed(7, 3, "value-orientation")


def test_decode_ack_rejects_truncated() -> None:
    try:
        decode_ack(memoryview(b"short"))
    except Exception as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("decode_ack accepted truncated payload")


def test_decode_ack_allows_absent_symmetric_metrics() -> None:
    payload = bytearray(ack_payload(3))
    struct.pack_into("<I", payload, 116, 0)
    payload[120:160] = bytes(40)

    assert decode_ack(memoryview(payload)).symmetric_selfplay is None


def serve_samples(
    socket_path: Path,
    *,
    produced_rows: list[int],
    responses: list[tuple[bytes, bytes]],
) -> threading.Thread:
    ready = threading.Event()

    def run() -> None:
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            listener.listen(1)
            ready.set()
            response_index = 0
            for produced in produced_rows:
                conn, _ = listener.accept()
                with conn:
                    frame_type, _payload = read_frame(conn, bytearray())
                    assert frame_type == 1
                    write_frame(conn, 2, ack_payload(produced))
                    if produced == 0:
                        continue
                    while response_index < len(responses):
                        frame_type, payload = read_frame(conn, bytearray())
                        assert frame_type == 3
                        assert struct.unpack_from("<I", payload, 0)[0] > 0
                        assert struct.unpack_from("<Q", payload, 4)[0] == 2
                        batch, targets = responses[response_index]
                        response_index += 1
                        write_frame(conn, 4, struct.pack("<I", len(batch)), batch, targets)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(timeout=1)
    return thread


def ack_payload(produced_rows: int) -> bytes:
    return (
        struct.pack("<I", SAMPLE_PROTOCOL_VERSION)
        + SCHEMA_HASH
        + struct.pack("<I", 2)
        + struct.pack("<Q", produced_rows)
        + struct.pack("<Q", 6)
        + struct.pack("<Q", 2)
        + struct.pack("<fffff", 87.5, 12.0, 0.25, 0.4, 42.5)
        + struct.pack("<f", 61.0)
        + bytes(20)
        + struct.pack("<ff", 0.75, 0.25)
        + struct.pack("<I", 1)
        + struct.pack("<10f", 0.4, 0.35, 0.25, 60.0, 62.0, 2.0, 50.0, 80.0, 81.0, 1.0)
        + b"e" * 16
        + b"v" * 16
        + b"a" * 32
        + schema_config().encode()
    )


def schema_config() -> FeatureSchemaConfig:
    return FeatureSchemaConfig(
        name="sample-test",
        node_vocab_size=7,
        node_attr_dim=1,
        edge_type_count=2,
        action_kind_vocab_size=8,
        max_nodes=3,
        max_edges=2,
        max_actions=3,
        max_subjects=2,
        expander_degree=0,
        expander_seed=0,
    )
