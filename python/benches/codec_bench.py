from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from gz.codec import BatchView, OutputEncoder  # noqa: E402
from gz.model.stub import stub  # noqa: E402
from gz.proto.frames import ENCODING_VERSION  # noqa: E402

NODES = 32
EDGES = 64
ACTIONS = 256
SUBJECTS = 8
ATTR_DIM = 1
SCHEMA_HASH = b"b" * 32


def main() -> int:
    for capacity, iterations in [(64, 200), (256, 80)]:
        batch = make_batch(capacity)
        encoder = OutputEncoder(capacity, ACTIONS)
        start = time.perf_counter()
        for _ in range(iterations):
            view = BatchView.parse(batch)
            values, logits = stub(view)
            encoder.encode(values, logits, view.row_count)
        elapsed = time.perf_counter() - start
        rows = capacity * iterations
        mb = len(batch) * iterations / (1024 * 1024)
        print(
            "B={capacity} batch_bytes={batch_bytes} iterations={iterations} "
            "rows_per_s={rows_per_s:.3f} mb_per_s={mb_per_s:.3f} "
            "us_per_batch={us_per_batch:.3f}".format(
                capacity=capacity,
                batch_bytes=len(batch),
                iterations=iterations,
                rows_per_s=rows / elapsed,
                mb_per_s=mb / elapsed,
                us_per_batch=elapsed * 1_000_000 / iterations,
            )
        )
    return 0


def make_batch(capacity: int) -> bytes:
    layout = _layout(capacity)
    out = bytearray(layout["total_len"])
    struct.pack_into(
        "<4sI32sIIIIIII",
        out,
        0,
        b"GZFB",
        ENCODING_VERSION,
        SCHEMA_HASH,
        capacity,
        capacity,
        NODES,
        EDGES,
        ACTIONS,
        SUBJECTS,
        ATTR_DIM,
    )

    _array(out, layout["node_count"], "<u4", (capacity,)).fill(NODES)
    tokens = _array(out, layout["node_tokens"], "<u2", (capacity, NODES))
    tokens[:] = (np.arange(NODES, dtype=np.uint16) % 31) + 1
    attrs = _array(out, layout["node_attrs"], "<f4", (capacity, NODES, ATTR_DIM))
    attrs[:, :, 0] = np.linspace(-1.0, 1.0, NODES, dtype=np.float32)

    _array(out, layout["edge_count"], "<u4", (capacity,)).fill(EDGES)
    edge_src = _array(out, layout["edge_src"], "<u4", (capacity, EDGES))
    edge_dst = _array(out, layout["edge_dst"], "<u4", (capacity, EDGES))
    edge_type = _array(out, layout["edge_type"], "u1", (capacity, EDGES))
    edge_src[:] = np.arange(EDGES, dtype=np.uint32) % NODES
    edge_dst[:] = (np.arange(EDGES, dtype=np.uint32) + 1) % NODES
    edge_type[:] = np.arange(EDGES, dtype=np.uint8) % 2

    _array(out, layout["action_count"], "<u4", (capacity,)).fill(ACTIONS)
    action_kind = _array(out, layout["action_kind"], "<u4", (capacity, ACTIONS))
    action_kind[:] = (np.arange(ACTIONS, dtype=np.uint32) % 10) + 2
    action_kind[:, -1] = 1
    action_prior = _array(out, layout["action_prior"], "<f4", (capacity, ACTIONS))
    action_prior[:] = np.linspace(-1.0, 1.0, ACTIONS, dtype=np.float32)
    action_prior[:, -1] = 0.0
    subject_count = _array(out, layout["subject_count"], "u1", (capacity, ACTIONS))
    subject_count.fill(1)
    subject_count[:, -1] = 0
    subjects = _array(out, layout["action_subjects"], "<u4", (capacity, ACTIONS, SUBJECTS))
    subjects.fill(0xFFFFFFFF)
    subjects[:, :, 0] = np.arange(ACTIONS, dtype=np.uint32) % NODES
    subjects[:, -1, :] = 0xFFFFFFFF
    position = _array(out, layout["position"], "<f4", (capacity, 4))
    position[:, :] = [0.0, 0.0, 1.0, 0.125]
    return bytes(out)


def _array(out: bytearray, offset: int, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(out, dtype=np.dtype(dtype), count=int(np.prod(shape)), offset=offset).reshape(shape)


def _layout(capacity: int) -> dict[str, int]:
    cursor = 68
    out = {}
    for name, size in [
        ("node_count", capacity * 4),
        ("node_tokens", capacity * NODES * 2),
        ("node_attrs", capacity * NODES * ATTR_DIM * 4),
        ("edge_count", capacity * 4),
        ("edge_src", capacity * EDGES * 4),
        ("edge_dst", capacity * EDGES * 4),
        ("edge_type", capacity * EDGES),
        ("action_count", capacity * 4),
        ("action_kind", capacity * ACTIONS * 4),
        ("action_prior", capacity * ACTIONS * 4),
        ("subject_count", capacity * ACTIONS),
        ("action_subjects", capacity * ACTIONS * SUBJECTS * 4),
        ("position", capacity * 16),
    ]:
        cursor = _align4(cursor)
        out[name] = cursor
        cursor += size
    out["total_len"] = _align4(cursor)
    return out


def _align4(value: int) -> int:
    return (value + 3) & ~3


if __name__ == "__main__":
    raise SystemExit(main())
