from __future__ import annotations

import struct
from pathlib import Path

import pytest

from gz.codec import TargetsView
from gz.codec.batch import EncodingError
from gz.proto.frames import ENCODING_VERSION

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_targets_view_parse_and_zero_copy() -> None:
    raw = bytearray(make_targets())
    view = TargetsView.parse(raw)

    assert view.capacity == 2
    assert view.row_count == 2
    assert view.max_actions == 3
    assert view.policy.tolist() == [[0.75, 0.25, 0.0], [1.0, 0.0, 0.0]]
    assert view.value.tolist() == [1.0, -1.0]
    assert view.value_valid.tolist() == [1, 1]
    assert view.reward.tolist() == [2.5, -3.0]

    struct.pack_into("<f", raw, _layout(2, 3)["policy"], 0.25)
    assert view.policy[0, 0] == pytest.approx(0.25)


def test_targets_view_rejects_bad_headers() -> None:
    valid = make_targets()
    with pytest.raises(EncodingError, match="bad target magic"):
        TargetsView.parse(b"BAD!" + valid[4:])
    with pytest.raises(EncodingError, match="unsupported target version"):
        bad = bytearray(valid)
        struct.pack_into("<I", bad, 4, ENCODING_VERSION + 1)
        TargetsView.parse(bad)
    with pytest.raises(EncodingError, match="zero target capacity"):
        bad = bytearray(valid)
        struct.pack_into("<I", bad, 8, 0)
        TargetsView.parse(bad)
    with pytest.raises(EncodingError, match="bad target length"):
        TargetsView.parse(valid[:-4])


def test_committed_gzft_fixture() -> None:
    view = TargetsView.parse((FIXTURES / "targets.gzft").read_bytes())

    assert view.capacity == 2
    assert view.row_count == 2
    assert view.max_actions == 3
    assert view.policy.tolist() == [[0.75, 0.25, 0.0], [1.0, 0.0, 0.0]]
    assert view.value.tolist() == [1.0, -1.0]
    assert view.value_valid.tolist() == [1, 1]
    assert view.reward.tolist() == [2.5, -3.0]


def make_targets() -> bytes:
    b, a = 2, 3
    layout = _layout(b, a)
    out = bytearray(layout["total_len"])
    struct.pack_into("<4sIIII", out, 0, b"GZFT", ENCODING_VERSION, b, 2, a)
    _f32(out, layout["policy"], [0.75, 0.25, 0.0, 1.0, 0.0, 0.0])
    _f32(out, layout["value"], [1.0, -1.0])
    out[layout["value_valid"] : layout["value_valid"] + 2] = b"\x01\x01"
    _f32(out, layout["reward"], [2.5, -3.0])
    return bytes(out)


def _layout(b: int, a: int) -> dict[str, int]:
    cursor = 20
    out = {}
    for name, size in [
        ("policy", b * a * 4),
        ("value", b * 4),
        ("value_valid", b),
        ("reward", b * 4),
    ]:
        cursor = _align4(cursor)
        out[name] = cursor
        cursor += size
    out["total_len"] = _align4(cursor)
    return out


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _f32(out: bytearray, offset: int, values: list[float]) -> None:
    for index, value in enumerate(values):
        struct.pack_into("<f", out, offset + index * 4, value)
