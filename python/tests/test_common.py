from __future__ import annotations

import hashlib

import pytest

from gz.common import (
    ActionSetHash,
    EngineIdentity,
    EngineId,
    EngineVersion,
    FeatureSchemaHash,
    file_blake2b,
    model_version,
)


def test_fixed_tags_hex_roundtrip_and_length_validation() -> None:
    tag = EngineId.from_hex("00" * 16)

    assert bytes(tag) == b"\x00" * 16
    assert str(tag) == "00" * 16
    assert EngineId.from_bytes(bytes(tag)) == tag
    with pytest.raises(ValueError):
        EngineId.from_bytes(b"\x00" * 15)


def test_model_version_is_deterministic_and_sensitive() -> None:
    schema = FeatureSchemaHash.from_bytes(b"s" * 32)
    first = model_version(b"a", schema, b"w")
    again = model_version(b"a", schema, b"w")
    changed = model_version(b"b", schema, b"w")

    assert first == again
    assert first != changed


def test_engine_identity_rejects_unspecified_all_zero_value() -> None:
    with pytest.raises(ValueError, match="must not be all zero"):
        EngineIdentity(
            EngineId.from_bytes(bytes(16)),
            EngineVersion.from_bytes(bytes(16)),
            ActionSetHash.from_bytes(bytes(32)),
        )


def test_file_blake2b_matches_hashlib(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"graphzero")

    expected = hashlib.blake2b(b"graphzero", digest_size=32).hexdigest()
    assert file_blake2b(path) == expected
