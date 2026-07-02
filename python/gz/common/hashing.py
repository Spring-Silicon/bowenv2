from __future__ import annotations

import hashlib
from pathlib import Path

from gz.common.tags import FeatureSchemaHash, ModelVersion

_MODEL_VERSION_DOMAIN = b"gz-model-version-v1"


def model_version(
    arch_config_hash: bytes,
    feature_schema_hash: bytes | FeatureSchemaHash,
    weights_hash: bytes,
) -> ModelVersion:
    hasher = hashlib.blake2b(digest_size=32)
    _update_chunk(hasher, _MODEL_VERSION_DOMAIN)
    _update_chunk(hasher, arch_config_hash)
    if isinstance(feature_schema_hash, FeatureSchemaHash):
        feature_schema_hash = bytes(feature_schema_hash)
    _update_chunk(hasher, feature_schema_hash)
    _update_chunk(hasher, weights_hash)
    return ModelVersion(hasher.digest()[:16])


def file_blake2b(path: str | Path) -> str:
    hasher = hashlib.blake2b(digest_size=32)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _update_chunk(hasher: object, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)
