from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gz.codec import FeatureSchemaConfig, SchemaConfigError
from gz.common import (
    ActionSetHash,
    EngineIdentity,
    EngineId,
    EngineVersion,
    FeatureSchemaHash,
    ModelVersion,
)

MANIFEST_VERSION = 1


class ManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WeightsInfo:
    filename: str
    bytes: int
    blake2b_256: str
    format: str = "safetensors"

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "bytes": self.bytes,
            "blake2b_256": self.blake2b_256,
            "format": self.format,
        }

    @classmethod
    def from_dict(cls, value: object) -> WeightsInfo:
        if not isinstance(value, dict):
            raise ManifestError("weights must be an object")
        if set(value) != {"filename", "bytes", "blake2b_256", "format"}:
            raise ManifestError("weights fields mismatch")
        filename = _str(value, "filename")
        if "/" in filename or filename in {"", ".", ".."}:
            raise ManifestError("bad weights filename")
        size = _int(value, "bytes")
        if size <= 0:
            raise ManifestError("weights bytes must be positive")
        digest = _hex(value, "blake2b_256", 32)
        fmt = _str(value, "format")
        if fmt != "safetensors":
            raise ManifestError("unsupported weights format")
        return cls(filename=filename, bytes=size, blake2b_256=digest, format=fmt)


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    model_version: ModelVersion
    arch_name: str
    arch_config: dict[str, Any]
    arch_config_hash: str
    feature_schema: FeatureSchemaConfig
    feature_schema_hash: FeatureSchemaHash
    engine_identity: EngineIdentity
    training_step: int
    run_id: str
    weights: WeightsInfo
    manifest_version: int = MANIFEST_VERSION

    @property
    def engine_id(self) -> EngineId:
        return self.engine_identity.engine_id

    @property
    def engine_version(self) -> EngineVersion:
        return self.engine_identity.engine_version

    @property
    def action_set_hash(self) -> ActionSetHash:
        return self.engine_identity.action_set_hash

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "model_version": self.model_version.hex(),
            "arch": {
                "name": self.arch_name,
                "config": self.arch_config,
                "arch_config_hash": self.arch_config_hash,
            },
            "feature_schema": self.feature_schema.to_dict(),
            "feature_schema_hash": self.feature_schema_hash.hex(),
            "engine_id": self.engine_id.hex(),
            "engine_version": self.engine_version.hex(),
            "action_set_hash": self.action_set_hash.hex(),
            "training_step": self.training_step,
            "run_id": self.run_id,
            "weights": self.weights.to_dict(),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"

    @classmethod
    def from_dict(cls, value: object) -> CheckpointManifest:
        if not isinstance(value, dict):
            raise ManifestError("manifest must be an object")
        required = {
            "manifest_version",
            "model_version",
            "arch",
            "feature_schema",
            "feature_schema_hash",
            "engine_id",
            "engine_version",
            "action_set_hash",
            "training_step",
            "run_id",
            "weights",
        }
        if set(value) != required:
            raise ManifestError("manifest fields mismatch")
        if _int(value, "manifest_version") != MANIFEST_VERSION:
            raise ManifestError("unsupported manifest version")
        arch = value["arch"]
        if not isinstance(arch, dict) or set(arch) != {"name", "config", "arch_config_hash"}:
            raise ManifestError("arch fields mismatch")
        config = arch["config"]
        if not isinstance(config, dict):
            raise ManifestError("arch config must be an object")
        try:
            feature_schema = FeatureSchemaConfig.from_dict(value["feature_schema"])
        except SchemaConfigError as error:
            raise ManifestError(str(error)) from error
        step = _int(value, "training_step")
        if step < 0:
            raise ManifestError("training_step must be non-negative")
        run_id = _str(value, "run_id")
        if not run_id:
            raise ManifestError("run_id must be non-empty")
        try:
            engine_identity = EngineIdentity.from_parts(
                EngineId.from_hex(_hex(value, "engine_id", 16)),
                EngineVersion.from_hex(_hex(value, "engine_version", 16)),
                ActionSetHash.from_hex(_hex(value, "action_set_hash", 32)),
            )
        except ValueError as error:
            raise ManifestError(str(error)) from error
        return cls(
            manifest_version=MANIFEST_VERSION,
            model_version=ModelVersion.from_hex(_hex(value, "model_version", 16)),
            arch_name=_str(arch, "name"),
            arch_config=dict(config),
            arch_config_hash=_hex(arch, "arch_config_hash", 32),
            feature_schema=feature_schema,
            feature_schema_hash=FeatureSchemaHash.from_hex(_hex(value, "feature_schema_hash", 32)),
            engine_identity=engine_identity,
            training_step=step,
            run_id=run_id,
            weights=WeightsInfo.from_dict(value["weights"]),
        )

    @classmethod
    def read(cls, path: str | Path) -> CheckpointManifest:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ManifestError("invalid manifest json") from error
        return cls.from_dict(value)


def _int(value: dict[str, object], name: str) -> int:
    field = value[name]
    if not isinstance(field, int):
        raise ManifestError(f"{name} must be an integer")
    return field


def _str(value: dict[str, object], name: str) -> str:
    field = value[name]
    if not isinstance(field, str):
        raise ManifestError(f"{name} must be a string")
    return field


def _hex(value: dict[str, object], name: str, byte_len: int) -> str:
    field = _str(value, name)
    if len(field) != byte_len * 2:
        raise ManifestError(f"{name} length mismatch")
    try:
        bytes.fromhex(field)
    except ValueError as error:
        raise ManifestError(f"{name} is not hex") from error
    return field
