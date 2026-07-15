from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from gz.checkpoints.manifest import CheckpointManifest, ManifestError
from gz.common import file_blake2b, model_version


@dataclass(frozen=True, slots=True)
class ResolvedCheckpoint:
    manifest: CheckpointManifest
    weights_path: Path


class CheckpointSource(ABC):
    @abstractmethod
    def resolve_latest(self) -> ResolvedCheckpoint:
        raise NotImplementedError


class DirectorySource(CheckpointSource):
    def __init__(self, root: str | Path, pointer: str | Path = "latest.json") -> None:
        self.root = Path(root)
        pointer_path = Path(pointer)
        self.pointer = pointer_path if pointer_path.is_absolute() else self.root / pointer_path

    def resolve_latest(self) -> ResolvedCheckpoint:
        latest_path = self.pointer
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ManifestError(f"missing {latest_path.name}") from error
        except json.JSONDecodeError as error:
            raise ManifestError(f"invalid {latest_path.name}") from error
        if not isinstance(latest, dict) or set(latest) != {"version_dir", "model_version"}:
            raise ManifestError("checkpoint pointer fields mismatch")
        version_dir = latest["version_dir"]
        model_version_hex = latest["model_version"]
        if not isinstance(version_dir, str) or "/" in version_dir or not version_dir:
            raise ManifestError("bad latest version_dir")
        if not isinstance(model_version_hex, str) or len(model_version_hex) != 32:
            raise ManifestError("bad latest model_version")
        resolved = self.resolve_version(version_dir)
        if resolved.manifest.model_version.hex() != model_version_hex:
            raise ManifestError("checkpoint pointer model_version mismatch")
        return resolved

    def resolve_version(self, version_dir: str) -> ResolvedCheckpoint:
        root = self.root / version_dir
        manifest = CheckpointManifest.read(root / "manifest.json")
        weights_path = root / manifest.weights.filename
        if not weights_path.is_file():
            raise ManifestError("missing weights file")
        if weights_path.stat().st_size != manifest.weights.bytes:
            raise ManifestError("weights byte length mismatch")
        weights_hash = file_blake2b(weights_path)
        if weights_hash != manifest.weights.blake2b_256:
            raise ManifestError("weights hash mismatch")
        expected = model_version(
            bytes.fromhex(manifest.arch_config_hash),
            manifest.feature_schema_hash,
            bytes.fromhex(weights_hash),
        )
        if expected != manifest.model_version:
            raise ManifestError("model_version mismatch")
        return ResolvedCheckpoint(manifest=manifest, weights_path=weights_path)
