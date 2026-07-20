from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gz.checkpoints.manifest import CheckpointManifest, ManifestError, WeightsInfo
from gz.checkpoints.source import DirectorySource
from gz.checkpoints.weights import save_state_dict
from gz.codec import FeatureSchemaConfig
from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash, file_blake2b, model_version


def publish_checkpoint(
    root: str | Path,
    state_dict: dict[str, Any],
    *,
    arch_name: str,
    arch_config: dict[str, Any],
    arch_config_hash: bytes,
    feature_schema: FeatureSchemaConfig,
    feature_schema_hash: FeatureSchemaHash,
    engine_id: EngineId,
    engine_version: EngineVersion,
    action_set_hash: ActionSetHash,
    training_step: int,
    run_id: str,
    checkpoint_pointers: Iterable[str] = (),
) -> CheckpointManifest:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    version_dir = _next_version_dir(root)
    tmp = root / f"{version_dir}.tmp"
    final = root / version_dir
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    weights_name = "model.safetensors"
    weights_path = tmp / weights_name
    save_state_dict(weights_path, state_dict)
    _fsync_file(weights_path)
    weights_hash = file_blake2b(weights_path)
    version = model_version(arch_config_hash, feature_schema_hash, bytes.fromhex(weights_hash))
    manifest = CheckpointManifest(
        model_version=version,
        arch_name=arch_name,
        arch_config=arch_config,
        arch_config_hash=arch_config_hash.hex(),
        feature_schema=feature_schema,
        feature_schema_hash=feature_schema_hash,
        engine_id=engine_id,
        engine_version=engine_version,
        action_set_hash=action_set_hash,
        training_step=training_step,
        run_id=run_id,
        weights=WeightsInfo(
            filename=weights_name,
            bytes=weights_path.stat().st_size,
            blake2b_256=weights_hash,
        ),
    )

    manifest_path = tmp / "manifest.json"
    manifest_path.write_bytes(manifest.to_json_bytes())
    _fsync_file(manifest_path)
    os.replace(tmp, final)
    _fsync_dir(root)

    # Write archival aliases before advancing latest. If publication is
    # interrupted between them, resume still starts from the older checkpoint
    # while the completed milestone remains pinned.
    for pointer_name in checkpoint_pointers:
        _write_checkpoint_pointer(
            root,
            pointer_name,
            version_dir,
            manifest.model_version.hex(),
        )
    _write_checkpoint_pointer(
        root,
        "latest.json",
        version_dir,
        manifest.model_version.hex(),
    )
    return manifest


def ensure_checkpoint_pointer(root: str | Path, pointer_name: str) -> CheckpointManifest:
    root = Path(root)
    pointer = root / pointer_name
    if pointer.is_file():
        return DirectorySource(root, pointer=pointer).resolve_latest().manifest
    manifest = DirectorySource(root).resolve_latest().manifest
    promote_checkpoint_pointer(root, pointer_name, manifest.model_version.hex())
    return manifest


def promote_checkpoint_pointer(
    root: str | Path,
    pointer_name: str,
    model_version_hex: str,
) -> None:
    root = Path(root)
    version_dir = None
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("version_"):
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = CheckpointManifest.read(manifest_path)
        if manifest.model_version.hex() == model_version_hex:
            version_dir = child.name
            break
    if version_dir is None:
        raise ValueError(f"checkpoint version not found: {model_version_hex}")
    _write_checkpoint_pointer(root, pointer_name, version_dir, model_version_hex)


def _write_checkpoint_pointer(
    root: Path,
    pointer_name: str,
    version_dir: str,
    model_version_hex: str,
) -> None:
    if Path(pointer_name).name != pointer_name:
        raise ValueError("checkpoint pointer must be a file name")
    pointer = {"version_dir": version_dir, "model_version": model_version_hex}
    pointer_path = root / pointer_name
    pointer_tmp = root / f"{pointer_name}.tmp"
    pointer_tmp.write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _fsync_file(pointer_tmp)
    os.replace(pointer_tmp, pointer_path)
    _fsync_dir(root)


def prune_checkpoints(
    root: str | Path,
    retain: int,
    *,
    protected_model_versions: Iterable[str] = (),
) -> tuple[str, ...]:
    """Delete old unreferenced checkpoint generations.

    A zero retention disables pruning. Positive retention keeps the newest N
    generations plus every generation referenced by a named checkpoint pointer
    or protected by an in-flight consumer.
    """
    if retain < 0:
        raise ValueError("checkpoint retention must be non-negative")
    if retain == 0:
        return ()

    root = Path(root)
    versions: list[tuple[int, Path, CheckpointManifest]] = []
    for child in root.iterdir():
        suffix = child.name.removeprefix("version_")
        if (
            not child.is_dir()
            or not child.name.startswith("version_")
            or not suffix.isdigit()
        ):
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            raise ManifestError(f"missing manifest for {child.name}")
        versions.append((int(suffix), child, CheckpointManifest.read(manifest_path)))
    versions.sort(key=lambda item: item[0])

    by_dir = {path.name: manifest for _, path, manifest in versions}
    protected_dirs = {path.name for _, path, _ in versions[-retain:]}
    pointer_names = ["latest.json", "best.json", "arena.json"]
    pointer_names.extend(
        sorted(
            child.name
            for child in root.iterdir()
            if child.is_file() and _is_permanent_checkpoint_pointer(child.name)
        )
    )
    for pointer_name in pointer_names:
        target = _checkpoint_pointer_target(root, pointer_name, by_dir)
        if target is not None:
            protected_dirs.add(target)

    requested_versions = set(protected_model_versions)
    available_versions = {
        manifest.model_version.hex() for manifest in by_dir.values()
    }
    missing_versions = requested_versions - available_versions
    if missing_versions:
        missing = ", ".join(sorted(missing_versions))
        raise ManifestError(f"protected checkpoint version not found: {missing}")
    for directory, manifest in by_dir.items():
        if manifest.model_version.hex() in requested_versions:
            protected_dirs.add(directory)

    removals = [path for _, path, _ in versions if path.name not in protected_dirs]
    _remove_prune_tombstones(root)
    if not removals:
        return ()

    tombstones: list[Path] = []
    for path in removals:
        tombstone = root / f".{path.name}.prune"
        os.replace(path, tombstone)
        tombstones.append(tombstone)
    _fsync_dir(root)
    for tombstone in tombstones:
        shutil.rmtree(tombstone)
    _fsync_dir(root)
    return tuple(path.name for path in removals)


def _checkpoint_pointer_target(
    root: Path,
    pointer_name: str,
    manifests: dict[str, CheckpointManifest],
) -> str | None:
    pointer_path = root / pointer_name
    if not pointer_path.exists():
        return None
    if not pointer_path.is_file():
        raise ManifestError(f"invalid {pointer_name}")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ManifestError(f"invalid {pointer_name}") from error
    if not isinstance(pointer, dict) or set(pointer) != {"version_dir", "model_version"}:
        raise ManifestError(f"{pointer_name} fields mismatch")
    version_dir = pointer["version_dir"]
    model_version_hex = pointer["model_version"]
    if not isinstance(version_dir, str) or version_dir not in manifests:
        raise ManifestError(f"{pointer_name} references missing checkpoint")
    if not isinstance(model_version_hex, str):
        raise ManifestError(f"bad {pointer_name} model_version")
    try:
        raw_model_version = bytes.fromhex(model_version_hex)
    except ValueError as error:
        raise ManifestError(f"bad {pointer_name} model_version") from error
    if len(raw_model_version) != 16:
        raise ManifestError(f"bad {pointer_name} model_version")
    if manifests[version_dir].model_version.hex() != model_version_hex:
        raise ManifestError(f"{pointer_name} model_version mismatch")
    return version_dir


def _remove_prune_tombstones(root: Path) -> None:
    removed = False
    for child in root.iterdir():
        name = child.name
        if not name.startswith(".version_") or not name.endswith(".prune"):
            continue
        suffix = name.removeprefix(".version_").removesuffix(".prune")
        if not suffix.isdigit():
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed = True
    if removed:
        _fsync_dir(root)


def _is_permanent_checkpoint_pointer(name: str) -> bool:
    if not name.startswith("step_") or not name.endswith(".json"):
        return False
    return name.removeprefix("step_").removesuffix(".json").isdigit()


def _next_version_dir(root: Path) -> str:
    next_index = 0
    for child in root.iterdir():
        name = child.name
        if not child.is_dir() or not name.startswith("version_") or name.endswith(".tmp"):
            continue
        suffix = name.removeprefix("version_")
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)
    return f"version_{next_index}"


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
