from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from gz.checkpoints.manifest import CheckpointManifest, WeightsInfo
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

    latest = {
        "version_dir": version_dir,
        "model_version": manifest.model_version.hex(),
    }
    latest_tmp = root / "latest.json.tmp"
    latest_tmp.write_text(json.dumps(latest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _fsync_file(latest_tmp)
    os.replace(latest_tmp, root / "latest.json")
    _fsync_dir(root)
    return manifest


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
