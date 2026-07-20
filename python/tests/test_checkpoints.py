from __future__ import annotations

import json
from pathlib import Path

import pytest

from gz.checkpoints import (
    DirectorySource,
    ManifestError,
    prune_checkpoints,
    publish_checkpoint,
)
from gz.checkpoints.manifest import CheckpointManifest
from gz.checkpoints.publish import ensure_checkpoint_pointer, promote_checkpoint_pointer
from gz.codec import FeatureSchemaConfig
from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash
from gz.model.exphormer import ArchConfig, build_model

torch = pytest.importorskip("torch")


def test_publish_and_resolve_roundtrip(tmp_path: Path) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)

    manifest = publish_checkpoint(
        tmp_path,
        model.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        training_step=7,
        run_id="run",
    )

    resolved = DirectorySource(tmp_path).resolve_latest()

    assert resolved.manifest == manifest
    assert resolved.weights_path.name == "model.safetensors"
    assert (tmp_path / "latest.json").is_file()


def test_weights_tampering_is_detected(tmp_path: Path) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)
    publish_checkpoint(
        tmp_path,
        model.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        training_step=0,
        run_id="run",
    )
    weights = tmp_path / "version_0" / "model.safetensors"
    with weights.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"X")

    with pytest.raises(ManifestError, match="weights hash mismatch"):
        DirectorySource(tmp_path).resolve_latest()


def test_manifest_validation_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"manifest_version": 1}), encoding="utf-8")

    with pytest.raises(ManifestError, match="manifest fields mismatch"):
        CheckpointManifest.read(path)


def test_latest_replace_preserves_old_version_and_model_version_is_stable(tmp_path: Path) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)
    kwargs = dict(
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        run_id="run",
    )
    first = publish_checkpoint(tmp_path, model.state_dict(), training_step=0, **kwargs)
    second = publish_checkpoint(tmp_path, model.state_dict(), training_step=1, **kwargs)
    state = model.state_dict()
    first_key = next(iter(state))
    state[first_key] = state[first_key] + 1.0
    third = publish_checkpoint(tmp_path, state, training_step=2, **kwargs)

    source = DirectorySource(tmp_path)
    assert source.resolve_version("version_0").manifest == first
    assert source.resolve_latest().manifest == third
    assert first.model_version == second.model_version
    assert third.model_version != first.model_version


def test_named_checkpoint_pointer_stays_frozen_until_exact_promotion(tmp_path: Path) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)
    kwargs = dict(
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        run_id="run",
    )
    first = publish_checkpoint(tmp_path, model.state_dict(), training_step=0, **kwargs)
    ensure_checkpoint_pointer(tmp_path, "best.json")
    state = model.state_dict()
    first_key = next(iter(state))
    state[first_key] = state[first_key] + 1.0
    second = publish_checkpoint(tmp_path, state, training_step=1, **kwargs)

    best = DirectorySource(tmp_path, pointer="best.json")
    assert best.resolve_latest().manifest == first
    assert DirectorySource(tmp_path).resolve_latest().manifest == second

    promote_checkpoint_pointer(tmp_path, "best.json", second.model_version.hex())
    assert best.resolve_latest().manifest == second


def test_prune_checkpoints_keeps_newest_named_and_in_flight_versions(
    tmp_path: Path,
) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)
    kwargs = dict(
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        run_id="run",
    )
    base_state = model.state_dict()
    first_key = next(iter(base_state))
    manifests = []
    for step in range(6):
        state = dict(base_state)
        state[first_key] = base_state[first_key] + float(step)
        manifests.append(
            publish_checkpoint(tmp_path, state, training_step=step, **kwargs)
        )
    promote_checkpoint_pointer(
        tmp_path,
        "best.json",
        manifests[0].model_version.hex(),
    )
    promote_checkpoint_pointer(
        tmp_path,
        "arena.json",
        manifests[1].model_version.hex(),
    )

    removed = prune_checkpoints(
        tmp_path,
        2,
        protected_model_versions=(manifests[2].model_version.hex(),),
    )

    assert removed == ("version_3",)
    assert {
        path.name
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.startswith("version_")
    } == {"version_0", "version_1", "version_2", "version_4", "version_5"}
    assert DirectorySource(tmp_path).resolve_latest().manifest == manifests[5]
    assert (
        DirectorySource(tmp_path, pointer="best.json").resolve_latest().manifest
        == manifests[0]
    )
    assert (
        DirectorySource(tmp_path, pointer="arena.json").resolve_latest().manifest
        == manifests[1]
    )
    assert DirectorySource(tmp_path).resolve_version("version_2").manifest == manifests[2]

    state = dict(base_state)
    state[first_key] = base_state[first_key] + 6.0
    publish_checkpoint(tmp_path, state, training_step=6, **kwargs)
    assert DirectorySource(tmp_path).resolve_latest().weights_path.parent.name == "version_6"


def test_permanent_step_pointer_survives_rolling_prune(tmp_path: Path) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)
    kwargs = dict(
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        run_id="run",
    )
    base_state = model.state_dict()
    first_key = next(iter(base_state))
    manifests = []
    for step in (1000, 1001, 1002):
        state = dict(base_state)
        state[first_key] = base_state[first_key] + float(step)
        manifests.append(
            publish_checkpoint(
                tmp_path,
                state,
                training_step=step,
                checkpoint_pointers=("step_1000.json",) if step == 1000 else (),
                **kwargs,
            )
        )

    removed = prune_checkpoints(tmp_path, 1)

    assert removed == ("version_1",)
    assert {
        path.name
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.startswith("version_")
    } == {"version_0", "version_2"}
    milestone = DirectorySource(tmp_path, pointer="step_1000.json").resolve_latest()
    assert milestone.manifest == manifests[0]
    assert DirectorySource(tmp_path).resolve_latest().manifest == manifests[2]


def test_prune_checkpoints_validates_pointers_before_deleting(tmp_path: Path) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)
    kwargs = dict(
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        run_id="run",
    )
    base_state = model.state_dict()
    first_key = next(iter(base_state))
    manifests = []
    for step in range(3):
        state = dict(base_state)
        state[first_key] = base_state[first_key] + float(step)
        manifests.append(
            publish_checkpoint(tmp_path, state, training_step=step, **kwargs)
        )
    (tmp_path / "best.json").write_text(
        json.dumps(
            {
                "version_dir": "version_99",
                "model_version": manifests[0].model_version.hex(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="best.json references missing checkpoint"):
        prune_checkpoints(tmp_path, 1)

    assert {
        path.name
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.startswith("version_")
    } == {"version_0", "version_1", "version_2"}


def schema_config() -> FeatureSchemaConfig:
    return FeatureSchemaConfig(
        name="test",
        node_vocab_size=8,
        node_attr_dim=0,
        edge_type_count=3,
        action_kind_vocab_size=8,
        max_nodes=4,
        max_edges=12,
        max_actions=4,
        max_subjects=2,
        expander_degree=2,
        expander_seed=0,
    )


def feature_hash() -> FeatureSchemaHash:
    return FeatureSchemaHash.from_bytes(b"f" * 32)
