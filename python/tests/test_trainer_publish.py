from __future__ import annotations

from pathlib import Path

import pytest

from gz.checkpoints import DirectorySource
from gz.codec import FeatureSchemaConfig
from gz.common import ActionSetHash, EngineIdentity, EngineId, EngineVersion, FeatureSchemaHash
from gz.model.exphormer import ArchConfig, build_model
from gz.trainer.publish import EmaWeights, publish_ema
from python.tests.test_checkpoints import schema_config

torch = pytest.importorskip("torch")


ENGINE_IDENTITY = EngineIdentity.from_parts(
    EngineId.from_bytes(b"e" * 16),
    EngineVersion.from_bytes(b"v" * 16),
    ActionSetHash.from_bytes(b"a" * 32),
)


def test_ema_weights_update_with_literal_arithmetic() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema = EmaWeights(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(4.0)

    ema.update(model)

    assert ema.state_dict()["weight"].item() == pytest.approx(3.0)


def test_publish_ema_roundtrips_and_version_changes(tmp_path: Path) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)
    ema = EmaWeights(model, decay=0.5)
    first = publish_ema(
        tmp_path,
        ema,
        schema=schema,
        schema_hash=FeatureSchemaHash.from_bytes(b"f" * 32),
        arch=arch,
        training_step=0,
        run_id="run",
        engine_identity=ENGINE_IDENTITY,
    )
    for tensor in model.state_dict().values():
        if tensor.is_floating_point():
            tensor.add_(1.0)
            break
    ema.update(model)
    second = publish_ema(
        tmp_path,
        ema,
        schema=schema,
        schema_hash=FeatureSchemaHash.from_bytes(b"f" * 32),
        arch=arch,
        training_step=1,
        run_id="run",
        engine_identity=ENGINE_IDENTITY,
    )

    assert DirectorySource(tmp_path).resolve_latest().manifest == second
    assert first.model_version != second.model_version


def test_ema_norms_report_param_and_update_magnitudes() -> None:
    torch = pytest.importorskip("torch")

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EmaWeights(model, 0.5)

    param_norm, update_norm = ema.norms(None)
    assert abs(param_norm - 2.0) < 1e-6  # sqrt(4 ones)
    assert update_norm == 0.0

    snapshot = ema.state_dict()
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)  # shadow -> 0.5*1 + 0.5*3 = 2.0 each

    param_norm, update_norm = ema.norms(snapshot)
    assert abs(param_norm - 4.0) < 1e-6  # sqrt(4 * 4)
    assert abs(update_norm - 2.0) < 1e-6  # delta 1.0 each -> sqrt(4)


def test_publish_ema_rejects_nonfinite_weights_before_writing(tmp_path: Path) -> None:
    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    model = build_model(schema, arch)
    ema = EmaWeights(model, decay=0.0)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    ema.update(model)

    with pytest.raises(RuntimeError, match="non-finite EMA tensor"):
        publish_ema(
            tmp_path,
            ema,
            schema=schema,
            schema_hash=FeatureSchemaHash.from_bytes(b"f" * 32),
            arch=arch,
            training_step=1,
            run_id="run",
            engine_identity=ENGINE_IDENTITY,
        )

    assert not (tmp_path / "latest.json").exists()
    assert not list(tmp_path.glob("version_*"))
