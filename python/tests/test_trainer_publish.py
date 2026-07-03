from __future__ import annotations

from pathlib import Path

import pytest

from gz.checkpoints import DirectorySource
from gz.codec import FeatureSchemaConfig
from gz.common import FeatureSchemaHash
from gz.model.exphormer import ArchConfig, build_model
from gz.trainer.publish import EmaWeights, publish_ema
from python.tests.test_checkpoints import schema_config

torch = pytest.importorskip("torch")


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
    )

    assert DirectorySource(tmp_path).resolve_latest().manifest == second
    assert first.model_version != second.model_version
