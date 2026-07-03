from __future__ import annotations

from pathlib import Path

import pytest

from gz.trainer.driver import load_config


def test_load_config_defaults_and_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[trainer]
batch = 4
total_steps = 3

[selfplay]
lanes = 1
workers_per_lane = 1

[paths]
run_dir = "run"
graphzero_bin = "graphzero-test"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.trainer.batch == 4
    assert config.trainer.total_steps == 3
    assert config.selfplay.lanes == 1
    assert config.paths.run_dir == Path("run")
    assert config.paths.replay_dir == Path("run/replay")
    assert config.paths.checkpoint_dir == Path("run/checkpoints")
    assert config.paths.graphzero_bin == "graphzero-test"


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text("[trainer]\nunknown = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config fields"):
        load_config(config_path)
