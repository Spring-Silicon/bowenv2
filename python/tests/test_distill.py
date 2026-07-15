from __future__ import annotations

from pathlib import Path

import pytest

from gz.trainer.distill import generate_dataset, load_distill_config


def write_config(path: Path, extra: str = "") -> None:
    path.write_text(
        f"""
[trainer]
value_weight = 0.0
value_batch = 0
max_reuse = 0.0

[selfplay]
max_candidates = 1023

[distill]
states = 17
workers = 3
teacher = "reducing-uniform"
seed = 9
max_steps = 64
position_features = true
{extra}

[paths]
run_dir = "runs/distill-test"
replay_dir = "runs/distill-data/replay"
graphzero_bin = "target/release/graphzero"
""",
        encoding="utf-8",
    )


def test_distill_config_reuses_run_config_and_parses_dataset_controls(tmp_path: Path) -> None:
    path = tmp_path / "distill.toml"
    write_config(path)

    run_config, distill = load_distill_config(path)

    assert distill.states == 17
    assert distill.workers == 3
    assert distill.teacher == "reducing-uniform"
    assert run_config.trainer.value_weight == 0.0
    assert run_config.paths.replay_dir == Path.cwd() / "runs/distill-data/replay"


def test_distill_config_rejects_value_training(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        "[trainer]\nvalue_weight = 1.0\n\n[distill]\nstates = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy distillation"):
        load_distill_config(path)


def test_generate_dataset_passes_the_complete_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "distill.toml"
    write_config(path)
    run_config, distill = load_distill_config(path)
    observed = None

    def fake_run(command: list[str], check: bool) -> None:
        nonlocal observed
        observed = (command, check)

    monkeypatch.setattr("gz.trainer.distill.subprocess.run", fake_run)

    generate_dataset(run_config, distill)

    assert observed is not None
    command, check = observed
    assert check is True
    assert command[:2] == ["target/release/graphzero", "distill-generate"]
    assert command[command.index("--states") + 1] == "17"
    assert command[command.index("--max-candidates") + 1] == "1023"
    assert command[command.index("--position-features") + 1] == "true"
