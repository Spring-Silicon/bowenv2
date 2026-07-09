from __future__ import annotations

from pathlib import Path

import pytest

from gz.trainer.driver import (
    WandbRun,
    _sample_window_rows,
    init_replay,
    load_config,
)


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
    # Paths are pinned to the trainer's cwd: children (the evaluator) run in
    # their own working directories, so relative paths must not cross over.
    assert config.paths.run_dir == Path.cwd() / "run"
    assert config.paths.replay_dir == Path.cwd() / "run/replay"
    assert config.paths.checkpoint_dir == Path.cwd() / "run/checkpoints"
    assert config.paths.sample_socket == Path.cwd() / "run/sample.sock"
    assert config.paths.graphzero_bin == "graphzero-test"


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text("[trainer]\nunknown = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config fields"):
        load_config(config_path)


def test_load_config_extends_one_base_with_recursive_table_merge(tmp_path: Path) -> None:
    base_dir = tmp_path / "bases"
    run_dir = tmp_path / "runs"
    base_dir.mkdir()
    run_dir.mkdir()
    (base_dir / "whittle.toml").write_text(
        """
[arch]
dim = 128
layers = 4
policy_head = "pointer"
value_input = "single"

[trainer]
batch = 256
window_rows = 50000
total_steps = 5000

[selfplay]
lanes = 44
workers_per_lane = 48
reference = "gated-policy"
root_mode = "fixed"

[paths]
run_dir = "runs/base"
graphzero_bin = "graphzero-base"
""",
        encoding="utf-8",
    )
    config_path = run_dir / "clean-arena.toml"
    config_path.write_text(
        """
extends = "../bases/whittle.toml"

[arch]
value_input = "pair"

[trainer]
window_rows = 10000

[selfplay]
no_backtrack = true

[paths]
run_dir = "runs/clean-arena"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.arch.dim == 128
    assert config.arch.policy_head == "pointer"
    assert config.arch.value_input == "pair"
    assert config.trainer.batch == 256
    assert config.trainer.window_rows == 10000
    assert config.selfplay.lanes == 44
    assert config.selfplay.no_backtrack is True
    assert config.paths.run_dir == Path.cwd() / "runs/clean-arena"
    assert config.paths.graphzero_bin == "graphzero-base"


def test_load_config_extends_is_one_layer_only(tmp_path: Path) -> None:
    base_dir = tmp_path / "bases"
    base_dir.mkdir()
    (base_dir / "root.toml").write_text("[paths]\nrun_dir = \"root\"\n", encoding="utf-8")
    (base_dir / "middle.toml").write_text(
        "extends = \"root.toml\"\n\n[paths]\nrun_dir = \"middle\"\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        "extends = \"bases/middle.toml\"\n\n[paths]\nrun_dir = \"run\"\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one layer"):
        load_config(config_path)


def test_load_config_extends_must_be_string(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text("extends = 7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extends must be a string"):
        load_config(config_path)


def test_load_config_parses_wandb_table(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[wandb]
project = "graphzero"
run_name = "curve-2"

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.wandb.project == "graphzero"
    assert config.wandb.run_name == "curve-2"
    assert config.wandb.entity == ""


def test_wandb_run_disabled_without_project(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text('[paths]\nrun_dir = "run"\n', encoding="utf-8")

    assert WandbRun.start(load_config(config_path)) is None


def test_wandb_run_maps_step_records_to_grouped_keys() -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.logged: list[tuple[dict, int]] = []
            self.finished = False

        def log(self, payload: dict, step: int) -> None:
            self.logged.append((payload, step))

        def finish(self) -> None:
            self.finished = True

    fake = FakeRun()
    run = WandbRun(fake)
    run.write(
        {
            "event": "step",
            "step": 7,
            "timestamp": 123.0,
            "policy_loss": 4.5,
            "rows_per_s": 200.0,
            "produced_rows": 4096,
        }
    )
    run.write({"event": "publish", "training_step": 10, "model_version": "ab"})
    run.finish()

    payload, step = run.run.logged[0]
    assert step == 7
    assert payload == {
        "train/policy_loss": 4.5,
        "perf/rows_per_s": 200.0,
        "perf/produced_rows": 4096,
    }
    assert "timestamp" not in payload
    publish_payload, publish_step = run.run.logged[1]
    assert publish_step == 10
    assert publish_payload == {"publish/count": 1, "publish/training_step": 10}
    assert fake.finished


def test_load_config_parses_arch_table(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        '[arch]\ndim = 64\nlayers = 2\n\n[paths]\nrun_dir = "run"\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.arch.dim == 64
    assert config.arch.layers == 2
    assert config.arch.heads == 4


def test_load_config_accepts_blind_pair_value_with_admission_stagger(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[selfplay]
position_features = false
admission_stagger_ms = 5000

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.selfplay.position_features is False
    assert config.selfplay.admission_stagger_ms == 5000


def test_load_config_requires_positive_min_startup_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        '[trainer]\nmin_startup_rows = 0\n\n[paths]\nrun_dir = "run"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_startup_rows"):
        load_config(config_path)


def test_sample_window_clamps_to_produced_rows() -> None:
    assert _sample_window_rows(10000, produced_rows=1) == 1
    assert _sample_window_rows(10000, produced_rows=1000) == 1000
    assert _sample_window_rows(10000, produced_rows=12000) == 10000


def test_init_replay_uses_schema_only_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        assert check is True

    monkeypatch.setattr("subprocess.run", fake_run)
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[selfplay]
max_candidates = 1023

[paths]
run_dir = "run"
graphzero_bin = "graphzero-test"
""",
        encoding="utf-8",
    )

    init_replay(load_config(config_path))

    assert calls == [
        [
            "graphzero-test",
            "replay-init",
            "--replay-dir",
            str(Path.cwd() / "run/replay"),
            "--max-candidates",
            "1023",
        ]
    ]


def test_memory_watchdog_aborts_below_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    from gz.trainer import driver

    monkeypatch.setattr(driver, "_mem_available_gb", lambda: 12.0)
    driver.check_memory(10.0)  # above floor: fine
    with pytest.raises(RuntimeError, match="12.0 GiB available"):
        driver.check_memory(40.0)
    driver.check_memory(0)  # disabled


def test_wandb_run_logs_graph_facts_once() -> None:
    class FakeConfig(dict):
        def update(self, values, allow_val_change=False):
            dict.update(self, values)

    class FakeRun:
        def __init__(self) -> None:
            self.logged: list[tuple[dict, int]] = []
            self.config = FakeConfig()

        def log(self, payload: dict, step: int) -> None:
            self.logged.append((payload, step))

    run = WandbRun(FakeRun())
    run.write(
        {
            "event": "graph",
            "root_cost": 150.0,
            "root_nodes": 200,
            "root_edges": 400,
            "root_candidates": 900,
        }
    )

    payload, step = run.run.logged[0]
    assert step == 0
    assert payload["graph/root_cost"] == 150.0
    assert run.run.config["graph"]["root_candidates"] == 900
