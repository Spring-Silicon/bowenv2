from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from gz.trainer.driver import (
    SamplePrefetcher,
    TrainerConfig,
    WandbRun,
    _checkpoint_due,
    _cumulative_reuse,
    _permanent_checkpoint_pointers,
    _required_episodes,
    _required_produced_rows,
    _resolved_trainer_seeds,
    _sample_training_batches,
    _sample_window_rows,
    _symmetric_step_fields,
    init_replay,
    load_config,
    spawn_torch_selfplay,
    trainer_loop_config,
)


def test_load_config_defaults_and_absolute_paths(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
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
    )
    config = load_config(path)
    assert config.trainer.batch == 4
    assert config.trainer.total_steps == 3
    assert config.selfplay.lanes == 1
    assert config.arch.name == "gz-graph-v2"
    assert config.arch.state_input == "joint-board"
    assert config.paths.run_dir == Path.cwd() / "run"
    assert config.paths.replay_dir == Path.cwd() / "run/replay"
    assert config.paths.checkpoint_dir == Path.cwd() / "run/checkpoints"
    assert config.paths.sample_socket == Path.cwd() / "run/sample.sock"
    assert config.paths.graphzero_bin == "graphzero-test"


def test_config_inheritance_is_recursive_and_one_layer_only(tmp_path: Path) -> None:
    (tmp_path / "base.toml").write_text(
        """
[trainer]
batch = 16
total_steps = 100
[selfplay]
lanes = 2
workers_per_lane = 3
[paths]
run_dir = "base"
graphzero_bin = "graphzero"
""",
        encoding="utf-8",
    )
    child = write_config(
        tmp_path,
        """
extends = "base.toml"
[trainer]
total_steps = 200
[paths]
run_dir = "child"
""",
        name="child.toml",
    )
    config = load_config(child)
    assert config.trainer.batch == 16
    assert config.trainer.total_steps == 200
    assert config.selfplay.workers_per_lane == 3
    assert config.paths.run_dir == Path.cwd() / "child"

    (tmp_path / "nested.toml").write_text(
        'extends = "base.toml"\n', encoding="utf-8"
    )
    grandchild = write_config(
        tmp_path,
        'extends = "nested.toml"\n',
        name="grandchild.toml",
    )
    with pytest.raises(ValueError, match="one layer"):
        load_config(grandchild)


def test_retired_symmetric_keys_are_accepted_only_at_fixed_values(tmp_path: Path) -> None:
    compatible = write_config(
        tmp_path,
        """
[trainer]
value_mirror = false
[selfplay]
training_mode = "symmetric-selfplay"
reference = "none"
root_mode = "generated"
reference_gamma = 0.0
reference_trajectory_pool = 0
value_reward = "sign"
[paths]
run_dir = "run"
graphzero_bin = "graphzero"
""",
    )
    assert load_config(compatible).selfplay.lanes == 2

    incompatible = compatible.read_text(encoding="utf-8").replace(
        "reference_gamma = 0.0", "reference_gamma = 0.2"
    )
    compatible.write_text(incompatible, encoding="utf-8")
    with pytest.raises(ValueError, match="retired selfplay setting"):
        load_config(compatible)

    compatible.write_text(
        incompatible.replace("reference_gamma = 0.2", "reference_gamma = 0.0").replace(
            "value_mirror = false", "value_mirror = true"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="value_mirror=true"):
        load_config(compatible)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("trainer", "value_trunk_grad_scale", "1.1", "value_trunk_grad_scale"),
        ("trainer", "compile_mode", '"unknown"', "compile_mode"),
        ("trainer", "checkpoint_retain", "-1", "checkpoint_retain"),
        ("trainer", "min_startup_rows", "0", "min_startup_rows"),
        ("selfplay", "length_tiebreak", "false", "length_tiebreak"),
    ],
)
def test_load_config_rejects_invalid_active_settings(
    tmp_path: Path,
    section: str,
    field: str,
    value: str,
    message: str,
) -> None:
    path = write_config(
        tmp_path,
        f"""
[{section}]
{field} = {value}
[paths]
run_dir = "run"
graphzero_bin = "graphzero"
""",
    )
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_load_config_rejects_unknown_fields_and_unsupported_arch(tmp_path: Path) -> None:
    unknown = write_config(
        tmp_path,
        """
[trainer]
not_a_setting = 1
[paths]
run_dir = "run"
graphzero_bin = "graphzero"
""",
    )
    with pytest.raises(ValueError, match="unknown config fields"):
        load_config(unknown)

    unsupported = unknown.read_text(encoding="utf-8").replace(
        "[trainer]\nnot_a_setting = 1", '[arch]\ntrunk = "sage"'
    )
    unknown.write_text(unsupported, encoding="utf-8")
    with pytest.raises(ValueError, match="trunk"):
        load_config(unknown)


def test_canonical_symmetric_config_resolves_retained_recipe() -> None:
    config = load_config(
        Path("configs/whittle-generated-exphormer-v2-symmetric-selfplay.toml")
    )
    assert config.arch.name == "gz-graph-v2"
    assert config.arch.trunk == "exphormer"
    assert config.arch.state_input == "joint-board"
    assert config.arch.value_input == "single"
    assert config.selfplay.length_tiebreak is True
    assert config.paths.replay_dir.as_posix().startswith("/opt/dlami/nvme/")


def test_trainer_seed_and_loop_settings_are_independent() -> None:
    assert _resolved_trainer_seeds(TrainerConfig(seed=7)) == (7, 7)
    config = TrainerConfig(
        seed=7,
        model_seed=11,
        data_seed=13,
        compile_model=True,
        compile_mode="reduce-overhead",
        value_trunk_grad_scale=0.1,
    )
    assert _resolved_trainer_seeds(config) == (11, 13)
    loop = trainer_loop_config(config, symmetric_mask_stop=False)
    assert loop.compile_model is True
    assert loop.compile_mode == "reduce-overhead"
    assert loop.value_trunk_grad_scale == 0.1
    assert loop.mask_stop_loss is False


def test_reuse_and_checkpoint_arithmetic() -> None:
    assert _sample_window_rows(300_000, 12_000) == 12_000
    assert _sample_window_rows(30_000, 100_000) == 30_000
    assert _cumulative_reuse(9, 512, 1024) == 5.0
    assert _required_produced_rows(7, 512, 8.0, 8) == 512
    assert _required_produced_rows(8, 512, 8.0, 8) == 1024
    assert _required_episodes(7, 8, 44) == 44
    assert _required_episodes(8, 8, 44) == 88

    config = TrainerConfig(
        publish_interval=8,
        permanent_checkpoint_interval=10,
    )
    assert _checkpoint_due(config, 8)
    assert _checkpoint_due(config, 10)
    assert not _checkpoint_due(config, 9)
    assert _permanent_checkpoint_pointers(config, 20) == ("step_20.json",)


def test_sample_training_batches_uses_independent_windows_and_streams() -> None:
    class Sampler:
        def __init__(self, result: str) -> None:
            self.result = result
            self.calls = []

        def sample(self, batch: int, window: int, seed: int):
            self.calls.append((batch, window, seed))
            return self.result

    policy = Sampler("policy")
    value = Sampler("value")
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = _sample_training_batches(
            policy,
            policy_batch=512,
            policy_window_rows=30_000,
            value_batch=64,
            value_window_rows=300_000,
            run_seed=42,
            step=5,
            produced_rows=50_000,
            value_sampler=value,
            value_executor=executor,
        )
    assert result.policy == "policy"
    assert result.value == "value"
    assert policy.calls[0][:2] == (512, 30_000)
    assert value.calls[0][:2] == (64, 50_000)
    assert policy.calls[0][2] != value.calls[0][2]


def test_prefetcher_obeys_row_and_episode_gate() -> None:
    class Sampler:
        def __init__(self) -> None:
            self.refreshes = 0
            self.samples = []

        def refresh(self):
            self.refreshes += 1
            return SimpleNamespace(
                produced_rows=0 if self.refreshes == 1 else 512,
                episodes=0 if self.refreshes == 1 else 4,
            )

        def sample(self, batch: int, window: int, seed: int):
            self.samples.append((batch, window, seed))
            return "batch"

    sampler = Sampler()
    prefetcher = SamplePrefetcher(
        sampler,
        batch=512,
        window_rows=30_000,
        value_batch=0,
        value_window_rows=30_000,
        seed=42,
        total_steps=1,
        max_reuse=8.0,
        reuse_gate_interval=8,
        reuse_gate_episodes=4,
    )
    prefetcher.start()
    try:
        assert prefetcher.next().policy == "batch"
    finally:
        prefetcher.stop()
        prefetcher.join()
    assert sampler.refreshes >= 2
    assert len(sampler.samples) == 1


def test_symmetric_step_fields_report_both_seats() -> None:
    metrics = SimpleNamespace(
        p1_win_rate_ema=0.4,
        p2_win_rate_ema=0.35,
        draw_rate_ema=0.25,
        seat_advantage_ema=0.05,
        p1_terminal_cost_ema=70.0,
        p2_terminal_cost_ema=72.0,
        mean_terminal_cost_ema=71.0,
        terminal_cost_margin_ema=4.0,
        terminal_cost_best=40.0,
        p1_episode_len_ema=50.0,
        p2_episode_len_ema=52.0,
        game_len_ema=102.0,
        episode_len_margin_ema=2.0,
    )
    ack = SimpleNamespace(
        symmetric_selfplay=metrics,
        value_sign_accuracy_early_ema=0.6,
        value_sign_accuracy_late_ema=0.7,
        episode_latency_ema=12.0,
    )
    fields = _symmetric_step_fields(ack, completed_games=123)
    assert fields["symmetric_games_completed"] == 123
    assert fields["symmetric_p1_win_rate_ema"] == 0.4
    assert fields["symmetric_p2_win_rate_ema"] == 0.35
    assert fields["symmetric_best_of_two_terminal_cost_ema"] == 69.0
    assert fields["symmetric_game_latency_s"] == 12.0


def test_process_commands_contain_only_active_symmetric_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        write_config(
            tmp_path,
            """
[trainer]
batch = 64
[selfplay]
lanes = 4
workers_per_lane = 3
tree_reuse = true
mask_stop = true
[paths]
run_dir = "run"
graphzero_bin = "graphzero-test"
""",
        )
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr("gz.trainer.driver.subprocess.run", run)
    monkeypatch.setattr("gz.trainer.driver.subprocess.Popen", run)
    init_replay(config)
    spawn_torch_selfplay(config)

    assert calls[0][0][:2] == ["graphzero-test", "replay-init"]
    command = calls[1][0]
    assert command[:2] == ["graphzero-test", "selfplay"]
    assert command[command.index("--lanes") + 1] == "4"
    assert command[command.index("--tree-reuse") + 1] == "true"
    assert "--reference" not in command
    assert "--training-mode" not in command
    assert calls[1][1]["start_new_session"] is True


def test_wandb_mapping_logs_only_explicit_metrics() -> None:
    class Run:
        def __init__(self) -> None:
            self.logs = []

        def log(self, payload, step):
            self.logs.append((payload, step))

    run = Run()
    writer = WandbRun(run)
    writer.write(
        {
            "event": "step",
            "step": 8,
            "policy_loss": 1.25,
            "value_loss": 0.5,
            "unknown": 99,
        }
    )
    assert run.logs == [
        ({"train/policy_loss": 1.25, "train/value_loss": 0.5}, 8)
    ]


def write_config(
    root: Path,
    body: str,
    *,
    name: str = "run.toml",
) -> Path:
    path = root / name
    path.write_text(body, encoding="utf-8")
    return path
