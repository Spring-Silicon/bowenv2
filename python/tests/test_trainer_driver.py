from __future__ import annotations

import io
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gz.trainer.driver import (
    ArenaCheckpointPublisher,
    OpponentTracker,
    SamplePrefetcher,
    SelfplayStatsTracker,
    TrainerConfig,
    WandbRun,
    _required_episodes,
    _required_produced_rows,
    _load_initial_checkpoint,
    _policy_arch_config,
    _resolved_trainer_seeds,
    _sample_training_batches,
    _sample_window_rows,
    init_replay,
    load_config,
    pump_selfplay_stderr,
    spawn_torch_selfplay,
    trainer_loop_config,
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
reference_gamma = 0.25

[paths]
run_dir = "run"
graphzero_bin = "graphzero-test"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.trainer.batch == 4
    assert config.trainer.total_steps == 3
    assert config.trainer.checkpoint_retain == 0
    assert config.selfplay.lanes == 1
    assert config.selfplay.reference_gamma == 0.25
    assert config.selfplay.reference_trajectory_pool == 0
    # Paths are pinned to the trainer's cwd: children (the evaluator) run in
    # their own working directories, so relative paths must not cross over.
    assert config.paths.run_dir == Path.cwd() / "run"
    assert config.paths.replay_dir == Path.cwd() / "run/replay"
    assert config.paths.checkpoint_dir == Path.cwd() / "run/checkpoints"
    assert config.paths.sample_socket == Path.cwd() / "run/sample.sock"
    assert config.paths.graphzero_bin == "graphzero-test"


def test_trainer_seed_overrides_are_independent_and_backward_compatible() -> None:
    assert _resolved_trainer_seeds(TrainerConfig(seed=42)) == (42, 42)
    assert _resolved_trainer_seeds(
        TrainerConfig(seed=42, model_seed=5, data_seed=17)
    ) == (5, 17)


def test_trainer_compile_config_reaches_loop(tmp_path: Path) -> None:
    config_path = tmp_path / "compile.toml"
    config_path.write_text(
        "[trainer]\ncompile_model = true\ncompile_mode = 'reduce-overhead'\n",
        encoding="utf-8",
    )

    trainer = load_config(config_path).trainer
    loop = trainer_loop_config(trainer, data_seed=7)

    assert loop.compile_model is True
    assert loop.compile_mode == "reduce-overhead"


def test_load_config_rejects_unknown_trainer_compile_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-compile.toml"
    config_path.write_text(
        "[trainer]\ncompile_mode = 'fastest-please'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown compile_mode"):
        load_config(config_path)


def test_load_config_rejects_negative_checkpoint_retention(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-retention.toml"
    config_path.write_text(
        "[trainer]\ncheckpoint_retain = -1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint_retain must be non-negative"):
        load_config(config_path)


def test_load_config_accepts_trainer_matmul_precision(tmp_path: Path) -> None:
    config_path = tmp_path / "matmul.toml"
    config_path.write_text(
        "[trainer]\nmatmul_precision = 'high'\n",
        encoding="utf-8",
    )

    assert load_config(config_path).trainer.matmul_precision == "high"


def test_load_config_rejects_unknown_trainer_matmul_precision(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-matmul.toml"
    config_path.write_text(
        "[trainer]\nmatmul_precision = 'fastest-please'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown matmul_precision"):
        load_config(config_path)


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text("[trainer]\nunknown = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config fields"):
        load_config(config_path)


def test_load_config_rejects_unknown_policy_initializer(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-policy-init.toml"
    config_path.write_text("[trainer]\npolicy_init = 'unknown'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported policy_init"):
        load_config(config_path)


def test_load_config_rejects_resume_with_initial_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-init.toml"
    config_path.write_text(
        "[trainer]\nresume = true\ninit_checkpoint = 'checkpoints'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(config_path)


def test_initial_checkpoint_loads_weights_without_resume_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from gz.checkpoints import publish_checkpoint
    from gz.common import ActionSetHash, EngineId, EngineVersion
    from gz.model.exphormer import ArchConfig, build_model
    from python.tests.test_checkpoints import feature_hash, schema_config

    schema = schema_config()
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    source = build_model(schema, arch)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    published = publish_checkpoint(
        tmp_path,
        source.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        training_step=73,
        run_id="distill",
    )
    target = build_model(schema, arch)

    resolved = _load_initial_checkpoint(target, tmp_path, feature_hash(), arch)

    assert resolved.manifest == published
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)


def test_policy_initial_checkpoint_preserves_fresh_value_head(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from gz.checkpoints import publish_checkpoint
    from gz.common import ActionSetHash, EngineId, EngineVersion
    from gz.model.exphormer import ArchConfig, build_model
    from python.tests.test_checkpoints import feature_hash, schema_config

    schema = schema_config()
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        value_input="pair",
    )
    source = build_model(schema, arch)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            parameter.fill_(0.0 if name.startswith("value.") else 0.25)
    publish_checkpoint(
        tmp_path,
        source.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        training_step=1280,
        run_id="distill",
    )
    target = build_model(schema, arch)
    fresh_value = {
        name: value.clone()
        for name, value in target.state_dict().items()
        if name.startswith("value.")
    }

    _load_initial_checkpoint(
        target,
        tmp_path,
        feature_hash(),
        arch,
        scope="policy",
    )

    for name, value in target.state_dict().items():
        expected = fresh_value[name] if name.startswith("value.") else source.state_dict()[name]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert any(torch.count_nonzero(value) for value in fresh_value.values())


def test_policy_initial_checkpoint_allows_different_value_head(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from gz.checkpoints import publish_checkpoint
    from gz.common import ActionSetHash, EngineId, EngineVersion
    from gz.model.exphormer import ArchConfig, build_model
    from python.tests.test_checkpoints import feature_hash, schema_config

    schema = schema_config()
    source_arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        value_input="pair",
        value_head="scalar",
    )
    target_arch = replace(source_arch, value_head="hl_gauss")
    source = build_model(schema, source_arch)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            parameter.fill_(0.0 if name.startswith("value.") else 0.25)
    publish_checkpoint(
        tmp_path,
        source.state_dict(),
        arch_name=source_arch.name,
        arch_config=source_arch.to_dict(),
        arch_config_hash=source_arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        training_step=1280,
        run_id="distill",
    )
    target = build_model(schema, target_arch)
    fresh_value = {
        name: value.clone()
        for name, value in target.state_dict().items()
        if name.startswith("value.")
    }

    _load_initial_checkpoint(
        target,
        tmp_path,
        feature_hash(),
        target_arch,
        scope="policy",
    )

    for name, value in target.state_dict().items():
        expected = fresh_value[name] if name.startswith("value.") else source.state_dict()[name]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_v2_rejects_v1_policy_initial_checkpoint(tmp_path: Path) -> None:
    from gz.checkpoints import publish_checkpoint
    from gz.common import ActionSetHash, EngineId, EngineVersion
    from gz.model.exphormer import ArchConfig, build_model
    from python.tests.test_checkpoints import feature_hash, schema_config

    schema = schema_config()
    source_arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        policy_head="pointer",
    )
    target_arch = replace(
        source_arch,
        name="gz-graph-v2",
        position_encoding="remaining_budget",
    )
    source = build_model(schema, source_arch)
    publish_checkpoint(
        tmp_path,
        source.state_dict(),
        arch_name=source_arch.name,
        arch_config=source_arch.to_dict(),
        arch_config_hash=source_arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_hash(),
        engine_id=EngineId.from_bytes(b"e" * 16),
        engine_version=EngineVersion.from_bytes(b"v" * 16),
        action_set_hash=ActionSetHash.from_bytes(b"a" * 32),
        training_step=1280,
        run_id="distill",
    )

    with pytest.raises(RuntimeError, match="arch name"):
        _load_initial_checkpoint(
            build_model(schema, target_arch),
            tmp_path,
            feature_hash(),
            target_arch,
            scope="policy",
        )


def test_policy_arch_compatibility_only_ignores_value_module() -> None:
    from gz.model.exphormer import ArchConfig

    source = ArchConfig()
    value_variant = replace(
        source,
        value_input="pair",
        value_activation="tanh",
        value_hidden=64,
        value_head="hl_gauss",
        value_bins=51,
        value_min=-2.0,
        value_max=2.0,
        value_sigma_ratio=1.0,
    )
    policy_variant = replace(source, subject_encoding="match")

    assert _policy_arch_config(source) == _policy_arch_config(value_variant)
    assert _policy_arch_config(source) != _policy_arch_config(policy_variant)


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
            "terminal_cost_ema": 62.5,
            "terminal_cost_best": 51.0,
            "admission_waiting": 17,
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
        "selfplay/terminal_cost_ema": 62.5,
        "selfplay/terminal_cost_best": 51.0,
        "admission/waiting_workers": 17,
        "perf/rows_per_s": 200.0,
        "perf/produced_rows": 4096,
    }
    assert "timestamp" not in payload
    publish_payload, publish_step = run.run.logged[1]
    assert publish_step == 10
    assert publish_payload == {"publish/count": 1, "publish/training_step": 10}
    assert fake.finished


def test_selfplay_stats_tracks_admission_controller_heartbeat() -> None:
    tracker = SelfplayStatsTracker()
    tracker.observe_admission(
        {
            "outstanding": "512",
            "reserved": "4",
            "waiting": "23",
            "max_waiting": "42",
            "bootstrap_grants": "1",
            "paced_grants": "20",
            "eval_capacity_milli": "12000500",
            "episode_work_milli": "3072500",
            "pressure_gain_milli": "1250",
            "gap_us": "256031",
        }
    )

    assert tracker.step_fields() == {
        "admission_outstanding": 512,
        "admission_reserved": 4,
        "admission_waiting": 23,
        "admission_max_waiting": 42,
        "admission_bootstrap_grants": 1,
        "admission_paced_grants": 20,
        "admission_eval_capacity": 12000.5,
        "admission_episode_work": 3072.5,
        "admission_pressure_gain": 1.25,
        "admission_gap_ms": 256.031,
    }


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


@pytest.mark.parametrize(
    ("arch_settings", "expected_head"),
    [
        ("value_activation = 'tanh'", "scalar"),
        ("value_head = 'hl_gauss'", "hl_gauss"),
    ],
)
def test_load_config_accepts_both_graded_value_losses(
    tmp_path: Path,
    arch_settings: str,
    expected_head: str,
) -> None:
    config_path = tmp_path / f"graded-{expected_head}.toml"
    config_path.write_text(
        f"""
[arch]
{arch_settings}

[selfplay]
value_reward = "graded"
value_reward_scale = 0.1

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.selfplay.value_reward == "graded"
    assert config.selfplay.value_reward_scale == pytest.approx(0.1)
    assert config.arch.value_head == expected_head


@pytest.mark.parametrize(
    ("arch_settings", "selfplay_settings", "message"),
    [
        ("", "value_reward = 'graded'", "graded scalar values require"),
        ("value_head = 'hl_gauss'", "", "requires value_reward = 'graded'"),
    ],
)
def test_load_config_rejects_incompatible_value_reward_and_head(
    tmp_path: Path,
    arch_settings: str,
    selfplay_settings: str,
    message: str,
) -> None:
    config_path = tmp_path / "invalid-value.toml"
    config_path.write_text(
        f"""
[arch]
{arch_settings}

[selfplay]
{selfplay_settings}

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_seed42_parity_config_resolves_reference_recipe() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/whittle-seed42-parity-s240.toml")

    assert (config.arch.trunk, config.arch.layers, config.arch.sage_layers) == ("sage", 3, 3)
    assert config.arch.subject_encoding == "match"
    assert config.arch.position_encoding == "policy_budget"
    assert config.arch.action_encoding == "candidate_only"
    assert (config.arch.profile, config.arch.value_hidden) == ("whittlezero", 256)
    assert (config.selfplay.lanes, config.selfplay.workers_per_lane) == (32, 1)
    assert config.selfplay.max_batch == 32
    assert config.trainer.max_reuse == 0.0
    assert config.trainer.reuse_gate_interval == 8
    assert config.trainer.reuse_gate_episodes == 32
    assert config.trainer.value_batch == 256
    assert config.trainer.value_window_rows == 5000
    assert (config.arch.policy_head, config.arch.value_input, config.arch.value_activation) == (
        "pointer",
        "pair",
        "tanh",
    )
    assert (config.selfplay.simulations, config.selfplay.max_considered) == (64, 16)
    assert config.selfplay.max_candidates == 2303
    assert (config.selfplay.c_visit, config.selfplay.c_scale) == (50.0, 0.3)
    assert config.selfplay.max_steps == 104
    assert config.selfplay.no_backtrack and not config.selfplay.mask_stop
    assert not config.selfplay.length_tiebreak and not config.selfplay.tree_reuse
    assert config.selfplay.reference_gamma == 0.2
    assert config.trainer.optimizer == "muon_mixed"
    assert config.trainer.total_steps == 240
    assert config.trainer.lr_decay_steps == 800
    assert config.trainer.min_lr_ratio == 0.1
    assert config.trainer.publish_interval == 8
    assert config.trainer.publish_lag_blocks == 1
    assert config.trainer.window_rows == 10000
    assert config.trainer.value_mirror


def test_generated_v2_base_bounds_actor_checkpoint_retention() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root / "configs/bases/whittle-generated-exphormer-v2-sampled-tree.toml"
    )

    assert (config.trainer.publish_interval, config.trainer.checkpoint_retain) == (8, 16)
    assert config.selfplay.reference_arena_interval == 128
    assert config.selfplay.challenger_max_batch == 128


def test_benchmark_cadence_ablation_changes_only_cadence_recipe() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/ablations/benchmark-cadence-01-s240.toml")

    assert (config.arch.trunk, config.arch.layers, config.arch.subject_encoding) == (
        "exphormer",
        4,
        "mean",
    )
    assert not config.selfplay.position_features
    assert (config.selfplay.simulations, config.selfplay.max_considered) == (48, 8)
    assert config.selfplay.max_candidates == 1023
    assert config.selfplay.reference_gamma == 0.0
    assert config.selfplay.reference_mask_stop is True
    assert (config.selfplay.lanes, config.selfplay.workers_per_lane) == (32, 1)
    assert config.selfplay.max_batch == 32
    assert config.trainer.optimizer == "adamw"
    assert config.trainer.batch == 256 and config.trainer.value_batch == 0
    assert config.trainer.max_reuse == 0.0
    assert (config.trainer.reuse_gate_interval, config.trainer.reuse_gate_episodes) == (8, 32)
    assert (config.trainer.publish_interval, config.trainer.publish_lag_blocks) == (8, 1)
    assert config.trainer.resume


def test_policy_budget_requires_exported_position_features(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
position_encoding = "policy_budget"
value_input = "pair"

[selfplay]
position_features = false

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy_budget"):
        load_config(config_path)


def test_remaining_budget_requires_exported_position_features(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
name = "gz-graph-v2"
policy_head = "pointer"
position_encoding = "remaining_budget"
value_input = "pair"

[selfplay]
position_features = false

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="remaining_budget"):
        load_config(config_path)


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


def test_load_config_accepts_adaptive_admission_smoothing(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[selfplay]
admission_smoothing = true

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.selfplay.admission_smoothing is True


def test_load_config_rejects_fixed_and_adaptive_admission_pacing(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[selfplay]
admission_stagger_ms = 100
admission_smoothing = true

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(config_path)


def test_load_config_requires_gated_policy_for_reference_trajectory_pool(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[selfplay]
reference = "policy"
root_mode = "fixed"
reference_trajectory_pool = 8

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference_trajectory_pool"):
        load_config(config_path)


def test_load_config_accepts_sampled_trajectory_pair_opponent(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[selfplay]
reference = "policy"
root_mode = "fixed"
policy_opponent_mode = "sampled-trajectory"

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.selfplay.policy_opponent_mode == "sampled-trajectory"


def test_load_config_rejects_sampled_trajectory_without_pair_value(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[selfplay]
reference = "policy"
root_mode = "fixed"
policy_opponent_mode = "sampled-trajectory"

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires value_input = 'pair'"):
        load_config(config_path)


def test_load_config_rejects_gamma_for_active_sampled_trajectory(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[selfplay]
reference = "policy"
root_mode = "fixed"
reference_gamma = 0.2
policy_opponent_mode = "sampled-trajectory"

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires reference_gamma = 0"):
        load_config(config_path)


def test_load_config_rejects_gated_policy_for_sampled_trajectory(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[selfplay]
reference = "gated-policy"
root_mode = "fixed"
policy_opponent_mode = "sampled-trajectory"

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires reference = 'policy'"):
        load_config(config_path)


def test_load_config_accepts_generated_arena_gated_greedy_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[trainer]
publish_interval = 32

[selfplay]
reference = "gated-policy"
root_mode = "generated"
policy_opponent_mode = "greedy-trajectory"
reference_arena_size = 128
reference_arena_seed = 910000001
reference_arena_interval = 160

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.selfplay.reference_arena_size == 128
    assert config.selfplay.reference_arena_seed == 910_000_001
    assert config.selfplay.reference_arena_interval == 160


def test_load_config_requires_arena_interval_to_align_with_actor_publications(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[trainer]
publish_interval = 32

[selfplay]
reference = "gated-policy"
root_mode = "generated"
policy_opponent_mode = "greedy-trajectory"
reference_arena_size = 128
reference_arena_interval = 100

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple of publish_interval"):
        load_config(config_path)


def test_load_config_accepts_fixed_gated_sampled_tree(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[trainer]
value_batch = 256
value_mirror = false

[selfplay]
reference = "gated-policy"
root_mode = "fixed"
policy_opponent_mode = "sampled-tree"
tree_reuse = false
reference_max_batch = 128
length_tiebreak = true

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.selfplay.policy_opponent_mode == "sampled-tree"
    assert config.selfplay.length_tiebreak is True
    assert config.trainer.value_batch == 256


def test_load_config_rejects_generated_gated_policy_without_arena(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[selfplay]
reference = "gated-policy"
root_mode = "generated"
policy_opponent_mode = "greedy-trajectory"

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generated-root arena"):
        load_config(config_path)


def test_load_config_requires_positive_min_startup_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        '[trainer]\nmin_startup_rows = 0\n\n[paths]\nrun_dir = "run"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_startup_rows"):
        load_config(config_path)


def test_publish_lag_requires_a_publish_aligned_gate(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[trainer]
publish_interval = 8
publish_lag_blocks = 1
reuse_gate_interval = 1

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="publish-aligned"):
        load_config(config_path)


def test_sample_window_clamps_to_produced_rows() -> None:
    assert _sample_window_rows(10000, produced_rows=1) == 1
    assert _sample_window_rows(10000, produced_rows=1000) == 1000
    assert _sample_window_rows(10000, produced_rows=12000) == 10000


def test_block_row_gate_admits_all_steps_together() -> None:
    reuse = 4096 / 3328

    assert _required_produced_rows(0, 512, reuse, 8) == 3328
    assert _required_produced_rows(7, 512, reuse, 8) == 3328
    assert _required_produced_rows(8, 512, reuse, 8) == 6656
    assert _required_produced_rows(15, 512, reuse, 8) == 6656


def test_block_episode_gate_waits_for_each_actor_wave() -> None:
    assert _required_episodes(0, 8, 32) == 32
    assert _required_episodes(7, 8, 32) == 32
    assert _required_episodes(8, 8, 32) == 64
    assert _required_episodes(15, 8, 32) == 64


def test_sample_training_batches_uses_independent_value_stream_and_window() -> None:
    calls: list[tuple[int, int, int]] = []

    class Sampler:
        def sample(self, batch: int, window: int, seed: int) -> tuple[int, int, int]:
            calls.append((batch, window, seed))
            return calls[-1]

    sampled = _sample_training_batches(
        Sampler(),
        policy_batch=512,
        policy_window_rows=10000,
        value_batch=256,
        value_window_rows=5000,
        run_seed=5,
        step=9,
        produced_rows=12000,
    )

    assert calls[0][:2] == (512, 10000)
    assert calls[1][:2] == (256, 5000)
    assert calls[0][2] != calls[1][2]
    assert sampled.policy == calls[0]
    assert sampled.value == calls[1]


def test_sampled_tree_requests_policy_and_value_streams() -> None:
    calls: list[tuple[int, str]] = []

    class Sampler:
        def sample(
            self, batch: int, _window: int, _seed: int, *, kind: str = "any"
        ) -> tuple[int, str]:
            calls.append((batch, kind))
            return calls[-1]

    _sample_training_batches(
        Sampler(),
        policy_batch=512,
        policy_window_rows=10000,
        value_batch=256,
        value_window_rows=10000,
        run_seed=5,
        step=0,
        produced_rows=10000,
        sampled_tree=True,
    )

    assert calls == [(512, "policy"), (256, "value")]


def test_canonical_muon_ab_changes_only_optimizer_and_run_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    adamw = load_config(
        root / "configs/ablations/muon-canonical-2026-07-13/00-adamw.toml"
    )
    muon = load_config(
        root / "configs/ablations/muon-canonical-2026-07-13/01-muon.toml"
    )

    assert (adamw.trainer.optimizer, adamw.trainer.lr) == ("adamw", 3e-4)
    assert (muon.trainer.optimizer, muon.trainer.lr) == ("muon_mixed", 0.02)
    assert adamw.trainer.adamw_lr == muon.trainer.adamw_lr == 3e-4
    for config in (adamw, muon):
        assert config.selfplay.max_batch == 128
        assert config.selfplay.reference_max_batch == 128
        assert (
            config.selfplay.challenger_max_batch
            or config.selfplay.reference_max_batch
            or config.selfplay.max_batch
        ) == 128
        assert str(config.paths.replay_dir).startswith(
            "/opt/dlami/nvme/graphzero-replay/"
        )
        assert config.paths.checkpoint_dir == config.paths.run_dir / "checkpoints"
        assert not str(config.paths.checkpoint_dir).startswith("/opt/dlami/nvme/")

    normalized = replace(
        adamw,
        trainer=replace(
            adamw.trainer,
            optimizer=muon.trainer.optimizer,
            lr=muon.trainer.lr,
        ),
        paths=muon.paths,
        wandb=muon.wandb,
    )
    assert normalized == muon


def test_muon_lr002_changes_only_muon_lr_and_run_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    canonical = load_config(
        root / "configs/ablations/muon-canonical-2026-07-13/01-muon.toml"
    )
    lower_lr = load_config(
        root / "configs/ablations/muon-canonical-2026-07-13/02-muon-lr002.toml"
    )

    assert canonical.trainer.lr == 0.02
    assert lower_lr.trainer.lr == 0.002
    normalized = replace(
        canonical,
        trainer=replace(canonical.trainer, lr=lower_lr.trainer.lr),
        paths=lower_lr.paths,
        wandb=lower_lr.wandb,
    )
    assert normalized == lower_lr


def test_muon_lr0002_changes_only_muon_lr_and_run_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    lr002 = load_config(
        root / "configs/ablations/muon-canonical-2026-07-13/02-muon-lr002.toml"
    )
    lr0002 = load_config(
        root / "configs/ablations/muon-canonical-2026-07-13/03-muon-lr0002.toml"
    )

    assert lr002.trainer.lr == 0.002
    assert lr0002.trainer.lr == 0.0002
    normalized = replace(
        lr002,
        trainer=replace(lr002.trainer, lr=lr0002.trainer.lr),
        paths=lr0002.paths,
        wandb=lr0002.wandb,
    )
    assert normalized == lr0002


def test_adamw_lr02_changes_only_optimizer_and_run_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    muon = load_config(
        root / "configs/ablations/muon-canonical-2026-07-13/01-muon.toml"
    )
    adamw = load_config(
        root / "configs/ablations/muon-canonical-2026-07-13/04-adamw-lr02.toml"
    )

    assert (muon.trainer.optimizer, muon.trainer.lr) == ("muon_mixed", 0.02)
    assert (adamw.trainer.optimizer, adamw.trainer.lr) == ("adamw", 0.02)
    normalized = replace(
        muon,
        trainer=replace(muon.trainer, optimizer=adamw.trainer.optimizer),
        paths=adamw.paths,
        wandb=adamw.wandb,
    )
    assert normalized == adamw


def test_low_lr_optimizer_ab_changes_only_optimizer_and_run_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "configs/ablations/optimizer-low-lr-ab-2026-07-13"
    muon = load_config(config_dir / "00-muon.toml")
    adamw = load_config(config_dir / "01-adamw.toml")

    assert (muon.trainer.optimizer, muon.trainer.lr) == ("muon_mixed", 2e-4)
    assert (adamw.trainer.optimizer, adamw.trainer.lr) == ("adamw", 2e-4)
    for config in (muon, adamw):
        assert config.trainer.adamw_lr == 2e-4
        assert config.trainer.lr_schedule == "constant"
        assert config.trainer.warmup_steps == 0
        assert config.trainer.min_lr_ratio == 0.0
        assert config.trainer.grad_clip == 3.0
        assert config.trainer.total_steps == 10000
        assert config.selfplay.policy_opponent_mode == "sampled-tree"
        assert config.selfplay.reference_gamma == 0.0
        assert config.selfplay.max_batch == 128
        assert config.selfplay.reference_max_batch == 128
        assert (
            config.selfplay.challenger_max_batch
            or config.selfplay.reference_max_batch
            or config.selfplay.max_batch
        ) == 128

    normalized = replace(
        muon,
        trainer=replace(muon.trainer, optimizer=adamw.trainer.optimizer),
        paths=adamw.paths,
        wandb=adamw.wandb,
    )
    assert normalized == adamw


def test_sequential_value_sampling_ab_changes_only_sampling_and_run_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    baseline = load_config(
        root / "configs/ablations/optimizer-low-lr-ab-2026-07-13/01-adamw.toml"
    )
    sequential = load_config(
        root
        / "configs/ablations/adamw-sampling-ab-2026-07-13/00-sequential-value.toml"
    )

    assert baseline.trainer.parallel_value_sampling is True
    assert sequential.trainer.parallel_value_sampling is False
    normalized = replace(
        baseline,
        trainer=replace(baseline.trainer, parallel_value_sampling=False),
        paths=sequential.paths,
        wandb=sequential.wandb,
    )
    assert normalized == sequential


def test_parallel_clip_one_ab_changes_only_clip_and_run_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    baseline = load_config(
        root / "configs/ablations/optimizer-low-lr-ab-2026-07-13/01-adamw.toml"
    )
    clip_one = load_config(
        root
        / "configs/ablations/adamw-sampling-ab-2026-07-13/01-parallel-clip1.toml"
    )

    assert baseline.trainer.parallel_value_sampling is True
    assert clip_one.trainer.parallel_value_sampling is True
    assert baseline.trainer.grad_clip == 3.0
    assert clip_one.trainer.grad_clip == 1.0
    normalized = replace(
        baseline,
        trainer=replace(baseline.trainer, grad_clip=1.0),
        paths=clip_one.paths,
        wandb=clip_one.wandb,
    )
    assert normalized == clip_one


def test_prefetcher_does_not_sample_a_block_before_its_reuse_gate() -> None:
    class Sampler:
        def __init__(self) -> None:
            self.produced_rows = 0
            self.episodes = 0
            self.sampled = threading.Event()

        def refresh(self):
            return SimpleNamespace(
                produced_rows=self.produced_rows,
                episodes=self.episodes,
            )

        def sample(self, batch: int, window: int, seed: int):
            self.sampled.set()
            return (batch, window, seed)

    sampler = Sampler()
    prefetcher = SamplePrefetcher(
        sampler,
        512,
        10000,
        256,
        5000,
        5,
        1,
        4096 / 3328,
        8,
        32,
    )
    prefetcher.start()
    try:
        assert not sampler.sampled.wait(0.05)
        sampler.produced_rows = 3328
        sampler.episodes = 32
        assert sampler.sampled.wait(1.0)
        sampled = prefetcher.next()
        assert sampled.policy[0] == 512
        assert sampled.value[0] == 256
    finally:
        prefetcher.stop()


def test_sampled_tree_reuse_gate_uses_policy_rows_not_total_rows() -> None:
    class Sampler:
        def __init__(self) -> None:
            self.produced_rows = 0
            self.produced_policy_rows = 0
            self.episodes = 0
            self.sampled = threading.Event()

        def refresh(self):
            return SimpleNamespace(
                produced_rows=self.produced_rows,
                produced_policy_rows=self.produced_policy_rows,
                episodes=self.episodes,
            )

        def sample(self, batch: int, window: int, seed: int, *, kind: str):
            self.sampled.set()
            return (batch, window, seed, kind)

    sampler = Sampler()
    prefetcher = SamplePrefetcher(
        sampler,
        512,
        10000,
        256,
        5000,
        5,
        1,
        4096 / 3328,
        8,
        32,
        sampled_tree=True,
    )
    prefetcher.start()
    try:
        sampler.produced_rows = 5000
        sampler.produced_policy_rows = 3000
        sampler.episodes = 32
        assert not sampler.sampled.wait(0.05)
        sampler.produced_policy_rows = 3328
        assert sampler.sampled.wait(1.0)
        sampled = prefetcher.next()
        assert sampled.policy[3] == "policy"
        assert sampled.value[3] == "value"
    finally:
        prefetcher.stop()


def test_prefetcher_overlaps_policy_and_value_sampling() -> None:
    started = threading.Barrier(2, timeout=1.0)

    class PolicySampler:
        def refresh(self):
            return SimpleNamespace(produced_rows=1000, episodes=1)

        def sample(self, batch: int, window: int, seed: int):
            started.wait()
            return ("policy", batch, window, seed)

    class ValueSampler:
        def sample(self, batch: int, window: int, seed: int):
            started.wait()
            return ("value", batch, window, seed)

    prefetcher = SamplePrefetcher(
        PolicySampler(),
        512,
        10000,
        256,
        5000,
        5,
        1,
        0.0,
        8,
        0,
        value_sampler=ValueSampler(),
    )
    prefetcher.start()
    try:
        sampled = prefetcher.next()
        assert sampled.policy[:3] == ("policy", 512, 1000)
        assert sampled.value[:3] == ("value", 256, 1000)
        assert sampled.policy[3] != sampled.value[3]
    finally:
        prefetcher.stop()


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


def test_spawn_torch_selfplay_passes_reference_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        pass

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr("gz.trainer.driver.subprocess.Popen", fake_popen)
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[trainer]
min_startup_rows = 1

[selfplay]
    reference = "gated-policy"
    root_mode = "fixed"
	    reference_gamma = 0.25
	    reference_trajectory_pool = 8
	    reference_mask_stop = true
    c_visit = 50.0
    c_scale = 0.3

[paths]
run_dir = "run"
graphzero_bin = "graphzero-test"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    spawn_torch_selfplay(config)

    command, kwargs = calls[0]
    index = command.index("--reference-gamma")
    assert command[index + 1] == "0.25"
    assert command[command.index("--reference-trajectory-pool") + 1] == "8"
    assert command[command.index("--reference-checkpoint-pointer") + 1] == str(
        config.paths.checkpoint_dir / "best.json"
    )
    assert command[command.index("--reference-mask-stop") + 1] == "true"
    assert command[command.index("--c-visit") + 1] == "50.0"
    assert command[command.index("--c-scale") + 1] == "0.3"
    assert command[command.index("--value-reward") + 1] == "sign"
    assert command[command.index("--value-reward-scale") + 1] == "0.1"
    assert command[command.index("--admission-smoothing") + 1] == "false"
    assert kwargs["start_new_session"] is True


def test_spawn_torch_selfplay_passes_generated_arena_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return object()

    monkeypatch.setattr("gz.trainer.driver.subprocess.Popen", fake_popen)
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[selfplay]
reference = "gated-policy"
root_mode = "generated"
policy_opponent_mode = "greedy-trajectory"
reference_arena_size = 128
reference_arena_seed = 99
challenger_max_batch = 16

[paths]
run_dir = "run"
graphzero_bin = "graphzero-test"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    spawn_torch_selfplay(config)

    command = calls[0]
    assert command[command.index("--reference-arena-size") + 1] == "128"
    assert command[command.index("--reference-arena-seed") + 1] == "99"
    assert command[command.index("--challenger-max-batch") + 1] == "16"
    assert command[command.index("--reference-checkpoint-pointer") + 1] == str(
        config.paths.checkpoint_dir / "best.json"
    )
    assert command[command.index("--reference-challenger-checkpoint-pointer") + 1] == str(
        config.paths.checkpoint_dir / "arena.json"
    )


def test_spawn_torch_selfplay_passes_fixed_sampled_tree_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "gz.trainer.driver.subprocess.Popen",
        lambda command, **_kwargs: calls.append(command) or object(),
    )
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[arch]
value_input = "pair"

[trainer]
value_batch = 256

[selfplay]
reference = "gated-policy"
root_mode = "fixed"
policy_opponent_mode = "sampled-tree"
tree_reuse = false
reference_max_batch = 128

[paths]
run_dir = "run"
graphzero_bin = "graphzero-test"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    spawn_torch_selfplay(config)

    command = calls[0]
    assert command[command.index("--reference-checkpoint-pointer") + 1] == str(
        config.paths.checkpoint_dir / "best.json"
    )
    assert command[command.index("--reference-max-batch") + 1] == "128"


def test_arena_checkpoint_publisher_pins_active_and_keeps_newest_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    promoted: list[tuple[Path, str, str]] = []
    monkeypatch.setattr(
        "gz.trainer.driver.promote_checkpoint_pointer",
        lambda root, pointer, model_version: promoted.append(
            (Path(root), pointer, model_version)
        ),
    )
    first = "01" * 16
    skipped = "02" * 16
    latest = "03" * 16
    publisher = ArenaCheckpointPublisher(tmp_path, completed_version="00" * 16)

    publisher.schedule(first, 32)
    publisher.schedule(skipped, 64)
    publisher.schedule(latest, 96)

    assert promoted == [(tmp_path, "arena.json", first)]
    assert publisher.protected_versions() == {first, latest}
    publisher.complete(first)
    assert promoted == [
        (tmp_path, "arena.json", first),
        (tmp_path, "arena.json", latest),
    ]
    assert publisher.protected_versions() == {latest}
    assert [event["model_version"] for event in publisher.drain_events()] == [
        first,
        latest,
    ]


def test_arena_acceptance_promotes_best_then_advances_queued_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = "01" * 16
    next_version = "02" * 16
    process = SimpleNamespace(
        stderr=io.BytesIO(
            (
                "event=arena_gate accepted=true challenger=0.2 best=0.1 margin=1.0 "
                f"arena_size=8 steps=24 version={version}\n"
            ).encode()
        )
    )
    promoted: list[tuple[Path, str, str]] = []
    monkeypatch.setattr(
        "gz.trainer.driver.promote_checkpoint_pointer",
        lambda root, pointer, model_version: promoted.append(
            (Path(root), pointer, model_version)
        ),
    )
    publisher = ArenaCheckpointPublisher(tmp_path)
    publisher.schedule(version, 32)
    publisher.schedule(next_version, 64)

    pump_selfplay_stderr(
        process,
        OpponentTracker(),
        SelfplayStatsTracker(),
        tmp_path,
        publisher,
    )

    assert promoted == [
        (tmp_path, "arena.json", version),
        (tmp_path, "best.json", version),
        (tmp_path, "arena.json", next_version),
    ]


def test_policy_acceptance_promotes_the_exact_checkpoint_from_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = "02" * 16
    process = SimpleNamespace(
        stderr=io.BytesIO(
            (
                "event=policy_gate accepted=true challenger=-42 best=-42 "
                f"steps=16 version={version}\n"
            ).encode()
        )
    )
    promoted: list[tuple[Path, str, str]] = []
    monkeypatch.setattr(
        "gz.trainer.driver.promote_checkpoint_pointer",
        lambda root, pointer, model_version: promoted.append(
            (Path(root), pointer, model_version)
        ),
    )

    pump_selfplay_stderr(
        process,
        OpponentTracker(),
        SelfplayStatsTracker(),
        tmp_path,
    )

    assert promoted == [(tmp_path, "best.json", version)]


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
