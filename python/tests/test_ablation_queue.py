from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from gz.trainer.distill import load_distill_config
from gz.trainer.driver import PathsConfig, load_config
from tools.run_ablation_queue import (
    _run_command,
    evaluate_early_stop,
    load_manifest,
    run_one,
    summarize_run,
    validate_manifest,
    wait_for_queue,
)


def test_overnight_manifest_loads_unique_run_directories() -> None:
    manifest = load_manifest(Path("configs/ablations/overnight-2026-07-10/queue.toml"))

    validate_manifest(manifest)
    assert len(manifest["runs"]) == 15
    assert len({spec["config"].paths.run_dir for spec in manifest["runs"]}) == 15
    assert manifest["runs"][0]["name"] == "baseline-seed42"


def test_10m_value_reward_queue_chains_distillation_before_triplet() -> None:
    queue_path = Path(
        "configs/ablations/value-reward-cadence8-10m-2026-07-14/queue.toml"
    )
    manifest = load_manifest(queue_path)

    validate_manifest(manifest)
    assert manifest["wait_for_state"] == Path(
        "runs/queues/whittle-generated-exphormer-v2-sampled-tree-2-"
        "cadence8-value-reward/state.json"
    ).resolve()
    assert [spec["kind"] for spec in manifest["runs"]] == [
        "distill",
        "train",
        "train",
        "train",
    ]
    assert manifest["runs"][0]["generate"] is True
    assert manifest["environment"]["_RJEM_MALLOC_CONF"] == (
        "background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000"
    )

    distill_run, distill = load_distill_config(manifest["runs"][0]["config_path"])
    assert (distill_run.arch.dim, distill_run.arch.layers) == (256, 5)
    assert (distill_run.arch.heads, distill_run.arch.ffn_dim) == (8, 896)
    assert distill_run.arch.name == "gz-graph-v2"
    assert (distill.states, distill_run.trainer.total_steps) == (100_000, 1280)

    training = [spec["config"] for spec in manifest["runs"][1:]]
    assert [config.selfplay.value_reward for config in training] == [
        "sign",
        "graded",
        "graded",
    ]
    assert [config.arch.value_head for config in training] == [
        "scalar",
        "scalar",
        "hl_gauss",
    ]
    assert all(config.trainer.publish_interval == 8 for config in training)
    assert all(config.selfplay.reference_arena_interval == 128 for config in training)
    assert all(config.trainer.checkpoint_retain == 16 for config in training)


def test_summarize_run_requires_final_publish(tmp_path: Path) -> None:
    source = load_config("configs/bases/benchmark-throughput-r8-screen.toml")
    paths = PathsConfig(
        replay_dir=tmp_path / "replay",
        checkpoint_dir=tmp_path / "checkpoints",
        run_dir=tmp_path,
        sample_socket=tmp_path / "sample.sock",
        graphzero_bin=source.paths.graphzero_bin,
    )
    config = replace(source, paths=paths)
    records = [
        {"event": "publish", "training_step": 0},
        {
            "event": "step",
            "step": 4991,
            "timestamp": 10.0,
            "produced_rows": 1234,
            "measure_finals": 20,
            "terminal_cost_ema": 80.0,
            "terminal_cost_best": 55.0,
            "stop_rate": 0.25,
            "episode_len_ema": 40.0,
            "measure_repeat_rate": 0.1,
        },
        {"event": "publish", "training_step": 5000, "timestamp": 20.0},
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_run(tmp_path / "config.toml", config)

    assert summary["complete"] is True
    assert summary["final_publish_step"] == 5000
    assert summary["best_terminal_cost"] == 55.0
    assert summary["tail_terminal_cost_ema"] == 80.0
    assert summary["metrics_duration_seconds"] == 10.0


def test_early_stop_requires_all_thresholds_and_evaluates_rule_once() -> None:
    records = [
        {
            "event": "step",
            "step": step,
            "terminal_cost_ema": terminal_cost,
            "policy_loss": policy_loss,
        }
        for step, terminal_cost, policy_loss in (
            (971, 140.0, 6.60),
            (981, 141.0, 6.61),
            (991, 142.0, 6.62),
            (1001, 143.0, 6.63),
            (1011, 144.0, 6.64),
        )
    ]
    rules = [
        {
            "step": 1000,
            "window": 5,
            "terminal_cost_ema_gt": 130.0,
            "policy_loss_gt": 6.55,
        }
    ]
    evaluated: set[int] = set()

    decision = evaluate_early_stop(records, rules, evaluated)

    assert decision is not None
    assert decision["last_step"] == 1011
    assert decision["terminal_cost_ema"] == 142.0
    assert decision["policy_loss"] == 6.62
    assert evaluate_early_stop(records, rules, evaluated) is None


def test_early_stop_does_not_fire_when_one_threshold_passes() -> None:
    records = [
        {
            "event": "step",
            "step": 1001,
            "terminal_cost_ema": 140.0,
            "policy_loss": 6.4,
        }
    ]
    rules = [
        {
            "step": 1000,
            "window": 1,
            "terminal_cost_ema_gt": 130.0,
            "policy_loss_gt": 6.55,
        }
    ]
    evaluated: set[int] = set()

    assert evaluate_early_stop(records, rules, evaluated) is None
    assert evaluated == {0}


def test_distill_run_command_generates_dataset() -> None:
    command = _run_command(
        {
            "kind": "distill",
            "generate": True,
            "config_path": Path("distill.toml"),
        }
    )

    assert command[1:] == [
        "-m",
        "gz.trainer.distill",
        "distill.toml",
        "--generate",
    ]


def test_wait_for_queue_accepts_completed_dependency(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "finished_at": 123.0,
                "runs": {"first": {"status": "complete"}},
            }
        ),
        encoding="utf-8",
    )

    state = wait_for_queue(state_path, 0.01, 1.0, require_success=True)

    assert state["finished_at"] == 123.0


def test_run_one_cleans_up_child_when_queue_is_interrupted(
    tmp_path: Path, monkeypatch: object
) -> None:
    source = load_config("configs/bases/benchmark-throughput-r8-screen.toml")
    paths = PathsConfig(
        replay_dir=tmp_path / "run" / "replay",
        checkpoint_dir=tmp_path / "run" / "checkpoints",
        run_dir=tmp_path / "run",
        sample_socket=tmp_path / "run" / "sample.sock",
        graphzero_bin=source.paths.graphzero_bin,
    )
    config = replace(source, paths=paths)
    process = _RunningProcess()
    interrupted = []
    cleaned = []
    monkeypatch.setattr("tools.run_ablation_queue._check_disk", lambda *_: None)
    monkeypatch.setattr("tools.run_ablation_queue._check_config_sources", lambda *_: None)
    monkeypatch.setattr("tools.run_ablation_queue._other_trainers", lambda: [])
    monkeypatch.setattr("tools.run_ablation_queue.subprocess.Popen", lambda *_, **__: process)
    monkeypatch.setattr(
        "tools.run_ablation_queue._interrupt_process",
        lambda child: (interrupted.append(child), setattr(child, "returncode", -2)),
    )
    monkeypatch.setattr(
        "tools.run_ablation_queue._cleanup_matching_processes", cleaned.append
    )
    monkeypatch.setattr(
        "tools.run_ablation_queue.time.sleep",
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    spec = {
        "name": "interrupted",
        "config_path": tmp_path / "config.toml",
        "config": config,
        "source_hashes": {},
        "timeout_seconds": 60.0,
        "early_stop": [],
    }

    try:
        run_one(spec, 0.0, 1.0, tmp_path / "results.jsonl", {"runs": {}})
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("run_one did not propagate queue interruption")

    assert interrupted == [process]
    assert cleaned == [paths.run_dir]


def test_run_one_applies_queue_environment(
    tmp_path: Path, monkeypatch: object
) -> None:
    source = load_config("configs/bases/benchmark-throughput-r8-screen.toml")
    paths = PathsConfig(
        replay_dir=tmp_path / "run" / "replay",
        checkpoint_dir=tmp_path / "run" / "checkpoints",
        run_dir=tmp_path / "run",
        sample_socket=tmp_path / "run" / "sample.sock",
        graphzero_bin=source.paths.graphzero_bin,
    )
    config = replace(source, paths=paths, trainer=replace(source.trainer, total_steps=0))
    captured: dict[str, object] = {}

    class _CompletedProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

    def popen(*args: object, **kwargs: object) -> _CompletedProcess:
        captured.update(kwargs)
        return _CompletedProcess()

    monkeypatch.setattr("tools.run_ablation_queue._check_disk", lambda *_: None)
    monkeypatch.setattr("tools.run_ablation_queue._check_config_sources", lambda *_: None)
    monkeypatch.setattr("tools.run_ablation_queue._other_trainers", lambda: [])
    monkeypatch.setattr("tools.run_ablation_queue.subprocess.Popen", popen)
    monkeypatch.setattr("tools.run_ablation_queue._cleanup_matching_processes", lambda *_: None)
    spec = {
        "name": "environment",
        "kind": "train",
        "config_path": tmp_path / "config.toml",
        "config": config,
        "source_hashes": {},
        "timeout_seconds": 60.0,
        "early_stop": [],
    }

    run_one(
        spec,
        0.0,
        1.0,
        tmp_path / "results.jsonl",
        {"runs": {}},
        {"_RJEM_MALLOC_CONF": "background_thread:true"},
    )

    assert captured["env"]["_RJEM_MALLOC_CONF"] == "background_thread:true"


class _RunningProcess:
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode
