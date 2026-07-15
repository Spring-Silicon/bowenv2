from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from typing import Any

from gz.trainer.driver import RunConfig, load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_URL = re.compile(r"https://wandb\.ai/[^\s]+/runs/[a-zA-Z0-9]+")


def _interrupt_on_termination(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _resolve(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_source_hashes(path: Path) -> dict[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    sources = [path]
    extends = data.get("extends")
    if extends is not None:
        if not isinstance(extends, str):
            raise ValueError(f"extends must be a string in {path}")
        sources.append((path.parent / extends).resolve())
    return {str(source): _sha256(source) for source in sources}


def load_manifest(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    queue = data.get("queue")
    if not isinstance(queue, dict):
        raise ValueError("manifest requires [queue]")
    manifest_dir = path.parent.resolve()
    results_dir = _resolve(REPO_ROOT, queue.get("results_dir"), "queue.results_dir")
    poll_seconds = float(queue.get("poll_seconds", 5.0))
    min_free_gb = float(queue.get("min_free_gb", 100.0))
    if poll_seconds <= 0.0 or min_free_gb < 0.0:
        raise ValueError("poll_seconds must be positive and min_free_gb non-negative")
    wait_for_state = queue.get("wait_for_state")
    if wait_for_state is not None:
        wait_for_state = _resolve(REPO_ROOT, wait_for_state, "queue.wait_for_state")
    wait_timeout_seconds = float(queue.get("wait_timeout_seconds", 172800.0))
    if wait_timeout_seconds <= 0.0:
        raise ValueError("queue.wait_timeout_seconds must be positive")
    environment = queue.get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and key and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("queue.environment must contain non-empty string keys and string values")

    observe = data.get("observe")
    if observe is not None:
        if not isinstance(observe, dict):
            raise ValueError("[observe] must be a table")
        observe = {
            "pid": int(observe["pid"]),
            "config": _resolve(manifest_dir, observe.get("config"), "observe.config"),
            "timeout_seconds": float(observe.get("timeout_seconds", 14400)),
        }

    raw_runs = data.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("manifest requires at least one [[runs]] entry")
    runs = []
    names: set[str] = set()
    run_dirs: set[Path] = set()
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, dict):
            raise ValueError(f"runs[{index}] must be a table")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"runs[{index}].name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate run name: {name}")
        names.add(name)
        kind = raw.get("kind", "train")
        if kind not in ("train", "distill"):
            raise ValueError(f"runs[{index}].kind must be 'train' or 'distill'")
        generate = raw.get("generate", False)
        if not isinstance(generate, bool):
            raise ValueError(f"runs[{index}].generate must be boolean")
        if generate and kind != "distill":
            raise ValueError(f"runs[{index}].generate requires kind = 'distill'")
        config_path = _resolve(manifest_dir, raw.get("config"), f"runs[{index}].config")
        config = load_config(config_path)
        if config.paths.run_dir in run_dirs:
            raise ValueError(f"duplicate run directory: {config.paths.run_dir}")
        run_dirs.add(config.paths.run_dir)
        timeout_seconds = float(raw.get("timeout_seconds", 10800))
        if timeout_seconds <= 0.0:
            raise ValueError(f"runs[{index}].timeout_seconds must be positive")
        early_stop = raw.get("early_stop", [])
        if not isinstance(early_stop, list):
            raise ValueError(f"runs[{index}].early_stop must be an array")
        early_stop_rules = []
        for rule_index, rule in enumerate(early_stop):
            if not isinstance(rule, dict):
                raise ValueError(
                    f"runs[{index}].early_stop[{rule_index}] must be a table"
                )
            step = int(rule.get("step", 0))
            window = int(rule.get("window", 5))
            terminal_cost_ema_gt = rule.get(
                "terminal_cost_ema_gt", rule.get("terminal_cost_mean_gt")
            )
            policy_loss_gt = rule.get("policy_loss_gt")
            if step <= 0 or window <= 0:
                raise ValueError(
                    f"runs[{index}].early_stop[{rule_index}] step and window must be positive"
                )
            if terminal_cost_ema_gt is None and policy_loss_gt is None:
                raise ValueError(
                    f"runs[{index}].early_stop[{rule_index}] requires a metric threshold"
                )
            early_stop_rules.append(
                {
                    "step": step,
                    "window": window,
                    "terminal_cost_ema_gt": (
                        float(terminal_cost_ema_gt)
                        if terminal_cost_ema_gt is not None
                        else None
                    ),
                    "policy_loss_gt": (
                        float(policy_loss_gt) if policy_loss_gt is not None else None
                    ),
                }
            )
        runs.append(
            {
                "name": name,
                "kind": kind,
                "generate": generate,
                "config_path": config_path,
                "config": config,
                "source_hashes": _config_source_hashes(config_path),
                "timeout_seconds": timeout_seconds,
                "early_stop": early_stop_rules,
            }
        )

    return {
        "name": str(queue.get("name", path.stem)),
        "results_dir": results_dir,
        "poll_seconds": poll_seconds,
        "min_free_gb": min_free_gb,
        "continue_on_failure": bool(queue.get("continue_on_failure", False)),
        "wait_for_state": wait_for_state,
        "wait_timeout_seconds": wait_timeout_seconds,
        "wait_for_success": bool(queue.get("wait_for_success", True)),
        "environment": environment,
        "observe": observe,
        "runs": runs,
    }


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid metrics JSON at {path}:{line_number}") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def evaluate_early_stop(
    records: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    evaluated: set[int],
) -> dict[str, Any] | None:
    steps = [record for record in records if record.get("event") == "step"]
    if not steps:
        return None
    last_step = int(steps[-1].get("step", -1))
    for index, rule in enumerate(rules):
        if index in evaluated or last_step < rule["step"]:
            continue
        window = steps[-rule["window"] :]
        terminal_cost_ema = _mean(window, "terminal_cost_ema")
        policy_loss = _mean(window, "policy_loss")
        required = []
        if rule["terminal_cost_ema_gt"] is not None:
            if terminal_cost_ema is None:
                continue
            required.append(terminal_cost_ema > rule["terminal_cost_ema_gt"])
        if rule["policy_loss_gt"] is not None:
            if policy_loss is None:
                continue
            required.append(policy_loss > rule["policy_loss_gt"])
        evaluated.add(index)
        if all(required):
            return {
                "rule_index": index,
                "rule": rule,
                "last_step": last_step,
                "terminal_cost_ema": terminal_cost_ema,
                "policy_loss": policy_loss,
            }
    return None


def summarize_run(config_path: Path, config: RunConfig) -> dict[str, Any]:
    records = _read_metrics(config.paths.run_dir / "metrics.jsonl")
    steps = [record for record in records if record.get("event") == "step"]
    publishes = [record for record in records if record.get("event") == "publish"]
    final_publish = max((int(record.get("training_step", -1)) for record in publishes), default=-1)
    tail = steps[-min(50, len(steps)) :]
    best_terminal = min(
        (float(record["terminal_cost_best"]) for record in steps if isinstance(record.get("terminal_cost_best"), (int, float))),
        default=None,
    )
    timestamps = [float(record["timestamp"]) for record in records if isinstance(record.get("timestamp"), (int, float))]
    log_path = config.paths.run_dir / "trainer.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    urls = WANDB_URL.findall(log_text)
    last = steps[-1] if steps else {}
    return {
        "config": str(config_path),
        "run_dir": str(config.paths.run_dir),
        "total_steps": config.trainer.total_steps,
        "complete": final_publish == config.trainer.total_steps,
        "final_publish_step": final_publish,
        "last_logged_step": int(last.get("step", -1)),
        "produced_rows": int(last.get("produced_rows", 0)),
        "measured_finals": int(last.get("measure_finals", 0)),
        "best_terminal_cost": best_terminal,
        "final_terminal_cost_ema": last.get("terminal_cost_ema"),
        "tail_terminal_cost_ema": _mean(tail, "terminal_cost_ema"),
        "tail_stop_rate": _mean(tail, "stop_rate"),
        "tail_episode_len": _mean(tail, "episode_len_ema"),
        "tail_repeat_rate": _mean(tail, "measure_repeat_rate"),
        "tail_learner_win_rate": _mean(tail, "learner_win_rate"),
        "tail_policy_loss": _mean(tail, "policy_loss"),
        "tail_value_loss": _mean(tail, "value_loss"),
        "samples_per_row": last.get("samples_per_row"),
        "metrics_duration_seconds": max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None,
        "wandb_url": urls[-1] if urls else None,
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_result(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return False
    fields = stat.split()
    return len(fields) > 2 and fields[2] != "Z"


def _interrupt_process(process: subprocess.Popen[bytes], grace_seconds: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10.0)


def _matching_process_groups(fragment: str) -> set[int]:
    groups: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if fragment not in command:
                continue
            if not (
                ("graphzero selfplay" in command)
                or ("-m gz.evaluator" in command)
                or ("graphzero replay-serve" in command)
            ):
                continue
            groups.add(os.getpgid(int(entry.name)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    groups.discard(os.getpgrp())
    return groups


def _cleanup_matching_processes(run_dir: Path) -> None:
    groups = _matching_process_groups(str(run_dir))
    for group in groups:
        try:
            os.killpg(group, signal.SIGINT)
        except ProcessLookupError:
            pass
    if groups:
        time.sleep(5.0)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _other_trainers() -> list[int]:
    trainers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError):
            continue
        if b"-m" in command and any(
            module in command for module in (b"gz.trainer", b"gz.trainer.distill")
        ):
            trainers.append(int(entry.name))
    return trainers


def _check_disk(path: Path, min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"only {free_gb:.1f} GiB free; floor is {min_free_gb:.1f} GiB")


def _check_config_sources(spec: dict[str, Any]) -> None:
    actual = _config_source_hashes(spec["config_path"])
    if actual != spec["source_hashes"]:
        raise RuntimeError(f"config sources changed after queue start: {spec['name']}")


def wait_for_queue(
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    require_success: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            state = None
        if isinstance(state, dict) and "finished_at" in state:
            runs = state.get("runs")
            if not isinstance(runs, dict):
                raise RuntimeError(f"dependency queue has invalid state: {state_path}")
            failed = {}
            for name, result in runs.items():
                if not isinstance(result, dict):
                    failed[name] = "invalid-state"
                elif result.get("status") not in ("complete", "skipped-complete"):
                    failed[name] = result.get("status")
            if require_success and failed:
                raise RuntimeError(f"dependency queue did not succeed: {failed}")
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError(f"dependency queue did not finish: {state_path}")
        time.sleep(poll_seconds)


def _run_command(spec: dict[str, Any]) -> list[str]:
    config_path = str(spec["config_path"])
    if spec.get("kind", "train") == "distill":
        command = [sys.executable, "-m", "gz.trainer.distill", config_path]
        if spec.get("generate", False):
            command.append("--generate")
        return command
    return [sys.executable, "-m", "gz.trainer", "--config", config_path]


def wait_for_observed(
    observed: dict[str, Any],
    poll_seconds: float,
    results_path: Path,
    state: dict[str, Any],
) -> None:
    config_path = observed["config"]
    config = load_config(config_path)
    deadline = time.monotonic() + observed["timeout_seconds"]
    while _pid_alive(observed["pid"]):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"observed PID {observed['pid']} did not finish before timeout")
        time.sleep(poll_seconds)
    summary = summarize_run(config_path, config)
    result = {"name": "observed-policy-budget", "status": "observed", **summary}
    _append_result(results_path, result)
    state["observed"] = result
    if not summary["complete"]:
        raise RuntimeError("observed policy-budget run ended without its final publish")


def run_one(
    spec: dict[str, Any],
    min_free_gb: float,
    poll_seconds: float,
    results_path: Path,
    state: dict[str, Any],
    environment_overrides: dict[str, str] | None = None,
) -> bool:
    config_path: Path = spec["config_path"]
    config: RunConfig = spec["config"]
    existing = summarize_run(config_path, config)
    if existing["complete"]:
        result = {"name": spec["name"], "status": "skipped-complete", **existing}
        _append_result(results_path, result)
        state["runs"][spec["name"]] = result
        return True
    if config.paths.run_dir.exists() and any(config.paths.run_dir.iterdir()):
        raise RuntimeError(f"incomplete non-empty run directory: {config.paths.run_dir}")

    _check_disk(REPO_ROOT, min_free_gb)
    _check_config_sources(spec)
    trainers = _other_trainers()
    if trainers:
        raise RuntimeError(f"refusing to overlap trainer processes: {trainers}")
    config.paths.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.paths.run_dir / "trainer.log"
    environment = os.environ.copy()
    environment.update(environment_overrides or {})
    python_path = [str(REPO_ROOT / "python"), str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    command = _run_command(spec)
    started = time.time()
    timed_out = False
    early_stop = None
    evaluated_rules: set[int] = set()
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + spec["timeout_seconds"]
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    _interrupt_process(process)
                    break
                try:
                    records = _read_metrics(config.paths.run_dir / "metrics.jsonl")
                except ValueError:
                    records = []
                early_stop = evaluate_early_stop(
                    records, spec["early_stop"], evaluated_rules
                )
                if early_stop is not None:
                    _interrupt_process(process)
                    break
                time.sleep(poll_seconds)
        finally:
            if process.poll() is None:
                _interrupt_process(process)
            _cleanup_matching_processes(config.paths.run_dir)
        exit_code = process.returncode

    summary = summarize_run(config_path, config)
    succeeded = exit_code == 0 and summary["complete"]
    result = {
        "name": spec["name"],
        "status": (
            "complete"
            if succeeded
            else (
                "early-stopped"
                if early_stop is not None
                else ("timeout" if timed_out else "failed")
            )
        ),
        "exit_code": exit_code,
        "wall_seconds": time.time() - started,
        "early_stop": early_stop,
        **summary,
    }
    _append_result(results_path, result)
    state["runs"][spec["name"]] = result
    return succeeded


def validate_manifest(manifest: dict[str, Any]) -> None:
    for spec in manifest["runs"]:
        config: RunConfig = spec["config"]
        if not math.isfinite(spec["timeout_seconds"]):
            raise ValueError(f"non-finite timeout for {spec['name']}")
        if config.paths.run_dir == manifest["results_dir"]:
            raise ValueError(f"run directory collides with results directory: {spec['name']}")


def manifest_snapshot(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPO_ROOT, text=True
    )
    binary = REPO_ROOT / "target/release/graphzero"
    return {
        "created_at": time.time(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "graphzero_binary_sha256": _sha256(binary),
        "environment": manifest["environment"],
        "runs": [
            {
                "name": spec["name"],
                "kind": spec["kind"],
                "generate": spec["generate"],
                "config_path": str(spec["config_path"]),
                "source_hashes": spec["source_hashes"],
                "effective_config": asdict(spec["config"]),
            }
            for spec in manifest["runs"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGHUP, _interrupt_on_termination)
    signal.signal(signal.SIGTERM, _interrupt_on_termination)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    if args.validate_only:
        for spec in manifest["runs"]:
            print(f"{spec['name']}\t{spec['config_path']}\t{spec['config'].paths.run_dir}")
        return 0

    results_dir: Path = manifest["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"
    state_path = results_dir / "state.json"
    _write_json(results_dir / "manifest-snapshot.json", manifest_snapshot(manifest_path, manifest))
    state: dict[str, Any] = {
        "manifest": str(manifest_path),
        "name": manifest["name"],
        "started_at": time.time(),
        "runs": {},
    }
    _write_json(state_path, state)

    if manifest["wait_for_state"] is not None:
        dependency_path: Path = manifest["wait_for_state"]
        state["waiting_for"] = str(dependency_path)
        _write_json(state_path, state)
        dependency = wait_for_queue(
            dependency_path,
            manifest["poll_seconds"],
            manifest["wait_timeout_seconds"],
            manifest["wait_for_success"],
        )
        state.pop("waiting_for", None)
        state["dependency_finished_at"] = dependency["finished_at"]
        _write_json(state_path, state)

    if manifest["observe"] is not None:
        wait_for_observed(manifest["observe"], manifest["poll_seconds"], results_path, state)
        _write_json(state_path, state)

    for spec in manifest["runs"]:
        state["active"] = spec["name"]
        _write_json(state_path, state)
        try:
            succeeded = run_one(
                spec,
                manifest["min_free_gb"],
                manifest["poll_seconds"],
                results_path,
                state,
                manifest["environment"],
            )
        except Exception as error:
            result = {"name": spec["name"], "status": "runner-error", "error": repr(error)}
            _append_result(results_path, result)
            state["runs"][spec["name"]] = result
            succeeded = False
        state.pop("active", None)
        _write_json(state_path, state)
        if not succeeded and not manifest["continue_on_failure"]:
            return 1

    state["finished_at"] = time.time()
    _write_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
