from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from gz.model.exphormer import ArchConfig, build_model, initialize_policy
from gz.checkpoints.publish import prune_checkpoints
from gz.trainer.data import TrainingStager
from gz.trainer.loop import LoopConfig, TrainerLoop
from gz.trainer.publish import EmaWeights, publish_ema
from gz.trainer.sampler import SampleAck, SampleClient, step_seed


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    lr: float = 3e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 200
    lr_decay_steps: int | None = None
    min_lr_ratio: float = 0.0
    batch: int = 256
    # Zero shares the policy batch, preserving the historical trainer path.
    # A positive value samples and stages an independent value batch.
    value_batch: int = 0
    window_rows: int = 200_000
    # Zero inherits window_rows. This counts source rows; paired value
    # training evaluates both orientations of every sampled source row.
    value_window_rows: int = 0
    total_steps: int = 1000
    publish_interval: int = 500
    # Newest ordinary actor checkpoints to retain. Named checkpoint pointers
    # and in-flight arena challengers are retained in addition; zero disables.
    checkpoint_retain: int = 0
    # Publish and permanently pin an exact checkpoint at each positive
    # multiple of this many optimizer steps. Zero disables milestone pins.
    permanent_checkpoint_interval: int = 1000
    # Hold each periodic checkpoint until the next training gate. A one-block
    # lag matches whittlezero's overlapped actor snapshot schedule.
    publish_lag_blocks: int = 0
    value_weight: float = 1.0
    # Scale value gradients entering the shared trunk while leaving the
    # private value head's gradients unchanged.
    value_trunk_grad_scale: float = 1.0
    value_final_weight: float = 1.0
    value_v8_weight: float = 0.0
    value_v32_weight: float = 0.0
    terminal_score_weight: float = 0.0
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    adamw_lr: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    policy_init: str = "default"
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    min_startup_rows: int = 256
    # Optional experiment controls. By default the legacy seed drives both
    # Torch (initialization/dropout) and replay/value-orientation sampling.
    seed: int = 0
    model_seed: int | None = None
    data_seed: int | None = None
    device: str = "cuda:1"
    startup_timeout: float = 60.0
    reconnect_limit: int = 5
    log_interval: int = 1
    step_sleep: float = 0.0
    min_available_gb: float = 40.0
    # Sample batch N+1 on a background thread while the GPU trains batch N,
    # taking the socket read/decode off the step critical path. Off = the
    # historical strictly-serial loop, kept for A/B comparison.
    prefetch: bool = True
    # Sample policy and value batches on separate replay connections. Off
    # keeps GPU prefetching but issues both requests sequentially on one client.
    parallel_value_sampling: bool = True
    # Compile static-shape model forward/backward graphs with TorchInductor.
    # Optimizer, EMA, and checkpoints continue to own the original module.
    compile_model: bool = False
    compile_mode: str = "default"
    matmul_precision: str = "highest"
    # Pace the trainer against fresh production: each gate waits until enough
    # source rows exist for its cumulative policy samples. Zero disables.
    max_reuse: float = 0.0
    # Number of optimizer steps admitted together by max_reuse. One is the
    # historical streaming gate; whittlezero admits eight after each wave.
    reuse_gate_interval: int = 1
    # Completed episodes required per admitted block. Zero disables the
    # episode-count gate; this preserves fixed actor-wave cadence when
    # episode lengths change.
    reuse_gate_episodes: int = 0
    # Continue an interrupted run in place: load the latest
    # published checkpoint (EMA weights seed both the live model and the
    # EMA -- an approximate resume; optimizer moments restart), and start
    # the step counter at the checkpoint's training_step.
    resume: bool = False
    # Seed a new run from another checkpoint directory. Only model weights
    # transfer; optimizer, EMA, counters, and training step restart at zero.
    init_checkpoint: str = ""
    # "all" restores every model tensor. "policy" transfers the trunk and
    # policy while preserving this run's freshly initialized value module.
    init_checkpoint_scope: str = "all"


@dataclass(frozen=True, slots=True)
class SelfplayConfig:
    lanes: int = 2
    workers_per_lane: int = 8
    simulations: int = 8
    max_considered: int = 8
    gumbel_scale: float = 1.0
    c_visit: float = 50.0
    c_scale: float = 1.0
    max_steps: int = 8
    max_candidates: int = 255
    max_row_backlog: int = 200_000
    replay_retain: int = 0
    eval_device: str = "cuda:0"
    eval_poll_interval: float = 10.0
    seed: int = 0
    max_batch: int = 16
    python_dir: str = "python"
    position_features: bool = True
    no_backtrack: bool = True
    gumbel_noise_overlap: float = 0.5
    mask_stop: bool = False
    length_tiebreak: bool = True
    tree_reuse: bool = True
    eval_processes: int = 1
    admission_stagger_ms: int = 0
    admission_smoothing: bool = False
    wave_batching: bool = False


@dataclass(frozen=True, slots=True)
class WandbConfig:
    project: str = ""
    entity: str = ""
    run_name: str = ""
    mode: str = ""
    # Resume this wandb run id in place (wandb.init(resume="must")).
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class PathsConfig:
    replay_dir: Path
    checkpoint_dir: Path
    run_dir: Path
    sample_socket: Path
    graphzero_bin: str


@dataclass(frozen=True, slots=True)
class RunConfig:
    trainer: TrainerConfig
    selfplay: SelfplayConfig
    paths: PathsConfig
    wandb: WandbConfig
    arch: ArchConfig

def _sample_window_rows(window_rows: int, produced_rows: int) -> int:
    return max(1, min(int(window_rows), int(produced_rows)))


def _cumulative_reuse(step: int, batch: int, produced_rows: int) -> float:
    if batch <= 0 or produced_rows <= 0:
        return 0.0
    return (step + 1) * batch / produced_rows


def _resolved_trainer_seeds(config: TrainerConfig) -> tuple[int, int]:
    model_seed = config.seed if config.model_seed is None else config.model_seed
    data_seed = config.seed if config.data_seed is None else config.data_seed
    return model_seed, data_seed


def _required_produced_rows(
    step: int,
    batch: int,
    max_reuse: float,
    gate_interval: int,
) -> int:
    if max_reuse <= 0.0:
        return 0
    block_end = (step // gate_interval + 1) * gate_interval
    return math.ceil(block_end * batch / max_reuse)


def _required_episodes(step: int, gate_interval: int, episodes_per_gate: int) -> int:
    if episodes_per_gate <= 0:
        return 0
    block = step // gate_interval + 1
    return block * episodes_per_gate


@dataclass(frozen=True, slots=True)
class SampledBatches:
    policy: object
    value: object | None


def _sample_training_batches(
    sampler: SampleClient,
    *,
    policy_batch: int,
    policy_window_rows: int,
    value_batch: int,
    value_window_rows: int,
    run_seed: int,
    step: int,
    produced_rows: int,
    value_sampler: SampleClient | None = None,
    value_executor: Executor | None = None,
) -> SampledBatches:
    if value_executor is not None and value_sampler is None:
        raise ValueError("parallel value sampling requires a separate sample client")
    value_future = None
    if value_batch > 0 and value_executor is not None:
        assert value_sampler is not None
        value_future = value_executor.submit(
            value_sampler.sample,
            value_batch,
            _sample_window_rows(value_window_rows, produced_rows),
            step_seed(run_seed, step, "value-sample"),
        )
    policy = sampler.sample(
        policy_batch,
        _sample_window_rows(policy_window_rows, produced_rows),
        step_seed(run_seed, step),
    )
    value = None
    if value_batch > 0:
        if value_future is not None:
            value = value_future.result()
        else:
            value = sampler.sample(
                value_batch,
                _sample_window_rows(value_window_rows, produced_rows),
                step_seed(run_seed, step, "value-sample"),
            )
    return SampledBatches(policy=policy, value=value)


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    _set_matmul_precision(config.trainer.matmul_precision)
    model_seed, data_seed = _resolved_trainer_seeds(config.trainer)
    for path in (config.paths.replay_dir, config.paths.checkpoint_dir, config.paths.run_dir):
        path.mkdir(parents=True, exist_ok=True)
    resume_resolved = None
    resume_start = 0
    if config.trainer.resume:
        from gz.checkpoints import DirectorySource

        resume_resolved = DirectorySource(str(config.paths.checkpoint_dir)).resolve_latest()
        resume_start = resume_resolved.manifest.training_step
        if resume_start >= config.trainer.total_steps:
            raise RuntimeError("resume checkpoint is at or past total_steps")
    metrics = MetricsWriter(config.paths.run_dir / "metrics.jsonl", WandbRun.start(config))

    arch = config.arch
    model = None
    ema = None
    published_snapshot = None
    if not config.trainer.resume:
        init_replay(config)
        serve = spawn_replay_serve(config)
        try:
            sampler = SampleClient(
                config.paths.sample_socket,
                startup_timeout=config.trainer.startup_timeout,
                reconnect_limit=config.trainer.reconnect_limit,
            )
            sampler.wait_until_ready(0, alive_check=lambda: check_child(serve, "replay-serve"))
            _seed_model(model_seed)
            model = build_model(sampler.feature_schema, arch)
            if config.trainer.init_checkpoint:
                resolved = _load_initial_checkpoint(
                    model,
                    config.trainer.init_checkpoint,
                    sampler.feature_schema_hash,
                    arch,
                    scope=config.trainer.init_checkpoint_scope,
                )
                metrics.write(
                    {
                        "event": "init_checkpoint",
                        "source_model_version": resolved.manifest.model_version.hex(),
                        "source_training_step": resolved.manifest.training_step,
                        "scope": config.trainer.init_checkpoint_scope,
                    }
                )
            else:
                initialize_policy(model, config.trainer.policy_init)
            model = model.to(config.trainer.device)
            ema = EmaWeights(model, config.trainer.ema_decay)
            first = publish_ema(
                config.paths.checkpoint_dir,
                ema,
                schema=sampler.feature_schema,
                schema_hash=sampler.feature_schema_hash,
                arch=arch,
                training_step=0,
                run_id=config.paths.run_dir.name,
            )
            param_norm, _ = ema.norms(None)
            published_snapshot = ema.state_dict()
            metrics.write(
                {
                    "event": "publish",
                    "training_step": 0,
                    "model_version": first.model_version.hex(),
                    "param_norm": param_norm,
                    "update_norm": 0.0,
                }
            )
        finally:
            sampler.close()
            stop_child(serve)

    _prune_training_checkpoints(config)

    def start_stage(
        stage_config: RunConfig,
    ) -> tuple[
        subprocess.Popen[bytes],
        threading.Thread,
        SelfplayStatsTracker,
        SampleClient,
    ]:
        child = spawn_torch_selfplay(stage_config)
        stage_stats = SelfplayStatsTracker()
        pump = threading.Thread(
            target=pump_selfplay_stderr,
            args=(
                child,
                stage_stats,
            ),
            daemon=True,
        )
        pump.start()
        stage_sampler = SampleClient(
            stage_config.paths.sample_socket,
            startup_timeout=config.trainer.startup_timeout,
            reconnect_limit=config.trainer.reconnect_limit,
        )
        try:
            stage_sampler.wait_until_ready(
                config.trainer.min_startup_rows,
                alive_check=lambda: check_child(child, "selfplay"),
            )
        except BaseException:
            stage_sampler.close()
            kill_child(child)
            pump.join(timeout=5.0)
            raise
        return child, pump, stage_stats, stage_sampler

    (
        selfplay,
        selfplay_pump,
        selfplay_stats,
        sampler,
    ) = start_stage(config)
    prefetcher = None
    try:
        if config.trainer.resume:
            from gz.checkpoints.weights import load_state_dict

            assert resume_resolved is not None
            resolved = resume_resolved
            if resolved.manifest.feature_schema_hash != sampler.feature_schema_hash:
                raise RuntimeError("resume checkpoint feature schema does not match the store")
            if ArchConfig.from_dict(resolved.manifest.arch_config) != arch:
                raise RuntimeError("resume checkpoint arch does not match [arch] config")
            model = build_model(sampler.feature_schema, arch).to(config.trainer.device)
            model.load_state_dict(load_state_dict(resolved.weights_path))
            ema = EmaWeights(model, config.trainer.ema_decay)
            published_snapshot = ema.state_dict()
            metrics.write(
                {
                    "event": "resume",
                    "training_step": resume_start,
                    "model_version": resolved.manifest.model_version.hex(),
                }
            )
        feature_schema = sampler.feature_schema
        feature_schema_hash = sampler.feature_schema_hash
        sample_max_batch = sampler.max_batch
        metrics.write(
            {
                "event": "value_tasks",
                "auxiliary_heads": arch.auxiliary_heads,
                "horizons": [8, 32],
                "score_scale": feature_schema.max_nodes,
                "value_final_weight": config.trainer.value_final_weight,
                "value_v8_weight": config.trainer.value_v8_weight,
                "value_v32_weight": config.trainer.value_v32_weight,
                "terminal_score_weight": config.trainer.terminal_score_weight,
            }
        )
        policy_stager = TrainingStager(
            feature_schema,
            sample_max_batch,
            config.trainer.device,
            validate_terminal_score=config.trainer.terminal_score_weight > 0.0,
        )
        value_stager = (
            TrainingStager(
                feature_schema,
                sample_max_batch,
                config.trainer.device,
                validate_terminal_score=config.trainer.terminal_score_weight > 0.0,
            )
            if config.trainer.value_batch > 0
            else None
        )
        loop = TrainerLoop(
            model,
            trainer_loop_config(
                config.trainer,
                symmetric_mask_stop=config.selfplay.mask_stop,
            ),
        )
        loop.step_index = resume_start
        window = PerfWindow(0, 0)

        def start_prefetch(
            stage_sampler: SampleClient,
            start_step: int,
        ) -> SamplePrefetcher | None:
            if not config.trainer.prefetch:
                return None
            result = SamplePrefetcher(
                stage_sampler,
                config.trainer.batch,
                config.trainer.window_rows,
                config.trainer.value_batch,
                config.trainer.value_window_rows or config.trainer.window_rows,
                data_seed,
                config.trainer.total_steps,
                config.trainer.max_reuse,
                config.trainer.reuse_gate_interval,
                config.trainer.reuse_gate_episodes,
                start_step=start_step,
                value_sampler=(
                    stage_sampler.fork()
                    if config.trainer.value_batch > 0
                    and config.trainer.parallel_value_sampling
                    else None
                ),
            )
            result.start()
            return result

        prefetcher = start_prefetch(sampler, resume_start)

        def publish_training_step(training_step: int) -> None:
            nonlocal published_snapshot, last_published_step
            if training_step == last_published_step:
                return
            if training_step < last_published_step:
                raise RuntimeError("training checkpoints must publish in step order")
            manifest = publish_ema(
                config.paths.checkpoint_dir,
                ema,
                schema=feature_schema,
                schema_hash=feature_schema_hash,
                arch=arch,
                training_step=training_step,
                run_id=config.paths.run_dir.name,
                checkpoint_pointers=_permanent_checkpoint_pointers(
                    config.trainer,
                    training_step,
                ),
            )
            pruned = _prune_training_checkpoints(config)
            param_norm, update_norm = ema.norms(published_snapshot)
            published_snapshot = ema.state_dict()
            metrics.write(
                {
                    "event": "publish",
                    "training_step": training_step,
                    "model_version": manifest.model_version.hex(),
                    "param_norm": param_norm,
                    "update_norm": update_norm,
                    "checkpoints_pruned": len(pruned),
                }
            )
            last_published_step = training_step

        last_published_step = resume_start
        produced_floor = 0
        episodes_floor = 0
        pending_publish_step = None
        for step in range(resume_start, config.trainer.total_steps):
            check_child(selfplay, "selfplay")
            if step % 50 == 0:
                check_memory(config.trainer.min_available_gb)
            # With prefetch, sample_ms measures the wait for the queued
            # batch: ~0 while sampling keeps up, the residual stall when it
            # does not. The reuse gate stalls inside the same window.
            sample_started = time.perf_counter()
            gate_step = step
            if config.trainer.max_reuse > 0 or config.trainer.reuse_gate_episodes > 0:
                needed_rows = _required_produced_rows(
                    gate_step,
                    config.trainer.batch,
                    config.trainer.max_reuse,
                    config.trainer.reuse_gate_interval,
                )
                needed_episodes = _required_episodes(
                    gate_step,
                    config.trainer.reuse_gate_interval,
                    config.trainer.reuse_gate_episodes,
                )
                while (
                    produced_floor < needed_rows
                    or episodes_floor < needed_episodes
                ):
                    ack = prefetcher.refresh() if prefetcher is not None else sampler.refresh()
                    produced_floor = ack.produced_rows
                    episodes_floor = ack.episodes
                    if (
                        produced_floor >= needed_rows
                        and episodes_floor >= needed_episodes
                    ):
                        break
                    check_child(selfplay, "selfplay")
                    time.sleep(0.1)
            if pending_publish_step is not None:
                publish_training_step(pending_publish_step)
                pending_publish_step = None
            if prefetcher is not None:
                samples = prefetcher.next()
            else:
                ack = sampler.refresh()
                samples = _sample_training_batches(
                    sampler,
                    policy_batch=config.trainer.batch,
                    policy_window_rows=config.trainer.window_rows,
                    value_batch=config.trainer.value_batch,
                    value_window_rows=(
                        config.trainer.value_window_rows or config.trainer.window_rows
                    ),
                    run_seed=data_seed,
                    step=step,
                    produced_rows=ack.produced_rows,
                )
            train_started = time.perf_counter()
            # Metrics force a host-device sync; off-interval steps skip them
            # entirely so consecutive steps pipeline on the GPU.
            metrics_step = step % config.trainer.log_interval == 0
            policy_training_batch = policy_stager.copy(
                samples.policy.batch,
                samples.policy.targets,
            )
            value_training_batch = None
            if value_stager is not None:
                assert samples.value is not None
                value_training_batch = value_stager.copy(
                    samples.value.batch,
                    samples.value.targets,
                )
            metrics_record = loop.train_step(
                policy_training_batch,
                value_training_batch,
                with_metrics=metrics_step,
            )
            ema.update(model)
            window.record(sample_started, train_started, time.perf_counter())
            if metrics_step:
                assert metrics_record is not None
                ack = prefetcher.refresh() if prefetcher is not None else sampler.refresh()
                produced = ack.produced_rows
                total_produced = produced
                total_episodes = ack.episodes
                value_batch = config.trainer.value_batch
                if value_batch == 0 and config.trainer.value_weight != 0.0:
                    value_batch = config.trainer.batch
                stop_rate = ack.episodes_stopped / ack.episodes if ack.episodes else 0.0
                record = {
                    "event": "step",
                    "timestamp": time.time(),
                    "step": metrics_record.step,
                    "policy_loss": metrics_record.policy_loss,
                    "value_loss": metrics_record.value_loss,
                    "value_final_loss": metrics_record.value_final_loss,
                    "value_v8_loss": metrics_record.value_v8_loss,
                    "value_v32_loss": metrics_record.value_v32_loss,
                    "terminal_score_loss": metrics_record.terminal_score_loss,
                    "terminal_score_mae": metrics_record.terminal_score_mae,
                    "terminal_score_bias": metrics_record.terminal_score_bias,
                    "loss": metrics_record.loss,
                    "grad_norm": metrics_record.grad_norm,
                    "lr": metrics_record.lr,
                    "fraction_valid": metrics_record.fraction_valid,
                    "label_mean": metrics_record.label_mean,
                    "terminal_cost_ema": ack.episode_cost_ema,
                    "terminal_cost_best": ack.best_cost,
                    "produced_rows": total_produced,
                    "policy_reuse": _cumulative_reuse(
                        gate_step,
                        config.trainer.batch,
                        produced,
                    ),
                    "value_reuse": _cumulative_reuse(
                        gate_step,
                        value_batch,
                        produced,
                    ),
                    "stop_rate": stop_rate,
                    "episode_len_ema": ack.episode_len_ema,
                    "stop_rate_ema": ack.stop_rate_ema,
                    **selfplay_stats.step_fields(),
                }
                record["value_accuracy"] = metrics_record.value_accuracy
                record["learner_win_rate"] = metrics_record.learner_win_rate
                # -1.0 = no labeled episode appended yet (unseeded EMA).
                if ack.learner_win_rate_ema >= 0.0:
                    record["learner_win_rate_ema"] = ack.learner_win_rate_ema
                if ack.value_sign_accuracy_early_ema >= 0.0:
                    record["value_sign_accuracy_early_ema"] = (
                        ack.value_sign_accuracy_early_ema
                    )
                if ack.value_sign_accuracy_late_ema >= 0.0:
                    record["value_sign_accuracy_late_ema"] = ack.value_sign_accuracy_late_ema
                if ack.episode_latency_ema >= 0.0:
                    record["episode_latency_s"] = ack.episode_latency_ema
                record.update(_symmetric_step_fields(ack, total_episodes))
                # Outcome gauges are per-store-open; a zero means unseeded
                # (no episode appended by this selfplay process yet).
                record.update(window.drain(total_produced, total_episodes))
                record.update(metrics_record.logging_fields())
                metrics.write(record)
            if _checkpoint_due(config.trainer, step + 1):
                if config.trainer.publish_lag_blocks:
                    pending_publish_step = step + 1
                else:
                    publish_training_step(step + 1)
            if config.trainer.step_sleep:
                time.sleep(config.trainer.step_sleep)
        if prefetcher is not None:
            prefetcher.stop()
        if pending_publish_step is not None:
            publish_training_step(pending_publish_step)
        elif not _checkpoint_due(config.trainer, config.trainer.total_steps):
            publish_training_step(config.trainer.total_steps)
    except BaseException:
        # wandb's atexit hook marks the run crashed; only the clean path
        # finishes it explicitly.
        if prefetcher is not None:
            prefetcher.stop()
        kill_child(selfplay)
        if prefetcher is not None:
            try:
                prefetcher.join()
            except BaseException:
                pass
        sampler.close()
        selfplay_pump.join(timeout=5.0)
        raise
    else:
        kill_child(selfplay)
        if prefetcher is not None:
            prefetcher.join()
        sampler.close()
        selfplay_pump.join(timeout=5.0)
        metrics.finish()


class SelfplayStatsTracker:
    """Parses eval_stats / measure_stats heartbeats off the selfplay
    stderr pump. The selfplay side emits cumulative counters every 30s;
    step_fields() reports window rates (delta since the last fold) plus
    the cumulative ledger, so batch fill and the measure repeat rate
    are live in wandb instead of dying with the killed process's exit
    summary."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.eval_batches = None
        self.eval_rows = None
        self.eval_at = None
        self.folded = None
        self.measure: dict[str, int] = {}
        self.admission: dict[str, int] = {}

    def observe_eval(self, fields: dict[str, str]) -> None:
        if fields.get("role", "current") != "current":
            return
        try:
            batches = int(fields["batches"])
            rows = int(fields["rows"])
        except (KeyError, ValueError):
            return
        with self.lock:
            self.eval_batches = batches
            self.eval_rows = rows
            self.eval_at = time.time()

    def observe_measure(self, fields: dict[str, str]) -> None:
        try:
            parsed = {
                key: int(fields[key])
                for key in ("appended", "dropped", "finals", "distinct")
            }
        except (KeyError, ValueError):
            return
        with self.lock:
            self.measure = parsed

    def observe_admission(self, fields: dict[str, str]) -> None:
        try:
            parsed = {
                key: int(fields[key])
                for key in (
                    "outstanding",
                    "reserved",
                    "waiting",
                    "max_waiting",
                    "bootstrap_grants",
                    "paced_grants",
                    "eval_capacity_milli",
                    "episode_work_milli",
                    "pressure_gain_milli",
                    "gap_us",
                )
            }
        except (KeyError, ValueError):
            return
        with self.lock:
            self.admission = parsed

    def step_fields(self) -> dict[str, object]:
        with self.lock:
            out: dict[str, object] = {}
            if self.eval_batches is not None:
                out["eval_batches_total"] = self.eval_batches
                out["eval_rows_total"] = self.eval_rows
                if self.folded is not None:
                    prev_batches, prev_rows, prev_at = self.folded
                    d_batches = self.eval_batches - prev_batches
                    d_rows = self.eval_rows - prev_rows
                    dt = self.eval_at - prev_at
                    if d_batches > 0 and dt > 0:
                        out["eval_mean_batch"] = d_rows / d_batches
                        out["eval_batches_per_s"] = d_batches / dt
                        out["eval_evals_per_s"] = d_rows / dt
                if self.folded is None or self.folded[0] != self.eval_batches:
                    self.folded = (self.eval_batches, self.eval_rows, self.eval_at)
            if self.measure:
                out["measure_finals"] = self.measure["finals"]
                out["measure_distinct_finals"] = self.measure["distinct"]
                if self.measure["finals"] > 0:
                    out["measure_repeat_rate"] = (
                        self.measure["finals"] - self.measure["distinct"]
                    ) / self.measure["finals"]
            if self.admission:
                for key in (
                    "outstanding",
                    "reserved",
                    "waiting",
                    "max_waiting",
                    "bootstrap_grants",
                    "paced_grants",
                ):
                    out[f"admission_{key}"] = self.admission[key]
                out["admission_eval_capacity"] = (
                    self.admission["eval_capacity_milli"] / 1_000
                )
                out["admission_episode_work"] = (
                    self.admission["episode_work_milli"] / 1_000
                )
                out["admission_pressure_gain"] = (
                    self.admission["pressure_gain_milli"] / 1_000
                )
                out["admission_gap_ms"] = self.admission["gap_us"] / 1_000
            return out


def parse_stat_fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.strip().split() if "=" in token)


def pump_selfplay_stderr(
    process: subprocess.Popen[bytes],
    stats: SelfplayStatsTracker,
) -> None:
    """Relays stderr and folds selfplay heartbeat counters."""
    assert process.stderr is not None
    for raw in iter(process.stderr.readline, b""):
        sys.stderr.buffer.write(raw)
        sys.stderr.buffer.flush()
        if raw.startswith(b"event=eval_stats "):
            stats.observe_eval(parse_stat_fields(raw.decode("utf-8", "replace")))
        elif raw.startswith(b"event=measure_stats "):
            stats.observe_measure(parse_stat_fields(raw.decode("utf-8", "replace")))
        elif raw.startswith(b"event=admission_stats "):
            stats.observe_admission(parse_stat_fields(raw.decode("utf-8", "replace")))


class MetricsWriter:
    def __init__(self, path: Path, wandb_run: WandbRun | None = None) -> None:
        self.handle = path.open("a", encoding="utf-8")
        self.wandb_run = wandb_run

    def write(self, record: dict[str, object]) -> None:
        self.handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.handle.flush()
        if self.wandb_run is not None:
            self.wandb_run.write(record)

    def finish(self) -> None:
        self.handle.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


class SamplePrefetcher:
    """Keeps one training pair in flight while the GPU trains.

    The policy client remains shared with refresh() under a lock. A distinct
    value client lets the replay service sample and collate both roles in
    parallel without interleaving frames on either socket.
    """

    def __init__(
        self,
        sampler: SampleClient,
        batch: int,
        window_rows: int,
        value_batch: int,
        value_window_rows: int,
        seed: int,
        total_steps: int,
        max_reuse: float,
        reuse_gate_interval: int,
        reuse_gate_episodes: int,
        start_step: int = 0,
        *,
        value_sampler: SampleClient | None = None,
    ) -> None:
        self._sampler = sampler
        self._batch = batch
        self._window_rows = window_rows
        self._value_batch = value_batch
        self._value_window_rows = value_window_rows
        self._seed = seed
        self._total_steps = total_steps
        self._max_reuse = max_reuse
        self._reuse_gate_interval = reuse_gate_interval
        self._reuse_gate_episodes = reuse_gate_episodes
        self._start_step = start_step
        self._value_sampler = value_sampler
        # Depth 2 rides out replay-store read spikes (compaction bursts)
        # without letting sample timing drift more than two steps.
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sample-prefetch", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Unblock a full queue so the thread can observe the stop flag.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

    def join(self, timeout: float = 10.0) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError("sample prefetcher did not stop")

    def next(self) -> object:
        result, error = self._queue.get()
        if error is not None:
            raise error
        return result

    def refresh(self) -> object:
        with self._lock:
            return self._sampler.refresh()

    def _run(self) -> None:
        value_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="value-sample")
            if self._value_sampler is not None
            else None
        )
        try:
            for step in range(self._start_step, self._total_steps):
                if self._stop.is_set():
                    return
                try:
                    needed_rows = max(
                        _required_produced_rows(
                            step,
                            self._batch,
                            self._max_reuse,
                            self._reuse_gate_interval,
                        ),
                        _required_produced_rows(
                            step,
                            self._value_batch,
                            self._max_reuse,
                            self._reuse_gate_interval,
                        ),
                    )
                    needed_episodes = _required_episodes(
                        step,
                        self._reuse_gate_interval,
                        self._reuse_gate_episodes,
                    )
                    while True:
                        with self._lock:
                            ack = self._sampler.refresh()
                        if (
                            ack.produced_rows >= needed_rows
                            and ack.episodes >= needed_episodes
                        ):
                            break
                        if self._stop.wait(0.1):
                            return
                    with self._lock:
                        result = _sample_training_batches(
                            self._sampler,
                            policy_batch=self._batch,
                            policy_window_rows=self._window_rows,
                            value_batch=self._value_batch,
                            value_window_rows=self._value_window_rows,
                            run_seed=self._seed,
                            step=step,
                            produced_rows=ack.produced_rows,
                            value_sampler=self._value_sampler,
                            value_executor=value_executor,
                        )
                except BaseException as error:  # surfaced on next()
                    self._queue.put((None, error))
                    return
                while not self._stop.is_set():
                    try:
                        self._queue.put((result, None), timeout=1.0)
                        break
                    except queue.Full:
                        continue
        finally:
            if value_executor is not None:
                value_executor.shutdown(wait=True, cancel_futures=True)


class PerfWindow:
    """Accumulates per-step timings between metric writes."""

    def __init__(self, produced_rows: int = 0, episodes: int = 0) -> None:
        self.window_started = time.perf_counter()
        self.last_produced = produced_rows
        self.last_episodes = episodes
        self.has_counter_baseline = False
        self.steps = 0
        self.sample_seconds = 0.0
        self.train_seconds = 0.0

    def record(self, sample_started: float, train_started: float, finished: float) -> None:
        self.steps += 1
        self.sample_seconds += train_started - sample_started
        self.train_seconds += finished - train_started

    def drain(self, produced: int, episodes: int) -> dict[str, float]:
        now = time.perf_counter()
        elapsed = max(now - self.window_started, 1e-9)
        steps = max(self.steps, 1)
        perf = {
            "steps_per_s": self.steps / elapsed,
            "rows_per_s": (
                max(produced - self.last_produced, 0) / elapsed
                if self.has_counter_baseline
                else 0.0
            ),
            "episodes_per_s": (
                max(episodes - self.last_episodes, 0) / elapsed
                if self.has_counter_baseline
                else 0.0
            ),
            "sample_ms": 1000.0 * self.sample_seconds / steps,
            "train_ms": 1000.0 * self.train_seconds / steps,
        }
        self.window_started = now
        self.last_produced = produced
        self.last_episodes = episodes
        self.has_counter_baseline = True
        self.steps = 0
        self.sample_seconds = 0.0
        self.train_seconds = 0.0
        return perf


# JSONL keys -> grouped wandb keys. Keeping diagnostics explicit here prevents
# experimental fields from silently flooding the human-facing dashboard.
WANDB_KEYS = {
    "policy_loss": "train/policy_loss",
    "value_loss": "train/value_loss",
    "value_final_loss": "train/value_final_loss",
    "value_v8_loss": "train/value_v8_loss",
    "value_v32_loss": "train/value_v32_loss",
    "terminal_score_loss": "train/terminal_score_loss",
    "terminal_score_mae": "train/terminal_score_mae_nodes",
    "terminal_score_bias": "train/terminal_score_bias_nodes",
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "grad_clip_scale": "train/grad_clip_scale",
    "lr": "train/lr",
    "value_accuracy": "train/value_accuracy",
    "value_mae": "train/value_mae",
    "value_rmse": "train/value_rmse",
    "fraction_valid": "train/fraction_valid",
    "label_mean": "train/label_mean",
    "learner_win_rate": "train/learner_win_rate",
    "aux_signal_v8_final_target_correlation": (
        "auxiliary/signal/v8_final_target_correlation"
    ),
    "aux_signal_v32_final_target_correlation": (
        "auxiliary/signal/v32_final_target_correlation"
    ),
    "aux_signal_v8_v32_target_correlation": (
        "auxiliary/signal/v8_v32_target_correlation"
    ),
    "aux_signal_terminal_score_correlation": (
        "auxiliary/signal/terminal_score_correlation"
    ),
    "aux_signal_early_v8_final_target_correlation": (
        "auxiliary/signal/early_v8_final_target_correlation"
    ),
    "aux_signal_early_v32_final_target_correlation": (
        "auxiliary/signal/early_v32_final_target_correlation"
    ),
    "aux_signal_early_v8_target_std": "auxiliary/signal/early_v8_target_std",
    "aux_signal_early_v32_target_std": "auxiliary/signal/early_v32_target_std",
    "aux_gradient_effective_auxiliary_norm": (
        "auxiliary/readout_gradient/effective_auxiliary_norm"
    ),
    "aux_gradient_auxiliary_to_final_norm_ratio": (
        "auxiliary/readout_gradient/auxiliary_to_final_norm_ratio"
    ),
    "aux_gradient_auxiliary_alignment_ratio": (
        "auxiliary/readout_gradient/auxiliary_alignment_ratio"
    ),
    "aux_gradient_final_auxiliary_cosine": (
        "auxiliary/readout_gradient/final_auxiliary_cosine"
    ),
    "aux_gradient_policy_auxiliary_cosine": (
        "auxiliary/readout_gradient/policy_auxiliary_cosine"
    ),
    "parameter_trunk_gradient_norm": "optimizer/parameter/trunk_gradient_norm",
    "parameter_trunk_update_to_parameter": (
        "optimizer/parameter/trunk_update_to_parameter"
    ),
    "parameter_value_final_update_to_parameter": (
        "optimizer/parameter/value_final_update_to_parameter"
    ),
    "parameter_value_horizons_update_to_parameter": (
        "optimizer/parameter/value_horizons_update_to_parameter"
    ),
    "parameter_terminal_score_update_to_parameter": (
        "optimizer/parameter/terminal_score_update_to_parameter"
    ),
    "learner_win_rate_ema": "selfplay/learner_win_rate_ema",
    "value_sign_accuracy_early_ema": "selfplay/value_sign_accuracy_early_ema",
    "value_sign_accuracy_late_ema": "selfplay/value_sign_accuracy_late_ema",
    "episode_latency_s": "lag/episode_latency_s",
    "eval_mean_batch": "eval/mean_batch",
    "eval_batches_per_s": "eval/batches_per_s",
    "eval_evals_per_s": "eval/evals_per_s",
    "measure_finals": "measure/finals",
    "measure_distinct_finals": "measure/distinct_finals",
    "measure_repeat_rate": "measure/repeat_rate",
    "admission_outstanding": "admission/outstanding_evals",
    "admission_reserved": "admission/reserved_evals",
    "admission_waiting": "admission/waiting_workers",
    "admission_max_waiting": "admission/max_waiting_workers",
    "admission_bootstrap_grants": "admission/bootstrap_grants",
    "admission_paced_grants": "admission/paced_grants",
    "admission_eval_capacity": "admission/eval_capacity",
    "admission_episode_work": "admission/evals_per_episode",
    "admission_pressure_gain": "admission/pressure_gain",
    "admission_gap_ms": "admission/gap_ms",
    "terminal_cost_ema": "selfplay/terminal_cost_ema",
    "terminal_cost_best": "selfplay/terminal_cost_best",
    "stop_rate": "selfplay/stop_rate",
    "episode_len_ema": "selfplay/episode_len_ema",
    "stop_rate_ema": "selfplay/stop_rate_ema",
    "symmetric_games_completed": "symmetric/games_completed",
    "symmetric_p1_win_rate_ema": "symmetric/p1_win_rate_ema",
    "symmetric_p2_win_rate_ema": "symmetric/p2_win_rate_ema",
    "symmetric_draw_rate_ema": "symmetric/draw_rate_ema",
    "symmetric_decisive_rate_ema": "symmetric/decisive_rate_ema",
    "symmetric_seat_advantage_ema": "symmetric/seat_advantage_ema",
    "symmetric_p1_terminal_cost_ema": "symmetric/p1_terminal_cost_ema",
    "symmetric_p2_terminal_cost_ema": "symmetric/p2_terminal_cost_ema",
    "symmetric_mean_terminal_cost_ema": "symmetric/mean_terminal_cost_ema",
    "symmetric_best_of_two_terminal_cost_ema": "symmetric/best_of_two_terminal_cost_ema",
    "symmetric_terminal_cost_margin_ema": "symmetric/terminal_cost_margin_ema",
    "symmetric_terminal_cost_best": "symmetric/terminal_cost_best",
    "symmetric_p1_rewrites_ema": "symmetric/p1_rewrites_ema",
    "symmetric_p2_rewrites_ema": "symmetric/p2_rewrites_ema",
    "symmetric_game_rewrites_ema": "symmetric/game_rewrites_ema",
    "symmetric_rewrite_margin_ema": "symmetric/rewrite_margin_ema",
    "symmetric_value_sign_accuracy_early_ema": "symmetric/value_sign_accuracy_early_ema",
    "symmetric_value_sign_accuracy_late_ema": "symmetric/value_sign_accuracy_late_ema",
    "symmetric_game_latency_s": "symmetric/game_latency_s",
    "reduction_ema": "graph/reduction_ema",
    "reduction_best": "graph/reduction_best",
    "steps_per_s": "perf/steps_per_s",
    "rows_per_s": "perf/rows_per_s",
    "episodes_per_s": "perf/episodes_per_s",
    "sample_ms": "perf/sample_ms",
    "train_ms": "perf/train_ms",
    "produced_rows": "perf/produced_rows",
    "policy_reuse": "perf/policy_reuse",
    "value_reuse": "perf/value_reuse",
}


class WandbRun:
    """Optional wandb mirror of the metrics JSONL. Never load-bearing:
    init failure logs one line and the run proceeds without it."""

    def __init__(self, run: object) -> None:
        self.run = run
        self.publishes = 0

    @classmethod
    def start(
        cls,
        config: RunConfig,
        extra_config: dict[str, object] | None = None,
    ) -> WandbRun | None:
        if not config.wandb.project:
            return None
        try:
            import wandb

            run_config = {
                "trainer": asdict(config.trainer),
                "selfplay": asdict(config.selfplay),
                "arch": asdict(config.arch),
                "run_dir": str(config.paths.run_dir),
            }
            if extra_config:
                run_config.update(extra_config)
            run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity or None,
                name=config.wandb.run_name or config.paths.run_dir.name,
                mode=config.wandb.mode or None,
                id=config.wandb.run_id or None,
                resume="must" if config.wandb.run_id else None,
                # A resumed run keeps its original config; re-sending it
                # would conflict on any knob the resume changed.
                config=None if config.wandb.run_id else run_config,
            )
        except Exception as error:
            print(f"event=wandb_disabled error={error}", file=sys.stderr, flush=True)
            return None
        return cls(run)

    def write(self, record: dict[str, object]) -> None:
        if record.get("event") == "step":
            payload = {
                WANDB_KEYS[key]: value
                for key, value in record.items()
                if key in WANDB_KEYS
            }
            self.run.log(payload, step=record["step"])
        elif record.get("event") == "graph":
            facts = {k: v for k, v in record.items() if k != "event"}
            self.run.config.update({"graph": facts}, allow_val_change=True)
            self.run.log({f"graph/{k}": v for k, v in facts.items()}, step=0)
        elif record.get("event") == "publish":
            self.publishes += 1
            payload = {
                "publish/count": self.publishes,
                "publish/training_step": record["training_step"],
            }
            for key in ("param_norm", "update_norm", "checkpoints_pruned"):
                if key in record:
                    payload[f"publish/{key}"] = record[key]
            self.run.log(payload, step=record["training_step"])
    def finish(self) -> None:
        self.run.finish()


def _validate(config: RunConfig) -> RunConfig:
    if config.trainer.lr_schedule not in ("cosine", "constant"):
        raise ValueError(f"unknown lr_schedule: {config.trainer.lr_schedule}")
    if config.trainer.min_startup_rows < 1:
        raise ValueError("min_startup_rows must be at least 1")
    if config.trainer.publish_interval < 1:
        raise ValueError("publish_interval must be positive")
    if config.trainer.checkpoint_retain < 0:
        raise ValueError("checkpoint_retain must be non-negative")
    if config.trainer.permanent_checkpoint_interval < 0:
        raise ValueError("permanent_checkpoint_interval must be non-negative")
    if config.trainer.publish_lag_blocks not in (0, 1):
        raise ValueError("publish_lag_blocks must be 0 or 1")
    if config.trainer.batch < 1:
        raise ValueError("batch must be positive")
    if config.trainer.value_batch < 0:
        raise ValueError("value_batch must be non-negative")
    if config.trainer.window_rows < 1:
        raise ValueError("window_rows must be positive")
    if config.trainer.value_window_rows < 0:
        raise ValueError("value_window_rows must be non-negative")
    if not math.isfinite(config.trainer.value_trunk_grad_scale) or not (
        0.0 <= config.trainer.value_trunk_grad_scale <= 1.0
    ):
        raise ValueError("value_trunk_grad_scale must be finite and in [0, 1]")
    task_weights = (
        config.trainer.value_final_weight,
        config.trainer.value_v8_weight,
        config.trainer.value_v32_weight,
        config.trainer.terminal_score_weight,
    )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in task_weights):
        raise ValueError("value task weights must be finite and non-negative")
    if not math.isclose(sum(task_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("value task weights must sum to one")
    auxiliary_weight = any(weight > 0.0 for weight in task_weights[1:])
    if auxiliary_weight and config.arch.auxiliary_heads != "v8-v32-score":
        raise ValueError("auxiliary task weights require v8-v32-score model heads")
    if config.trainer.compile_mode not in (
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ):
        raise ValueError(f"unknown compile_mode: {config.trainer.compile_mode}")
    if config.trainer.matmul_precision not in ("highest", "high", "medium"):
        raise ValueError(
            f"unknown matmul_precision: {config.trainer.matmul_precision}"
        )
    if config.trainer.reuse_gate_interval < 1:
        raise ValueError("reuse_gate_interval must be positive")
    if config.trainer.reuse_gate_episodes < 0:
        raise ValueError("reuse_gate_episodes must be non-negative")
    if config.trainer.publish_lag_blocks and (
        config.trainer.reuse_gate_interval != config.trainer.publish_interval
        or (
            config.trainer.max_reuse == 0.0
            and config.trainer.reuse_gate_episodes == 0
        )
    ):
        raise ValueError(
            "publish_lag_blocks requires a publish-aligned reuse gate"
        )
    if not math.isfinite(config.trainer.max_reuse) or config.trainer.max_reuse < 0.0:
        raise ValueError("max_reuse must be finite and non-negative")
    if config.trainer.lr_decay_steps is not None and config.trainer.lr_decay_steps < 1:
        raise ValueError("lr_decay_steps must be positive")
    if not 0.0 <= config.trainer.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if config.trainer.optimizer not in ("adamw", "muon_mixed"):
        raise ValueError(f"unknown optimizer: {config.trainer.optimizer}")
    if config.trainer.adamw_lr is not None and config.trainer.adamw_lr <= 0.0:
        raise ValueError("adamw_lr must be positive")
    if not 0.0 <= config.trainer.momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if config.trainer.ns_steps < 1:
        raise ValueError("ns_steps must be positive")
    if config.trainer.policy_init not in ("default", "neutral"):
        raise ValueError(f"unsupported policy_init: {config.trainer.policy_init}")
    if config.trainer.policy_init == "neutral" and config.arch.policy_head != "pointer":
        raise ValueError("policy_init = 'neutral' requires policy_head = 'pointer'")
    if config.trainer.resume and config.trainer.init_checkpoint:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    if config.trainer.init_checkpoint_scope not in ("all", "policy"):
        raise ValueError("init_checkpoint_scope must be 'all' or 'policy'")
    if not config.trainer.init_checkpoint and config.trainer.init_checkpoint_scope != "all":
        raise ValueError("init_checkpoint_scope requires init_checkpoint")
    for name, seed in (
        ("seed", config.trainer.seed),
        ("model_seed", config.trainer.model_seed),
        ("data_seed", config.trainer.data_seed),
    ):
        if seed is not None and not 0 <= seed < 2**64:
            raise ValueError(f"{name} must fit an unsigned 64-bit integer")
    if (
        not math.isfinite(config.selfplay.c_visit)
        or config.selfplay.c_visit < 0.0
        or not math.isfinite(config.selfplay.c_scale)
        or config.selfplay.c_scale < 0.0
    ):
        raise ValueError("c_visit and c_scale must be finite and non-negative")
    if (
        config.arch.position_encoding == "policy_budget"
        and not config.selfplay.position_features
    ):
        raise ValueError(
            f"position_encoding = '{config.arch.position_encoding}' requires position_features = true"
        )
    for name, value in (
        ("lanes", config.selfplay.lanes),
        ("workers_per_lane", config.selfplay.workers_per_lane),
        ("simulations", config.selfplay.simulations),
        ("max_considered", config.selfplay.max_considered),
        ("max_steps", config.selfplay.max_steps),
        ("max_candidates", config.selfplay.max_candidates),
        ("max_batch", config.selfplay.max_batch),
        ("eval_processes", config.selfplay.eval_processes),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if config.selfplay.eval_processes > config.selfplay.lanes:
        raise ValueError("eval_processes cannot exceed lanes")
    if config.selfplay.admission_stagger_ms < 0:
        raise ValueError("admission_stagger_ms must be non-negative")
    if config.selfplay.admission_smoothing and config.selfplay.admission_stagger_ms:
        raise ValueError(
            "admission_smoothing and admission_stagger_ms are mutually exclusive"
        )
    if config.selfplay.max_row_backlog < 1:
        raise ValueError("max_row_backlog must be positive")
    if config.selfplay.replay_retain < 0:
        raise ValueError("replay_retain must be non-negative")
    if (
        not math.isfinite(config.selfplay.gumbel_scale)
        or config.selfplay.gumbel_scale < 0.0
    ):
        raise ValueError("gumbel_scale must be finite and non-negative")
    if (
        not math.isfinite(config.selfplay.gumbel_noise_overlap)
        or config.selfplay.gumbel_noise_overlap >= 1.0
    ):
        raise ValueError("gumbel_noise_overlap must be < 1")
    if (
        not math.isfinite(config.selfplay.eval_poll_interval)
        or config.selfplay.eval_poll_interval < 0.0
    ):
        raise ValueError("eval_poll_interval must be finite and non-negative")
    if not 0 <= config.selfplay.seed < 2**64:
        raise ValueError("selfplay seed must fit an unsigned 64-bit integer")
    if not config.selfplay.mask_stop and not config.selfplay.position_features:
        raise ValueError("STOP-enabled symmetric selfplay requires position_features = true")
    if not config.selfplay.length_tiebreak:
        raise ValueError("symmetric selfplay requires length_tiebreak = true")
    if config.arch.state_input != "joint-board":
        raise ValueError("symmetric selfplay requires state_input = 'joint-board'")
    if config.arch.value_input != "single":
        raise ValueError("symmetric selfplay requires value_input = 'single'")
    return config


def _symmetric_step_fields(ack: SampleAck, completed_games: int) -> dict[str, float | int]:
    fields: dict[str, float | int] = {
        "symmetric_games_completed": completed_games,
    }
    metrics = ack.symmetric_selfplay
    if metrics is None:
        return fields
    fields.update(
        {
            "symmetric_p1_win_rate_ema": metrics.p1_win_rate_ema,
            "symmetric_p2_win_rate_ema": metrics.p2_win_rate_ema,
            "symmetric_draw_rate_ema": metrics.draw_rate_ema,
            "symmetric_decisive_rate_ema": max(0.0, 1.0 - metrics.draw_rate_ema),
            "symmetric_seat_advantage_ema": metrics.seat_advantage_ema,
            "symmetric_p1_terminal_cost_ema": metrics.p1_terminal_cost_ema,
            "symmetric_p2_terminal_cost_ema": metrics.p2_terminal_cost_ema,
            "symmetric_mean_terminal_cost_ema": metrics.mean_terminal_cost_ema,
            "symmetric_best_of_two_terminal_cost_ema": metrics.mean_terminal_cost_ema
            - 0.5 * metrics.terminal_cost_margin_ema,
            "symmetric_terminal_cost_margin_ema": metrics.terminal_cost_margin_ema,
            "symmetric_terminal_cost_best": metrics.terminal_cost_best,
            "symmetric_p1_rewrites_ema": metrics.p1_episode_len_ema,
            "symmetric_p2_rewrites_ema": metrics.p2_episode_len_ema,
            "symmetric_game_rewrites_ema": metrics.game_len_ema,
            "symmetric_rewrite_margin_ema": metrics.episode_len_margin_ema,
        }
    )
    if ack.value_sign_accuracy_early_ema >= 0.0:
        fields["symmetric_value_sign_accuracy_early_ema"] = (
            ack.value_sign_accuracy_early_ema
        )
    if ack.value_sign_accuracy_late_ema >= 0.0:
        fields["symmetric_value_sign_accuracy_late_ema"] = (
            ack.value_sign_accuracy_late_ema
        )
    if ack.episode_latency_ema >= 0.0:
        fields["symmetric_game_latency_s"] = ack.episode_latency_ema
    return fields


def _prune_training_checkpoints(config: RunConfig) -> tuple[str, ...]:
    return prune_checkpoints(
        config.paths.checkpoint_dir,
        config.trainer.checkpoint_retain,
    )


def _checkpoint_due(config: TrainerConfig, training_step: int) -> bool:
    return training_step > 0 and (
        training_step % config.publish_interval == 0
        or bool(
            config.permanent_checkpoint_interval
            and training_step % config.permanent_checkpoint_interval == 0
        )
    )


def _permanent_checkpoint_pointers(
    config: TrainerConfig,
    training_step: int,
) -> tuple[str, ...]:
    interval = config.permanent_checkpoint_interval
    if training_step <= 0 or interval == 0 or training_step % interval:
        return ()
    return (f"step_{training_step}.json",)


def load_config(path: str | Path) -> RunConfig:
    data = _load_config_table(Path(path))
    trainer = _dataclass_from_dict(
        TrainerConfig,
        _symmetric_trainer_config(data.get("trainer", {})),
    )
    selfplay = _dataclass_from_dict(
        SelfplayConfig,
        _symmetric_selfplay_config(data.get("selfplay", {})),
    )
    wandb = _dataclass_from_dict(WandbConfig, data.get("wandb", {}))
    arch = _dataclass_from_dict(ArchConfig, data.get("arch", {}))
    raw_paths = data.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ValueError("[paths] must be a table")
    run_dir = Path(str(raw_paths.get("run_dir", "runs/train-whittle")))
    replay_dir = Path(str(raw_paths.get("replay_dir", run_dir / "replay")))
    checkpoint_dir = Path(str(raw_paths.get("checkpoint_dir", run_dir / "checkpoints")))
    sample_socket = Path(str(raw_paths.get("sample_socket", run_dir / "sample.sock")))
    graphzero_bin = str(raw_paths.get("graphzero_bin", os.environ.get("GRAPHZERO_BIN", "graphzero")))
    # Children run in their own working directories (the evaluator runs in
    # python_dir), so relative config paths must be pinned to the trainer's
    # cwd before they cross a process boundary.
    run_dir = run_dir.absolute()
    replay_dir = replay_dir.absolute()
    checkpoint_dir = checkpoint_dir.absolute()
    sample_socket = sample_socket.absolute()
    return _validate(RunConfig(
        trainer=trainer,
        selfplay=selfplay,
        paths=PathsConfig(
            replay_dir=replay_dir,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            sample_socket=sample_socket,
            graphzero_bin=graphzero_bin,
        ),
        wandb=wandb,
        arch=arch,
    ))


def _symmetric_selfplay_config(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError("[selfplay] must be a table")
    required_legacy_values = {
        "training_mode": "symmetric-selfplay",
        "reference": "none",
        "root_mode": "generated",
        "reference_ema_decay": 0.0,
        "reference_gamma": 0.0,
        "reference_trajectory_pool": 0,
        "reference_arena_size": 0,
        "reference_arena_interval": 0,
        "reference_max_batch": 0,
        "challenger_max_batch": 0,
        "value_reward": "sign",
    }
    for key, expected in required_legacy_values.items():
        if key in data and data[key] != expected:
            raise ValueError(f"retired selfplay setting {key}={data[key]!r} is unsupported")
    retired = required_legacy_values.keys() | {
        "reference_arena_seed",
        "policy_opponent_mode",
        "reference_mask_stop",
        "value_reward_scale",
    }
    return {key: value for key, value in data.items() if key not in retired}


def _symmetric_trainer_config(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError("[trainer] must be a table")
    if "value_mirror" in data and data["value_mirror"] is not False:
        raise ValueError("retired trainer setting value_mirror=true is unsupported")
    return {key: value for key, value in data.items() if key != "value_mirror"}


def _load_config_table(path: Path) -> dict[str, object]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    extends = data.pop("extends", None)
    if extends is None:
        return data
    if not isinstance(extends, str):
        raise ValueError("extends must be a string")

    base_path = (path.parent / extends).resolve()
    base = tomllib.loads(base_path.read_text(encoding="utf-8"))
    if "extends" in base:
        raise ValueError("config inheritance is limited to one layer")
    return _merge_config_tables(base, data)


def _merge_config_tables(base: dict[str, object], child: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in child.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_config_tables(base_value, value)
        else:
            merged[key] = value
    return merged


def init_replay(config: RunConfig) -> None:
    command = [
        config.paths.graphzero_bin,
        "replay-init",
        "--replay-dir",
        str(config.paths.replay_dir),
        "--max-candidates",
        str(config.selfplay.max_candidates),
    ]
    subprocess.run(command, check=True)


def spawn_replay_serve(config: RunConfig) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            config.paths.graphzero_bin,
            "replay-serve",
            "--replay-dir",
            str(config.paths.replay_dir),
            "--socket",
            str(config.paths.sample_socket),
            "--max-batch",
            str(max(config.trainer.batch, config.trainer.value_batch)),
        ]
    )


def spawn_torch_selfplay(config: RunConfig) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            config.paths.graphzero_bin,
            "selfplay",
            "--replay-dir",
            str(config.paths.replay_dir),
            "--episodes",
            "0",
            "--lanes",
            str(config.selfplay.lanes),
            "--workers-per-lane",
            str(config.selfplay.workers_per_lane),
            "--position-features",
            "true" if config.selfplay.position_features else "false",
            "--no-backtrack",
            "true" if config.selfplay.no_backtrack else "false",
            "--gumbel-noise-overlap",
            str(config.selfplay.gumbel_noise_overlap),
            "--tree-reuse",
            "true" if config.selfplay.tree_reuse else "false",
            "--mask-stop",
            "true" if config.selfplay.mask_stop else "false",
            "--length-tiebreak",
            "true" if config.selfplay.length_tiebreak else "false",
            "--eval-processes",
            str(config.selfplay.eval_processes),
            "--admission-stagger-ms",
            str(config.selfplay.admission_stagger_ms),
            "--admission-smoothing",
            "true" if config.selfplay.admission_smoothing else "false",
            "--wave-batching",
            "true" if config.selfplay.wave_batching else "false",
            "--evaluator",
            "torch",
            "--python-dir",
            config.selfplay.python_dir,
            "--checkpoint-dir",
            str(config.paths.checkpoint_dir),
            "--eval-device",
            config.selfplay.eval_device,
            "--eval-poll-interval",
            str(config.selfplay.eval_poll_interval),
            "--seed",
            str(config.selfplay.seed),
            "--max-steps",
            str(config.selfplay.max_steps),
            "--simulations",
            str(config.selfplay.simulations),
            "--max-considered",
            str(config.selfplay.max_considered),
            "--gumbel-scale",
            str(config.selfplay.gumbel_scale),
            "--c-visit",
            str(config.selfplay.c_visit),
            "--c-scale",
            str(config.selfplay.c_scale),
            "--max-candidates",
            str(config.selfplay.max_candidates),
            "--max-batch",
            str(config.selfplay.max_batch),
            "--serve-socket",
            str(config.paths.sample_socket),
            # Sampled GZFB/GZFT batches are encoded at the serve capacity, and
            # the trainer stages at trainer.batch — they must be one knob.
            "--serve-max-batch",
            str(config.trainer.batch),
            "--replay-backlog",
            str(config.selfplay.max_row_backlog),
            *(
                ["--replay-retain", str(config.selfplay.replay_retain)]
                if config.selfplay.replay_retain
                else []
            ),
        ],
        # Selfplay spawns the evaluator child; a new session lets kill_child
        # take down the whole group instead of orphaning the evaluator (and
        # its GPU memory) when selfplay is SIGKILLed.
        start_new_session=True,
        # The pump thread relays evaluator/selfplay heartbeats unchanged.
        stderr=subprocess.PIPE,
    )


def _seed_model(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def _set_matmul_precision(precision: str) -> None:
    import torch

    torch.set_float32_matmul_precision(precision)


def trainer_loop_config(
    config: TrainerConfig,
    *,
    symmetric_mask_stop: bool = True,
) -> LoopConfig:
    return LoopConfig(
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        lr_decay_steps=config.lr_decay_steps,
        min_lr_ratio=config.min_lr_ratio,
        value_weight=config.value_weight,
        value_trunk_grad_scale=config.value_trunk_grad_scale,
        value_final_weight=config.value_final_weight,
        value_v8_weight=config.value_v8_weight,
        value_v32_weight=config.value_v32_weight,
        terminal_score_weight=config.terminal_score_weight,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        adamw_lr=config.adamw_lr,
        momentum=config.momentum,
        nesterov=config.nesterov,
        ns_steps=config.ns_steps,
        grad_clip=config.grad_clip,
        mask_stop_loss=symmetric_mask_stop,
        compile_model=config.compile_model,
        compile_mode=config.compile_mode,
    )


def _load_initial_checkpoint(
    model: object,
    source: str | Path,
    feature_schema_hash: object,
    arch: ArchConfig,
    *,
    scope: str = "all",
) -> object:
    from gz.checkpoints import DirectorySource
    from gz.checkpoints.weights import load_state_dict

    resolved = DirectorySource(Path(source).absolute()).resolve_latest()
    if resolved.manifest.feature_schema_hash != feature_schema_hash:
        raise RuntimeError("initial checkpoint feature schema does not match the store")
    if resolved.manifest.arch_name != arch.name:
        raise RuntimeError("initial checkpoint arch name does not match [arch] config")
    source_arch = ArchConfig.from_dict(resolved.manifest.arch_config)
    if scope == "all" and source_arch != arch:
        raise RuntimeError("initial checkpoint arch does not match [arch] config")
    if scope == "policy" and _policy_arch_config(source_arch) != _policy_arch_config(arch):
        raise RuntimeError("initial checkpoint policy arch does not match [arch] config")
    state = load_state_dict(resolved.weights_path)
    if scope == "all":
        model.load_state_dict(state)
    elif scope == "policy":
        policy_state = {
            name: tensor for name, tensor in state.items() if not _is_value_parameter(name)
        }
        incompatible = model.load_state_dict(policy_state, strict=False)
        expected_missing = {
            name for name in model.state_dict() if _is_value_parameter(name)
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError("policy checkpoint scope did not isolate the value module")
    else:
        raise ValueError(f"unsupported initial checkpoint scope: {scope}")
    return resolved


def _policy_arch_config(arch: ArchConfig) -> dict[str, object]:
    value_fields = {
        "value_input",
        "value_activation",
        "value_hidden",
        "value_head",
        "value_bins",
        "value_min",
        "value_max",
        "value_sigma_ratio",
        "auxiliary_heads",
    }
    return {
        name: value
        for name, value in arch.to_dict().items()
        if name not in value_fields
    }


def _is_value_parameter(name: str) -> bool:
    return name.startswith(("value.", "horizon_value.", "terminal_score."))


def check_memory(min_available_gb: float) -> None:
    """Aborts the run before memory pressure can freeze a swapless box:
    the kernel thrashes long before the OOM killer fires."""
    if min_available_gb <= 0:
        return
    available = _mem_available_gb()
    if available is not None and available < min_available_gb:
        raise RuntimeError(
            f"aborting: {available:.1f} GiB available < {min_available_gb} GiB floor"
        )


def _mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def check_child(child: subprocess.Popen[bytes], name: str) -> None:
    status = child.poll()
    if status is not None:
        raise RuntimeError(f"{name} exited with status {status}")


def stop_child(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        kill_child(child)


def kill_child(child: subprocess.Popen[bytes]) -> None:
    try:
        # Children spawned with start_new_session lead their own group;
        # kill the group so their own children (the evaluator) die too.
        if os.getpgid(child.pid) == child.pid:
            os.killpg(child.pid, signal.SIGKILL)
        elif child.poll() is None:
            child.send_signal(signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def _dataclass_from_dict(cls: object, data: object) -> object:
    if not isinstance(data, dict):
        raise ValueError("config section must be a table")
    fields = cls.__dataclass_fields__
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"unknown config fields for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)
