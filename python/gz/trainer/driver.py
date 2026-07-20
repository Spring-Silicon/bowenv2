from __future__ import annotations

import math
import os
import queue
import signal
import subprocess
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from gz.checkpoints import DirectorySource, ResolvedCheckpoint
from gz.checkpoints.publish import prune_checkpoints
from gz.common import FeatureSchemaHash
from gz.model.exphormer import ArchConfig, build_model, initialize_policy
from gz.trainer.config import (
    RunConfig,
    TrainerConfig,
    load_config,
    resolved_trainer_seeds,
)
from gz.trainer.data import TrainingStager
from gz.trainer.loop import LoopConfig, TrainerLoop
from gz.trainer.publish import EmaWeights, publish_ema
from gz.trainer.sampler import SampleAck, SampleClient, step_seed
from gz.trainer.telemetry import (
    MetricsWriter,
    PerfWindow,
    SelfplayStatsTracker,
    WandbRun,
    pump_selfplay_stderr,
    symmetric_step_fields,
)


def _sample_window_rows(window_rows: int, produced_rows: int) -> int:
    return max(1, min(int(window_rows), int(produced_rows)))


def _cumulative_reuse(step: int, batch: int, produced_rows: int) -> float:
    if batch <= 0 or produced_rows <= 0:
        return 0.0
    return (step + 1) * batch / produced_rows


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
    model_seed, data_seed = resolved_trainer_seeds(config.trainer)
    for path in (config.paths.replay_dir, config.paths.checkpoint_dir, config.paths.run_dir):
        path.mkdir(parents=True, exist_ok=True)
    resume_resolved = None
    learner_manifest = None
    resume_start = 0
    if config.trainer.resume:
        resume_resolved = DirectorySource(str(config.paths.checkpoint_dir)).resolve_latest()
        learner_manifest = resume_resolved.manifest
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
            learner_manifest = first
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

    assert learner_manifest is not None
    actor_checkpoint = _resolve_actor_checkpoint(
        config,
        learner_manifest.feature_schema_hash,
    )
    metrics.write(
        {
            "event": "actor_checkpoint",
            "checkpoint_dir": str(config.paths.actor_checkpoint_dir),
            "checkpoint_pointer": config.selfplay.actor_checkpoint_pointer,
            "model_version": actor_checkpoint.manifest.model_version.hex(),
            "training_step": actor_checkpoint.manifest.training_step,
            "arch_config_hash": actor_checkpoint.manifest.arch_config_hash,
            "frozen": config.selfplay.eval_poll_interval == 0.0,
        }
    )

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
                "soft_policy_weight": config.trainer.soft_policy_weight,
                "soft_policy_temperature": config.trainer.soft_policy_temperature,
                "soft_policy_trunk_grad_scale": (
                    config.trainer.soft_policy_trunk_grad_scale
                ),
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
                    "soft_policy_loss": metrics_record.soft_policy_loss,
                    "soft_policy_kl": metrics_record.soft_policy_kl,
                    "soft_policy_target_entropy": (
                        metrics_record.soft_policy_target_entropy
                    ),
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
                record.update(symmetric_step_fields(ack, total_episodes))
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
            "--eval-processes",
            str(config.selfplay.eval_processes),
            "--admission-stagger-ms",
            str(config.selfplay.admission_stagger_ms),
            "--admission-smoothing",
            "true" if config.selfplay.admission_smoothing else "false",
            "--evaluator",
            "torch",
            "--python-dir",
            config.selfplay.python_dir,
            "--checkpoint-dir",
            str(config.paths.actor_checkpoint_dir),
            "--checkpoint-pointer",
            config.selfplay.actor_checkpoint_pointer,
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


def _resolve_actor_checkpoint(
    config: RunConfig,
    expected_schema_hash: FeatureSchemaHash,
) -> ResolvedCheckpoint:
    resolved = DirectorySource(
        config.paths.actor_checkpoint_dir,
        pointer=config.selfplay.actor_checkpoint_pointer,
    ).resolve_latest()
    if resolved.manifest.feature_schema_hash != expected_schema_hash:
        raise RuntimeError("actor checkpoint feature schema does not match the learner")
    return resolved


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
        soft_policy_weight=config.soft_policy_weight,
        soft_policy_temperature=config.soft_policy_temperature,
        soft_policy_trunk_grad_scale=config.soft_policy_trunk_grad_scale,
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
            name: tensor
            for name, tensor in state.items()
            if not _is_auxiliary_parameter(name)
        }
        incompatible = model.load_state_dict(policy_state, strict=False)
        expected_missing = {
            name for name in model.state_dict() if _is_auxiliary_parameter(name)
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError("policy checkpoint scope did not isolate auxiliary modules")
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


def _is_auxiliary_parameter(name: str) -> bool:
    return name.startswith(
        (
            "value.",
            "horizon_value.",
            "terminal_score.",
            "soft_policy.",
            "soft_policy_kind_embedding.",
            "policy.soft_pointer_key.",
        )
    )


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
