from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EpisodeLengthStage:
    start_step: int
    max_steps: int


@dataclass(frozen=True, slots=True)
class EpisodeLengthSchedule:
    mode: str
    stages: tuple[EpisodeLengthStage, ...]

    def stage_index(self, training_step: int) -> int:
        if training_step < 0:
            raise ValueError("training_step must be non-negative")
        index = 0
        for candidate, stage in enumerate(self.stages[1:], start=1):
            if training_step < stage.start_step:
                break
            index = candidate
        return index


@dataclass(frozen=True, slots=True)
class EpisodeLengthProgress:
    stage_index: int = 0
    completed_rows: int = 0
    completed_policy_rows: int = 0
    completed_value_rows: int = 0
    completed_episodes: int = 0


class ReplayCounters(Protocol):
    produced_rows: int
    produced_policy_rows: int
    produced_value_rows: int
    episodes: int


def parse_episode_length_schedule(
    raw: object,
    *,
    final_max_steps: int,
    total_steps: int,
) -> EpisodeLengthSchedule:
    if raw is None:
        schedule = EpisodeLengthSchedule(
            mode="fixed",
            stages=(EpisodeLengthStage(start_step=0, max_steps=final_max_steps),),
        )
        validate_episode_length_schedule(
            schedule,
            final_max_steps=final_max_steps,
            total_steps=total_steps,
        )
        return schedule
    if not isinstance(raw, dict):
        raise ValueError("[episode_length_schedule] must be a table")
    mode = raw.get("mode", "explicit")
    if not isinstance(mode, str):
        raise ValueError("episode_length_schedule.mode must be a string")

    if mode == "explicit":
        unknown = set(raw) - {"mode", "stages"}
        if unknown:
            raise ValueError(f"unknown episode_length_schedule fields: {sorted(unknown)}")
        raw_stages = raw.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("explicit episode length schedule requires [[...stages]]")
        stages = []
        for index, raw_stage in enumerate(raw_stages):
            if not isinstance(raw_stage, dict):
                raise ValueError(f"episode length stage {index} must be a table")
            unknown = set(raw_stage) - {"start_step", "max_steps"}
            if unknown:
                raise ValueError(
                    f"unknown episode length stage fields at index {index}: {sorted(unknown)}"
                )
            stages.append(
                EpisodeLengthStage(
                    start_step=_schedule_int(raw_stage, "start_step"),
                    max_steps=_schedule_int(raw_stage, "max_steps"),
                )
            )
    elif mode in ("linear", "exponential"):
        allowed = {"mode", "start", "interval_steps", "maximum"}
        allowed.add("increment" if mode == "linear" else "factor")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown episode_length_schedule fields: {sorted(unknown)}")
        initial = _schedule_int(raw, "start")
        interval = _schedule_int(raw, "interval_steps")
        maximum = _schedule_int(raw, "maximum", default=final_max_steps)
        if mode == "linear":
            increment = _schedule_int(raw, "increment")
            if increment <= 0:
                raise ValueError("linear episode length increment must be positive")

            def next_length(current: int) -> int:
                return min(current + increment, maximum)

        else:
            factor = raw.get("factor")
            if (
                isinstance(factor, bool)
                or not isinstance(factor, (int, float))
                or not math.isfinite(float(factor))
                or float(factor) <= 1.0
            ):
                raise ValueError("exponential episode length factor must be finite and > 1")

            def next_length(current: int) -> int:
                return min(max(current + 1, math.ceil(current * float(factor))), maximum)

        if interval <= 0:
            raise ValueError("episode length interval_steps must be positive")
        if initial <= 0 or maximum <= 0 or initial > maximum:
            raise ValueError("episode length start and maximum must satisfy 0 < start <= maximum")
        stages = [EpisodeLengthStage(start_step=0, max_steps=initial)]
        while stages[-1].max_steps < maximum:
            next_start = stages[-1].start_step + interval
            if next_start >= total_steps:
                raise ValueError(
                    "episode length schedule does not reach maximum before total_steps"
                )
            stages.append(
                EpisodeLengthStage(
                    start_step=next_start,
                    max_steps=next_length(stages[-1].max_steps),
                )
            )
    else:
        raise ValueError(f"unknown episode length schedule mode: {mode}")

    schedule = EpisodeLengthSchedule(mode=mode, stages=tuple(stages))
    validate_episode_length_schedule(
        schedule,
        final_max_steps=final_max_steps,
        total_steps=total_steps,
    )
    return schedule


def validate_episode_length_schedule(
    schedule: EpisodeLengthSchedule,
    *,
    final_max_steps: int,
    total_steps: int,
) -> None:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if final_max_steps < 1:
        raise ValueError("selfplay.max_steps must be positive")
    if not schedule.stages or schedule.stages[0].start_step != 0:
        raise ValueError("episode length schedule must start at training step 0")
    previous = schedule.stages[0]
    if previous.max_steps < 1:
        raise ValueError("episode length max_steps must be positive")
    for stage in schedule.stages[1:]:
        if stage.start_step <= previous.start_step:
            raise ValueError("episode length stage start_step values must increase")
        if stage.start_step >= total_steps:
            raise ValueError("episode length stage must start before total_steps")
        if stage.max_steps <= previous.max_steps:
            raise ValueError("episode length stage max_steps values must increase")
        previous = stage
    if schedule.stages[-1].max_steps != final_max_steps:
        raise ValueError(
            "final episode length stage max_steps must equal selfplay.max_steps"
        )


def episode_length_state_path(run_dir: Path) -> Path:
    return run_dir / "episode-length-schedule.json"


def write_episode_length_progress(
    run_dir: Path,
    schedule: EpisodeLengthSchedule,
    progress: EpisodeLengthProgress,
) -> None:
    path = episode_length_state_path(run_dir)
    payload = {
        "version": 1,
        "stages": [asdict(stage) for stage in schedule.stages],
        **asdict(progress),
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_episode_length_progress(
    run_dir: Path,
    schedule: EpisodeLengthSchedule,
) -> EpisodeLengthProgress:
    path = episode_length_state_path(run_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            "scheduled resume is missing episode-length-schedule.json"
        ) from error
    expected_stages = [asdict(stage) for stage in schedule.stages]
    if payload.get("version") != 1 or payload.get("stages") != expected_stages:
        raise RuntimeError("episode length schedule state does not match the config")
    try:
        progress = EpisodeLengthProgress(
            stage_index=int(payload["stage_index"]),
            completed_rows=int(payload["completed_rows"]),
            completed_policy_rows=int(payload["completed_policy_rows"]),
            completed_value_rows=int(payload["completed_value_rows"]),
            completed_episodes=int(payload["completed_episodes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("invalid episode length schedule state") from error
    if (
        not 0 <= progress.stage_index < len(schedule.stages)
        or progress.completed_rows < 0
        or progress.completed_policy_rows < 0
        or progress.completed_value_rows < 0
        or progress.completed_episodes < 0
    ):
        raise RuntimeError("invalid episode length schedule progress counters")
    return progress


def advance_episode_length_progress(
    progress: EpisodeLengthProgress,
    counters: ReplayCounters,
) -> EpisodeLengthProgress:
    return EpisodeLengthProgress(
        stage_index=progress.stage_index + 1,
        completed_rows=progress.completed_rows + counters.produced_rows,
        completed_policy_rows=(
            progress.completed_policy_rows + counters.produced_policy_rows
        ),
        completed_value_rows=progress.completed_value_rows + counters.produced_value_rows,
        completed_episodes=progress.completed_episodes + counters.episodes,
    )


def _schedule_int(
    table: dict[str, object], key: str, *, default: int | None = None
) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"episode length schedule {key} must be an integer")
    return value
