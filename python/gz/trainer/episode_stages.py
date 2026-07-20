from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

from gz.trainer.episode_schedule import (
    EpisodeLengthProgress,
    ReplayCounters,
    advance_episode_length_progress,
    episode_length_state_path,
    load_episode_length_progress,
    write_episode_length_progress,
)
from gz.trainer.sampler import step_seed

if TYPE_CHECKING:
    from pathlib import Path

    from gz.trainer.driver import RunConfig


@dataclass(slots=True)
class EpisodeStagePlan:
    config: RunConfig
    stage_index: int
    progress: EpisodeLengthProgress

    @classmethod
    def load(
        cls,
        config: RunConfig,
        resume_start: int,
        read_replay_counters: Callable[[RunConfig], ReplayCounters],
    ) -> EpisodeStagePlan:
        stage_index = config.episode_length_schedule.stage_index(resume_start)
        progress = EpisodeLengthProgress()
        plan = cls(config=config, stage_index=stage_index, progress=progress)
        if not plan.scheduled:
            return plan
        if config.trainer.resume:
            progress = load_progress(config)
            if progress.stage_index > stage_index:
                raise RuntimeError(
                    "episode length schedule state is ahead of the resume checkpoint"
                )
            while progress.stage_index < stage_index:
                completed = episode_length_stage_config(config, progress.stage_index)
                progress = advance_episode_length_progress(
                    progress,
                    read_replay_counters(completed),
                )
                write_progress(config, progress)
            plan.progress = progress
        elif state_path(config).exists():
            raise RuntimeError(
                "episode length schedule state already exists; set trainer.resume = true"
            )
        return plan

    @property
    def scheduled(self) -> bool:
        return len(self.config.episode_length_schedule.stages) > 1

    @property
    def active_config(self) -> RunConfig:
        return episode_length_stage_config(self.config, self.stage_index)

    def write_progress(self) -> None:
        if self.scheduled:
            write_progress(self.config, self.progress)

    def advance(self, next_stage_index: int, counters: ReplayCounters) -> None:
        if next_stage_index != self.stage_index + 1:
            raise RuntimeError("episode length schedule skipped a stage")
        progress = advance_episode_length_progress(self.progress, counters)
        if progress.stage_index != next_stage_index:
            raise RuntimeError("episode length progress advanced to the wrong stage")
        self.stage_index = next_stage_index
        self.progress = progress
        write_progress(self.config, progress)


def episode_length_stage_config(config: RunConfig, stage_index: int) -> RunConfig:
    stage = config.episode_length_schedule.stages[stage_index]
    if len(config.episode_length_schedule.stages) == 1:
        return config
    stage_seed = (
        config.selfplay.seed
        if stage_index == 0
        else step_seed(
            config.selfplay.seed,
            stage.start_step,
            "episode-length-selfplay",
        )
    )
    return replace(
        config,
        selfplay=replace(
            config.selfplay,
            max_steps=stage.max_steps,
            seed=stage_seed,
        ),
        paths=replace(
            config.paths,
            replay_dir=(
                config.paths.replay_dir
                / f"stage-{stage_index:02d}-max-{stage.max_steps:04d}"
            ),
        ),
    )


def state_path(config: RunConfig) -> Path:
    return episode_length_state_path(config.paths.run_dir)


def write_progress(config: RunConfig, progress: EpisodeLengthProgress) -> None:
    write_episode_length_progress(
        config.paths.run_dir,
        config.episode_length_schedule,
        progress,
    )


def load_progress(config: RunConfig) -> EpisodeLengthProgress:
    return load_episode_length_progress(
        config.paths.run_dir,
        config.episode_length_schedule,
    )
