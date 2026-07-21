from __future__ import annotations

from pathlib import Path

from gz.checkpoints import DirectorySource, ResolvedCheckpoint
from gz.checkpoints.publish import prune_checkpoints
from gz.common import EngineIdentity, FeatureSchemaHash
from gz.model.exphormer import ArchConfig
from gz.trainer.config import RunConfig, TrainerConfig


def validate_checkpoint_engine_identity(
    manifest: object,
    expected: EngineIdentity,
) -> None:
    if manifest.engine_identity != expected:
        raise RuntimeError("checkpoint engine identity does not match the replay store")


def prune_training_checkpoints(config: RunConfig) -> tuple[str, ...]:
    return prune_checkpoints(
        config.paths.checkpoint_dir,
        config.trainer.checkpoint_retain,
    )


def checkpoint_due(config: TrainerConfig, training_step: int) -> bool:
    return training_step > 0 and (
        training_step % config.publish_interval == 0
        or bool(
            config.permanent_checkpoint_interval
            and training_step % config.permanent_checkpoint_interval == 0
        )
    )


def permanent_checkpoint_pointers(
    config: TrainerConfig,
    training_step: int,
) -> tuple[str, ...]:
    interval = config.permanent_checkpoint_interval
    if training_step <= 0 or interval == 0 or training_step % interval:
        return ()
    return (f"step_{training_step}.json",)


def resolve_actor_checkpoint(
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


def load_initial_checkpoint(
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
    source_arch = ArchConfig.from_manifest_dict(resolved.manifest.arch_config)
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
        for name, value in arch.to_manifest_dict().items()
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
