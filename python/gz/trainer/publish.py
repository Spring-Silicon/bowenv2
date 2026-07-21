from __future__ import annotations

from pathlib import Path

from gz.checkpoints import CheckpointManifest, publish_checkpoint
from gz.codec import FeatureSchemaConfig
from gz.common import EngineIdentity, FeatureSchemaHash
from gz.model.exphormer import ArchConfig


class EmaWeights:
    def __init__(self, model: object, decay: float) -> None:
        if decay < 0.0 or decay >= 1.0:
            raise ValueError("ema decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def update(self, model: object) -> None:
        import torch

        # Fused multi-tensor update: the per-tensor loop launched two tiny
        # kernels per parameter every step and was the trainer's single
        # largest line (~19% of step wall). _foreach_ batches the same
        # mul/add arithmetic into a handful of launches, bit-identically.
        float_shadows = []
        float_lives = []
        for name, tensor in model.state_dict().items():
            live = tensor.detach()
            shadow = self.shadow[name]
            if live.is_floating_point():
                float_shadows.append(shadow)
                float_lives.append(live)
            else:
                shadow.copy_(live)
        if float_shadows:
            torch._foreach_mul_(float_shadows, self.decay)
            torch._foreach_add_(float_shadows, float_lives, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def assert_finite(self) -> None:
        import torch

        for name, tensor in self.shadow.items():
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"refusing to publish non-finite EMA tensor: {name}")

    def norms(self, previous: dict[str, object] | None) -> tuple[float, float]:
        """(L2 norm of the EMA weights, L2 norm of the delta vs `previous`).
        The update norm is 0.0 when there is no previous snapshot."""
        import torch

        with torch.no_grad():
            param_sq = 0.0
            delta_sq = 0.0
            for name, tensor in self.shadow.items():
                if not tensor.is_floating_point():
                    continue
                param_sq += float(tensor.float().pow(2).sum())
                if previous is not None:
                    delta_sq += float((tensor.float() - previous[name].float()).pow(2).sum())
            return param_sq**0.5, delta_sq**0.5


def publish_ema(
    checkpoint_dir: str | Path,
    ema: EmaWeights,
    *,
    schema: FeatureSchemaConfig,
    schema_hash: FeatureSchemaHash,
    arch: ArchConfig,
    training_step: int,
    run_id: str,
    engine_identity: EngineIdentity,
    checkpoint_pointers: tuple[str, ...] = (),
) -> CheckpointManifest:
    ema.assert_finite()
    return publish_checkpoint(
        checkpoint_dir,
        ema.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_manifest_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=schema_hash,
        engine_identity=engine_identity,
        training_step=training_step,
        run_id=run_id,
        checkpoint_pointers=checkpoint_pointers,
    )
