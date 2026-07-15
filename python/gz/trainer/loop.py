from __future__ import annotations

import math
from dataclasses import dataclass

from gz.trainer.data import TrainingBatch
from gz.trainer.optim import build_optimizer
from gz.trainer.sampler import step_seed


@dataclass(frozen=True, slots=True)
class LoopConfig:
    lr: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 1000
    lr_decay_steps: int | None = None
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.0
    value_weight: float = 1.0
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    adamw_lr: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    run_seed: int = 0
    # Train both orientations of pair-value data. A separate value batch
    # samples one orientation per row, matching whittlezero's mirrored replay;
    # the legacy shared batch evaluates both orientations of every row.
    value_mirror: bool = False
    compile_model: bool = False
    compile_mode: str = "default"


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    policy_loss: float
    value_loss: float
    loss: float
    grad_norm: float
    lr: float
    value_accuracy: float
    fraction_valid: float
    label_mean: float
    learner_win_rate: float


class TrainerLoop:
    def __init__(self, model: object, config: LoopConfig) -> None:
        torch = _torch()
        self.model = model
        self.config = config
        self.optimizer = build_optimizer(
            model,
            name=config.optimizer,
            lr=config.lr,
            adamw_lr=config.adamw_lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_steps=config.ns_steps,
        )
        self._model_forward = model
        self._policy_logits = model.policy_logits
        self._value_only = model.value_only
        if config.compile_model:
            compile_options = {
                "fullgraph": True,
                "dynamic": False,
                "mode": config.compile_mode,
            }
            self._model_forward = torch.compile(model, **compile_options)
            self._policy_logits = torch.compile(model.policy_logits, **compile_options)
            self._value_only = torch.compile(model.value_only, **compile_options)
        self.step_index = 0
        # bf16 autocast on CUDA, matching the evaluator's serving numerics.
        # Params and optimizer state stay f32; no GradScaler is needed for
        # bf16 (full f32 exponent range).
        self.device_type = next(model.parameters()).device.type

    def train_step(
        self,
        batch: TrainingBatch,
        value_batch: TrainingBatch | None = None,
        with_metrics: bool = True,
    ) -> StepMetrics | None:
        """One optimizer step. With `with_metrics=False` the step enqueues no
        host-device synchronization at all (no `.item()`/`.cpu()`), so
        back-to-back steps pipeline on the GPU; callers request metrics only
        on the steps they log."""
        torch = _torch()
        functional = torch.nn.functional
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        separate_value_batch = self.config.value_weight != 0.0 and value_batch is not None
        if separate_value_batch:
            value_batch = _trim_to_row_count(value_batch)
            pair_mode = getattr(getattr(self.model, "arch", None), "value_input", None) == "pair"
            value_flip = None
            if pair_mode and self.config.value_mirror:
                value_flip = pair_value_flip(
                    torch,
                    value_batch,
                    self.config.run_seed,
                    self.step_index,
                )
            with torch.autocast(
                device_type=self.device_type,
                dtype=torch.bfloat16,
                enabled=self.device_type == "cuda",
            ):
                logits = self._policy_logits(batch.features)
                policy_loss = policy_ce_loss(
                    logits, batch.policy, batch.features.action_count, batch.row_count
                )
            policy_loss.backward()
            policy_loss = policy_loss.detach()
            del logits

            with torch.autocast(
                device_type=self.device_type,
                dtype=torch.bfloat16,
                enabled=self.device_type == "cuda",
            ):
                value_raw = self._value_only(value_batch.features, value_flip=value_flip)
                value = flipped_value_targets(torch, value_batch.value, value_flip)
                value_loss = value_head_loss(
                    self.model,
                    value_raw,
                    value,
                    value_batch.value_valid,
                    value_batch.row_count,
                )
                value_prediction = decode_value_output(self.model, value_raw)
            (self.config.value_weight * value_loss).backward()
            value_loss = value_loss.detach()
            value_prediction = value_prediction.detach()
            del value_raw
            loss = policy_loss + self.config.value_weight * value_loss
        else:
            with torch.autocast(
                device_type=self.device_type,
                dtype=torch.bfloat16,
                enabled=self.device_type == "cuda",
            ):
                if self.config.value_weight == 0.0:
                    # Policy distillation should not encode the opponent graph or
                    # execute a value head whose loss is identically zero.
                    value_batch = batch
                    logits = self._policy_logits(batch.features)
                    value_raw = batch.value.new_zeros(batch.value.shape)
                    value = batch.value
                    value_prediction = value_raw
                    policy_loss = policy_ce_loss(
                        logits,
                        batch.policy,
                        batch.features.action_count,
                        batch.row_count,
                    )
                    value_loss = logits.sum() * 0.0
                else:
                    pair_mode = (
                        getattr(getattr(self.model, "arch", None), "value_input", None)
                        == "pair"
                    )
                    value_batch = batch
                    mirror = self.config.value_mirror and pair_mode
                    value_flip = None
                    if pair_mode and not mirror:
                        value_flip = pair_value_flip(
                            torch,
                            value_batch,
                            self.config.run_seed,
                            self.step_index,
                        )
                    value_raw, logits = self._model_forward(
                        batch.features, value_flip=value_flip, value_mirror=mirror
                    )
                    policy_loss = policy_ce_loss(
                        logits, batch.policy, batch.features.action_count, batch.row_count
                    )
                    if mirror:
                        # whittlezero's mirrored stream: every pair trains both
                        # orientations (targets z and -z); the swapped example is
                        # masked to rows that actually carry an opponent state.
                        canonical, mirrored = value_raw[0], value_raw[1]
                        value_loss = value_head_loss(
                            self.model,
                            canonical,
                            value_batch.value,
                            value_batch.value_valid,
                            value_batch.row_count,
                        )
                        present = getattr(value_batch.features, "opponent_state_present", None)
                        if present is not None:
                            mirrored_valid = value_batch.value_valid * (present > 0).to(
                                value_batch.value_valid.dtype
                            )
                            value_loss = 0.5 * value_loss + 0.5 * value_head_loss(
                                self.model,
                                mirrored,
                                -value_batch.value,
                                mirrored_valid,
                                value_batch.row_count,
                            )
                        value_raw = canonical
                        value = value_batch.value
                    else:
                        value = flipped_value_targets(torch, value_batch.value, value_flip)
                        value_loss = value_head_loss(
                            self.model,
                            value_raw,
                            value,
                            value_batch.value_valid,
                            value_batch.row_count,
                        )
                    value_prediction = decode_value_output(self.model, value_raw)
                loss = policy_loss + self.config.value_weight * value_loss
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip,
            error_if_nonfinite=True,
        )
        lr = lr_at_step(
            self.config.lr,
            self.step_index + 1,
            self.config.warmup_steps,
            self.config.lr_decay_steps or self.config.total_steps,
            self.config.lr_schedule,
            self.config.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        self.optimizer.step()
        self.step_index += 1

        if not with_metrics:
            return None

        with torch.no_grad():
            row_mask = _row_mask(
                torch,
                value_batch.row_count,
                value_prediction.shape[0],
                value_prediction.device,
            )
            valid = row_mask & (value_batch.value_valid > 0)
            valid_count = valid.sum()
            if bool(valid_count.item()):
                prediction = torch.where(value_prediction[valid] >= 0, 1.0, -1.0)
                label = torch.where(value[valid] >= 0, 1.0, -1.0)
                value_accuracy = (prediction == label).float().mean()
                label_mean = value[valid].mean()
                # Fraction of stored learner-perspective labels that beat
                # the reference. This deliberately ignores orientation flips.
                learner_win_rate = (value_batch.value[valid] > 0).float().mean()
            else:
                value_accuracy = value_prediction.new_tensor(0.0)
                label_mean = value_prediction.new_tensor(0.0)
                learner_win_rate = value_prediction.new_tensor(0.0)
            fraction_valid = valid.float().mean()
        return StepMetrics(
            step=self.step_index,
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            loss=float(loss.detach().cpu()),
            grad_norm=float(grad_norm.detach().cpu()),
            lr=lr,
            value_accuracy=float(value_accuracy.detach().cpu()),
            fraction_valid=float(fraction_valid.detach().cpu()),
            label_mean=float(label_mean.detach().cpu()),
            learner_win_rate=float(learner_win_rate.detach().cpu()),
        )


def policy_ce_loss(logits: object, policy: object, action_count: object, row_count: int) -> object:
    torch = _torch()
    action_index = torch.arange(logits.shape[1], device=logits.device)
    action_mask = action_index.unsqueeze(0) < action_count.unsqueeze(1)
    row_mask = _row_mask(torch, row_count, logits.shape[0], logits.device)
    masked_logits = logits.masked_fill(~action_mask, -1.0e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    policy_masked = torch.where(action_mask, policy, torch.zeros_like(policy))
    per_row = -(policy_masked * log_probs).sum(dim=1)
    valid = row_mask & (policy_masked.sum(dim=1) > 0)
    weight = valid.to(per_row.dtype)
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def value_bce_loss(value_raw: object, value: object, value_valid: object, row_count: int) -> object:
    torch = _torch()
    functional = torch.nn.functional
    row_mask = _row_mask(torch, row_count, value_raw.shape[0], value_raw.device)
    valid = row_mask & (value_valid > 0)
    weight = valid.to(value_raw.dtype)
    # Fully tensorized: a data-dependent host branch here would synchronize
    # the CUDA stream between the forward pass and backward, stalling the
    # GPU mid-step. Invalid rows may carry arbitrary label bytes, so their
    # targets are zeroed before the pointwise loss and their terms weighted
    # out; zero valid rows yields loss 0 with finite gradients.
    target = torch.where(valid, (value + 1.0) * 0.5, torch.zeros_like(value))
    per_row = functional.binary_cross_entropy_with_logits(
        2.0 * value_raw, target, reduction="none"
    )
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def value_mse_loss(value_raw: object, value: object, value_valid: object, row_count: int) -> object:
    # whittlezero's value loss: MSE against the +/-1 target on the
    # tanh-bounded head. Masking mirrors value_bce_loss -- fully
    # tensorized, invalid rows zeroed and weighted out.
    torch = _torch()
    row_mask = _row_mask(torch, row_count, value_raw.shape[0], value_raw.device)
    valid = row_mask & (value_valid > 0)
    weight = valid.to(value_raw.dtype)
    target = torch.where(valid, value, torch.zeros_like(value))
    per_row = (value_raw - target) ** 2
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def hl_gauss_target_probs(
    target: object,
    bin_edges: object,
    sigma: float,
) -> object:
    torch = _torch()
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("HL-Gauss sigma must be finite and positive")
    edges = bin_edges.to(device=target.device, dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    target = torch.minimum(torch.maximum(target, edges[0]), edges[-1])
    denominator = math.sqrt(2.0) * sigma
    cdf = torch.special.erf((edges - target.unsqueeze(-1)) / denominator)
    normalizer = cdf[..., -1] - cdf[..., 0]
    probabilities = cdf[..., 1:] - cdf[..., :-1]
    return probabilities / normalizer.clamp_min(1.0e-12).unsqueeze(-1)


def value_hl_gauss_loss(
    value_logits: object,
    value: object,
    value_valid: object,
    row_count: int,
    bin_edges: object,
    sigma: float,
) -> object:
    torch = _torch()
    row_mask = _row_mask(torch, row_count, value_logits.shape[0], value_logits.device)
    valid = row_mask & (value_valid > 0)
    target = torch.where(valid, value, torch.zeros_like(value))
    target_probs = hl_gauss_target_probs(target, bin_edges, sigma)
    per_row = -(target_probs * torch.log_softmax(value_logits.float(), dim=-1)).sum(dim=-1)
    weight = valid.to(per_row.dtype)
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def value_head_loss(
    model: object,
    value_raw: object,
    value: object,
    value_valid: object,
    row_count: int,
) -> object:
    arch = getattr(model, "arch", None)
    if getattr(arch, "value_head", "scalar") == "hl_gauss":
        sigma = arch.value_sigma_ratio * (arch.value_max - arch.value_min) / arch.value_bins
        return value_hl_gauss_loss(
            value_raw,
            value,
            value_valid,
            row_count,
            model.value_bin_edges,
            sigma,
        )
    if getattr(arch, "value_activation", "logit") == "tanh":
        return value_mse_loss(value_raw, value, value_valid, row_count)
    return value_bce_loss(value_raw, value, value_valid, row_count)


def decode_value_output(model: object, value_raw: object) -> object:
    if getattr(getattr(model, "arch", None), "value_head", "scalar") == "hl_gauss":
        return model.decode_value(value_raw)
    return value_raw


def pair_value_flip(torch: object, batch: TrainingBatch, run_seed: int, step: int) -> object:
    if getattr(batch.features, "opponent_state_present", None) is None:
        return None
    device = batch.value.device
    generator = torch.Generator(device=device)
    generator.manual_seed(step_seed(run_seed, step, "value-orientation"))
    row_mask = _row_mask(torch, batch.row_count, batch.value.shape[0], device)
    present = batch.features.opponent_state_present > 0
    return (torch.rand(batch.value.shape, generator=generator, device=device) < 0.5) & row_mask & present


def flipped_value_targets(torch: object, value: object, value_flip: object) -> object:
    if value_flip is None:
        return value
    return torch.where(value_flip, -value, value)


def lr_at_step(
    base_lr: float,
    step: int,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
    min_lr_ratio: float = 0.0,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    if total_steps <= warmup_steps:
        return base_lr
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _row_mask(torch: object, row_count: int, capacity: int, device: object) -> object:
    return torch.arange(capacity, device=device) < row_count


def _trim_to_row_count(batch: TrainingBatch) -> TrainingBatch:
    row_count = batch.row_count
    if batch.value.shape[0] == row_count:
        return batch
    features = type(batch.features)(*(tensor[:row_count] for tensor in batch.features))
    return TrainingBatch(
        features=features,
        policy=batch.policy[:row_count],
        value=batch.value[:row_count],
        value_valid=batch.value_valid[:row_count],
        reward=batch.reward[:row_count],
        row_count=row_count,
    )


def _torch():
    import torch

    return torch
