from __future__ import annotations

import math
from dataclasses import dataclass

from gz.trainer.data import TrainingBatch
from gz.trainer.diagnostics import (
    AuxiliarySignalMetrics,
    ParameterUpdateMetrics,
    ReadoutGradientMetrics,
    auxiliary_signal_metrics as _auxiliary_signal_metrics,
    diagnostic_logging_fields as _diagnostic_logging_fields,
    finish_parameter_metrics as _finish_parameter_metrics,
    finish_readout_gradient_metrics as _finish_readout_gradient_metrics,
    readout_gradient_tensors as _readout_gradient_tensors,
    snapshot_parameter_metrics as _snapshot_parameter_metrics,
)
from gz.trainer.optim import build_optimizer


@dataclass(frozen=True, slots=True)
class LoopConfig:
    lr: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 1000
    lr_decay_steps: int | None = None
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.0
    value_weight: float = 1.0
    value_trunk_grad_scale: float = 1.0
    value_final_weight: float = 1.0
    value_v8_weight: float = 0.0
    value_v32_weight: float = 0.0
    terminal_score_weight: float = 0.0
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    adamw_lr: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    mask_stop_loss: bool = False
    compile_model: bool = False
    compile_mode: str = "default"


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    policy_loss: float
    value_loss: float
    value_final_loss: float
    value_v8_loss: float
    value_v32_loss: float
    terminal_score_loss: float
    terminal_score_mae: float
    terminal_score_bias: float
    loss: float
    grad_norm: float
    grad_clip_scale: float
    lr: float
    value_accuracy: float
    value_mae: float
    value_rmse: float
    fraction_valid: float
    label_mean: float
    learner_win_rate: float
    auxiliary_signals: AuxiliarySignalMetrics | None
    readout_gradients: ReadoutGradientMetrics | None
    parameter_updates: ParameterUpdateMetrics | None

    def logging_fields(self) -> dict[str, float | int]:
        return _diagnostic_logging_fields(
            self.grad_clip_scale,
            self.auxiliary_signals,
            self.readout_gradients,
            self.parameter_updates,
        )


class TrainerLoop:
    def __init__(self, model: object, config: LoopConfig) -> None:
        torch = _torch()
        self.model = model
        self.config = config
        _validate_task_weights(config)
        self._auxiliary_layout = (
            getattr(getattr(model, "arch", None), "auxiliary_heads", "none")
            == "v8-v32-score"
        )
        self._auxiliary_tasks = any(
            weight > 0.0
            for weight in (
                config.value_v8_weight,
                config.value_v32_weight,
                config.terminal_score_weight,
            )
        )
        if self._auxiliary_tasks and not self._auxiliary_layout:
            raise ValueError("auxiliary task weights require v8-v32-score model heads")
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
        self._training_forward = getattr(model, "training_forward", None)
        self._training_values = getattr(model, "training_values", None)
        if self._auxiliary_tasks and (
            self._training_forward is None or self._training_values is None
        ):
            raise ValueError("model does not expose auxiliary training outputs")
        if config.compile_model:
            compile_options = {
                "fullgraph": True,
                "dynamic": False,
                "mode": config.compile_mode,
            }
            self._model_forward = torch.compile(model, **compile_options)
            # Keep a graph boundary between the trunk and policy head. A single
            # bf16 Inductor graph spanning both produces incorrect backward
            # gradients on the 10M joint-board model.
            def policy_trunk(batch: object) -> object:
                graph, node_roles = model._model_graph(batch)
                return model._encode_graph(graph, node_roles)

            compiled_policy_trunk = torch.compile(policy_trunk, **compile_options)
            compiled_policy_head = torch.compile(model._policy_logits, **compile_options)

            def split_policy_logits(batch: object) -> object:
                h, g_readout, node_mask = compiled_policy_trunk(batch)
                return compiled_policy_head(batch, h, g_readout, node_mask)

            self._policy_logits = split_policy_logits
            self._value_only = torch.compile(model.value_only, **compile_options)
            if self._auxiliary_tasks:
                compiled_auxiliary_heads = torch.compile(
                    model._training_value_outputs,
                    **compile_options,
                )

                def split_training_forward(
                    batch: object,
                    value_trunk_grad_scale: float = 1.0,
                ) -> tuple[object, object, object, object, object]:
                    h, g_readout, node_mask = compiled_policy_trunk(batch)
                    value_outputs = compiled_auxiliary_heads(
                        g_readout,
                        value_trunk_grad_scale,
                    )
                    logits = compiled_policy_head(batch, h, g_readout, node_mask)
                    return (*value_outputs, logits, g_readout)

                def split_training_values(
                    batch: object,
                    value_trunk_grad_scale: float = 1.0,
                ) -> tuple[object, object, object, object]:
                    _, g_readout, _ = compiled_policy_trunk(batch)
                    return (
                        *compiled_auxiliary_heads(
                            g_readout,
                            value_trunk_grad_scale,
                        ),
                        g_readout,
                    )

                self._training_forward = split_training_forward
                self._training_values = split_training_values
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
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metric_zero = batch.value.new_zeros(())
        value_final_loss = metric_zero
        value_v8_loss = metric_zero
        value_v32_loss = metric_zero
        terminal_score_loss = metric_zero
        score_prediction = None
        horizon_prediction = None
        readout_gradient_tensors = None
        separate_value_batch = self.config.value_weight != 0.0 and value_batch is not None
        if separate_value_batch:
            value_batch = _trim_to_row_count(value_batch)
            with torch.autocast(
                device_type=self.device_type,
                dtype=torch.bfloat16,
                enabled=self.device_type == "cuda",
            ):
                logits = self._policy_logits(batch.features)
                policy_loss = policy_ce_loss(
                    logits,
                    batch.policy,
                    batch.features.action_count,
                    batch.row_count,
                    action_kind=getattr(batch.features, "action_kind", None),
                    mask_stop=self.config.mask_stop_loss,
                )
            policy_loss.backward()
            policy_loss = policy_loss.detach()
            del logits

            with torch.autocast(
                device_type=self.device_type,
                dtype=torch.bfloat16,
                enabled=self.device_type == "cuda",
            ):
                if self._auxiliary_tasks:
                    assert self._training_values is not None
                    value_raw, horizon_raw, score_raw, value_readout = self._training_values(
                        value_batch.features,
                        value_trunk_grad_scale=self.config.value_trunk_grad_scale,
                    )
                else:
                    value_raw = self._value_only(
                        value_batch.features,
                        value_trunk_grad_scale=self.config.value_trunk_grad_scale,
                    )
                value = value_batch.value
                if self._auxiliary_tasks:
                    (
                        value_loss,
                        value_final_loss,
                        value_v8_loss,
                        value_v32_loss,
                        terminal_score_loss,
                        score_prediction,
                    ) = auxiliary_task_loss(
                        self.model,
                        value_raw,
                        horizon_raw,
                        score_raw,
                        value_batch,
                        self.config,
                    )
                else:
                    value_loss = value_head_loss(
                        value_raw,
                        value,
                        value_batch.value_valid,
                        value_batch.row_count,
                    )
                    value_final_loss = value_loss
                value_prediction = value_raw
            if self._auxiliary_tasks and with_metrics:
                readout_gradient_tensors = _readout_gradient_tensors(
                    torch,
                    value_readout,
                    final_loss=value_final_loss,
                    v8_loss=value_v8_loss,
                    v32_loss=value_v32_loss,
                    terminal_score_loss=terminal_score_loss,
                    policy_loss=None,
                    config=self.config,
                )
                horizon_prediction = horizon_raw.detach()
            (self.config.value_weight * value_loss).backward()
            value_loss = value_loss.detach()
            value_prediction = value_prediction.detach()
            if score_prediction is not None:
                score_prediction = score_prediction.detach()
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
                        action_kind=getattr(batch.features, "action_kind", None),
                        mask_stop=self.config.mask_stop_loss,
                    )
                    value_loss = logits.sum() * 0.0
                else:
                    value_batch = batch
                    if self._auxiliary_tasks:
                        assert self._training_forward is not None
                        value_raw, horizon_raw, score_raw, logits, value_readout = (
                            self._training_forward(
                                batch.features,
                                value_trunk_grad_scale=(
                                    self.config.value_trunk_grad_scale
                                ),
                            )
                        )
                    else:
                        value_raw, logits = self._model_forward(
                            batch.features,
                            value_trunk_grad_scale=self.config.value_trunk_grad_scale,
                        )
                    policy_loss = policy_ce_loss(
                        logits,
                        batch.policy,
                        batch.features.action_count,
                        batch.row_count,
                        action_kind=getattr(batch.features, "action_kind", None),
                        mask_stop=self.config.mask_stop_loss,
                    )
                    if self._auxiliary_tasks:
                        value = value_batch.value
                        (
                            value_loss,
                            value_final_loss,
                            value_v8_loss,
                            value_v32_loss,
                            terminal_score_loss,
                            score_prediction,
                        ) = auxiliary_task_loss(
                            self.model,
                            value_raw,
                            horizon_raw,
                            score_raw,
                            value_batch,
                            self.config,
                        )
                    else:
                        value = value_batch.value
                        value_loss = value_head_loss(
                            value_raw,
                            value,
                            value_batch.value_valid,
                            value_batch.row_count,
                        )
                        value_final_loss = value_loss
                    value_prediction = value_raw
                loss = policy_loss + self.config.value_weight * value_loss
            if (
                self._auxiliary_tasks
                and self.config.value_weight != 0.0
                and with_metrics
            ):
                readout_gradient_tensors = _readout_gradient_tensors(
                    torch,
                    value_readout,
                    final_loss=value_final_loss,
                    v8_loss=value_v8_loss,
                    v32_loss=value_v32_loss,
                    terminal_score_loss=terminal_score_loss,
                    policy_loss=policy_loss,
                    config=self.config,
                )
                horizon_prediction = horizon_raw.detach()
            loss.backward()
            value_prediction = value_prediction.detach()
            if score_prediction is not None:
                score_prediction = score_prediction.detach()
        if with_metrics and self._auxiliary_layout:
            parameter_snapshot, parameter_totals = _snapshot_parameter_metrics(self.model)
        else:
            parameter_snapshot = None
            parameter_totals = None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip,
            error_if_nonfinite=True,
        )
        grad_clip_scale_tensor = torch.clamp(
            self.config.grad_clip / (grad_norm.float() + 1.0e-6),
            max=1.0,
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

        parameter_updates = (
            _finish_parameter_metrics(
                self.model,
                parameter_snapshot,
                parameter_totals,
            )
            if parameter_snapshot is not None and parameter_totals is not None
            else None
        )
        readout_gradients = (
            _finish_readout_gradient_metrics(readout_gradient_tensors)
            if readout_gradient_tensors is not None
            else None
        )

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
                sign_valid = valid & (value != 0)
                if bool(sign_valid.sum().item()):
                    prediction = torch.where(value_prediction[sign_valid] >= 0, 1.0, -1.0)
                    label = torch.where(value[sign_valid] > 0, 1.0, -1.0)
                    value_accuracy = (prediction == label).float().mean()
                else:
                    value_accuracy = value_prediction.new_tensor(0.0)
                label_mean = value[valid].mean()
                # Fraction of stored learner-perspective labels that beat
                # the reference. This deliberately ignores orientation flips.
                learner_win_rate = (value_batch.value[valid] > 0).float().mean()
                value_error = value_prediction[valid].float() - value[valid].float()
                value_mae = value_error.abs().mean()
                value_rmse = value_error.square().mean().sqrt()
            else:
                value_accuracy = value_prediction.new_tensor(0.0)
                label_mean = value_prediction.new_tensor(0.0)
                learner_win_rate = value_prediction.new_tensor(0.0)
                value_mae = value_prediction.new_tensor(0.0)
                value_rmse = value_prediction.new_tensor(0.0)
            fraction_valid = valid.float().mean()
            if score_prediction is not None:
                score_valid = row_mask & (value_batch.horizon_value_valid > 0)
                terminal_nodes = -value_batch.reward.float()
                score_error = (
                    score_prediction.float() * float(self.model.schema.max_nodes)
                    - terminal_nodes
                )
                score_weight = score_valid.to(score_error.dtype)
                score_count = score_weight.sum().clamp(min=1.0)
                terminal_score_mae = (score_error.abs() * score_weight).sum() / score_count
                terminal_score_bias = (score_error * score_weight).sum() / score_count
            else:
                terminal_score_mae = value_prediction.new_tensor(0.0)
                terminal_score_bias = value_prediction.new_tensor(0.0)
            auxiliary_signals = (
                _auxiliary_signal_metrics(
                    torch,
                    self.model,
                    value_batch,
                    score_prediction,
                )
                if horizon_prediction is not None and score_prediction is not None
                else None
            )
        return StepMetrics(
            step=self.step_index,
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            value_final_loss=float(value_final_loss.detach().cpu()),
            value_v8_loss=float(value_v8_loss.detach().cpu()),
            value_v32_loss=float(value_v32_loss.detach().cpu()),
            terminal_score_loss=float(terminal_score_loss.detach().cpu()),
            terminal_score_mae=float(terminal_score_mae.detach().cpu()),
            terminal_score_bias=float(terminal_score_bias.detach().cpu()),
            loss=float(loss.detach().cpu()),
            grad_norm=float(grad_norm.detach().cpu()),
            grad_clip_scale=float(grad_clip_scale_tensor.detach().cpu()),
            lr=lr,
            value_accuracy=float(value_accuracy.detach().cpu()),
            value_mae=float(value_mae.detach().cpu()),
            value_rmse=float(value_rmse.detach().cpu()),
            fraction_valid=float(fraction_valid.detach().cpu()),
            label_mean=float(label_mean.detach().cpu()),
            learner_win_rate=float(learner_win_rate.detach().cpu()),
            auxiliary_signals=auxiliary_signals,
            readout_gradients=readout_gradients,
            parameter_updates=parameter_updates,
        )


def policy_ce_loss(
    logits: object,
    policy: object,
    action_count: object,
    row_count: int,
    *,
    action_kind: object | None = None,
    mask_stop: bool = False,
) -> object:
    torch = _torch()
    action_index = torch.arange(logits.shape[1], device=logits.device)
    action_mask = action_index.unsqueeze(0) < action_count.unsqueeze(1)
    if mask_stop:
        if action_kind is None:
            raise ValueError("mask_stop requires action_kind")
        action_mask = action_mask & (action_kind != 1)
    row_mask = _row_mask(torch, row_count, logits.shape[0], logits.device)
    masked_logits = logits.masked_fill(~action_mask, -1.0e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    policy_masked = torch.where(action_mask, policy, torch.zeros_like(policy))
    per_row = -(policy_masked * log_probs).sum(dim=1)
    valid = row_mask & (policy_masked.sum(dim=1) > 0)
    weight = valid.to(per_row.dtype)
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def value_mse_loss(value_raw: object, value: object, value_valid: object, row_count: int) -> object:
    # Fully tensorized masked MSE for bounded sign and auxiliary labels.
    torch = _torch()
    row_mask = _row_mask(torch, row_count, value_raw.shape[0], value_raw.device)
    valid = row_mask & (value_valid > 0)
    weight = valid.to(value_raw.dtype)
    target = torch.where(valid, value, torch.zeros_like(value))
    per_row = (value_raw - target) ** 2
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def value_head_loss(
    value_raw: object,
    value: object,
    value_valid: object,
    row_count: int,
) -> object:
    return value_mse_loss(value_raw, value, value_valid, row_count)


def auxiliary_task_loss(
    model: object,
    value_raw: object,
    horizon_raw: object,
    score_raw: object,
    batch: TrainingBatch,
    config: LoopConfig,
) -> tuple[object, object, object, object, object, object]:
    torch = _torch()
    final_loss = value_head_loss(
        value_raw,
        batch.value,
        batch.value_valid,
        batch.row_count,
    )
    v8_loss = value_mse_loss(
        horizon_raw[:, 0],
        batch.horizon_value[:, 0],
        batch.horizon_value_valid,
        batch.row_count,
    )
    v32_loss = value_mse_loss(
        horizon_raw[:, 1],
        batch.horizon_value[:, 1],
        batch.horizon_value_valid,
        batch.row_count,
    )
    score_prediction = torch.sigmoid(score_raw)
    score_target = -batch.reward / float(model.schema.max_nodes)
    score_loss = value_mse_loss(
        score_prediction,
        score_target,
        batch.horizon_value_valid,
        batch.row_count,
    )
    weighted_losses = (
        (config.value_final_weight, final_loss),
        (config.value_v8_weight, v8_loss),
        (config.value_v32_weight, v32_loss),
        (config.terminal_score_weight, score_loss),
    )
    combined = sum(weight * loss for weight, loss in weighted_losses if weight > 0.0)
    return combined, final_loss, v8_loss, v32_loss, score_loss, score_prediction


def _validate_task_weights(config: LoopConfig) -> None:
    weights = (
        config.value_final_weight,
        config.value_v8_weight,
        config.value_v32_weight,
        config.terminal_score_weight,
    )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("value task weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("value task weights must sum to one")


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
        horizon_value=batch.horizon_value[:row_count],
        horizon_value_valid=batch.horizon_value_valid[:row_count],
        reward=batch.reward[:row_count],
        row_count=row_count,
    )


def _torch():
    import torch

    return torch
