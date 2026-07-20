from __future__ import annotations

import math
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from gz.trainer.data import TrainingBatch
from gz.trainer.loop import (
    LoopConfig,
    TrainerLoop,
    auxiliary_task_loss,
    hl_gauss_target_probs,
    policy_ce_loss,
    value_bce_loss,
    value_hl_gauss_loss,
    value_head_loss,
    value_mse_loss,
)

torch = pytest.importorskip("torch")


def test_policy_ce_loss_matches_literal_and_ignores_padded_slots() -> None:
    logits = torch.tensor([[math.log(2.0), math.log(1.0), 100.0]], dtype=torch.float32)
    policy = torch.tensor([[0.25, 0.75, 1000.0]], dtype=torch.float32)
    action_count = torch.tensor([2], dtype=torch.int64)

    loss = policy_ce_loss(logits, policy, action_count, row_count=1)

    assert float(loss) == pytest.approx(0.25 * -math.log(2.0 / 3.0) + 0.75 * -math.log(1.0 / 3.0))

    changed = policy.clone()
    changed[0, 2] = -5000.0
    assert float(policy_ce_loss(logits, changed, action_count, row_count=1)) == pytest.approx(float(loss))


def test_policy_ce_loss_can_exclude_reserved_stop_slot() -> None:
    logits = torch.tensor(
        [[math.log(3.0), math.log(1.0), 100.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    policy = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    action_count = torch.tensor([3], dtype=torch.int64)
    action_kind = torch.tensor([[2, 3, 1]], dtype=torch.int64)

    loss = policy_ce_loss(
        logits,
        policy,
        action_count,
        row_count=1,
        action_kind=action_kind,
        mask_stop=True,
    )

    expected = 0.5 * -math.log(3.0 / 4.0) + 0.5 * -math.log(1.0 / 4.0)
    assert float(loss) == pytest.approx(expected)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 2]) == 0.0


def test_value_bce_loss_matches_literal_with_tie() -> None:
    value_raw = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    value = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    valid = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    loss = value_bce_loss(value_raw, value, valid, row_count=3)

    expected = (math.log(2.0) + math.log1p(math.exp(-2.0)) + math.log1p(math.exp(-2.0))) / 3.0
    assert float(loss) == pytest.approx(expected)


def test_value_bce_loss_zero_valid_has_finite_gradient() -> None:
    value_raw = torch.tensor([1.0, -1.0], dtype=torch.float32, requires_grad=True)
    value = torch.tensor([1.0, -1.0], dtype=torch.float32)
    valid = torch.tensor([0.0, 0.0], dtype=torch.float32)

    loss = value_bce_loss(value_raw, value, valid, row_count=2)
    loss.backward()

    assert float(loss) == 0.0
    assert value_raw.grad is not None
    assert value_raw.grad.tolist() == [0.0, 0.0]


def test_value_mse_loss_matches_literal_and_masks_invalid() -> None:
    value_raw = torch.tensor([0.5, -0.25, 0.9], dtype=torch.float32)
    value = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)
    valid = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

    loss = value_mse_loss(value_raw, value, valid, row_count=3)

    assert float(loss) == pytest.approx((0.5**2 + 0.75**2) / 2.0)


def test_value_mse_loss_zero_valid_has_finite_gradient() -> None:
    value_raw = torch.tensor([1.0, -1.0], dtype=torch.float32, requires_grad=True)
    value = torch.tensor([1.0, -1.0], dtype=torch.float32)
    valid = torch.tensor([0.0, 0.0], dtype=torch.float32)

    loss = value_mse_loss(value_raw, value, valid, row_count=2)
    loss.backward()

    assert float(loss) == 0.0
    assert value_raw.grad.tolist() == [0.0, 0.0]


def test_auxiliary_task_loss_uses_normalized_weights_and_terminal_node_target() -> None:
    model = SimpleNamespace(
        arch=SimpleNamespace(value_head="scalar", value_activation="tanh"),
        schema=SimpleNamespace(max_nodes=100),
    )
    batch = TrainingBatch(
        features=SimpleNamespace(),
        policy=torch.zeros((2, 1)),
        value=torch.tensor([1.0, -1.0]),
        value_valid=torch.ones(2),
        horizon_value=torch.tensor([[0.5, -0.5], [1.0, -1.0]]),
        horizon_value_valid=torch.ones(2),
        reward=torch.tensor([-73.0, -20.0]),
        row_count=2,
    )
    score_probability = torch.tensor([0.73, 0.20])
    score_raw = torch.logit(score_probability)
    config = LoopConfig(
        value_final_weight=0.5,
        value_v8_weight=0.2,
        value_v32_weight=0.2,
        terminal_score_weight=0.1,
    )

    combined, final, v8, v32, score, score_prediction = auxiliary_task_loss(
        model,
        torch.zeros(2),
        torch.zeros((2, 2)),
        score_raw,
        batch,
        config,
    )

    assert float(final) == pytest.approx(1.0)
    assert float(v8) == pytest.approx(0.625)
    assert float(v32) == pytest.approx(0.625)
    assert float(score) == pytest.approx(0.0, abs=1e-12)
    assert float(combined) == pytest.approx(0.75)
    torch.testing.assert_close(score_prediction * 100.0, torch.tensor([73.0, 20.0]))


@pytest.mark.parametrize(
    "config",
    [
        LoopConfig(value_final_weight=0.5),
        LoopConfig(value_final_weight=1.1, value_v8_weight=-0.1),
        LoopConfig(value_final_weight=float("nan")),
    ],
)
def test_trainer_rejects_invalid_value_task_weights(config: LoopConfig) -> None:
    with pytest.raises(ValueError, match="value task weights"):
        TrainerLoop(object(), config)


def test_terminal_reward_objective_uses_raw_scalar_mse() -> None:
    model = SimpleNamespace(
        arch=SimpleNamespace(value_head="scalar", value_activation="logit")
    )
    value_raw = torch.tensor([-80.0, -60.0, 100.0], requires_grad=True)
    value = torch.tensor([-82.0, -55.0, -100.0])
    valid = torch.tensor([1.0, 1.0, 0.0])

    loss = value_head_loss(
        model,
        value_raw,
        value,
        valid,
        row_count=3,
        objective="terminal_reward",
    )
    loss.backward()

    assert float(loss) == pytest.approx((2.0**2 + 5.0**2) / 2.0)
    assert value_raw.grad.tolist() == pytest.approx([2.0, -5.0, 0.0])


def test_hl_gauss_targets_are_normalized_symmetric_histograms() -> None:
    bins = 101
    edges = torch.linspace(-1.0, 1.0, bins + 1)
    sigma = 0.75 * 2.0 / bins

    probabilities = hl_gauss_target_probs(
        torch.tensor([-0.5, 0.0, 0.5]),
        edges,
        sigma,
    )

    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(probabilities[0], probabilities[2].flip(0), rtol=1e-5, atol=1e-6)
    assert int(probabilities[1].argmax()) == bins // 2


def test_hl_gauss_loss_matches_uniform_cross_entropy() -> None:
    bins = 101
    logits = torch.zeros((3, bins), requires_grad=True)
    values = torch.tensor([-0.75, 0.0, 0.75])
    valid = torch.ones(3)

    loss = value_hl_gauss_loss(
        logits,
        values,
        valid,
        row_count=3,
        bin_edges=torch.linspace(-1.0, 1.0, bins + 1),
        sigma=0.75 * 2.0 / bins,
    )
    loss.backward()

    assert float(loss) == pytest.approx(math.log(bins), rel=1e-6)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_hl_gauss_loss_zero_valid_has_finite_zero_gradient() -> None:
    bins = 101
    logits = torch.randn((2, bins), requires_grad=True)
    values = torch.tensor([float("nan"), float("nan")])

    loss = value_hl_gauss_loss(
        logits,
        values,
        torch.zeros(2),
        row_count=2,
        bin_edges=torch.linspace(-1.0, 1.0, bins + 1),
        sigma=0.75 * 2.0 / bins,
    )
    loss.backward()

    assert float(loss) == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_trainer_loop_dispatches_graded_targets_to_hl_gauss_head() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value_logits = torch.nn.Parameter(torch.zeros(101))
            self.policy_weight = torch.nn.Parameter(torch.tensor(0.0))
            self.arch = SimpleNamespace(
                value_input="single",
                value_head="hl_gauss",
                value_bins=101,
                value_min=-1.0,
                value_max=1.0,
                value_sigma_ratio=0.75,
            )
            edges = torch.linspace(-1.0, 1.0, 102)
            self.register_buffer("value_bin_edges", edges)
            self.register_buffer("value_bin_centers", (edges[:-1] + edges[1:]) * 0.5)

        def forward(
            self,
            features,
            value_flip=None,
            value_mirror=False,
            value_trunk_grad_scale=1.0,
        ):
            del value_flip, value_mirror, value_trunk_grad_scale
            return (
                self.value_logits.expand(features.action_count.shape[0], -1),
                self.policy_logits(features),
            )

        def policy_logits(self, features):
            return self.policy_weight.expand(features.action_count.shape[0], 2)

        def value_only(
            self,
            features,
            value_flip=None,
            value_trunk_grad_scale=1.0,
        ):
            del value_flip, value_trunk_grad_scale
            return self.value_logits.expand(features.action_count.shape[0], -1)

        def decode_value(self, logits):
            return (torch.softmax(logits.float(), dim=-1) * self.value_bin_centers).sum(dim=-1)

    class Features(NamedTuple):
        action_count: object

    batch = TrainingBatch(
        features=Features(action_count=torch.full((3,), 2, dtype=torch.int64)),
        policy=torch.full((3, 2), 0.5),
        value=torch.tensor([-0.5, 0.0, 0.5]),
        value_valid=torch.ones(3),
        horizon_value=torch.zeros((3, 2)),
        horizon_value_valid=torch.zeros(3),
        reward=torch.zeros(3),
        row_count=3,
    )
    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(lr=1e-3, warmup_steps=0, total_steps=1),
    )

    metrics = loop.train_step(batch)

    assert metrics is not None
    assert metrics.value_loss == pytest.approx(math.log(101), rel=1e-6)
    assert model.value_logits.grad is not None
    assert torch.isfinite(model.value_logits.grad).all()


def test_independent_value_batch_uses_split_model_paths() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.arch = SimpleNamespace(value_input="pair", value_activation="tanh")
            self.policy_calls = 0
            self.value_calls = 0
            self.value_flip = None
            self.value_mirror = None
            self.value_trunk_grad_scale = None
            self.policy_grad = None

        def forward(self, *args, **kwargs):
            raise AssertionError("combined forward must not serve an independent value batch")

        def policy_logits(self, features):
            self.policy_calls += 1
            own = self.weight.expand(features.action_count.shape[0])
            return torch.stack((own, -own), dim=1)

        def value_only(
            self,
            features,
            value_flip=None,
            value_mirror=False,
            value_trunk_grad_scale=1.0,
        ):
            self.value_calls += 1
            assert self.weight.grad is not None
            self.policy_grad = self.weight.grad.detach().clone()
            self.value_flip = value_flip
            self.value_mirror = value_mirror
            self.value_trunk_grad_scale = value_trunk_grad_scale
            canonical = torch.tanh(self.weight).expand(features.action_count.shape[0])
            if value_mirror:
                return torch.stack((canonical, -canonical))
            return canonical

    class Features(NamedTuple):
        action_count: object
        opponent_state_present: object

    def batch(capacity: int, row_count: int | None = None) -> TrainingBatch:
        features = Features(
            action_count=torch.full((capacity,), 2, dtype=torch.int64),
            opponent_state_present=torch.ones(capacity),
        )
        return TrainingBatch(
            features=features,
            policy=torch.full((capacity, 2), 0.5),
            value=-torch.ones(capacity),
            value_valid=torch.ones(capacity),
            horizon_value=torch.zeros((capacity, 2)),
            horizon_value_valid=torch.zeros(capacity),
            reward=-torch.full((capacity,), 104.0),
            row_count=capacity if row_count is None else row_count,
        )

    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(
            lr=1e-3,
            warmup_steps=0,
            total_steps=1,
            optimizer="adamw",
            value_mirror=True,
            value_trunk_grad_scale=0.1,
        ),
    )

    metrics = loop.train_step(batch(2), batch(2, row_count=1))

    assert metrics is not None
    assert (model.policy_calls, model.value_calls) == (1, 1)
    assert model.policy_grad is not None and model.policy_grad.abs().item() > 0.0
    assert model.value_flip is None
    assert model.value_mirror is True
    assert model.value_trunk_grad_scale == 0.1


def test_trainer_uses_auxiliary_training_path_and_reports_each_task(monkeypatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = torch.nn.Parameter(torch.tensor(0.1))
            self.trunk = torch.nn.Parameter(torch.tensor(0.25))
            self.value = torch.nn.Parameter(torch.tensor(0.25))
            self.horizon_value = torch.nn.Parameter(torch.tensor([0.25, 0.25]))
            self.terminal_score = torch.nn.Parameter(torch.tensor(0.25))
            self.arch = SimpleNamespace(
                value_input="single",
                value_head="scalar",
                value_activation="tanh",
                auxiliary_heads="v8-v32-score",
            )
            self.schema = SimpleNamespace(max_nodes=100)
            self.training_calls = 0
            self.trunk_scale = None

        def forward(self, *args, **kwargs):
            raise AssertionError("main-only forward must not serve auxiliary training")

        def policy_logits(self, *args, **kwargs):
            raise AssertionError("split policy path is not used for a shared batch")

        def value_only(self, *args, **kwargs):
            raise AssertionError("main-only value path must not serve auxiliary training")

        def training_forward(self, features, value_trunk_grad_scale=1.0):
            self.training_calls += 1
            self.trunk_scale = value_trunk_grad_scale
            rows = features.action_count.shape[0]
            centered_trunk = self.trunk - 0.25
            detached = centered_trunk.detach()
            readout = detached + (
                centered_trunk - detached
            ) * value_trunk_grad_scale
            value = torch.tanh(self.value - 0.25 + readout).expand(rows)
            horizon = torch.tanh(self.horizon_value - 0.25 + readout).expand(rows, 2)
            score = (self.terminal_score - 0.25 + readout).expand(rows)
            policy = torch.stack(
                (
                    (self.policy + centered_trunk).expand(rows),
                    -self.policy.expand(rows),
                ),
                dim=1,
            )
            return value, horizon, score, policy, self.trunk

        def training_values(self, features, value_trunk_grad_scale=1.0):
            raise AssertionError("value-only auxiliary path is not used for a shared batch")

    class Features(NamedTuple):
        action_count: object
        position: object

    batch = TrainingBatch(
        features=Features(
            action_count=torch.full((2,), 2, dtype=torch.int64),
            position=torch.tensor(
                [[0.0, 0.0, 1.0, 0.01], [20.0, 0.0, 0.8, 0.01]]
            ),
        ),
        policy=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        value=torch.tensor([1.0, -1.0]),
        value_valid=torch.ones(2),
        horizon_value=torch.tensor([[0.5, -0.5], [1.0, -1.0]]),
        horizon_value_valid=torch.ones(2),
        reward=torch.tensor([-50.0, -50.0]),
        row_count=2,
    )
    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(
            lr=1e-3,
            lr_schedule="constant",
            warmup_steps=0,
            total_steps=1,
            value_trunk_grad_scale=0.1,
            value_final_weight=0.5,
            value_v8_weight=0.2,
            value_v32_weight=0.2,
            terminal_score_weight=0.1,
        ),
    )
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    metrics = loop.train_step(batch)

    assert metrics is not None
    assert model.training_calls == 1
    assert model.trunk_scale == 0.1
    assert metrics.value_final_loss == pytest.approx(1.0)
    assert metrics.value_v8_loss == pytest.approx(0.625)
    assert metrics.value_v32_loss == pytest.approx(0.625)
    assert metrics.terminal_score_loss == pytest.approx(0.0, abs=1e-12)
    assert metrics.value_loss == pytest.approx(0.75)
    assert metrics.terminal_score_mae == pytest.approx(0.0, abs=1e-5)
    assert metrics.terminal_score_bias == pytest.approx(0.0, abs=1e-5)
    assert metrics.auxiliary_signals is not None
    assert metrics.auxiliary_signals.v8_final_target_correlation == pytest.approx(-1.0)
    assert metrics.auxiliary_signals.v32_final_target_correlation == pytest.approx(1.0)
    assert metrics.auxiliary_signals.early_v8_target_std == pytest.approx(0.25)
    assert metrics.readout_gradients is not None
    assert metrics.readout_gradients.auxiliary_alignment_ratio == pytest.approx(0.0)
    assert metrics.readout_gradients.policy_auxiliary_cosine == pytest.approx(0.0)
    assert metrics.parameter_updates is not None

    def relative_update(parameter_name: str) -> float:
        before = parameters_before[parameter_name].float()
        after = dict(model.named_parameters())[parameter_name].detach().float()
        return float(
            torch.linalg.vector_norm(after - before)
            / torch.linalg.vector_norm(before)
        )

    assert metrics.parameter_updates.trunk_update_to_parameter == pytest.approx(
        relative_update("trunk")
    )
    assert metrics.parameter_updates.value_final_update_to_parameter == pytest.approx(
        relative_update("value")
    )
    assert metrics.parameter_updates.value_horizons_update_to_parameter == pytest.approx(
        relative_update("horizon_value")
    )
    assert metrics.parameter_updates.terminal_score_update_to_parameter == pytest.approx(
        relative_update("terminal_score")
    )
    fields = metrics.logging_fields()
    assert len(fields) == 19
    assert fields["aux_signal_early_v8_target_std"] == pytest.approx(0.25)
    assert fields["aux_gradient_auxiliary_alignment_ratio"] == pytest.approx(0.0)
    assert fields["parameter_trunk_update_to_parameter"] == pytest.approx(
        relative_update("trunk")
    )

    def forbidden_diagnostic(*args, **kwargs):
        raise AssertionError("detailed diagnostics must stay off the non-logging path")

    monkeypatch.setattr("gz.trainer.loop._readout_gradient_tensors", forbidden_diagnostic)
    monkeypatch.setattr("gz.trainer.loop._snapshot_parameter_metrics", forbidden_diagnostic)
    assert loop.train_step(batch, with_metrics=False) is None


def test_compile_model_wraps_static_fullgraph_training_paths(monkeypatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.policy_stages = []

        def forward(
            self,
            features,
            value_flip=None,
            value_mirror=False,
            value_trunk_grad_scale=1.0,
        ):
            del value_flip, value_mirror, value_trunk_grad_scale
            value = self.weight.expand(features.action_count.shape[0])
            return value, self.policy_logits(features)

        def policy_logits(self, features):
            return self.weight.expand(features.action_count.shape[0], 2)

        def _model_graph(self, features):
            self.policy_stages.append("graph")
            return features, "roles"

        def _encode_graph(self, graph, node_roles):
            assert node_roles == "roles"
            self.policy_stages.append("trunk")
            batch = graph.action_count.shape[0]
            return self.weight.expand(batch, 1, 1), self.weight.expand(batch, 1), None

        def _policy_logits(self, features, h, g_readout, node_mask):
            del h, g_readout
            assert node_mask is None
            self.policy_stages.append("head")
            return self.weight.expand(features.action_count.shape[0], 2)

        def value_only(
            self,
            features,
            value_flip=None,
            value_trunk_grad_scale=1.0,
        ):
            del value_flip, value_trunk_grad_scale
            return self.weight.expand(features.action_count.shape[0])

    calls = []

    def compile_path(path, **options):
        calls.append((path, options))
        return path

    monkeypatch.setattr(torch, "compile", compile_path)
    model = Model()

    loop = TrainerLoop(
        model,
        LoopConfig(compile_model=True, compile_mode="reduce-overhead"),
    )

    assert len(calls) == 4
    assert calls[0][0] is model
    assert calls[2][0] == model._policy_logits
    assert calls[3][0] == model.value_only
    assert all(
        call[1]
        == {
            "fullgraph": True,
            "dynamic": False,
            "mode": "reduce-overhead",
        }
        for call in calls
    )
    features = SimpleNamespace(action_count=torch.ones(2, dtype=torch.int64))
    torch.testing.assert_close(loop._policy_logits(features), model.policy_logits(features))
    assert model.policy_stages == ["graph", "trunk", "head"]


def test_compile_model_preserves_auxiliary_readout_for_diagnostics(monkeypatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.arch = SimpleNamespace(
                auxiliary_heads="v8-v32-score",
                value_input="single",
            )

        def forward(self, features, **kwargs):
            del kwargs
            readout = self.weight.expand(features.action_count.shape[0], 1)
            return readout[:, 0], self._policy_logits(features, None, readout, None)

        def policy_logits(self, features):
            readout = self.weight.expand(features.action_count.shape[0], 1)
            return self._policy_logits(features, None, readout, None)

        def value_only(self, features, **kwargs):
            del kwargs
            return self.weight.expand(features.action_count.shape[0])

        def training_forward(self, *args, **kwargs):
            raise AssertionError("compiled split path must replace training_forward")

        def training_values(self, *args, **kwargs):
            raise AssertionError("compiled split path must replace training_values")

        def _model_graph(self, features):
            return features, None

        def _encode_graph(self, graph, node_roles):
            assert node_roles is None
            rows = graph.action_count.shape[0]
            readout = self.weight.expand(rows, 1)
            return readout.unsqueeze(1), readout, None

        def _policy_logits(self, features, h, readout, node_mask):
            del h, node_mask
            return readout.expand(features.action_count.shape[0], 2)

        def _training_value_outputs(self, readout, value_trunk_grad_scale):
            del value_trunk_grad_scale
            return (
                readout[:, 0],
                readout.expand(readout.shape[0], 2),
                readout[:, 0],
            )

    calls = []

    def compile_path(path, **options):
        calls.append((path, options))
        return path

    monkeypatch.setattr(torch, "compile", compile_path)
    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(
            compile_model=True,
            value_final_weight=0.5,
            value_v8_weight=0.2,
            value_v32_weight=0.2,
            terminal_score_weight=0.1,
        ),
    )
    features = SimpleNamespace(action_count=torch.ones(2, dtype=torch.int64))

    assert len(calls) == 5
    assert loop._training_forward is not None
    value, horizon, score, logits, readout = loop._training_forward(features)
    assert (value.shape, horizon.shape, score.shape, logits.shape, readout.shape) == (
        (2,),
        (2, 2),
        (2,),
        (2, 2),
        (2, 1),
    )
    assert loop._training_values is not None
    value, horizon, score, readout = loop._training_values(features)
    assert (value.shape, horizon.shape, score.shape, readout.shape) == (
        (2,),
        (2, 2),
        (2,),
        (2, 1),
    )


def test_zero_value_weight_uses_policy_only_model_path() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_weight = torch.nn.Parameter(torch.tensor(0.1))
            self.value_weight = torch.nn.Parameter(torch.tensor(0.0))
            self.arch = SimpleNamespace(
                auxiliary_heads="v8-v32-score",
                value_input="single",
            )

        def forward(self, *args, **kwargs):
            raise AssertionError("combined forward must not run for policy-only training")

        def value_only(self, *args, **kwargs):
            raise AssertionError("value head must not run for policy-only training")

        def training_forward(self, *args, **kwargs):
            raise AssertionError("auxiliary heads must not run for policy-only training")

        def training_values(self, *args, **kwargs):
            raise AssertionError("auxiliary heads must not run for policy-only training")

        def policy_logits(self, features):
            return torch.stack(
                (
                    self.policy_weight.expand(features.action_count.shape[0]),
                    -self.policy_weight.expand(features.action_count.shape[0]),
                ),
                dim=1,
            )

    class Features(NamedTuple):
        action_count: object

    batch = TrainingBatch(
        features=Features(action_count=torch.full((2,), 2, dtype=torch.int64)),
        policy=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        value=torch.zeros(2),
        value_valid=torch.zeros(2),
        horizon_value=torch.zeros((2, 2)),
        horizon_value_valid=torch.zeros(2),
        reward=-torch.full((2,), 100.0),
        row_count=2,
    )
    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(
            lr=1e-3,
            warmup_steps=0,
            total_steps=1,
            value_weight=0.0,
            value_final_weight=0.5,
            value_v8_weight=0.2,
            value_v32_weight=0.2,
            terminal_score_weight=0.1,
        ),
    )

    metrics = loop.train_step(batch)

    assert metrics is not None
    assert metrics.value_loss == 0.0
    assert model.policy_weight.grad is not None
    assert model.policy_weight.grad.item() != 0.0
    assert model.value_weight.grad is None
    assert model.value_weight.item() == 0.0
    assert metrics.auxiliary_signals is None
    assert metrics.readout_gradients is None
    assert metrics.parameter_updates is not None


def test_nonfinite_gradient_aborts_before_optimizer_step() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def policy_logits(self, features):
            logits = self.weight * torch.tensor(float("nan"))
            return logits.expand(features.action_count.shape[0], 2)

        def value_only(self, *args, **kwargs):
            raise AssertionError("value head must not run for policy-only training")

    class Features(NamedTuple):
        action_count: object

    batch = TrainingBatch(
        features=Features(action_count=torch.full((2,), 2, dtype=torch.int64)),
        policy=torch.full((2, 2), 0.5),
        value=torch.zeros(2),
        value_valid=torch.zeros(2),
        horizon_value=torch.zeros((2, 2)),
        horizon_value_valid=torch.zeros(2),
        reward=torch.zeros(2),
        row_count=2,
    )
    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(
            lr=1e-3,
            warmup_steps=0,
            total_steps=1,
            value_weight=0.0,
        ),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        loop.train_step(batch)

    assert model.weight.item() == pytest.approx(0.1)


def test_constant_schedule_holds_base_lr_after_warmup() -> None:
    from gz.trainer.loop import lr_at_step

    assert lr_at_step(3e-4, 5, 10, 1000, "constant") == pytest.approx(1.5e-4)  # warmup ramp
    assert lr_at_step(3e-4, 500, 10, 1000, "constant") == pytest.approx(3e-4)
    assert lr_at_step(3e-4, 999999, 10, 1000, "constant") == pytest.approx(3e-4)
    # cosine still anneals
    assert lr_at_step(3e-4, 1000, 10, 1000, "cosine") < 1e-8
    assert lr_at_step(3e-4, 1000, 10, 1000, "cosine", 0.1) == pytest.approx(3e-5)
