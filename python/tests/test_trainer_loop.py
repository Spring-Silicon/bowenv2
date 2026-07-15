from __future__ import annotations

import math
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from gz.trainer.data import TrainingBatch
from gz.trainer.loop import (
    LoopConfig,
    TrainerLoop,
    hl_gauss_target_probs,
    policy_ce_loss,
    value_bce_loss,
    value_hl_gauss_loss,
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

        def forward(self, features, value_flip=None, value_mirror=False):
            del value_flip, value_mirror
            return (
                self.value_logits.expand(features.action_count.shape[0], -1),
                self.policy_logits(features),
            )

        def policy_logits(self, features):
            return self.policy_weight.expand(features.action_count.shape[0], 2)

        def value_only(self, features, value_flip=None):
            del value_flip
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
            self.policy_grad = None

        def forward(self, *args, **kwargs):
            raise AssertionError("combined forward must not serve an independent value batch")

        def policy_logits(self, features):
            self.policy_calls += 1
            own = self.weight.expand(features.action_count.shape[0])
            return torch.stack((own, -own), dim=1)

        def value_only(self, features, value_flip=None):
            self.value_calls += 1
            assert self.weight.grad is not None
            self.policy_grad = self.weight.grad.detach().clone()
            self.value_flip = value_flip
            return torch.tanh(self.weight).expand(features.action_count.shape[0])

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
        ),
    )

    metrics = loop.train_step(batch(2), batch(2, row_count=1))

    assert metrics is not None
    assert (model.policy_calls, model.value_calls) == (1, 1)
    assert model.policy_grad is not None and model.policy_grad.abs().item() > 0.0
    assert model.value_flip is not None
    assert model.value_flip.shape == (1,)


def test_compile_model_wraps_static_fullgraph_training_paths(monkeypatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, features, value_flip=None, value_mirror=False):
            del value_flip, value_mirror
            value = self.weight.expand(features.action_count.shape[0])
            return value, self.policy_logits(features)

        def policy_logits(self, features):
            return self.weight.expand(features.action_count.shape[0], 2)

        def value_only(self, features, value_flip=None):
            del value_flip
            return self.weight.expand(features.action_count.shape[0])

    calls = []

    def compile_path(path, **options):
        calls.append((path, options))
        return path

    monkeypatch.setattr(torch, "compile", compile_path)
    model = Model()

    TrainerLoop(
        model,
        LoopConfig(compile_model=True, compile_mode="reduce-overhead"),
    )

    assert [call[0] for call in calls] == [model, model.policy_logits, model.value_only]
    assert all(
        call[1]
        == {
            "fullgraph": True,
            "dynamic": False,
            "mode": "reduce-overhead",
        }
        for call in calls
    )


def test_zero_value_weight_uses_policy_only_model_path() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_weight = torch.nn.Parameter(torch.tensor(0.1))
            self.value_weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, *args, **kwargs):
            raise AssertionError("combined forward must not run for policy-only training")

        def value_only(self, *args, **kwargs):
            raise AssertionError("value head must not run for policy-only training")

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
        ),
    )

    metrics = loop.train_step(batch)

    assert metrics is not None
    assert metrics.value_loss == 0.0
    assert model.policy_weight.grad is not None
    assert model.policy_weight.grad.item() != 0.0
    assert model.value_weight.grad is None
    assert model.value_weight.item() == 0.0


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
