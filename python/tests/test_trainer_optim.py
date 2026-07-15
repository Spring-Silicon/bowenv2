from __future__ import annotations

import pytest

from gz.trainer.optim import MixedOptimizer, build_optimizer

torch = pytest.importorskip("torch")


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 3)
        self.attr_proj = torch.nn.Linear(3, 2, bias=False)
        self.position_proj = torch.nn.Linear(4, 2)
        self.linear = torch.nn.Linear(3, 2)
        self.norm = torch.nn.LayerNorm(2)
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )
        self.stop = torch.nn.Linear(2, 1)

    def forward(self, indices: object) -> object:
        return self.norm(self.linear(self.embedding(indices))).sum()


def test_muon_mixed_routes_matrix_weights_and_steps_every_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingMuon(torch.optim.Optimizer):
        options: dict[str, object] = {}

        def __init__(self, params: object, **options: object) -> None:
            type(self).options = options
            super().__init__(params, options)

        @torch.no_grad()
        def step(self, closure: object = None) -> object:
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-group["lr"])
            return loss

    monkeypatch.setattr(torch.optim, "Muon", RecordingMuon, raising=False)
    model = TinyModel()
    optimizer = build_optimizer(
        model,
        name="muon_mixed",
        lr=0.02,
        adamw_lr=3e-4,
        weight_decay=1e-4,
        momentum=0.95,
        nesterov=True,
        ns_steps=2,
    )
    assert isinstance(optimizer, MixedOptimizer)
    routed = {
        id(parameter): group["optimizer_kind"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert len(routed) == len(list(model.parameters()))
    assert routed[id(model.linear.weight)] == "muon"
    assert routed[id(model.embedding.weight)] == "adamw"
    assert all(
        routed[id(parameter)] == "adamw" for parameter in model.attr_proj.parameters()
    )
    assert all(
        routed[id(parameter)] == "adamw"
        for parameter in model.position_proj.parameters()
    )
    assert routed[id(model.norm.weight)] == "adamw"
    assert all(routed[id(parameter)] == "adamw" for parameter in model.policy.parameters())
    assert all(routed[id(parameter)] == "adamw" for parameter in model.value.parameters())
    assert all(routed[id(parameter)] == "adamw" for parameter in model.stop.parameters())
    learning_rates = {
        group["optimizer_kind"]: group["lr"] for group in optimizer.param_groups
    }
    assert learning_rates == {"muon": 0.02, "adamw": 3e-4}
    assert RecordingMuon.options == {
        "lr": 0.02,
        "momentum": 0.95,
        "weight_decay": 1e-4,
        "nesterov": True,
        "ns_steps": 2,
        "adjust_lr_fn": "original",
    }
    assert optimizer.optimizers[1].defaults["betas"] == (0.9, 0.95)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert all(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_muon_mixed_requires_official_torch_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(torch.optim, "Muon", raising=False)

    with pytest.raises(
        RuntimeError,
        match=r"torch\.optim\.Muon from PyTorch 2\.9 or newer",
    ):
        build_optimizer(
            TinyModel(),
            name="muon_mixed",
            lr=0.02,
            adamw_lr=3e-4,
            weight_decay=1e-4,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
        )


@pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"),
    reason="torch.optim.Muon requires PyTorch 2.9 or newer",
)
def test_muon_mixed_steps_with_official_torch_optimizer() -> None:
    model = TinyModel()
    optimizer = build_optimizer(
        model,
        name="muon_mixed",
        lr=0.02,
        adamw_lr=3e-4,
        weight_decay=1e-4,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )

    assert isinstance(optimizer.optimizers[0], torch.optim.Muon)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert all(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
