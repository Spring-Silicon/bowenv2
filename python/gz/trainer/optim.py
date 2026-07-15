from __future__ import annotations


class MixedOptimizer:
    def __init__(self, optimizers: list[object]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, object]]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, object]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError("mixed optimizer state count mismatch")
        for optimizer, optimizer_state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(optimizer_state)


def build_optimizer(
    model: object,
    *,
    name: str,
    lr: float,
    adamw_lr: float | None,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
) -> object:
    torch = _torch()
    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        _mark_groups(optimizer, "adamw", 1.0)
        return optimizer
    if name != "muon_mixed":
        raise ValueError(f"unknown optimizer: {name}")

    excluded = _muon_excluded_parameter_ids(model, torch)
    muon_parameters = []
    adamw_parameters = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim == 2
            and id(parameter) not in excluded
            and not parameter_name.endswith(".score")
        ):
            muon_parameters.append(parameter)
        else:
            adamw_parameters.append(parameter)

    optimizers = []
    if muon_parameters:
        # The split Muon/AdamW rates follow Keller's original shape adjustment.
        muon = _torch_muon(torch)(
            muon_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn="original",
        )
        _mark_groups(muon, "muon", 1.0)
        optimizers.append(muon)
    if adamw_parameters:
        resolved_adamw_lr = lr if adamw_lr is None else adamw_lr
        adamw = torch.optim.AdamW(
            adamw_parameters,
            lr=resolved_adamw_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
        _mark_groups(adamw, "adamw", resolved_adamw_lr / lr)
        optimizers.append(adamw)
    return MixedOptimizer(optimizers)


def _muon_excluded_parameter_ids(model: object, torch: object) -> set[int]:
    excluded = set()
    for module in model.modules():
        if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
            excluded.update(id(parameter) for parameter in module.parameters(recurse=False))
    # These are the graph model's first learned projections. Muon is for
    # hidden transformations; input layers stay on the auxiliary AdamW path.
    for module_name in ("attr_proj", "position_proj"):
        module = getattr(model, module_name, None)
        if isinstance(module, torch.nn.Module):
            excluded.update(id(parameter) for parameter in module.parameters())
    # Muon is reserved for hidden transformations. The policy/value outputs
    # (and WhittleZero's retained STOP head) are classifier heads and stay on
    # the auxiliary AdamW optimizer, including their 2D matrices.
    for module_name in ("policy", "value", "stop"):
        module = getattr(model, module_name, None)
        if isinstance(module, torch.nn.Module):
            excluded.update(id(parameter) for parameter in module.parameters())
    for name, parameter in model.named_parameters():
        if name.endswith("global_tokens"):
            excluded.add(id(parameter))
    return excluded


def _mark_groups(optimizer: object, kind: str, lr_scale: float) -> None:
    for group in optimizer.param_groups:
        group["optimizer_kind"] = kind
        group["lr_scale"] = lr_scale


def _torch_muon(torch: object) -> type:
    muon = getattr(torch.optim, "Muon", None)
    if muon is None:
        raise RuntimeError(
            "muon_mixed requires torch.optim.Muon from PyTorch 2.9 or newer; "
            f"installed torch is {torch.__version__}"
        )
    return muon


def _torch() -> object:
    import torch

    return torch
