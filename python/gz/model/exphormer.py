from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from gz.codec import BatchView, FeatureSchemaConfig


@dataclass(frozen=True, slots=True)
class ArchConfig:
    name: str = "gz-graph-v1"
    dim: int = 128
    layers: int = 4
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    aggregation: str = "attention"
    global_tokens: int = 1
    value_input: str = "single"
    policy_head: str = "mlp"
    trunk: str = "exphormer"
    sage_layers: int = 3
    value_activation: str = "logit"
    subject_encoding: str = "mean"
    position_encoding: str = "shared"
    action_encoding: str = "kind_prior"
    profile: str = "graphzero"
    value_hidden: int = 0
    value_head: str = "scalar"
    value_bins: int = 101
    value_min: float = -1.0
    value_max: float = 1.0
    value_sigma_ratio: float = 0.75

    def __post_init__(self) -> None:
        if self.name not in {"gz-graph-v1", "gz-graph-v2"}:
            raise ValueError("unsupported graph arch name")
        if self.dim <= 0 or self.layers <= 0 or self.heads <= 0 or self.ffn_dim <= 0:
            raise ValueError("arch dimensions must be positive")
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout out of range")
        if self.activation not in {"gelu", "relu"}:
            raise ValueError("unsupported activation")
        if self.aggregation not in {"attention", "gine"}:
            raise ValueError("unsupported aggregation")
        if self.global_tokens <= 0:
            raise ValueError("global_tokens must be positive")
        if self.value_input not in {"single", "scalar", "pair"}:
            raise ValueError("unsupported value_input")
        if self.policy_head not in {"mlp", "pointer"}:
            raise ValueError("unsupported policy_head")
        if self.trunk not in {"exphormer", "sage"}:
            raise ValueError("unsupported trunk")
        if self.sage_layers <= 0:
            raise ValueError("sage_layers must be positive")
        if self.value_activation not in {"logit", "tanh"}:
            raise ValueError("unsupported value_activation")
        if self.subject_encoding not in {"mean", "match"}:
            raise ValueError("unsupported subject_encoding")
        if self.position_encoding not in {"shared", "policy_budget", "remaining_budget"}:
            raise ValueError("unsupported position_encoding")
        if self.action_encoding not in {"kind_prior", "candidate_only"}:
            raise ValueError("unsupported action_encoding")
        if self.profile not in {"graphzero", "whittlezero"}:
            raise ValueError("unsupported profile")
        if self.value_hidden < 0:
            raise ValueError("value_hidden must be non-negative")
        if self.value_head not in {"scalar", "hl_gauss"}:
            raise ValueError("unsupported value_head")
        if self.value_bins <= 1:
            raise ValueError("value_bins must be greater than one")
        if not math.isfinite(self.value_min) or not math.isfinite(self.value_max):
            raise ValueError("value support must be finite")
        if self.value_max <= self.value_min:
            raise ValueError("value_max must be greater than value_min")
        if not math.isfinite(self.value_sigma_ratio) or self.value_sigma_ratio <= 0.0:
            raise ValueError("value_sigma_ratio must be finite and positive")
        if self.position_encoding == "remaining_budget" and self.name != "gz-graph-v2":
            raise ValueError("remaining_budget position encoding requires gz-graph-v2")
        if self.name == "gz-graph-v2" and (
            self.profile != "graphzero"
            or self.trunk != "exphormer"
            or self.policy_head != "pointer"
            or self.position_encoding != "remaining_budget"
        ):
            raise ValueError("gz-graph-v2 requires the GraphZero Exphormer pointer architecture")
        if self.profile == "whittlezero" and (
            self.name != "gz-graph-v1"
            or self.trunk != "sage"
            or self.global_tokens != 1
            or self.policy_head != "pointer"
            or self.subject_encoding != "match"
            or self.position_encoding != "policy_budget"
            or self.action_encoding != "candidate_only"
        ):
            raise ValueError("whittlezero profile requires the legacy pointer architecture")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dim": self.dim,
            "layers": self.layers,
            "heads": self.heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "aggregation": self.aggregation,
            "global_tokens": self.global_tokens,
            "value_input": self.value_input,
            "policy_head": self.policy_head,
            "trunk": self.trunk,
            "sage_layers": self.sage_layers,
            "value_activation": self.value_activation,
            "subject_encoding": self.subject_encoding,
            "position_encoding": self.position_encoding,
            "action_encoding": self.action_encoding,
            "profile": self.profile,
            "value_hidden": self.value_hidden,
            "value_head": self.value_head,
            "value_bins": self.value_bins,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "value_sigma_ratio": self.value_sigma_ratio,
        }

    def encode(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def hash(self) -> bytes:
        hasher = hashlib.blake2b(digest_size=32)
        _update_chunk(hasher, b"gz-arch-config-v1")
        _update_chunk(hasher, self.encode())
        return hasher.digest()

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ArchConfig:
        fields = {
            "name",
            "dim",
            "layers",
            "heads",
            "ffn_dim",
            "dropout",
            "activation",
            "aggregation",
            "global_tokens",
            "value_input",
            "policy_head",
            "trunk",
            "sage_layers",
            "value_activation",
            "subject_encoding",
            "position_encoding",
            "action_encoding",
            "profile",
            "value_hidden",
            "value_head",
            "value_bins",
            "value_min",
            "value_max",
            "value_sigma_ratio",
        }
        optional = {
            "value_input",
            "policy_head",
            "trunk",
            "sage_layers",
            "value_activation",
            "subject_encoding",
            "position_encoding",
            "action_encoding",
            "profile",
            "value_hidden",
            "value_head",
            "value_bins",
            "value_min",
            "value_max",
            "value_sigma_ratio",
        }
        keys = set(value)
        if not (fields - optional <= keys <= fields):
            raise ValueError("arch config fields mismatch")
        return cls(
            name=_str(value, "name"),
            dim=_int(value, "dim"),
            layers=_int(value, "layers"),
            heads=_int(value, "heads"),
            ffn_dim=_int(value, "ffn_dim"),
            dropout=_float(value, "dropout"),
            activation=_str(value, "activation"),
            aggregation=_str(value, "aggregation"),
            global_tokens=_int(value, "global_tokens"),
            value_input=_str(value, "value_input", "single"),
            policy_head=_str(value, "policy_head", "mlp"),
            trunk=_str(value, "trunk", "exphormer"),
            sage_layers=_int(value, "sage_layers", 3),
            value_activation=_str(value, "value_activation", "logit"),
            subject_encoding=_str(value, "subject_encoding", "mean"),
            position_encoding=_str(value, "position_encoding", "shared"),
            action_encoding=_str(value, "action_encoding", "kind_prior"),
            profile=_str(value, "profile", "graphzero"),
            value_hidden=_int(value, "value_hidden", 0),
            value_head=_str(value, "value_head", "scalar"),
            value_bins=_int(value, "value_bins", 101),
            value_min=_float(value, "value_min", -1.0),
            value_max=_float(value, "value_max", 1.0),
            value_sigma_ratio=_float(value, "value_sigma_ratio", 0.75),
        )


class GraphBatchTensors(NamedTuple):
    node_count: object
    node_tokens: object
    node_attrs: object
    edge_count: object
    edge_src: object
    edge_dst: object
    edge_type: object
    action_count: object
    action_kind: object
    action_prior: object
    subject_count: object
    action_subjects: object
    position: object
    opponent_reward: object
    opponent_present: object
    opponent_state_present: object
    opponent_node_count: object
    opponent_node_tokens: object
    opponent_node_attrs: object
    opponent_edge_count: object
    opponent_edge_src: object
    opponent_edge_dst: object
    opponent_edge_type: object
    opponent_position: object


class GraphStateTensors(NamedTuple):
    node_count: object
    node_tokens: object
    node_attrs: object
    edge_count: object
    edge_src: object
    edge_dst: object
    edge_type: object
    position: object


def build_model(schema: FeatureSchemaConfig, arch: ArchConfig):
    return _model_class()(schema, arch)


def initialize_policy(model: object, mode: str) -> None:
    if mode == "default":
        return
    if mode != "neutral":
        raise ValueError(f"unsupported policy initializer: {mode}")
    if model.arch.policy_head != "pointer":
        raise ValueError("neutral policy initialization requires a pointer head")
    key = (
        model.policy.policy_attention.key
        if model.arch.profile == "whittlezero"
        else model.policy.pointer_key
    )
    _torch().nn.init.zeros_(key.weight)


def initialize_value(model: object, mode: str) -> None:
    if mode == "default":
        return
    if mode != "zero":
        raise ValueError(f"unsupported value initializer: {mode}")
    output = model.value[-1]
    for parameter in output.parameters():
        _torch().nn.init.zeros_(parameter)


def build_pair_serving_models(model: object) -> tuple[object, object]:
    if model.arch.value_input != "pair":
        raise ValueError("pair serving models require a pair value head")
    serving_class, opponent_class = _pair_serving_model_classes()
    return serving_class(model).eval(), opponent_class(model).eval()


def tensors_from_batch(view: BatchView, device: str | object, pinned_staging: bool = True) -> GraphBatchTensors:
    return BatchStager.from_view(view, device=device, pinned_staging=pinned_staging).copy(view)


class BatchStager:
    def __init__(
        self,
        schema: FeatureSchemaConfig,
        capacity: int,
        device: str | object,
        pinned_staging: bool = True,
        transfer_stream: object = None,
    ) -> None:
        torch = _torch()
        self.schema = schema
        self.capacity = capacity
        self.device = torch.device(device)
        self.pin = bool(pinned_staging and self.device.type == "cuda")
        if transfer_stream is not None and self.device.type != "cuda":
            raise ValueError("transfer stream requires a CUDA device")
        self.transfer_stream = transfer_stream
        self.ready_event = torch.cuda.Event() if transfer_stream is not None else None
        b = capacity
        n = schema.max_nodes
        e = schema.max_edges
        a = schema.max_actions
        s = schema.max_subjects
        d = schema.node_attr_dim
        index = torch.int32
        self.node_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.node_tokens = _StagedTensor((b, n), index, self.device, self.pin, transfer_stream)
        self.node_attrs = _StagedTensor((b, n, d), torch.float32, self.device, self.pin, transfer_stream)
        self.edge_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.edge_src = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.edge_dst = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.edge_type = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.action_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.action_kind = _StagedTensor((b, a), index, self.device, self.pin, transfer_stream)
        self.action_prior = _StagedTensor((b, a), torch.float32, self.device, self.pin, transfer_stream)
        self.subject_count = _StagedTensor((b, a), index, self.device, self.pin, transfer_stream)
        self.action_subjects = _StagedTensor((b, a, s), index, self.device, self.pin, transfer_stream)
        self.position = _StagedTensor((b, 4), torch.float32, self.device, self.pin, transfer_stream)
        self.opponent_reward = _StagedTensor((b,), torch.float32, self.device, self.pin, transfer_stream)
        self.opponent_present = _StagedTensor((b,), torch.float32, self.device, self.pin, transfer_stream)
        self.opponent_state_present = _StagedTensor((b,), torch.float32, self.device, self.pin, transfer_stream)
        self.opponent_node_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.opponent_node_tokens = _StagedTensor((b, n), index, self.device, self.pin, transfer_stream)
        self.opponent_node_attrs = _StagedTensor((b, n, d), torch.float32, self.device, self.pin, transfer_stream)
        self.opponent_edge_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.opponent_edge_src = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.opponent_edge_dst = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.opponent_edge_type = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.opponent_position = _StagedTensor((b, 4), torch.float32, self.device, self.pin, transfer_stream)

    @classmethod
    def from_view(cls, view: BatchView, device: str | object, pinned_staging: bool = True) -> BatchStager:
        schema = FeatureSchemaConfig(
            name="batch-view",
            node_vocab_size=max(2, int(view.node_tokens.max(initial=0)) + 1),
            node_attr_dim=view.dims.node_attr_dim,
            edge_type_count=max(1, int(view.edge_type.max(initial=0)) + 1),
            action_kind_vocab_size=max(3, int(view.action_kind.max(initial=0)) + 1),
            max_nodes=view.dims.max_nodes,
            max_edges=view.dims.max_edges,
            max_actions=view.dims.max_actions,
            max_subjects=view.dims.max_subjects,
            opponent_reward_scale=256.0,
            expander_degree=0,
            expander_seed=0,
        )
        return cls(schema, view.batch_capacity, device, pinned_staging)

    def copy(self, view: BatchView, *, copy_opponent: bool = True) -> GraphBatchTensors:
        self._check_view(view)
        self.node_count.copy(view.node_count)
        self.node_tokens.copy(view.node_tokens)
        if view.node_attrs is None:
            self.node_attrs.zero_()
        else:
            self.node_attrs.copy(view.node_attrs)
        self.edge_count.copy(view.edge_count)
        self.edge_src.copy(view.edge_src)
        self.edge_dst.copy(view.edge_dst)
        self.edge_type.copy(view.edge_type)
        self.action_count.copy(view.action_count)
        self.action_kind.copy(view.action_kind)
        self.action_prior.copy(view.action_prior)
        self.subject_count.copy(view.subject_count)
        self.action_subjects.copy(view.action_subjects)
        self.position.copy(view.position)
        self.opponent_reward.copy(view.opponent_reward)
        self.opponent_present.copy(view.opponent_present)
        self.opponent_state_present.copy(view.opponent_state_present)
        if copy_opponent:
            self.opponent_node_count.copy(view.opponent_node_count)
            self.opponent_node_tokens.copy(view.opponent_node_tokens)
            if view.opponent_node_attrs is None:
                self.opponent_node_attrs.zero_()
            else:
                self.opponent_node_attrs.copy(view.opponent_node_attrs)
            self.opponent_edge_count.copy(view.opponent_edge_count)
            self.opponent_edge_src.copy(view.opponent_edge_src)
            self.opponent_edge_dst.copy(view.opponent_edge_dst)
            self.opponent_edge_type.copy(view.opponent_edge_type)
            self.opponent_position.copy(view.opponent_position)
        self._record_ready()
        return self.tensors()

    def dummy(self) -> GraphBatchTensors:
        self.node_count.fill_(1)
        self.node_tokens.zero_()
        self.node_tokens.cpu[..., 0] = 1
        self.node_attrs.zero_()
        self.edge_count.zero_()
        self.edge_src.zero_()
        self.edge_dst.zero_()
        self.edge_type.zero_()
        self.action_count.fill_(1)
        self.action_kind.zero_()
        self.action_kind.cpu[..., 0] = 1
        self.action_prior.zero_()
        self.subject_count.zero_()
        self.action_subjects.fill_(0xFFFF)
        self.position.zero_()
        self.opponent_reward.zero_()
        self.opponent_present.zero_()
        self.opponent_state_present.zero_()
        self.opponent_node_count.zero_()
        self.opponent_node_tokens.zero_()
        self.opponent_node_attrs.zero_()
        self.opponent_edge_count.zero_()
        self.opponent_edge_src.zero_()
        self.opponent_edge_dst.zero_()
        self.opponent_edge_type.zero_()
        self.opponent_position.zero_()
        for tensor in self._all():
            tensor.sync()
        self._record_ready()
        return self.tensors()

    def _record_ready(self) -> None:
        if self.ready_event is not None:
            self.ready_event.record(self.transfer_stream)

    def tensors(self) -> GraphBatchTensors:
        return GraphBatchTensors(
            node_count=self.node_count.device_tensor,
            node_tokens=self.node_tokens.device_tensor,
            node_attrs=self.node_attrs.device_tensor,
            edge_count=self.edge_count.device_tensor,
            edge_src=self.edge_src.device_tensor,
            edge_dst=self.edge_dst.device_tensor,
            edge_type=self.edge_type.device_tensor,
            action_count=self.action_count.device_tensor,
            action_kind=self.action_kind.device_tensor,
            action_prior=self.action_prior.device_tensor,
            subject_count=self.subject_count.device_tensor,
            action_subjects=self.action_subjects.device_tensor,
            position=self.position.device_tensor,
            opponent_reward=self.opponent_reward.device_tensor,
            opponent_present=self.opponent_present.device_tensor,
            opponent_state_present=self.opponent_state_present.device_tensor,
            opponent_node_count=self.opponent_node_count.device_tensor,
            opponent_node_tokens=self.opponent_node_tokens.device_tensor,
            opponent_node_attrs=self.opponent_node_attrs.device_tensor,
            opponent_edge_count=self.opponent_edge_count.device_tensor,
            opponent_edge_src=self.opponent_edge_src.device_tensor,
            opponent_edge_dst=self.opponent_edge_dst.device_tensor,
            opponent_edge_type=self.opponent_edge_type.device_tensor,
            opponent_position=self.opponent_position.device_tensor,
        )

    def _check_view(self, view: BatchView) -> None:
        dims = view.dims
        if view.batch_capacity != self.capacity:
            raise ValueError("batch capacity mismatch")
        if dims.max_nodes != self.schema.max_nodes:
            raise ValueError("max_nodes mismatch")
        if dims.max_edges != self.schema.max_edges:
            raise ValueError("max_edges mismatch")
        if dims.max_actions != self.schema.max_actions:
            raise ValueError("max_actions mismatch")
        if dims.max_subjects != self.schema.max_subjects:
            raise ValueError("max_subjects mismatch")
        if dims.node_attr_dim != self.schema.node_attr_dim:
            raise ValueError("node_attr_dim mismatch")

    def _all(self) -> tuple[_StagedTensor, ...]:
        return (
            self.node_count,
            self.node_tokens,
            self.node_attrs,
            self.edge_count,
            self.edge_src,
            self.edge_dst,
            self.edge_type,
            self.action_count,
            self.action_kind,
            self.action_prior,
            self.subject_count,
            self.action_subjects,
            self.position,
            self.opponent_reward,
            self.opponent_present,
            self.opponent_state_present,
            self.opponent_node_count,
            self.opponent_node_tokens,
            self.opponent_node_attrs,
            self.opponent_edge_count,
            self.opponent_edge_src,
            self.opponent_edge_dst,
            self.opponent_edge_type,
            self.opponent_position,
        )


class _StagedTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: object,
        device: object,
        pin: bool,
        transfer_stream: object = None,
    ) -> None:
        torch = _torch()
        self.cpu = torch.empty(shape, dtype=dtype, pin_memory=pin)
        self.device_tensor = torch.empty(shape, dtype=dtype, device=device)
        self.non_blocking = pin
        self.transfer_stream = transfer_stream

    def copy(self, array: np.ndarray) -> None:
        np.copyto(self.cpu.numpy(), array, casting="unsafe")
        self.sync()

    def zero_(self) -> None:
        self.cpu.zero_()
        self.sync()

    def fill_(self, value: int | float) -> None:
        self.cpu.fill_(value)
        self.sync()

    def sync(self) -> None:
        if self.transfer_stream is None:
            self.device_tensor.copy_(self.cpu, non_blocking=self.non_blocking)
            return
        torch = _torch()
        with torch.cuda.stream(self.transfer_stream):
            self.device_tensor.copy_(self.cpu, non_blocking=self.non_blocking)


@lru_cache(maxsize=1)
def _model_class():
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional

    class GraphModel(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.schema = schema
            self.arch = arch
            if arch.position_encoding == "policy_budget":
                position_dim = 1
            elif arch.position_encoding == "remaining_budget":
                position_dim = 2
            else:
                position_dim = 4
            if arch.profile == "whittlezero":
                if schema.name != "whittle-v2" or schema.node_attr_dim != 3:
                    raise ValueError("whittlezero profile requires the whittle-v2 feature schema")
                # Match WhittleNet construction order so a shared torch seed
                # produces the same initialized policy.
                self.position_proj = nn.Linear(position_dim, arch.dim)
                self.node_proj = nn.Linear(16, arch.dim)
                self.node_embedding = None
                self.attr_proj = None
            else:
                self.node_embedding = nn.Embedding(schema.node_vocab_size, arch.dim, padding_idx=0)
                self.attr_proj = nn.Linear(schema.node_attr_dim, arch.dim, bias=False) if schema.node_attr_dim else None
                self.position_proj = nn.Linear(position_dim, arch.dim)
                self.node_proj = None
            if arch.trunk == "sage":
                self.sage = nn.ModuleList([DirectionalSageLayer(arch) for _ in range(arch.sage_layers)])
                self.global_tokens = nn.Parameter(torch.zeros(arch.global_tokens, arch.dim))
                if arch.profile == "whittlezero":
                    layer = nn.TransformerEncoderLayer(
                        d_model=arch.dim,
                        nhead=arch.heads,
                        dim_feedforward=arch.ffn_dim,
                        dropout=arch.dropout,
                        batch_first=True,
                        norm_first=True,
                    )
                    self.transformer = nn.TransformerEncoder(layer, num_layers=arch.layers)
                    self.layers = None
                else:
                    self.transformer = None
                    self.layers = nn.ModuleList([TransformerBlock(arch) for _ in range(arch.layers)])
            else:
                self.sage = None
                self.transformer = None
                self.global_tokens = nn.Parameter(torch.zeros(arch.global_tokens, arch.dim))
                self.layers = nn.ModuleList([GraphLayer(schema, arch) for _ in range(arch.layers)])
            if arch.name == "gz-graph-v2":
                self.node_output_norm = nn.LayerNorm(arch.dim)
                self.global_output_norm = nn.LayerNorm(arch.dim)
            else:
                self.node_output_norm = None
                self.global_output_norm = None
            if arch.action_encoding == "candidate_only":
                self.kind_embedding = nn.Embedding(schema.action_kind_vocab_size - 2, arch.dim)
            else:
                self.kind_embedding = nn.Embedding(schema.action_kind_vocab_size, arch.dim, padding_idx=0)
            if arch.subject_encoding == "match":
                self.match = MatchAttention(schema, arch)
            else:
                self.match = None
            action_feat_dim = _action_feat_dim(arch)
            if arch.policy_head == "pointer":
                self.policy = (
                    WhittlePointerPolicyHead(arch, action_feat_dim)
                    if arch.profile == "whittlezero"
                    else PointerPolicyHead(arch, action_feat_dim)
                )
            else:
                self.policy = _mlp(nn, action_feat_dim, arch.ffn_dim, 1, arch.activation, arch.dropout)
            if arch.profile == "whittlezero":
                hidden = arch.value_hidden or 256
                # WhittleNet retains this legacy head even when its TSP pointer
                # emits STOP directly. Keeping it preserves initialization and
                # optimizer parity.
                self.stop = nn.Sequential(nn.Linear(arch.dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            else:
                self.stop = None
            value_dim = arch.dim
            if arch.value_input == "scalar":
                value_dim += 2
            elif arch.value_input == "pair":
                value_dim *= 2
            if arch.profile == "whittlezero":
                hidden = arch.value_hidden or 256
                self.value = nn.Sequential(
                    nn.Linear(value_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, arch.value_bins if arch.value_head == "hl_gauss" else 1),
                )
            else:
                self.value = _mlp(
                    nn,
                    value_dim,
                    arch.value_hidden or arch.ffn_dim,
                    arch.value_bins if arch.value_head == "hl_gauss" else 1,
                    arch.activation,
                    arch.dropout,
                )
            if arch.value_head == "hl_gauss":
                edges = torch.linspace(
                    arch.value_min,
                    arch.value_max,
                    arch.value_bins + 1,
                    dtype=torch.float32,
                )
                centers = (edges[:-1] + edges[1:]) * 0.5
                self.register_buffer("value_bin_edges", edges, persistent=False)
                self.register_buffer("value_bin_centers", centers, persistent=False)
            else:
                self.register_buffer("value_bin_edges", torch.empty(0), persistent=False)
                self.register_buffer("value_bin_centers", torch.empty(0), persistent=False)

        def forward(self, batch: GraphBatchTensors, value_flip: object = None, value_mirror: bool = False):
            graph = _self_graph(batch)
            h, g_readout, node_mask = self._encode_graph(graph)
            opponent_readout = None
            if self.arch.value_input == "pair":
                opponent_readout = self.opponent_readout(batch)
            return self._forward_heads(
                batch,
                h,
                g_readout,
                node_mask,
                opponent_readout,
                value_flip,
                value_mirror,
            )

        def policy_logits(self, batch: GraphBatchTensors):
            h, g_readout, node_mask = self._encode_graph(_self_graph(batch))
            return self._policy_logits(batch, h, g_readout, node_mask)

        def value_only(self, batch: GraphBatchTensors, value_flip: object = None):
            if self.arch.value_input != "pair":
                _, g_readout, _ = self._encode_graph(_self_graph(batch))
                return self._value_output(batch, g_readout, None, None, False)

            own = _self_graph(batch)
            opponent = _opponent_graph(batch)
            if value_flip is not None:
                own, opponent = _orient_graph_pair(torch, own, opponent, value_flip)
            pair_capacity = batch.node_count.shape[0]
            _, pair_readout, _ = self._encode_graph(_concat_graphs(torch, own, opponent))
            return self._value_output(
                batch,
                pair_readout[:pair_capacity],
                pair_readout[pair_capacity:],
                None,
                False,
            )

        def forward_with_opponent(self, batch: GraphBatchTensors, opponent_readout: object):
            graph = _self_graph(batch)
            h, g_readout, node_mask = self._encode_graph(graph)
            return self._forward_heads(
                batch,
                h,
                g_readout,
                node_mask,
                opponent_readout,
                None,
                False,
            )

        def opponent_readout(self, batch: GraphBatchTensors):
            _, readout, _ = self._encode_graph(_opponent_graph(batch))
            return readout

        def _forward_heads(
            self,
            batch: GraphBatchTensors,
            h: object,
            g_readout: object,
            node_mask: object,
            opponent_readout: object,
            value_flip: object,
            value_mirror: bool,
        ):
            logits = self._policy_logits(batch, h, g_readout, node_mask)
            value_raw = self._value_output(
                batch,
                g_readout,
                opponent_readout,
                value_flip,
                value_mirror,
            )
            return value_raw, logits

        def _policy_logits(self, batch, h, g_readout, node_mask):
            policy_readout = g_readout
            if self.arch.position_encoding == "policy_budget":
                policy_readout = policy_readout + self.position_proj(batch.position[:, 2:3])
            if self.arch.action_encoding == "candidate_only":
                is_candidate = batch.action_kind >= 2
                kind_index = (batch.action_kind - 2).clamp(0, self.kind_embedding.num_embeddings - 1)
                kind = self.kind_embedding(kind_index) * is_candidate.unsqueeze(-1)
            else:
                kind = self.kind_embedding(batch.action_kind.clamp(0, self.schema.action_kind_vocab_size - 1))
            if self.match is not None:
                subject_feat = self.match(h, node_mask, batch.action_subjects, batch.subject_count, kind)
            else:
                subject_feat = _subject_pool(torch, h, node_mask, batch.action_subjects, batch.subject_count)
            readout = policy_readout.unsqueeze(1).expand(-1, batch.action_kind.shape[1], -1)
            action_parts = (kind, subject_feat, readout)
            if self.arch.action_encoding == "kind_prior":
                action_parts = (kind, batch.action_prior.unsqueeze(-1), subject_feat, readout)
            action_feat = torch.cat(action_parts, dim=-1)
            if self.arch.policy_head == "pointer":
                action_index = torch.arange(action_feat.shape[1], device=action_feat.device)
                action_mask = action_index.unsqueeze(0) < batch.action_count.unsqueeze(1)
                logits = self.policy(policy_readout, action_feat, action_mask)
            else:
                logits = self.policy(action_feat).squeeze(-1)

            return logits

        def _value_output(
            self,
            batch,
            g_readout,
            opponent_readout,
            value_flip,
            value_mirror,
        ):
            value_input = g_readout
            mirrored_input = None
            if self.arch.value_input == "scalar":
                opponent = torch.stack((batch.opponent_reward, batch.opponent_present), dim=-1).to(g_readout.dtype)
                value_input = torch.cat((g_readout, opponent), dim=-1)
            elif self.arch.value_input == "pair":
                present = (batch.opponent_state_present > 0).unsqueeze(-1)
                opponent_readout = torch.where(present, opponent_readout, torch.zeros_like(opponent_readout))
                if value_flip is not None:
                    flip = value_flip.to(torch.bool).unsqueeze(-1) & present
                    left = torch.where(flip, opponent_readout, g_readout)
                    right = torch.where(flip, g_readout, opponent_readout)
                    value_input = torch.cat((left, right), dim=-1)
                else:
                    value_input = torch.cat((g_readout, opponent_readout), dim=-1)
                if value_mirror:
                    # whittlezero's mirrored emission, realized at train
                    # time: the swapped orientation of every pair as a
                    # second value example (target -z). The trunk readouts
                    # are shared; only the value MLP runs twice.
                    mirrored_input = torch.cat((opponent_readout, g_readout), dim=-1)
            value_raw = self._value_head_output(value_input)
            if mirrored_input is not None:
                mirrored_raw = self._value_head_output(mirrored_input)
                return torch.stack((value_raw, mirrored_raw))
            return value_raw

        def _value_head_output(self, value_input):
            value_raw = self.value(value_input)
            if self.arch.value_head == "hl_gauss":
                return value_raw
            value_raw = value_raw.squeeze(-1)
            if self.arch.value_activation == "tanh":
                # whittlezero's bounded value head: the serving-side search
                # and the MSE training target share the [-1, 1] scale.
                value_raw = torch.tanh(value_raw)
            return value_raw

        def decode_value(self, value_raw):
            if self.arch.value_head != "hl_gauss":
                return value_raw
            probabilities = torch.softmax(value_raw.float(), dim=-1)
            centers = self.value_bin_centers.to(
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            return (probabilities * centers).sum(dim=-1)

        def _encode_graph(self, graph: GraphStateTensors):
            b, n = graph.node_tokens.shape
            device = graph.node_tokens.device
            node_index = torch.arange(n, device=device)
            node_mask = node_index.unsqueeze(0) < graph.node_count.unsqueeze(1)
            if self.arch.profile == "whittlezero":
                h = self.node_proj(_whittle_node_features(torch, graph, node_mask))
            else:
                h = self.node_embedding(graph.node_tokens.clamp(0, self.schema.node_vocab_size - 1))
                if self.attr_proj is not None:
                    h = h + self.attr_proj(graph.node_attrs)
                h = h * node_mask.unsqueeze(-1)

            g = self.global_tokens.unsqueeze(0).expand(b, -1, -1)
            if self.arch.position_encoding == "shared":
                g = g + self.position_proj(graph.position).unsqueeze(1)
            elif self.arch.position_encoding == "remaining_budget":
                g = g + self.position_proj(_remaining_budget_position(torch, graph.position)).unsqueeze(1)
            if self.sage is not None:
                for layer in self.sage:
                    h = layer(h, graph, node_mask)
                seq = torch.cat((g, h), dim=1)
                ones = node_mask.new_ones((b, g.shape[1]))
                seq_mask = torch.cat((ones, node_mask), dim=1)
                if self.transformer is not None:
                    seq = self.transformer(seq, src_key_padding_mask=~seq_mask)
                else:
                    for layer in self.layers:
                        seq = layer(seq, seq_mask)
                g = seq[:, : g.shape[1]]
                h = seq[:, g.shape[1] :] * node_mask.unsqueeze(-1)
            else:
                for layer in self.layers:
                    h, g = layer(h, g, graph, node_mask)

            if self.node_output_norm is not None:
                h = self.node_output_norm(h) * node_mask.unsqueeze(-1)
                g = self.global_output_norm(g)

            return h, g.mean(dim=1), node_mask

    class MatchAttention(nn.Module):
        # whittlezero's policy_match_attention: the candidate's subject
        # nodes keep their pattern roles. Slot embeddings are added to the
        # gathered subject embeddings, the kind/rule embedding queries a
        # single-head attention over them, and the candidate contributes
        # [match root, attended] instead of an order-blind mean -- absorb,
        # distribute, and consensus candidates over the same node set with
        # different role assignments stop aliasing.
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.slot_embedding = nn.Embedding(max(1, schema.max_subjects), arch.dim)
            self.query = nn.Linear(arch.dim, arch.dim, bias=False)
            self.key = nn.Linear(arch.dim, arch.dim, bias=False)
            self.value = nn.Linear(arch.dim, arch.dim, bias=False)

        def forward(self, h, node_mask, action_subjects, subject_count, kind):
            gathered, valid = _gather_subjects(torch, h, node_mask, action_subjects, subject_count)
            root_valid = valid[:, :, 0].unsqueeze(-1)
            root = torch.where(root_valid, gathered[:, :, 0, :], torch.zeros_like(gathered[:, :, 0, :]))
            slots = torch.arange(gathered.shape[2], device=h.device)
            keyed = gathered + self.slot_embedding(slots).view(1, 1, gathered.shape[2], -1)
            query = self.query(kind).unsqueeze(2)
            scores = (query * self.key(keyed)).sum(dim=-1) / math.sqrt(keyed.shape[-1])
            scores = scores.masked_fill(~valid, -1.0e9)
            weights = torch.softmax(scores, dim=2)
            attended = (weights.unsqueeze(-1) * self.value(keyed)).sum(dim=2)
            has_match = valid.any(dim=2, keepdim=True)
            attended = torch.where(has_match, attended, torch.zeros_like(attended))
            return torch.cat((root, attended), dim=-1)

    class PointerPolicyHead(nn.Module):
        # whittlezero's tsp_pointer scorer: a multi-head glimpse over the
        # action tokens refines the graph readout into a board query, and
        # a single-head dot product against the same tokens produces the
        # logits, tanh-bounded to +/-CLIP. Scores are relative across the
        # action set; the per-candidate MLP scores each action in
        # isolation and its logit scale is unbounded.
        CLIP = 10.0

        def __init__(self, arch: ArchConfig, action_feat_dim: int) -> None:
            super().__init__()
            dim = arch.dim
            self.heads = arch.heads
            self.token_proj = nn.Linear(action_feat_dim, dim)
            self.glimpse_query = nn.Linear(dim, dim, bias=False)
            self.glimpse_key = nn.Linear(dim, dim, bias=False)
            self.glimpse_value = nn.Linear(dim, dim, bias=False)
            self.glimpse_unify = nn.Linear(dim, dim)
            self.board_ffn = nn.Sequential(
                nn.Linear(dim, arch.ffn_dim),
                _activation_module(nn, arch.activation),
                nn.Linear(arch.ffn_dim, dim),
            )
            if arch.name == "gz-graph-v2":
                self.pointer_board_norm = nn.LayerNorm(dim)
                self.pointer_token_norm = nn.LayerNorm(dim)
            else:
                self.pointer_board_norm = None
                self.pointer_token_norm = None
            self.pointer_key = nn.Linear(dim, dim, bias=False)

        def forward(self, readout, action_feat, action_mask):
            b, a, _ = action_feat.shape
            tokens = self.token_proj(action_feat)
            dim = tokens.shape[-1]
            split = dim // self.heads
            query = self.glimpse_query(readout).view(b, self.heads, split)
            keys = self.glimpse_key(tokens).view(b, a, self.heads, split)
            values = self.glimpse_value(tokens).view(b, a, self.heads, split)
            scores = torch.einsum("bhs,bahs->bha", query, keys) / math.sqrt(split)
            # -1e9, not -inf: rows past row_count have zero valid actions,
            # and an all--inf softmax row is NaN.
            scores = scores.masked_fill(~action_mask.unsqueeze(1), -1.0e9)
            board = torch.einsum("bha,bahs->bhs", torch.softmax(scores, dim=-1), values)
            board = self.glimpse_unify(board.reshape(b, dim))
            board = board + self.board_ffn(board)
            if self.pointer_board_norm is not None:
                board = self.pointer_board_norm(board)
                tokens = self.pointer_token_norm(tokens)
            raw = torch.einsum("bd,bad->ba", board, self.pointer_key(tokens)) / math.sqrt(dim)
            return self.CLIP * torch.tanh(raw)

    class SingleQueryAttention(nn.Module):
        def __init__(self, dim: int, heads: int, *, project_query: bool) -> None:
            super().__init__()
            self.heads = heads
            self.split = dim // heads
            self.key = nn.Linear(dim, dim, bias=False)
            self.value = nn.Linear(dim, dim, bias=False)
            self.query = nn.Linear(dim, dim, bias=False) if project_query else None
            self.unify = nn.Linear(dim, dim)

        def forward(self, query, tokens, mask):
            b, a, d = tokens.shape
            if self.query is not None:
                query = self.query(query)
            query = query.view(b, self.heads, self.split)
            keys = self.key(tokens).view(b, a, self.heads, self.split)
            values = self.value(tokens).view(b, a, self.heads, self.split)
            raw = torch.einsum("bhd,bahd->bha", query, keys) / math.sqrt(self.split)
            valid = mask.unsqueeze(1)
            masked = raw.masked_fill(~valid, -1.0e9)
            masked = torch.where(valid.any(dim=-1, keepdim=True), masked, torch.zeros_like(masked))
            weights = torch.softmax(masked, dim=-1) * valid.to(raw.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            out = torch.einsum("bha,bahd->bhd", weights, values).reshape(b, d)
            return self.unify(out), raw

    class WhittlePointerPolicyHead(nn.Module):
        CLIP = 10.0

        def __init__(self, arch: ArchConfig, action_feat_dim: int) -> None:
            super().__init__()
            self.token_proj = nn.Linear(action_feat_dim, arch.dim)
            self.board_attention = SingleQueryAttention(arch.dim, arch.heads, project_query=True)
            self.board_ffn = nn.Sequential(
                nn.Linear(arch.dim, arch.ffn_dim),
                nn.GELU(),
                nn.Linear(arch.ffn_dim, arch.dim),
            )
            self.policy_attention = SingleQueryAttention(arch.dim, 1, project_query=False)

        def forward(self, readout, action_feat, action_mask):
            tokens = self.token_proj(action_feat)
            board, _ = self.board_attention(readout, tokens, action_mask)
            board = board + self.board_ffn(board)
            _, raw = self.policy_attention(board, tokens, action_mask)
            return self.CLIP * torch.tanh(raw[:, 0, :])

    class DirectionalSageLayer(nn.Module):
        # whittlezero's DirectionalSAGELayer (model/graphsage.py) over gz
        # edge lists: sigmoid messages from each edge endpoint, max-pooled
        # per direction via scatter-amax, combined with the node state and
        # L2-normalized. Edge types are ignored, matching the original.
        # Sigmoid messages are strictly positive, so a zeros-initialized
        # amax scatter reproduces the original's masked-max-else-zero.
        def __init__(self, arch: ArchConfig) -> None:
            super().__init__()
            self.real_edges_only = arch.profile == "whittlezero"
            self.src_msg = nn.Linear(arch.dim, arch.dim)
            self.dst_msg = nn.Linear(arch.dim, arch.dim)
            self.combine = nn.Linear(arch.dim * 3, arch.dim)

        def forward(self, h, graph: GraphStateTensors, node_mask):
            b, n, d = h.shape
            src, dst, mask = _valid_edges(torch, graph, node_mask)
            if self.real_edges_only:
                mask = mask & (graph.edge_type < 2)
            weight = mask.unsqueeze(-1).to(h.dtype)
            fwd = torch.sigmoid(self.src_msg(_gather_nodes(torch, h, src))) * weight
            rev = torch.sigmoid(self.dst_msg(_gather_nodes(torch, h, dst))) * weight
            fwd_agg = torch.zeros((b, n, d), dtype=h.dtype, device=h.device)
            fwd_agg.scatter_reduce_(1, dst.unsqueeze(-1).expand(-1, -1, d), fwd, reduce="amax", include_self=True)
            rev_agg = torch.zeros((b, n, d), dtype=h.dtype, device=h.device)
            rev_agg.scatter_reduce_(1, src.unsqueeze(-1).expand(-1, -1, d), rev, reduce="amax", include_self=True)
            out = torch.relu(self.combine(torch.cat((h, fwd_agg, rev_agg), dim=-1)))
            out = functional.normalize(out, p=2.0, dim=-1, eps=1e-8)
            return out * node_mask.unsqueeze(-1)

    class TransformerBlock(nn.Module):
        # Pre-norm encoder block (whittlezero's build_transformer_encoder
        # with norm_first=True), run over [global tokens | nodes] with the
        # padded nodes masked out of the keys.
        def __init__(self, arch: ArchConfig) -> None:
            super().__init__()
            self.norm_attn = nn.LayerNorm(arch.dim)
            self.norm_ffn = nn.LayerNorm(arch.dim)
            self.attn = DenseAttention(arch)
            self.drop = nn.Dropout(arch.dropout)
            self.ffn = _mlp(nn, arch.dim, arch.ffn_dim, arch.dim, arch.activation, arch.dropout)

        def forward(self, seq, seq_mask):
            x = self.norm_attn(seq)
            seq = seq + self.drop(self.attn(x, x, seq_mask))
            seq = seq + self.ffn(self.norm_ffn(seq))
            return seq

    class GraphLayer(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.norm_edge = nn.LayerNorm(arch.dim)
            self.norm_exchange_h = nn.LayerNorm(arch.dim)
            self.norm_exchange_g = nn.LayerNorm(arch.dim)
            self.norm_read_h = nn.LayerNorm(arch.dim)
            self.norm_read_g = nn.LayerNorm(arch.dim)
            self.norm_ffn_h = nn.LayerNorm(arch.dim)
            self.norm_ffn_g = nn.LayerNorm(arch.dim)
            self.edge = EdgeAttention(schema, arch) if arch.aggregation == "attention" else EdgeGine(schema, arch)
            self.exchange = DenseAttention(arch)
            self.read = DenseAttention(arch)
            self.ffn_h = _mlp(nn, arch.dim, arch.ffn_dim, arch.dim, arch.activation, arch.dropout)
            self.ffn_g = _mlp(nn, arch.dim, arch.ffn_dim, arch.dim, arch.activation, arch.dropout)

        def forward(self, h, g, graph: GraphStateTensors, node_mask):
            h_mask = node_mask.unsqueeze(-1)
            h = h + self.edge(self.norm_edge(h), graph, node_mask) * h_mask
            h = h + self.exchange(self.norm_exchange_h(h), self.norm_exchange_g(g), None) * h_mask
            g = g + self.read(self.norm_read_g(g), self.norm_read_h(h), node_mask)
            h = h + self.ffn_h(self.norm_ffn_h(h)) * h_mask
            g = g + self.ffn_g(self.norm_ffn_g(g))
            h = h * h_mask
            return h, g

    class EdgeAttention(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.edge_type_count = schema.edge_type_count
            self.heads = arch.heads
            self.head_dim = arch.dim // arch.heads
            self.q_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.k_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.v_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.o_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.edge_embedding = nn.Embedding(max(1, 2 * schema.edge_type_count), arch.dim)

        def forward(self, h, graph: GraphStateTensors, node_mask):
            b, n, d = h.shape
            src, dst, typ, mask = _mirrored_edges(torch, graph, node_mask, self.edge_type_count)
            q = self.q_proj(h).reshape(b, n, self.heads, self.head_dim)
            k = self.k_proj(h).reshape(b, n, self.heads, self.head_dim)
            v = self.v_proj(h).reshape(b, n, self.heads, self.head_dim)
            q_dst = _gather_nodes(torch, q.reshape(b, n, d), dst).reshape(b, -1, self.heads, self.head_dim)
            k_src = _gather_nodes(torch, k.reshape(b, n, d), src).reshape(b, -1, self.heads, self.head_dim)
            v_src = _gather_nodes(torch, v.reshape(b, n, d), src).reshape(b, -1, self.heads, self.head_dim)
            e = self.edge_embedding(typ).reshape(b, -1, self.heads, self.head_dim)
            score = (q_dst * k_src * e).sum(dim=-1) / math.sqrt(self.head_dim)
            score = score.masked_fill(~mask.unsqueeze(-1), -1.0e9)
            scatter_index = dst.unsqueeze(-1).expand(-1, -1, self.heads)
            amax = torch.full((b, n, self.heads), -1.0e9, dtype=score.dtype, device=score.device)
            amax.scatter_reduce_(1, scatter_index, score, reduce="amax", include_self=True)
            edge_amax = torch.gather(amax, 1, scatter_index)
            weight = torch.exp(score - edge_amax) * mask.unsqueeze(-1).to(score.dtype)
            denom = torch.zeros((b, n, self.heads), dtype=score.dtype, device=score.device)
            denom.scatter_add_(1, scatter_index, weight)
            msg = weight.unsqueeze(-1) * v_src
            out = torch.zeros((b, n, self.heads, self.head_dim), dtype=h.dtype, device=h.device)
            out.scatter_add_(1, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.heads, self.head_dim), msg)
            out = out / denom.clamp_min(1.0e-6).unsqueeze(-1)
            return self.o_proj(out.reshape(b, n, d))

    class EdgeGine(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.edge_type_count = schema.edge_type_count
            self.k_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.edge_embedding = nn.Embedding(max(1, 2 * schema.edge_type_count), arch.dim)
            self.eps = nn.Parameter(torch.zeros(()))
            self.out = _mlp(nn, arch.dim, arch.ffn_dim, arch.dim, arch.activation, arch.dropout)
            self.activation = _activation(functional, arch.activation)

        def forward(self, h, graph: GraphStateTensors, node_mask):
            b, n, d = h.shape
            src, dst, typ, mask = _mirrored_edges(torch, graph, node_mask, self.edge_type_count)
            src_h = _gather_nodes(torch, self.k_proj(h), src)
            msg = self.activation(src_h + self.edge_embedding(typ)) * mask.unsqueeze(-1).to(h.dtype)
            out = torch.zeros((b, n, d), dtype=h.dtype, device=h.device)
            out.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, d), msg)
            return self.out((1.0 + self.eps) * h + out)

    class DenseAttention(nn.Module):
        def __init__(self, arch: ArchConfig) -> None:
            super().__init__()
            self.heads = arch.heads
            self.head_dim = arch.dim // arch.heads
            self.q = nn.Linear(arch.dim, arch.dim, bias=False)
            self.k = nn.Linear(arch.dim, arch.dim, bias=False)
            self.v = nn.Linear(arch.dim, arch.dim, bias=False)
            self.o = nn.Linear(arch.dim, arch.dim, bias=False)

        def forward(self, query, source, source_mask):
            b, q_len, d = query.shape
            k_len = source.shape[1]
            q = self.q(query).reshape(b, q_len, self.heads, self.head_dim).transpose(1, 2)
            k = self.k(source).reshape(b, k_len, self.heads, self.head_dim).transpose(1, 2)
            v = self.v(source).reshape(b, k_len, self.heads, self.head_dim).transpose(1, 2)
            score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if source_mask is not None:
                score = score.masked_fill(~source_mask.unsqueeze(1).unsqueeze(2), -1.0e9)
            weight = torch.softmax(score, dim=-1)
            out = torch.matmul(weight, v).transpose(1, 2).reshape(b, q_len, d)
            return self.o(out)

    return GraphModel


@lru_cache(maxsize=1)
def _pair_serving_model_classes():
    nn = _torch().nn

    class PairServingModel(nn.Module):
        def __init__(self, model: object) -> None:
            super().__init__()
            self.model = model

        def forward(self, batch: GraphBatchTensors, opponent_readout: object):
            return self.model.forward_with_opponent(batch, opponent_readout)

    class OpponentReadoutModel(nn.Module):
        def __init__(self, model: object) -> None:
            super().__init__()
            self.model = model

        def forward(self, batch: GraphBatchTensors):
            return self.model.opponent_readout(batch)

    return PairServingModel, OpponentReadoutModel


def _remaining_budget_position(torch: object, position: object):
    remaining = (position[..., 2] - position[..., 1] * position[..., 3]).clamp(0.0, 1.0)
    return torch.stack((remaining, position[..., 3]), dim=-1)


def _self_graph(batch: GraphBatchTensors) -> GraphStateTensors:
    return GraphStateTensors(
        node_count=batch.node_count,
        node_tokens=batch.node_tokens,
        node_attrs=batch.node_attrs,
        edge_count=batch.edge_count,
        edge_src=batch.edge_src,
        edge_dst=batch.edge_dst,
        edge_type=batch.edge_type,
        position=batch.position,
    )


def _opponent_graph(batch: GraphBatchTensors) -> GraphStateTensors:
    return GraphStateTensors(
        node_count=batch.opponent_node_count,
        node_tokens=batch.opponent_node_tokens,
        node_attrs=batch.opponent_node_attrs,
        edge_count=batch.opponent_edge_count,
        edge_src=batch.opponent_edge_src,
        edge_dst=batch.opponent_edge_dst,
        edge_type=batch.opponent_edge_type,
        position=batch.opponent_position,
    )


def _orient_graph_pair(
    torch: object,
    own: GraphStateTensors,
    opponent: GraphStateTensors,
    flip: object,
) -> tuple[GraphStateTensors, GraphStateTensors]:
    def select(left: object, right: object) -> object:
        mask = flip.reshape((flip.shape[0],) + (1,) * (left.ndim - 1))
        return torch.where(mask, right, left)

    return (
        GraphStateTensors(*(select(left, right) for left, right in zip(own, opponent, strict=True))),
        GraphStateTensors(*(select(right, left) for left, right in zip(own, opponent, strict=True))),
    )


def _concat_graphs(
    torch: object,
    left: GraphStateTensors,
    right: GraphStateTensors,
) -> GraphStateTensors:
    return GraphStateTensors(
        *(torch.cat((left_value, right_value), dim=0) for left_value, right_value in zip(left, right, strict=True))
    )


def _valid_edges(torch: object, graph: GraphStateTensors, node_mask: object):
    e = graph.edge_src.shape[1]
    edge_index = torch.arange(e, device=graph.edge_src.device)
    mask = edge_index.unsqueeze(0) < graph.edge_count.unsqueeze(1)
    mask = mask & (graph.edge_src < graph.node_count.unsqueeze(1))
    mask = mask & (graph.edge_dst < graph.node_count.unsqueeze(1))
    dummy = torch.arange(e, dtype=graph.edge_src.dtype, device=graph.edge_src.device)
    dummy = (dummy % node_mask.shape[1]).unsqueeze(0)
    src = (
        torch.where(mask, graph.edge_src, dummy)
        .clamp(0, node_mask.shape[1] - 1)
        .to(torch.int64)
    )
    dst = (
        torch.where(mask, graph.edge_dst, dummy)
        .clamp(0, node_mask.shape[1] - 1)
        .to(torch.int64)
    )
    return src, dst, mask


def _mirrored_edges(
    torch: object,
    graph: GraphStateTensors,
    node_mask: object,
    edge_type_count: int,
):
    e = graph.edge_src.shape[1]
    edge_index = torch.arange(e, device=graph.edge_src.device)
    base_mask = edge_index.unsqueeze(0) < graph.edge_count.unsqueeze(1)
    src_valid = graph.edge_src < graph.node_count.unsqueeze(1)
    dst_valid = graph.edge_dst < graph.node_count.unsqueeze(1)
    type_valid = graph.edge_type < edge_type_count
    base_mask = base_mask & src_valid & dst_valid & type_valid
    dummy_node = torch.arange(e, dtype=graph.edge_src.dtype, device=graph.edge_src.device)
    dummy_node = (dummy_node % node_mask.shape[1]).unsqueeze(0)
    base_src = torch.where(base_mask, graph.edge_src, dummy_node)
    base_dst = torch.where(base_mask, graph.edge_dst, dummy_node)
    dummy_type = torch.arange(e, dtype=graph.edge_type.dtype, device=graph.edge_type.device)
    dummy_type = (dummy_type % edge_type_count).unsqueeze(0)
    base_type = torch.where(base_mask, graph.edge_type, dummy_type)
    src = (
        torch.cat((base_src, base_dst), dim=1)
        .clamp(0, node_mask.shape[1] - 1)
        .to(torch.int64)
    )
    dst = (
        torch.cat((base_dst, base_src), dim=1)
        .clamp(0, node_mask.shape[1] - 1)
        .to(torch.int64)
    )
    typ = torch.cat((base_type, base_type + edge_type_count), dim=1).clamp(
        0, max(0, 2 * edge_type_count - 1)
    )
    mask = torch.cat((base_mask, base_mask), dim=1)
    return src, dst, typ, mask


def _gather_nodes(torch: object, h: object, index: object):
    d = h.shape[-1]
    index = index.to(torch.int64)
    return torch.gather(h, 1, index.unsqueeze(-1).expand(-1, -1, d))


def _whittle_node_features(torch: object, graph: GraphStateTensors, node_mask: object):
    tokens = graph.node_tokens.to(torch.int64)
    op = torch.zeros_like(tokens)
    op = torch.where(tokens == 17, 1, op)
    op = torch.where(tokens == 18, 2, op)
    op = torch.where(tokens == 20, 3, op)
    op = torch.where(tokens == 21, 4, op)
    op = torch.where(tokens == 19, 5, op)
    op = torch.where(tokens == 22, 6, op)
    op_features = torch.nn.functional.one_hot(op, num_classes=7).to(graph.node_attrs.dtype)

    input_mask = (tokens >= 1) & (tokens <= 6)
    input_slots = torch.nn.functional.one_hot((tokens - 1).clamp(0, 5), num_classes=6)
    input_slots = input_slots.to(graph.node_attrs.dtype) * input_mask.unsqueeze(-1)
    features = torch.cat((op_features, input_slots, graph.node_attrs), dim=-1)
    return features * node_mask.unsqueeze(-1)


def _action_feat_dim(arch: ArchConfig) -> int:
    # kind + subject encoding + readout; match mode contributes [root,
    # attended] where mean mode contributes one pooled block.
    subject_dim = arch.dim * 2 if arch.subject_encoding == "match" else arch.dim
    prior_dim = 1 if arch.action_encoding == "kind_prior" else 0
    return arch.dim * 2 + subject_dim + prior_dim


def _gather_subjects(
    torch: object,
    h: object,
    node_mask: object,
    action_subjects: object,
    subject_count: object,
):
    b, n, d = h.shape
    a = action_subjects.shape[1]
    s = action_subjects.shape[2]
    subject_index = torch.arange(s, device=h.device)
    valid = subject_index.reshape(1, 1, s) < subject_count.unsqueeze(-1)
    valid = valid & (action_subjects < node_mask.sum(dim=1).reshape(b, 1, 1))
    dummy = torch.arange(a * s, dtype=action_subjects.dtype, device=h.device)
    dummy = (dummy % n).reshape(1, a, s)
    safe = torch.where(valid, action_subjects, dummy).clamp(0, n - 1)
    # Gather over h's node dim directly: routing the gather through an
    # (b, a, n, d) expand made the backward materialize that full tensor
    # (tens of GiB at wide action masks) before reducing it.
    flat = safe.to(torch.int64).reshape(b, a * s, 1).expand(b, a * s, d)
    gathered = torch.gather(h, 1, flat).reshape(b, a, s, d)
    return gathered, valid


def _subject_pool(torch: object, h: object, node_mask: object, action_subjects: object, subject_count: object):
    gathered, valid = _gather_subjects(torch, h, node_mask, action_subjects, subject_count)
    weight = valid.unsqueeze(-1).to(h.dtype)
    denom = weight.sum(dim=2).clamp_min(1.0)
    return (gathered * weight).sum(dim=2) / denom


def _mlp(nn: object, in_dim: int, hidden_dim: int, out_dim: int, activation: str, dropout: float):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        _activation_module(nn, activation),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _activation_module(nn: object, activation: str):
    if activation == "gelu":
        return nn.GELU()
    if activation == "relu":
        return nn.ReLU()
    raise ValueError("unsupported activation")


def _activation(functional: object, activation: str):
    if activation == "gelu":
        return functional.gelu
    if activation == "relu":
        return functional.relu
    raise ValueError("unsupported activation")


def _torch():
    return importlib.import_module("torch")


def _update_chunk(hasher: object, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)


def _int(value: dict[str, object], name: str, default: int | None = None) -> int:
    field = value[name] if default is None else value.get(name, default)
    if not isinstance(field, int):
        raise ValueError(f"{name} must be an integer")
    return field


def _float(value: dict[str, object], name: str, default: float | None = None) -> float:
    field = value[name] if default is None else value.get(name, default)
    if not isinstance(field, (float, int)):
        raise ValueError(f"{name} must be numeric")
    return float(field)


def _str(value: dict[str, object], name: str, default: str | None = None) -> str:
    field = value.get(name, default)
    if not isinstance(field, str):
        raise ValueError(f"{name} must be a string")
    return field
