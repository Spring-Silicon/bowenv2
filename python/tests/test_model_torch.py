from __future__ import annotations

import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from gz.codec import BatchView, FeatureSchemaConfig
from gz.model.exphormer import (
    ArchConfig,
    BatchStager,
    build_model,
    build_pair_serving_models,
    initialize_policy,
    initialize_value,
)
from python.tests.test_codec import _bf16, _layout, _u16, make_batch

torch = pytest.importorskip("torch")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_zero_value_initializer_is_neutral_and_trainable_under_mirroring() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(
        view,
        node_vocab_size=7,
        edge_type_count=2,
        action_kind_vocab_size=8,
    )
    model = build_model(
        schema,
        ArchConfig(
            dim=16,
            layers=1,
            heads=4,
            ffn_dim=32,
            value_input="pair",
            value_activation="tanh",
        ),
    )
    policy_before = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("value.")
    }
    hidden_before = {
        name: value.clone()
        for name, value in model.value.state_dict().items()
        if not name.startswith("3.")
    }

    initialize_value(model, "zero")

    assert all(torch.count_nonzero(value) == 0 for value in model.value[-1].state_dict().values())
    for name, expected in hidden_before.items():
        torch.testing.assert_close(model.value.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in policy_before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    left = torch.randn(8, model.arch.dim)
    right = torch.randn(8, model.arch.dim)
    target = torch.tensor([1.0, -1.0] * 4)
    canonical = torch.tanh(model.value(torch.cat((left, right), dim=1))).squeeze(1)
    mirrored = torch.tanh(model.value(torch.cat((right, left), dim=1))).squeeze(1)
    assert torch.count_nonzero(canonical) == 0
    loss = 0.5 * (canonical - target).square().mean()
    loss = loss + 0.5 * (mirrored + target).square().mean()
    loss.backward()
    assert model.value[-1].weight.grad is not None
    assert torch.count_nonzero(model.value[-1].weight.grad) > 0


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_padding_invariance(aggregation: str) -> None:
    small = BatchView.parse(make_batch(attr_dim=1, capacity=2))
    padded = BatchView.parse(make_batch(attr_dim=1, capacity=3))
    schema = schema_for_view(padded, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).eval()

    small_values, small_logits = run_model(model, schema, small)
    padded_values, padded_logits = run_model(model, schema, padded)

    torch.testing.assert_close(padded_values[:2], small_values, rtol=0, atol=1e-7)
    torch.testing.assert_close(padded_logits[:2], small_logits, rtol=0, atol=1e-7)


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_batch_independence(aggregation: str) -> None:
    original = BatchView.parse(make_batch(attr_dim=1))
    mutated_bytes = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    struct.pack_into("<I", mutated_bytes, layout["node_count"] + 4, 3)
    struct.pack_into("<H", mutated_bytes, layout["node_tokens"] + 3 * 2, 6)
    struct.pack_into("<H", mutated_bytes, layout["node_tokens"] + 4 * 2, 5)
    struct.pack_into("<H", mutated_bytes, layout["node_tokens"] + 5 * 2, 4)
    mutated = BatchView.parse(mutated_bytes)
    schema = schema_for_view(original, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, original)
    mutated_values, mutated_logits = run_model(model, schema, mutated)

    torch.testing.assert_close(mutated_values[:1], values[:1], rtol=0, atol=0)
    torch.testing.assert_close(mutated_logits[:1], logits[:1], rtol=0, atol=0)


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_masks_reject_padding_edges_and_subjects(aggregation: str) -> None:
    baseline = BatchView.parse(make_batch(attr_dim=0))
    mutated_bytes = bytearray(make_batch(attr_dim=0))
    layout = _layout(2, 3, 2, 3, 2, 0)
    struct.pack_into("<I", mutated_bytes, layout["edge_count"], 2)
    struct.pack_into("<H", mutated_bytes, layout["edge_src"] + 2, 2)
    struct.pack_into("<H", mutated_bytes, layout["edge_dst"] + 2, 1)
    mutated_bytes[layout["edge_type"] + 1] = 1
    mutated_bytes[layout["subject_count"]] = 2
    struct.pack_into("<H", mutated_bytes, layout["action_subjects"] + 2, 2)
    mutated = BatchView.parse(mutated_bytes)
    schema = schema_for_view(baseline, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, baseline)
    mutated_values, mutated_logits = run_model(model, schema, mutated)

    torch.testing.assert_close(mutated_values[:1], values[:1], rtol=0, atol=0)
    torch.testing.assert_close(mutated_logits[:1, :1], logits[:1, :1], rtol=0, atol=0)
    assert torch.isfinite(logits[0, 1])


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_masked_padding_contents_do_not_change_outputs(aggregation: str) -> None:
    baseline = BatchView.parse(make_batch(attr_dim=0))
    mutated_bytes = bytearray(make_batch(attr_dim=0))
    layout = _layout(2, 3, 2, 3, 2, 0)
    _u16(mutated_bytes, layout["edge_src"], [0, 2, 2, 1])
    _u16(mutated_bytes, layout["edge_dst"], [1, 2, 1, 2])
    mutated_bytes[layout["edge_type"] : layout["edge_type"] + 4] = bytes(
        [1, 0, 1, 0]
    )
    _u16(mutated_bytes, layout["opponent_edge_src"], [2, 1, 1, 2])
    _u16(mutated_bytes, layout["opponent_edge_dst"], [1, 2, 0, 2])
    mutated_bytes[
        layout["opponent_edge_type"] : layout["opponent_edge_type"] + 4
    ] = bytes([1, 0, 1, 0])
    _u16(
        mutated_bytes,
        layout["action_subjects"],
        [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    )
    mutated = BatchView.parse(mutated_bytes)
    schema = schema_for_view(
        baseline,
        node_vocab_size=7,
        edge_type_count=2,
        action_kind_vocab_size=8,
    )
    model = build_model(schema, make_arch(aggregation)).eval()

    expected = run_model(model, schema, baseline)
    actual = run_model(model, schema, mutated)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match", "v2"])
def test_compile_fullgraph_and_no_recompile_for_row_count_change(aggregation: str) -> None:
    # Each variant compiles a fresh model; without a reset the process-wide
    # dynamo cache fills up and later in-process compiles (the serving
    # backend tests) fall over.
    torch._dynamo.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    view = BatchView.parse(make_batch(attr_dim=1))
    changed = bytearray(make_batch(attr_dim=1))
    struct.pack_into("<I", changed, 44, 1)
    changed_view = BatchView.parse(changed)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).to(device).eval()
    stager = BatchStager(schema, view.batch_capacity, device)
    tensors = stager.copy(view)
    eager = model(tensors)
    compiled = torch.compile(model, fullgraph=True)
    actual = compiled(tensors)

    torch.testing.assert_close(actual[0], eager[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[1], eager[1], rtol=1e-2, atol=1e-2)

    counter = torch._dynamo.testing.CompileCounter()
    counted = torch.compile(model, backend=counter, fullgraph=True)
    counted(stager.copy(view))
    counted(stager.copy(changed_view))
    assert counter.frame_count == 1


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_expander_fixture_flows_through_model(aggregation: str) -> None:
    view = BatchView.parse((FIXTURES / "batch_expander.gzfb").read_bytes())
    schema = schema_for_view(view, node_vocab_size=8, edge_type_count=3, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, view)

    assert values.shape == (2,)
    assert logits.shape == (2, 4)
    assert torch.isfinite(values[: view.row_count]).all()
    assert torch.isfinite(logits[: view.row_count, : view.action_count[0]]).all()
    assert view.edge_type[0, : view.edge_count[0]].tolist().count(2) == 3


def test_scalar_value_head_uses_opponent_features() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    changed_bytes = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    changed_bytes[layout["opponent_present"]] = 0
    changed_bytes[layout["opponent_present"] + 1] = 0
    changed = BatchView.parse(changed_bytes)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        aggregation="attention",
        value_input="scalar",
    )
    model = build_model(schema, arch).eval()
    with torch.no_grad():
        for param in model.value.parameters():
            param.zero_()
        model.value[0].weight[0, arch.dim + 1] = 1.0
        model.value[3].weight[0, 0] = 1.0

    values, logits = run_model(model, schema, view)
    changed_values, changed_logits = run_model(model, schema, changed)

    assert values.shape == (2,)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(changed_logits, logits, rtol=0, atol=0)
    assert not torch.equal(changed_values, values)


def test_pair_value_head_uses_opponent_state() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    changed_bytes = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    changed_bytes[layout["opponent_state_present"]] = 0
    changed_bytes[layout["opponent_state_present"] + 1] = 0
    changed = BatchView.parse(changed_bytes)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        aggregation="attention",
        value_input="pair",
    )
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, view)
    changed_values, changed_logits = run_model(model, schema, changed)

    assert values.shape == (2,)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(changed_logits, logits, rtol=0, atol=0)
    assert not torch.equal(changed_values, values)


def test_pair_serving_split_matches_full_forward() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        aggregation="attention",
        value_input="pair",
    )
    model = build_model(schema, arch).eval()
    tensors = tensors_of(schema, view)
    serving, opponent = build_pair_serving_models(model)

    with torch.inference_mode():
        expected = model(tensors)
        actual = serving(tensors, opponent(tensors))

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_pair_serving_split_compiles_fullgraph() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = "python"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from python.tests.test_model_torch import _assert_pair_serving_split_compiles; "
            "_assert_pair_serving_split_compiles()",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_pair_serving_split_compiles() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        name="gz-graph-v2",
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        aggregation="attention",
        value_input="pair",
        policy_head="pointer",
        position_encoding="remaining_budget",
    )
    model = build_model(schema, arch).to(device).eval()
    tensors = BatchStager(schema, view.batch_capacity, device).copy(view)
    serving, opponent = build_pair_serving_models(model)
    compiled_serving = torch.compile(serving, fullgraph=True, mode="reduce-overhead")
    compiled_opponent = torch.compile(opponent, fullgraph=True, mode="reduce-overhead")

    with torch.inference_mode():
        opponent_readout = compiled_opponent(tensors).clone()
        actual = compiled_serving(tensors, opponent_readout)
        expected = model(tensors)

    torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=1e-2)


def test_batch_stager_keeps_indexes_int32() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    tensors = BatchStager(schema, view.batch_capacity, "cpu").copy(view)

    for name in (
        "node_count",
        "node_tokens",
        "edge_count",
        "edge_src",
        "edge_dst",
        "edge_type",
        "action_count",
        "action_kind",
        "subject_count",
        "action_subjects",
        "opponent_node_count",
        "opponent_node_tokens",
        "opponent_edge_count",
        "opponent_edge_src",
        "opponent_edge_dst",
        "opponent_edge_type",
    ):
        assert getattr(tensors, name).dtype == torch.int32


def test_pointer_policy_head_bounded_and_masks_padded_actions() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    changed_bytes = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    # Mutate action slots past each row's action_count (row 0 counts 2,
    # row 1 counts 1): padded slots must not leak through the glimpse.
    _u16(changed_bytes, layout["action_kind"] + 2 * 2, [3])
    _bf16(changed_bytes, layout["action_prior"] + 2 * 2, [0.75])
    _u16(changed_bytes, layout["action_kind"] + 4 * 2, [5])
    _bf16(changed_bytes, layout["action_prior"] + 4 * 2, [0.5])
    changed = BatchView.parse(changed_bytes)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        aggregation="attention",
        policy_head="pointer",
    )
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, view)
    changed_values, changed_logits = run_model(model, schema, changed)

    assert values.shape == (2,)
    assert logits.shape == (2, 3)
    assert logits.abs().max() <= 10.0
    torch.testing.assert_close(changed_values, values, rtol=0, atol=0)
    torch.testing.assert_close(changed_logits[0, :2], logits[0, :2], rtol=0, atol=0)
    torch.testing.assert_close(changed_logits[1, :1], logits[1, :1], rtol=0, atol=0)
    assert ArchConfig.from_dict(make_arch("attention").to_dict()).policy_head == "mlp"
    legacy = {k: v for k, v in make_arch("attention").to_dict().items() if k != "policy_head"}
    assert ArchConfig.from_dict(legacy).policy_head == "mlp"


@pytest.mark.parametrize("profile", ["graphzero", "whittlezero"])
def test_neutral_pointer_initialization_is_uniform_and_trainable(profile: str) -> None:
    attr_dim = 3 if profile == "whittlezero" else 1
    view = BatchView.parse(make_batch(attr_dim=attr_dim))
    schema = schema_for_view(
        view,
        name="whittle-v2" if profile == "whittlezero" else "test",
        node_vocab_size=23 if profile == "whittlezero" else 7,
        edge_type_count=3 if profile == "whittlezero" else 2,
        action_kind_vocab_size=46 if profile == "whittlezero" else 8,
    )
    kwargs = {}
    if profile == "whittlezero":
        kwargs = {
            "trunk": "sage",
            "subject_encoding": "match",
            "position_encoding": "policy_budget",
            "action_encoding": "candidate_only",
            "profile": "whittlezero",
        }
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        policy_head="pointer",
        **kwargs,
    )
    model = build_model(schema, arch)

    initialize_policy(model, "neutral")
    logits = model.policy_logits(tensors_of(schema, view))
    loss = torch.nn.functional.cross_entropy(logits[0, :2].unsqueeze(0), torch.tensor([0]))
    loss.backward()
    key = model.policy.policy_attention.key if profile == "whittlezero" else model.policy.pointer_key

    torch.testing.assert_close(logits[0, :2], torch.zeros(2), rtol=0, atol=0)
    assert torch.count_nonzero(key.weight) == 0
    assert key.weight.grad is not None
    assert key.weight.grad.abs().sum() > 0


def test_tanh_value_activation_bounds_values() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, value_activation="tanh")
    model = build_model(schema, arch).eval()

    values, _ = run_model(model, schema, view)

    assert values.abs().max() < 1.0
    assert ArchConfig.from_dict(arch.to_dict()) == arch
    legacy = {k: v for k, v in arch.to_dict().items() if k != "value_activation"}
    assert ArchConfig.from_dict(legacy).value_activation == "logit"


def test_hl_gauss_value_head_emits_logits_and_decodes_to_support() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        value_head="hl_gauss",
        value_bins=101,
        value_min=-1.0,
        value_max=1.0,
        value_sigma_ratio=0.75,
    )
    model = build_model(schema, arch).eval()

    value_logits, policy_logits = run_model(model, schema, view)
    decoded = model.decode_value(value_logits)

    assert value_logits.shape == (view.batch_capacity, 101)
    assert policy_logits.shape == (view.batch_capacity, view.max_actions)
    assert decoded.shape == (view.batch_capacity,)
    assert torch.isfinite(value_logits).all()
    assert torch.all((-1.0 <= decoded) & (decoded <= 1.0))
    assert model.value_bin_edges.shape == (102,)
    assert model.value_bin_centers.shape == (101,)

    legacy = {
        key: value
        for key, value in arch.to_dict().items()
        if key not in {"value_head", "value_bins", "value_min", "value_max", "value_sigma_ratio"}
    }
    parsed = ArchConfig.from_dict(legacy)
    assert parsed.value_head == "scalar"
    assert (parsed.value_bins, parsed.value_min, parsed.value_max, parsed.value_sigma_ratio) == (
        101,
        -1.0,
        1.0,
        0.75,
    )


def test_policy_budget_encoding_changes_policy_without_leaking_into_value() -> None:
    base = bytearray(make_batch(attr_dim=1))
    budget_changed = bytearray(base)
    nonbudget_changed = bytearray(base)
    layout = _layout(2, 3, 2, 3, 2, 1)
    _bf16(budget_changed, layout["position"], [2.0, 3.0, 0.25, 0.125, 1.0, 0.0, 1.0, 0.5])
    _bf16(nonbudget_changed, layout["position"], [9.0, 8.0, 0.75, 0.75, 1.0, 0.0, 1.0, 0.5])
    view = BatchView.parse(base)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        value_input="pair",
        policy_head="pointer",
        position_encoding="policy_budget",
    )
    model = build_model(schema, arch).eval()
    with torch.no_grad():
        model.position_proj.weight.fill_(1.0)
        model.position_proj.bias.zero_()
        base_value, base_logits = run_model(model, schema, view)
        budget_value, budget_logits = run_model(model, schema, BatchView.parse(budget_changed))
        nonbudget_value, nonbudget_logits = run_model(model, schema, BatchView.parse(nonbudget_changed))

    torch.testing.assert_close(budget_value, base_value, rtol=0, atol=0)
    torch.testing.assert_close(nonbudget_value, base_value, rtol=0, atol=0)
    assert not torch.equal(budget_logits[0, :2], base_logits[0, :2])
    torch.testing.assert_close(nonbudget_logits, base_logits, rtol=0, atol=0)
    assert ArchConfig.from_dict(arch.to_dict()) == arch


def test_v2_arch_requires_remaining_budget_exphormer_pointer() -> None:
    arch = ArchConfig(
        name="gz-graph-v2",
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        policy_head="pointer",
        position_encoding="remaining_budget",
    )

    assert ArchConfig.from_dict(arch.to_dict()) == arch
    with pytest.raises(ValueError, match="remaining_budget.*gz-graph-v2"):
        replace(arch, name="gz-graph-v1")
    with pytest.raises(ValueError, match="gz-graph-v2 requires"):
        replace(arch, position_encoding="shared")


def test_v2_position_uses_effective_remaining_budget_not_raw_root_step() -> None:
    baseline = bytearray(make_batch(attr_dim=1))
    equivalent = bytearray(baseline)
    exhausted = bytearray(baseline)
    layout = _layout(2, 3, 2, 3, 2, 1)
    _bf16(baseline, layout["position"], [4.0, 2.0, 0.75, 0.125, 1.0, 0.0, 1.0, 0.5])
    _bf16(equivalent, layout["position"], [4096.0, 0.0, 0.5, 0.125, 1.0, 0.0, 1.0, 0.5])
    _bf16(exhausted, layout["position"], [4.0, 8.0, 0.5, 0.125, 1.0, 0.0, 1.0, 0.5])
    view = BatchView.parse(baseline)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        name="gz-graph-v2",
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        policy_head="pointer",
        position_encoding="remaining_budget",
    )
    model = build_model(schema, arch).eval()
    projected_inputs = []
    hook = model.position_proj.register_forward_pre_hook(
        lambda _module, args: projected_inputs.append(args[0].detach().clone())
    )
    with torch.no_grad():
        baseline_output = run_model(model, schema, view)
        equivalent_output = run_model(model, schema, BatchView.parse(equivalent))
        run_model(model, schema, BatchView.parse(exhausted))
    hook.remove()

    torch.testing.assert_close(projected_inputs[0][0], torch.tensor([0.5, 0.125]), rtol=0, atol=0)
    torch.testing.assert_close(projected_inputs[1][0], projected_inputs[0][0], rtol=0, atol=0)
    torch.testing.assert_close(projected_inputs[2][0], torch.tensor([0.0, 0.125]), rtol=0, atol=0)
    torch.testing.assert_close(equivalent_output[0], baseline_output[0], rtol=0, atol=0)
    torch.testing.assert_close(equivalent_output[1], baseline_output[1], rtol=0, atol=0)


def test_v2_normalizes_trunk_and_final_pointer_operands() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        name="gz-graph-v2",
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        policy_head="pointer",
        position_encoding="remaining_budget",
    )
    model = build_model(schema, arch).eval()
    captured = {}

    def capture(name: str):
        return lambda _module, _args, output: captured.__setitem__(name, output.detach())

    hooks = [
        model.node_output_norm.register_forward_hook(capture("nodes")),
        model.global_output_norm.register_forward_hook(capture("globals")),
        model.policy.pointer_board_norm.register_forward_hook(capture("board")),
        model.policy.pointer_token_norm.register_forward_hook(capture("tokens")),
    ]
    with torch.no_grad():
        run_model(model, schema, view)
    for hook in hooks:
        hook.remove()

    for output in captured.values():
        assert bool(torch.isfinite(output).all())
        torch.testing.assert_close(
            output.mean(dim=-1),
            torch.zeros_like(output[..., 0]),
            rtol=0,
            atol=1.0e-5,
        )
        assert float(output.square().mean(dim=-1).max()) <= 1.0001


def test_v2_raw_root_step_cannot_change_outputs_or_gradients() -> None:
    baseline = bytearray(make_batch(attr_dim=1))
    large_root = bytearray(baseline)
    layout = _layout(2, 3, 2, 3, 2, 1)
    _bf16(baseline, layout["position"], [4.0, 2.0, 0.75, 0.125, 1.0, 0.0, 1.0, 0.5])
    _bf16(large_root, layout["position"], [65536.0, 2.0, 0.75, 0.125, 1.0, 0.0, 1.0, 0.5])
    view = BatchView.parse(baseline)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        name="gz-graph-v2",
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        policy_head="pointer",
        position_encoding="remaining_budget",
    )
    model = build_model(schema, arch).train()

    def gradients(batch: BatchView):
        model.zero_grad(set_to_none=True)
        value, logits = run_model(model, schema, batch)
        action_index = torch.arange(logits.shape[1])
        action_mask = action_index.unsqueeze(0) < tensors_of(schema, batch).action_count.unsqueeze(1)
        loss = value.float().square().mean() + logits[action_mask].float().square().mean()
        loss.backward()
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    baseline_gradients = gradients(view)
    large_root_gradients = gradients(BatchView.parse(large_root))

    assert baseline_gradients.keys() == large_root_gradients.keys()
    for name, baseline_gradient in baseline_gradients.items():
        assert bool(torch.isfinite(baseline_gradient).all()), name
        torch.testing.assert_close(large_root_gradients[name], baseline_gradient, rtol=0, atol=0)


def test_candidate_only_encoding_uses_zero_stop_token_and_ignores_static_prior() -> None:
    base = bytearray(make_batch(attr_dim=1))
    changed_prior = bytearray(base)
    layout = _layout(2, 3, 2, 3, 2, 1)
    _bf16(changed_prior, layout["action_prior"], [0.75, -0.5, 0.25, 0.0])
    view = BatchView.parse(base)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        policy_head="pointer",
        subject_encoding="match",
        action_encoding="candidate_only",
    )
    model = build_model(schema, arch).eval()
    captured = []
    hook = model.policy.token_proj.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    with torch.no_grad():
        model.kind_embedding.weight.fill_(1.0)
        _, baseline = run_model(model, schema, view)
        _, changed = run_model(model, schema, BatchView.parse(changed_prior))
    hook.remove()

    # Row 0 has one engine candidate followed by STOP. Whittle's pointer
    # input gives STOP no rule token; graph context remains in the last block.
    assert captured[0].shape[-1] == 4 * arch.dim
    torch.testing.assert_close(captured[0][0, 0, : arch.dim], torch.ones(arch.dim))
    torch.testing.assert_close(captured[0][0, 1, : 3 * arch.dim], torch.zeros(3 * arch.dim))
    torch.testing.assert_close(changed, baseline, rtol=0, atol=0)
    assert ArchConfig.from_dict(arch.to_dict()) == arch


def test_whittlezero_profile_ignores_expander_edges() -> None:
    baseline = bytearray(make_batch(attr_dim=3))
    changed = bytearray(baseline)
    layout = _layout(2, 3, 2, 3, 2, 3)
    struct.pack_into("<I", baseline, layout["edge_count"], 2)
    struct.pack_into("<I", changed, layout["edge_count"], 2)
    _u16(baseline, layout["edge_src"], [0, 0])
    _u16(baseline, layout["edge_dst"], [1, 2])
    _u16(changed, layout["edge_src"], [0, 1])
    _u16(changed, layout["edge_dst"], [1, 0])
    baseline[layout["edge_type"] + 1] = 2
    changed[layout["edge_type"] + 1] = 2
    view = BatchView.parse(baseline)
    schema = schema_for_view(
        view,
        name="whittle-v2",
        node_vocab_size=23,
        edge_type_count=3,
        action_kind_vocab_size=8,
    )
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        trunk="sage",
        sage_layers=1,
        policy_head="pointer",
        subject_encoding="match",
        position_encoding="policy_budget",
        action_encoding="candidate_only",
        profile="whittlezero",
        value_hidden=32,
    )
    model = build_model(schema, arch).eval()

    baseline_out = run_model(model, schema, view)
    changed_out = run_model(model, schema, BatchView.parse(changed))

    torch.testing.assert_close(changed_out[0], baseline_out[0], rtol=0, atol=0)
    torch.testing.assert_close(changed_out[1], baseline_out[1], rtol=0, atol=0)


def test_whittlezero_profile_matches_reference_parameter_count() -> None:
    view = BatchView.parse(make_batch(attr_dim=3))
    schema = schema_for_view(
        view,
        name="whittle-v2",
        node_vocab_size=23,
        edge_type_count=3,
        action_kind_vocab_size=46,
    )
    schema = replace(schema, max_subjects=8)
    arch = ArchConfig(
        dim=128,
        layers=3,
        heads=4,
        ffn_dim=512,
        dropout=0.1,
        trunk="sage",
        sage_layers=3,
        policy_head="pointer",
        value_input="pair",
        value_activation="tanh",
        subject_encoding="match",
        position_encoding="policy_budget",
        action_encoding="candidate_only",
        profile="whittlezero",
        value_hidden=256,
    )

    model = build_model(schema, arch)

    assert sum(parameter.numel() for parameter in model.parameters()) == 1_311_746


def test_match_encoding_distinguishes_subject_order() -> None:
    # Two candidates over the same node set in different roles must score
    # differently under match encoding; the mean pool aliases them.
    base = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    base[layout["subject_count"]] = 2
    _u16(base, layout["action_subjects"], [0, 1])
    swapped = bytearray(base)
    _u16(swapped, layout["action_subjects"], [1, 0])
    schema = schema_for_view(BatchView.parse(base), node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)

    match_model = build_model(schema, make_arch("match")).eval()
    mean_model = build_model(schema, make_arch("attention")).eval()
    with torch.no_grad():
        _, match_a = match_model(tensors_of(schema, BatchView.parse(base)))
        _, match_b = match_model(tensors_of(schema, BatchView.parse(swapped)))
        _, mean_a = mean_model(tensors_of(schema, BatchView.parse(base)))
        _, mean_b = mean_model(tensors_of(schema, BatchView.parse(swapped)))

    assert not torch.equal(match_a[0, :1], match_b[0, :1]), "match encoding sees role order"
    torch.testing.assert_close(mean_a[0, :1], mean_b[0, :1], rtol=0, atol=0)


def tensors_of(schema: FeatureSchemaConfig, view: BatchView):
    return BatchStager(schema, view.batch_capacity, "cpu").copy(view)


def test_value_mirror_returns_both_orientations() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, value_input="pair")
    model = build_model(schema, arch).eval()
    tensors = tensors_of(schema, view)
    with torch.no_grad():
        mirrored, _ = model(tensors, value_mirror=True)
        canonical, _ = model(tensors)

    assert mirrored.shape[0] == 2
    torch.testing.assert_close(mirrored[0], canonical, rtol=0, atol=0)
    # The swapped orientation equals canonical exactly when self and
    # opponent readouts coincide; on real batches they differ per row
    # only where an opponent state is present.
    present = tensors.opponent_state_present > 0
    if bool(present.any()):
        assert not torch.equal(mirrored[1][present], mirrored[0][present])


def test_hl_gauss_value_mirror_returns_two_categorical_distributions() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        value_input="pair",
        value_head="hl_gauss",
        value_bins=11,
    )
    model = build_model(schema, arch).eval()
    tensors = tensors_of(schema, view)

    with torch.no_grad():
        mirrored, _ = model(tensors, value_mirror=True)
        canonical, _ = model(tensors)

    assert mirrored.shape == (2, view.batch_capacity, 11)
    assert canonical.shape == (view.batch_capacity, 11)
    torch.testing.assert_close(mirrored[0], canonical, rtol=0, atol=0)


def test_separate_policy_and_value_entrypoints_match_full_eval_forward() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        value_input="pair",
    )
    model = build_model(schema, arch).eval()
    tensors = tensors_of(schema, view)

    with torch.no_grad():
        full_value, full_logits = model(tensors)
        policy_logits = model.policy_logits(tensors)
        value_only = model.value_only(tensors)

    torch.testing.assert_close(policy_logits, full_logits, rtol=0, atol=0)
    torch.testing.assert_close(value_only, full_value, rtol=1e-6, atol=1e-6)


def test_sage_trunk_arch_round_trip_and_legacy_defaults() -> None:
    arch = make_arch("sage")
    assert ArchConfig.from_dict(arch.to_dict()) == arch
    legacy = {k: v for k, v in make_arch("attention").to_dict().items() if k not in {"trunk", "sage_layers"}}
    parsed = ArchConfig.from_dict(legacy)
    assert parsed.trunk == "exphormer"
    assert parsed.sage_layers == 3


def run_model(model: object, schema: FeatureSchemaConfig, view: BatchView):
    stager = BatchStager(schema, view.batch_capacity, "cpu")
    return model(stager.copy(view))


def make_arch(aggregation: str) -> ArchConfig:
    # "sage" is the whittlezero SAGE+transformer trunk and "match" its
    # role-preserving subject encoding; the exphormer trunk variants
    # select the edge aggregation instead.
    if aggregation == "sage":
        return ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, trunk="sage", sage_layers=2)
    if aggregation == "match":
        return ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, subject_encoding="match")
    if aggregation == "v2":
        return ArchConfig(
            name="gz-graph-v2",
            dim=16,
            layers=1,
            heads=4,
            ffn_dim=32,
            dropout=0.0,
            policy_head="pointer",
            position_encoding="remaining_budget",
        )
    return ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, aggregation=aggregation)


def schema_for_view(
    view: BatchView,
    *,
    name: str = "test",
    node_vocab_size: int,
    edge_type_count: int,
    action_kind_vocab_size: int,
) -> FeatureSchemaConfig:
    return FeatureSchemaConfig(
        name=name,
        node_vocab_size=node_vocab_size,
        node_attr_dim=view.dims.node_attr_dim,
        edge_type_count=edge_type_count,
        action_kind_vocab_size=action_kind_vocab_size,
        max_nodes=view.dims.max_nodes,
        max_edges=view.dims.max_edges,
        max_actions=view.dims.max_actions,
        max_subjects=view.dims.max_subjects,
        expander_degree=0,
        expander_seed=0,
    )
