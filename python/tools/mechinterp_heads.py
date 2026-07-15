"""Mechanistic probes for GraphZero pointer-policy and pair-value heads.

The tool combines fixed-root counterfactual batches with rows sampled from a
completed run's retained replay window.  It is intentionally offline: the
checkpoint and replay store are read without modifying the training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as functional

from gz.checkpoints.source import DirectorySource
from gz.checkpoints.weights import load_state_dict
from gz.codec import BatchView
from gz.model import exphormer
from gz.model.exphormer import ArchConfig, BatchStager, build_model
from gz.trainer.sampler import SampleClient, SampleResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoints", default="0,1,5,10,20,40")
    parser.add_argument(
        "--patch-checkpoints",
        default="34,37",
        help="Early,late checkpoint versions used for causal policy patching.",
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--window-rows", type=int, default=16000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--graphzero-bin", type=Path, default=Path("target/release/graphzero"))
    parser.add_argument(
        "--position-features",
        choices=("on", "off"),
        default="off",
        help="Whether fixed probes expose their clock vectors to the model.",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.batch <= 0 or args.batch > 256:
        parser.error("--batch must be in [1, 256]")
    if args.window_rows <= 0:
        parser.error("--window-rows must be positive")
    return args


def load_model(checkpoint_dir: Path, version: int, device: str):
    resolved = DirectorySource(checkpoint_dir).resolve_version(f"version_{version}")
    manifest = resolved.manifest
    arch = ArchConfig.from_dict(dict(manifest.arch_config))
    model = build_model(manifest.feature_schema, arch).to(device)
    model.load_state_dict(load_state_dict(resolved.weights_path))
    model.eval()
    return model, manifest


def stage_view(view: BatchView, schema, device: str, expose_position: bool):
    stager = BatchStager(schema, view.batch_capacity, device, pinned_staging=False)
    batch = stager.copy(view)
    if not expose_position:
        batch = batch._replace(
            position=torch.zeros_like(batch.position),
            opponent_position=torch.zeros_like(batch.opponent_position),
        )
    return batch


def finite_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or left[valid].std() == 0.0 or right[valid].std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def fixed_probe(
    model,
    schema,
    probe_dir: Path,
    meta: dict[str, object],
    device: str,
    expose_position: bool,
) -> dict[str, object]:
    views = {
        name: BatchView.parse((probe_dir / f"{name}.gzfb").read_bytes())
        for name in ("sweep", "opponents", "orientation")
    }
    outputs: dict[str, tuple[BatchView, torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for name, view in views.items():
            batch = stage_view(view, schema, device, expose_position)
            values, logits = model(batch)
            outputs[name] = (view, values[: view.row_count].cpu(), logits[: view.row_count].cpu())

    root_view, sweep_values, sweep_logits = outputs["sweep"]
    action_count = int(root_view.action_count[0])
    root_logits = sweep_logits[0, :action_count]
    root_probs = torch.softmax(root_logits, dim=-1).numpy()
    candidate_meta = meta["candidates"]
    deltas = np.asarray([entry["delta"] for entry in candidate_meta], dtype=np.float64)
    valid_delta = np.isfinite(deltas)
    candidate_probs = root_probs[:-1]
    shrink = deltas < 0.0
    grow = deltas > 0.0
    neutral = deltas == 0.0
    top = np.argsort(candidate_probs)[::-1][:8]

    costs = np.asarray([entry["cost"] for entry in meta["sweep"]], dtype=np.float64)
    values = sweep_values[: len(costs)].numpy()
    opponent_values = outputs["opponents"][1].numpy()
    orientation_values = outputs["orientation"][1].numpy()
    entropy = float(-(root_probs * np.log(np.maximum(root_probs, 1.0e-30))).sum())

    return {
        "policy": {
            "action_count": action_count,
            "entropy": entropy,
            "effective_actions": math.exp(entropy),
            "stop_probability": float(root_probs[-1]),
            "shrink_mass": float(candidate_probs[shrink].sum()),
            "grow_mass": float(candidate_probs[grow].sum()),
            "neutral_mass": float(candidate_probs[neutral].sum()),
            "expected_immediate_cost_delta": float(
                np.sum(candidate_probs[valid_delta] * deltas[valid_delta])
            ),
            "probability_vs_negative_delta_corr": finite_corr(
                candidate_probs[valid_delta], -deltas[valid_delta]
            ),
            "logit_saturation_fraction": float((root_logits.abs() >= 9.5).float().mean()),
            "top_candidates": [
                {
                    "index": int(index),
                    "rule": candidate_meta[int(index)]["rule"],
                    "delta": float(deltas[index]),
                    "probability": float(candidate_probs[index]),
                    "logit": float(root_logits[index]),
                }
                for index in top
            ],
        },
        "value": {
            "cost_correlation": finite_corr(values, -costs),
            "root": float(values[0]),
            "best": float(values[int(costs.argmin())]),
            "worst": float(values[int(costs.argmax())]),
            "best_cost": float(costs.min()),
            "worst_cost": float(costs.max()),
            "opponent_absent": float(opponent_values[0]),
            "opponent_worse": float(opponent_values[1]),
            "opponent_self": float(opponent_values[2]),
            "opponent_best": float(opponent_values[3]),
            "best_vs_root": float(orientation_values[0]),
            "root_vs_best": float(orientation_values[1]),
            "orientation_sum": float(orientation_values[0] + orientation_values[1]),
        },
    }


def parameter_changes(initial, final) -> dict[str, object]:
    prefixes = ("policy", "value", "node_embedding", "kind_embedding", "position_proj")
    groups: dict[str, list[str]] = {name: [] for name in (*prefixes, "trunk")}
    initial_state = initial.state_dict()
    final_state = final.state_dict()
    for name in initial_state:
        group = next((prefix for prefix in prefixes if name.startswith(prefix)), "trunk")
        groups[group].append(name)

    result: dict[str, object] = {}
    for group, names in groups.items():
        base_sq = sum(float(initial_state[name].float().square().sum()) for name in names)
        delta_sq = sum(
            float((final_state[name].float() - initial_state[name].float()).square().sum())
            for name in names
        )
        result[group] = {
            "parameters": sum(initial_state[name].numel() for name in names),
            "initial_norm": math.sqrt(base_sq),
            "update_norm": math.sqrt(delta_sq),
            "relative_update": math.sqrt(delta_sq / base_sq) if base_sq else float("nan"),
        }
    return result


def cross_checkpoint_policy_patching(
    models: dict[int, object],
    probe_dir: Path,
    meta: dict[str, object],
    device: str,
    expose_position: bool,
    early_version: int = 34,
    late_version: int = 37,
) -> dict[str, object] | None:
    if early_version not in models or late_version not in models:
        return None
    early = models[early_version]
    late = models[late_version]
    if early.match is not None or late.match is not None:
        raise ValueError("cross-checkpoint patching currently requires mean subject encoding")
    view = BatchView.parse((probe_dir / "sweep.gzfb").read_bytes())
    batch = stage_view(view, early.schema, device, expose_position)
    encoded = {}
    with torch.no_grad():
        for version, model in ((early_version, early), (late_version, late)):
            encoded[version] = model._encode_graph(exphormer._self_graph(batch))

        configurations = (
            ("early", early_version, early_version, early_version),
            ("late", late_version, late_version, late_version),
            ("late_encoder", late_version, early_version, early_version),
            ("late_action_stack", early_version, late_version, late_version),
            ("late_kind_only", early_version, late_version, early_version),
            ("late_pointer_only", early_version, early_version, late_version),
            ("late_encoder_and_kind", late_version, late_version, early_version),
            ("late_encoder_and_pointer", late_version, early_version, late_version),
        )
        rows = []
        candidate_meta = meta["candidates"]
        deltas = np.asarray([entry["delta"] for entry in candidate_meta], dtype=np.float64)
        for name, encoder_version, kind_version, pointer_version in configurations:
            h, readout, node_mask = encoded[encoder_version]
            kind_model = models[kind_version]
            pointer_model = models[pointer_version]
            kind = kind_model.kind_embedding(
                batch.action_kind.clamp(0, early.schema.action_kind_vocab_size - 1)
            )
            subjects = exphormer._subject_pool(
                torch,
                h,
                node_mask,
                batch.action_subjects,
                batch.subject_count,
            )
            repeated_readout = readout.unsqueeze(1).expand(-1, batch.action_kind.shape[1], -1)
            features = torch.cat(
                (kind, batch.action_prior.unsqueeze(-1), subjects, repeated_readout), dim=-1
            )
            action_index = torch.arange(features.shape[1], device=features.device)
            action_mask = action_index.unsqueeze(0) < batch.action_count.unsqueeze(1)
            logits = pointer_model.policy(readout, features, action_mask)
            action_count = int(batch.action_count[0])
            root_logits = logits[0, :action_count]
            probabilities = torch.softmax(root_logits, dim=-1).cpu().numpy()
            candidate_probabilities = probabilities[:-1]
            entropy = float(
                -(probabilities * np.log(np.maximum(probabilities, 1.0e-30))).sum()
            )
            top = int(candidate_probabilities.argmax())
            rows.append(
                {
                    "name": name,
                    "encoder_version": encoder_version,
                    "kind_version": kind_version,
                    "pointer_version": pointer_version,
                    "shrink_mass": float(candidate_probabilities[deltas < 0.0].sum()),
                    "grow_mass": float(candidate_probabilities[deltas > 0.0].sum()),
                    "expected_immediate_cost_delta": float(
                        np.sum(candidate_probabilities * deltas)
                    ),
                    "entropy": entropy,
                    "top_rule": candidate_meta[top]["rule"],
                    "top_probability": float(candidate_probabilities[top]),
                }
            )
    return {
        "early_version": early_version,
        "late_version": late_version,
        "rows": rows,
    }


def value_weight_channels(model) -> dict[str, float]:
    weight = model.value[0].weight.detach().float()
    dim = weight.shape[1] // 2
    self_weight = weight[:, :dim]
    opponent_weight = weight[:, dim:]
    sum_channel = 0.5 * (self_weight + opponent_weight)
    difference_channel = 0.5 * (self_weight - opponent_weight)
    cosine = functional.cosine_similarity(
        self_weight.flatten(), -opponent_weight.flatten(), dim=0
    )
    return {
        "self_weight_norm": float(self_weight.norm()),
        "opponent_weight_norm": float(opponent_weight.norm()),
        "sum_channel_norm": float(sum_channel.norm()),
        "difference_channel_norm": float(difference_channel.norm()),
        "sum_to_difference_norm_ratio": float(sum_channel.norm() / difference_channel.norm()),
        "self_vs_negative_opponent_cosine": float(cosine),
    }


@contextmanager
def replay_client(
    graphzero_bin: Path, replay_dir: Path, max_batch: int
) -> Iterator[tuple[SampleClient, object]]:
    with tempfile.TemporaryDirectory(prefix="gz-mechinterp-") as temporary:
        socket_path = Path(temporary) / "sample.sock"
        process = subprocess.Popen(
            [
                str(graphzero_bin),
                "replay-serve",
                "--replay-dir",
                str(replay_dir),
                "--socket",
                str(socket_path),
                "--max-batch",
                str(max_batch),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client = SampleClient(socket_path, startup_timeout=30.0)
        deadline = time.monotonic() + 30.0
        try:
            while True:
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout is not None else ""
                    raise RuntimeError(f"replay-serve exited during startup: {output}")
                try:
                    ack = client.connect()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out starting replay-serve")
                    time.sleep(0.05)
            yield client, ack
        finally:
            client.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)


def sample_replay(
    client: SampleClient,
    sample_count: int,
    batch_size: int,
    window_rows: int,
    seed: int,
) -> list[SampleResult]:
    batches = []
    remaining = sample_count
    batch_index = 0
    while remaining:
        size = min(batch_size, remaining)
        batches.append(client.sample(size, window_rows, seed + batch_index))
        remaining -= size
        batch_index += 1
    return batches


def head_features(model, batch):
    graph = exphormer._self_graph(batch)
    h, readout, node_mask = model._encode_graph(graph)
    opponent = model.opponent_readout(batch)
    opponent = torch.where(
        (batch.opponent_state_present > 0).unsqueeze(-1),
        opponent,
        torch.zeros_like(opponent),
    )
    kind = model.kind_embedding(
        batch.action_kind.clamp(0, model.schema.action_kind_vocab_size - 1)
    )
    if model.match is None:
        subjects = exphormer._subject_pool(
            torch,
            h,
            node_mask,
            batch.action_subjects,
            batch.subject_count,
        )
    else:
        subjects = model.match(
            h,
            node_mask,
            batch.action_subjects,
            batch.subject_count,
            kind,
        )
    prior = batch.action_prior.unsqueeze(-1)
    repeated_readout = readout.unsqueeze(1).expand(-1, batch.action_kind.shape[1], -1)
    action_features = torch.cat((kind, prior, subjects, repeated_readout), dim=-1)
    action_index = torch.arange(action_features.shape[1], device=action_features.device)
    action_mask = action_index.unsqueeze(0) < batch.action_count.unsqueeze(1)
    return readout, opponent, action_features, action_mask


def policy_probabilities(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~action_mask, -1.0e9), dim=-1)


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (left + right)
    left_term = left * (left.clamp_min(1.0e-30).log() - midpoint.clamp_min(1.0e-30).log())
    right_term = right * (right.clamp_min(1.0e-30).log() - midpoint.clamp_min(1.0e-30).log())
    return 0.5 * (left_term.sum(dim=-1) + right_term.sum(dim=-1))


def policy_intervention(model, readout, features, action_mask, block: str):
    dim = model.arch.dim
    changed = features.clone()
    query = readout
    if block == "kind":
        changed[..., :dim] = 0.0
    elif block == "subjects":
        changed[..., dim + 1 : -dim] = 0.0
    elif block == "graph_context":
        changed[..., -dim:] = 0.0
        query = torch.zeros_like(readout)
    else:
        raise ValueError(f"unknown policy intervention: {block}")
    return model.policy(query, changed, action_mask)


def pointer_attention(model, readout, features, action_mask):
    policy = model.policy
    tokens = policy.token_proj(features)
    batch_size, action_count, dim = tokens.shape
    split = dim // policy.heads
    query = policy.glimpse_query(readout).view(batch_size, policy.heads, split)
    keys = policy.glimpse_key(tokens).view(batch_size, action_count, policy.heads, split)
    scores = torch.einsum("bhs,bahs->bha", query, keys) / math.sqrt(split)
    scores = scores.masked_fill(~action_mask.unsqueeze(1), -1.0e9)
    return torch.softmax(scores, dim=-1)


def ridge_regression_metrics(
    features: np.ndarray,
    target: np.ndarray,
    seed: int,
    classification: bool = False,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(features).all(axis=1) & np.isfinite(target)
    features = features[valid]
    target = target[valid]
    rng = np.random.default_rng(seed)
    if groups is None:
        order = rng.permutation(len(target))
        split = max(1, int(0.8 * len(order)))
        train, test = order[:split], order[split:]
    else:
        groups = np.asarray(groups)[valid]
        unique_groups = rng.permutation(np.unique(groups))
        if len(unique_groups) < 2:
            return {"r2": float("nan"), "sign_accuracy": float("nan")}
        split = min(len(unique_groups) - 1, max(1, int(0.8 * len(unique_groups))))
        train_groups = unique_groups[:split]
        train = np.flatnonzero(np.isin(groups, train_groups))
        test = np.flatnonzero(~np.isin(groups, train_groups))
    if len(test) == 0:
        return {"r2": float("nan"), "sign_accuracy": float("nan")}
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    train_x = (features[train] - mean) / scale
    test_x = (features[test] - mean) / scale
    target_mean = target[train].mean()
    train_y = target[train] - target_mean
    ridge = 1.0e-2
    gram = train_x.T @ train_x
    weight = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), train_x.T @ train_y)
    prediction = test_x @ weight + target_mean
    residual = float(np.square(target[test] - prediction).sum())
    total = float(np.square(target[test] - target[test].mean()).sum())
    result = {"r2": 1.0 - residual / total if total > 0.0 else float("nan")}
    if classification:
        result["sign_accuracy"] = float(
            (np.where(prediction >= 0.0, 1.0, -1.0) == target[test]).mean()
        )
    return result


def representation_probes(
    self_readout: np.ndarray,
    opponent_readout: np.ndarray,
    value_target: np.ndarray,
    learner_reward: np.ndarray,
    opponent_reward: np.ndarray,
    self_node_count: np.ndarray,
    self_edge_count: np.ndarray,
    opponent_node_count: np.ndarray,
    opponent_edge_count: np.ndarray,
    graph_pair_groups: np.ndarray,
    seed: int,
) -> dict[str, object]:
    return {
        "structure": {
            "self_node_count": ridge_regression_metrics(
                self_readout, self_node_count, seed, groups=graph_pair_groups
            ),
            "self_edge_count": ridge_regression_metrics(
                self_readout, self_edge_count, seed, groups=graph_pair_groups
            ),
            "opponent_node_count": ridge_regression_metrics(
                opponent_readout, opponent_node_count, seed, groups=graph_pair_groups
            ),
            "opponent_edge_count": ridge_regression_metrics(
                opponent_readout, opponent_edge_count, seed, groups=graph_pair_groups
            ),
        },
        "self_readout_to_learner_reward": ridge_regression_metrics(
            self_readout, learner_reward, seed, groups=graph_pair_groups
        ),
        "opponent_readout_to_opponent_reward": ridge_regression_metrics(
            opponent_readout, opponent_reward, seed, groups=graph_pair_groups
        ),
        "value_label": {
            "self_only": ridge_regression_metrics(
                self_readout,
                value_target,
                seed,
                classification=True,
                groups=graph_pair_groups,
            ),
            "opponent_only": ridge_regression_metrics(
                opponent_readout,
                value_target,
                seed,
                classification=True,
                groups=graph_pair_groups,
            ),
            "difference": ridge_regression_metrics(
                self_readout - opponent_readout,
                value_target,
                seed,
                classification=True,
                groups=graph_pair_groups,
            ),
            "sum": ridge_regression_metrics(
                self_readout + opponent_readout,
                value_target,
                seed,
                classification=True,
                groups=graph_pair_groups,
            ),
            "concatenated": ridge_regression_metrics(
                np.concatenate((self_readout, opponent_readout), axis=1),
                value_target,
                seed,
                classification=True,
                groups=graph_pair_groups,
            ),
        },
    }


def graph_pair_signature(view: BatchView, index: int) -> str:
    hasher = hashlib.blake2b(digest_size=16)

    def graph(
        node_count,
        node_tokens,
        node_attrs,
        edge_count,
        edge_src,
        edge_dst,
        edge_type,
        position,
    ) -> None:
        nodes = int(node_count[index])
        edges = int(edge_count[index])
        hasher.update(np.asarray([nodes, edges], dtype="<u4").tobytes())
        hasher.update(np.ascontiguousarray(node_tokens[index, :nodes]).tobytes())
        if node_attrs is not None:
            hasher.update(np.ascontiguousarray(node_attrs[index, :nodes]).tobytes())
        hasher.update(np.ascontiguousarray(edge_src[index, :edges]).tobytes())
        hasher.update(np.ascontiguousarray(edge_dst[index, :edges]).tobytes())
        hasher.update(np.ascontiguousarray(edge_type[index, :edges]).tobytes())
        hasher.update(np.ascontiguousarray(position[index]).tobytes())

    graph(
        view.node_count,
        view.node_tokens,
        view.node_attrs,
        view.edge_count,
        view.edge_src,
        view.edge_dst,
        view.edge_type,
        view.position,
    )
    hasher.update(bytes([int(view.opponent_state_present[index])]))
    graph(
        view.opponent_node_count,
        view.opponent_node_tokens,
        view.opponent_node_attrs,
        view.opponent_edge_count,
        view.opponent_edge_src,
        view.opponent_edge_dst,
        view.opponent_edge_type,
        view.opponent_position,
    )
    return hasher.hexdigest()


def replay_analysis(model, replay_batches: list[SampleResult], device: str, seed: int):
    vocab = model.schema.action_kind_vocab_size
    kind_model_mass = torch.zeros(vocab, dtype=torch.float64)
    kind_target_mass = torch.zeros(vocab, dtype=torch.float64)
    kind_logit_sum = torch.zeros(vocab, dtype=torch.float64)
    kind_count = torch.zeros(vocab, dtype=torch.float64)
    attention_kind_mass = torch.zeros((model.policy.heads, vocab), dtype=torch.float64)
    attention_entropy: list[np.ndarray] = []
    attention_stop: list[np.ndarray] = []
    row_metrics: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "policy_cross_entropy",
            "policy_kl",
            "policy_entropy",
            "target_entropy",
            "policy_top1_match",
            "policy_stop_mass",
            "target_stop_mass",
            "policy_stop_top1",
            "target_stop_top1",
            "value_prediction",
            "value_target",
            "value_raw",
            "value_swap",
            "value_tie",
            "learner_reward",
            "opponent_reward",
            "self_node_count",
            "self_edge_count",
            "opponent_node_count",
            "opponent_edge_count",
            "value_self_attribution_l1",
            "value_opponent_attribution_l1",
        )
    }
    intervention_metrics = {
        block: {"js": [], "top1_flip": [], "cross_entropy": []}
        for block in ("kind", "subjects", "graph_context")
    }
    component_square_sum = {name: 0.0 for name in ("kind", "prior", "subjects", "readout")}
    component_count = 0
    valid_logit_count = 0
    saturated_logit_count = 0
    hidden_abs_contribution = torch.zeros(model.arch.ffn_dim, dtype=torch.float64)
    hidden_signed_contribution = torch.zeros(model.arch.ffn_dim, dtype=torch.float64)
    replay_position_max = 0.0
    replay_opponent_position_max = 0.0
    self_readouts = []
    opponent_readouts = []
    graph_pair_signatures = []

    for sampled in replay_batches:
        view = sampled.batch
        row_count = view.row_count
        batch = stage_view(view, model.schema, device, expose_position=True)
        replay_position_max = max(replay_position_max, float(batch.position[:row_count].abs().max()))
        replay_opponent_position_max = max(
            replay_opponent_position_max,
            float(batch.opponent_position[:row_count].abs().max()),
        )
        with torch.no_grad():
            readout, opponent, action_features, action_mask = head_features(model, batch)
            readout = readout[:row_count]
            opponent = opponent[:row_count]
            action_features = action_features[:row_count]
            action_mask = action_mask[:row_count]
            counts = batch.action_count[:row_count]
            logits = model.policy(readout, action_features, action_mask)
            probabilities = policy_probabilities(logits, action_mask)
            target = torch.as_tensor(
                np.array(sampled.targets.policy[:row_count], copy=True),
                dtype=torch.float32,
                device=logits.device,
            )
            target = target * action_mask
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
            log_probabilities = probabilities.clamp_min(1.0e-30).log()
            target_log = target.clamp_min(1.0e-30).log()
            cross_entropy = -(target * log_probabilities).sum(dim=-1)
            target_entropy = -(target * target_log).sum(dim=-1)
            policy_entropy = -(probabilities * log_probabilities).sum(dim=-1)
            policy_kl = cross_entropy - target_entropy
            model_top = probabilities.argmax(dim=-1)
            target_top = target.argmax(dim=-1)
            stop_index = counts.to(torch.int64) - 1

            row_metrics["policy_cross_entropy"].append(cross_entropy.cpu().numpy())
            row_metrics["policy_kl"].append(policy_kl.cpu().numpy())
            row_metrics["policy_entropy"].append(policy_entropy.cpu().numpy())
            row_metrics["target_entropy"].append(target_entropy.cpu().numpy())
            row_metrics["policy_top1_match"].append((model_top == target_top).float().cpu().numpy())
            row_metrics["policy_stop_mass"].append(
                probabilities.gather(1, stop_index.unsqueeze(1)).squeeze(1).cpu().numpy()
            )
            row_metrics["target_stop_mass"].append(
                target.gather(1, stop_index.unsqueeze(1)).squeeze(1).cpu().numpy()
            )
            row_metrics["policy_stop_top1"].append((model_top == stop_index).float().cpu().numpy())
            row_metrics["target_stop_top1"].append((target_top == stop_index).float().cpu().numpy())

            valid_kinds = batch.action_kind[:row_count].clamp(0, vocab - 1).to(torch.int64)
            flat_kind = valid_kinds[action_mask]
            kind_model_mass.scatter_add_(
                0, flat_kind.cpu(), probabilities[action_mask].double().cpu()
            )
            kind_target_mass.scatter_add_(0, flat_kind.cpu(), target[action_mask].double().cpu())
            kind_logit_sum.scatter_add_(0, flat_kind.cpu(), logits[action_mask].double().cpu())
            kind_count.scatter_add_(
                0, flat_kind.cpu(), torch.ones_like(flat_kind, dtype=torch.float64).cpu()
            )

            valid_logits = logits[action_mask]
            valid_logit_count += valid_logits.numel()
            saturated_logit_count += int((valid_logits.abs() >= 9.5).sum())

            for block in intervention_metrics:
                changed_logits = policy_intervention(
                    model, readout, action_features, action_mask, block
                )
                changed_probabilities = policy_probabilities(changed_logits, action_mask)
                intervention_metrics[block]["js"].append(
                    js_divergence(probabilities, changed_probabilities).cpu().numpy()
                )
                intervention_metrics[block]["top1_flip"].append(
                    (changed_probabilities.argmax(dim=-1) != model_top).float().cpu().numpy()
                )
                intervention_metrics[block]["cross_entropy"].append(
                    (-(target * changed_probabilities.clamp_min(1.0e-30).log()).sum(dim=-1))
                    .cpu()
                    .numpy()
                )

            dim = model.arch.dim
            slices = {
                "kind": slice(0, dim),
                "prior": slice(dim, dim + 1),
                "subjects": slice(dim + 1, action_features.shape[-1] - dim),
                "readout": slice(action_features.shape[-1] - dim, action_features.shape[-1]),
            }
            token_weight = model.policy.token_proj.weight
            for name, section in slices.items():
                contribution = functional.linear(
                    action_features[..., section], token_weight[:, section], None
                )
                component_square_sum[name] += float(
                    contribution[action_mask].float().square().sum()
                )
            component_count += int(action_mask.sum())

            attention = pointer_attention(model, readout, action_features, action_mask)
            attention_entropy.append(
                (-(attention * attention.clamp_min(1.0e-30).log()).sum(dim=-1))
                .cpu()
                .numpy()
            )
            attention_stop.append(
                attention.gather(
                    2,
                    stop_index.view(-1, 1, 1).expand(-1, model.policy.heads, 1),
                )
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            for head in range(model.policy.heads):
                attention_kind_mass[head].scatter_add_(
                    0,
                    flat_kind.cpu(),
                    attention[:, head, :][action_mask].double().cpu(),
                )

            pair = torch.cat((readout, opponent), dim=-1)
            raw = model.value(pair).squeeze(-1)
            value = torch.tanh(raw) if model.arch.value_activation == "tanh" else raw
            swapped_raw = model.value(torch.cat((opponent, readout), dim=-1)).squeeze(-1)
            swapped = (
                torch.tanh(swapped_raw)
                if model.arch.value_activation == "tanh"
                else swapped_raw
            )
            tie_raw = model.value(torch.cat((readout, readout), dim=-1)).squeeze(-1)
            tie = torch.tanh(tie_raw) if model.arch.value_activation == "tanh" else tie_raw
            hidden = model.value[1](model.value[0](pair))
            unit_contribution = hidden * model.value[3].weight.squeeze(0)
            hidden_abs_contribution += unit_contribution.abs().sum(dim=0).double().cpu()
            hidden_signed_contribution += unit_contribution.sum(dim=0).double().cpu()

            value_target = np.array(sampled.targets.value[:row_count], copy=True)
            learner_reward = np.array(sampled.targets.reward[:row_count], copy=True)
            opponent_reward = (
                np.array(view.opponent_reward[:row_count], copy=True)
                * model.schema.opponent_reward_scale
            )
            row_metrics["value_prediction"].append(value.cpu().numpy())
            row_metrics["value_target"].append(value_target)
            row_metrics["value_raw"].append(raw.cpu().numpy())
            row_metrics["value_swap"].append(swapped.cpu().numpy())
            row_metrics["value_tie"].append(tie.cpu().numpy())
            row_metrics["learner_reward"].append(learner_reward)
            row_metrics["opponent_reward"].append(opponent_reward)
            row_metrics["self_node_count"].append(
                np.array(view.node_count[:row_count], dtype=np.float32, copy=True)
            )
            row_metrics["self_edge_count"].append(
                np.array(view.edge_count[:row_count], dtype=np.float32, copy=True)
            )
            row_metrics["opponent_node_count"].append(
                np.array(view.opponent_node_count[:row_count], dtype=np.float32, copy=True)
            )
            row_metrics["opponent_edge_count"].append(
                np.array(view.opponent_edge_count[:row_count], dtype=np.float32, copy=True)
            )
            graph_pair_signatures.extend(
                graph_pair_signature(view, index) for index in range(row_count)
            )
            self_readouts.append(readout.cpu().numpy())
            opponent_readouts.append(opponent.cpu().numpy())

        pair_for_gradient = torch.cat((readout.detach(), opponent.detach()), dim=-1).requires_grad_()
        raw_for_gradient = model.value(pair_for_gradient).squeeze(-1)
        gradient = torch.autograd.grad(raw_for_gradient.sum(), pair_for_gradient)[0]
        attribution = (pair_for_gradient * gradient).abs()
        dim = model.arch.dim
        row_metrics["value_self_attribution_l1"].append(
            attribution[:, :dim].sum(dim=-1).detach().cpu().numpy()
        )
        row_metrics["value_opponent_attribution_l1"].append(
            attribution[:, dim:].sum(dim=-1).detach().cpu().numpy()
        )

    merged = {name: np.concatenate(values) for name, values in row_metrics.items()}
    self_readout = np.concatenate(self_readouts)
    opponent_readout = np.concatenate(opponent_readouts)
    policy_cross_entropy = merged["policy_cross_entropy"]
    value_prediction = merged["value_prediction"]
    value_target = merged["value_target"]
    reward_margin = merged["learner_reward"] - merged["opponent_reward"]
    reward_tie = np.abs(reward_margin) < 1.0e-6
    non_tie = ~reward_tie
    readout_distance = np.linalg.norm(self_readout - opponent_readout, axis=1)
    pair_labels: dict[str, list[int]] = {}
    for signature, target in zip(graph_pair_signatures, value_target):
        counts = pair_labels.setdefault(signature, [0, 0])
        counts[1 if target > 0.0 else 0] += 1
    conflicting_pairs = {
        signature: counts
        for signature, counts in pair_labels.items()
        if counts[0] > 0 and counts[1] > 0
    }
    conflicting_rows = sum(sum(counts) for counts in conflicting_pairs.values())
    deterministic_pair_correct = sum(max(counts) for counts in pair_labels.values())

    kind_rows = []
    total_rows = len(value_target)
    for kind in range(vocab):
        if kind_count[kind] == 0:
            continue
        kind_rows.append(
            {
                "kind": kind,
                "occurrences": int(kind_count[kind]),
                "model_mass_per_row": float(kind_model_mass[kind] / total_rows),
                "target_mass_per_row": float(kind_target_mass[kind] / total_rows),
                "mean_logit": float(kind_logit_sum[kind] / kind_count[kind]),
            }
        )
    kind_rows.sort(key=lambda row: row["model_mass_per_row"], reverse=True)

    attention_entropy_array = np.concatenate(attention_entropy)
    attention_stop_array = np.concatenate(attention_stop)
    head_rows = []
    for head in range(model.policy.heads):
        top_kinds = torch.argsort(attention_kind_mass[head], descending=True)[:5]
        head_rows.append(
            {
                "head": head,
                "entropy": float(attention_entropy_array[:, head].mean()),
                "effective_actions": float(
                    np.exp(attention_entropy_array[:, head]).mean()
                ),
                "stop_mass": float(attention_stop_array[:, head].mean()),
                "top_kinds": [
                    {
                        "kind": int(kind),
                        "mass_per_row": float(attention_kind_mass[head, kind] / total_rows),
                    }
                    for kind in top_kinds
                ],
            }
        )

    hidden_total = float(hidden_abs_contribution.sum())
    hidden_order = torch.argsort(hidden_abs_contribution, descending=True)
    cumulative = torch.cumsum(hidden_abs_contribution[hidden_order], dim=0)
    units_for_half = int((cumulative < 0.5 * hidden_total).sum()) + 1
    top_hidden = [
        {
            "unit": int(unit),
            "absolute_share": float(hidden_abs_contribution[unit] / hidden_total),
            "mean_signed_contribution": float(hidden_signed_contribution[unit] / total_rows),
        }
        for unit in hidden_order[:10]
    ]

    interventions = {}
    baseline_ce = float(policy_cross_entropy.mean())
    for block, metrics in intervention_metrics.items():
        cross_entropy = np.concatenate(metrics["cross_entropy"])
        interventions[block] = {
            "mean_js_divergence": float(np.concatenate(metrics["js"]).mean()),
            "top1_flip_rate": float(np.concatenate(metrics["top1_flip"]).mean()),
            "cross_entropy": float(cross_entropy.mean()),
            "cross_entropy_delta": float(cross_entropy.mean() - baseline_ce),
        }

    value_self_attribution = merged["value_self_attribution_l1"]
    value_opponent_attribution = merged["value_opponent_attribution_l1"]
    value_mse = np.square(value_prediction - value_target)
    sign_prediction = np.where(value_prediction >= 0.0, 1.0, -1.0)
    return {
        "sample_count": total_rows,
        "position_max_abs": replay_position_max,
        "opponent_position_max_abs": replay_opponent_position_max,
        "policy": {
            "cross_entropy": baseline_ce,
            "kl_target_to_model": float(merged["policy_kl"].mean()),
            "model_entropy": float(merged["policy_entropy"].mean()),
            "target_entropy": float(merged["target_entropy"].mean()),
            "top1_agreement": float(merged["policy_top1_match"].mean()),
            "model_stop_mass": float(merged["policy_stop_mass"].mean()),
            "target_stop_mass": float(merged["target_stop_mass"].mean()),
            "model_stop_top1_rate": float(merged["policy_stop_top1"].mean()),
            "target_stop_top1_rate": float(merged["target_stop_top1"].mean()),
            "logit_saturation_fraction": saturated_logit_count / valid_logit_count,
            "interventions": interventions,
            "token_projection_component_rms": {
                name: math.sqrt(square_sum / component_count)
                for name, square_sum in component_square_sum.items()
            },
            "kinds": kind_rows,
            "glimpse_heads": head_rows,
        },
        "value": {
            "mse": float(value_mse.mean()),
            "sign_accuracy": float((sign_prediction == value_target).mean()),
            "positive_label_fraction": float((value_target > 0.0).mean()),
            "prediction_mean": float(value_prediction.mean()),
            "prediction_abs_mean": float(np.abs(value_prediction).mean()),
            "prediction_saturation_fraction": float((np.abs(value_prediction) >= 0.95).mean()),
            "positive_label_prediction_mean": float(value_prediction[value_target > 0].mean()),
            "negative_label_prediction_mean": float(value_prediction[value_target < 0].mean()),
            "reward_margin_correlation": finite_corr(value_prediction, reward_margin),
            "reward_tie_fraction": float(reward_tie.mean()),
            "reward_margin_mean": float(reward_margin.mean()),
            "reward_margin_std": float(reward_margin.std()),
            "learner_reward_unique": sorted(np.unique(merged["learner_reward"]).tolist()),
            "opponent_reward_unique": sorted(np.unique(merged["opponent_reward"]).tolist()),
            "non_tie_count": int(non_tie.sum()),
            "non_tie_sign_accuracy": (
                float((sign_prediction[non_tie] == value_target[non_tie]).mean())
                if non_tie.any()
                else float("nan")
            ),
            "swap_negative_correlation": finite_corr(
                merged["value_swap"], -value_prediction
            ),
            "swap_antisymmetry_error_mean_abs": float(
                np.abs(value_prediction + merged["value_swap"]).mean()
            ),
            "swap_sign_reversal_rate": float(
                (np.signbit(value_prediction) != np.signbit(merged["value_swap"])).mean()
            ),
            "self_pair_value_mean_abs": float(np.abs(merged["value_tie"]).mean()),
            "self_opponent_readout_distance_mean": float(readout_distance.mean()),
            "self_opponent_readout_identical_fraction": float((readout_distance < 1.0e-6).mean()),
            "unique_graph_pairs": len(pair_labels),
            "conflicting_label_graph_pairs": len(conflicting_pairs),
            "rows_in_conflicting_graph_pairs_fraction": conflicting_rows / total_rows,
            "graph_pair_majority_label_accuracy": deterministic_pair_correct / total_rows,
            "self_attribution_l1_mean": float(value_self_attribution.mean()),
            "opponent_attribution_l1_mean": float(value_opponent_attribution.mean()),
            "opponent_attribution_fraction": float(
                value_opponent_attribution.mean()
                / (value_self_attribution.mean() + value_opponent_attribution.mean())
            ),
            "hidden_units_for_half_absolute_contribution": units_for_half,
            "top_hidden_units": top_hidden,
        },
        "readouts": {
            "self": self_readout,
            "opponent": opponent_readout,
            "value_target": value_target,
            "learner_reward": merged["learner_reward"],
            "opponent_reward": merged["opponent_reward"],
            "self_node_count": merged["self_node_count"],
            "self_edge_count": merged["self_edge_count"],
            "opponent_node_count": merged["opponent_node_count"],
            "opponent_edge_count": merged["opponent_edge_count"],
            "graph_pair_groups": np.asarray(graph_pair_signatures),
        },
    }


def policy_gate_events(run_dir: Path, manifests: dict[int, object]) -> list[dict[str, object]]:
    by_model_version = {
        manifest.model_version.hex(): version for version, manifest in manifests.items()
    }
    pattern = re.compile(
        r"accepted=(true|false) challenger=(-?[0-9.]+) best=(-?[0-9.]+) "
        r"steps=([0-9]+) version=([0-9a-f]+)"
    )
    events = []
    log_paths = sorted(run_dir.glob("trainer*.log"), key=lambda path: path.stat().st_mtime)
    for log_path in log_paths:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if "event=policy_gate" not in line:
                continue
            match = pattern.search(line)
            if match is None:
                continue
            version = by_model_version.get(match.group(5))
            if version is None:
                continue
            events.append(
                {
                    "version": version,
                    "training_step": manifests[version].training_step,
                    "accepted": match.group(1) == "true",
                    "challenger_cost": -float(match.group(2)),
                    "incumbent_cost": -float(match.group(3)),
                    "rollout_steps": int(match.group(4)),
                    "log": log_path.name,
                }
            )
    return events


def readouts_for_model(model, replay_batches: list[SampleResult], device: str):
    self_readouts = []
    opponent_readouts = []
    with torch.no_grad():
        for sampled in replay_batches:
            batch = stage_view(sampled.batch, model.schema, device, expose_position=True)
            row_count = sampled.batch.row_count
            readout, opponent, _, _ = head_features(model, batch)
            self_readouts.append(readout[:row_count].cpu().numpy())
            opponent_readouts.append(opponent[:row_count].cpu().numpy())
    return np.concatenate(self_readouts), np.concatenate(opponent_readouts)


def kind_names(probe_dir: Path, meta: dict[str, object]) -> dict[int, str]:
    view = BatchView.parse((probe_dir / "sweep.gzfb").read_bytes())
    names = {1: "STOP"}
    for index, candidate in enumerate(meta["candidates"]):
        kind = int(view.action_kind[0, index])
        name = str(candidate["rule"])
        previous = names.setdefault(kind, name)
        if previous != name:
            names[kind] = f"{previous}|{name}"
    return names


def label_kinds(result: dict[str, object], names: dict[int, str]) -> None:
    replay_policy = result["replay"]["policy"]
    for row in replay_policy["kinds"]:
        row["name"] = names.get(row["kind"], f"kind_{row['kind']}")
    for head in replay_policy["glimpse_heads"]:
        for row in head["top_kinds"]:
            row["name"] = names.get(row["kind"], f"kind_{row['kind']}")


def markdown_report(result: dict[str, object]) -> str:
    lines = [
        "# Policy and Value Head Mechanistic Probes",
        "",
        f"Run: `{result['run_dir']}`",
        f"Retained replay sample: {result['replay']['sample_count']} rows",
        "",
        (
            "Fixed probes use the seed-42 production root and retain both graph clock vectors."
            if result["position_features"] == "on"
            else "Fixed probes use the seed-42 production root and force both graph clock vectors to zero."
        ),
        "",
        "## Checkpoint trajectory",
        "",
        "| Version | Step | Shrink mass | Grow mass | E[immediate delta] | Policy entropy | Value/cost corr | Swap error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in result["checkpoints"]:
        policy = checkpoint["fixed_probe"]["policy"]
        value = checkpoint["fixed_probe"]["value"]
        lines.append(
            f"| {checkpoint['version']} | {checkpoint['training_step']} | "
            f"{policy['shrink_mass']:.4f} | {policy['grow_mass']:.4f} | "
            f"{policy['expected_immediate_cost_delta']:+.3f} | {policy['entropy']:.3f} | "
            f"{value['cost_correlation']:+.3f} | {abs(value['orientation_sum']):.3f} |"
        )

    transition_events = [
        event for event in result["policy_gate_events"] if event["version"] >= 34
    ]
    if transition_events:
        lines.extend(
            [
                "",
                "### Gate around the strategy transition",
                "",
                "| Version | Step | Challenger cost | Incumbent cost | Accepted | Rollout steps |",
                "|---:|---:|---:|---:|---|---:|",
            ]
        )
        for event in transition_events:
            lines.append(
                f"| {event['version']} | {event['training_step']} | "
                f"{event['challenger_cost']:.0f} | {event['incumbent_cost']:.0f} | "
                f"{str(event['accepted']).lower()} | {event['rollout_steps']} |"
            )

    patching = result.get("cross_checkpoint_policy_patching")
    if patching is not None:
        lines.extend(
            [
                "",
                "### Cross-checkpoint causal patching",
                "",
                "| Configuration | Encoder | Kind embedding | Pointer | Shrink mass | Grow mass | E[delta] | Top rule |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in patching["rows"]:
            lines.append(
                f"| {row['name']} | {row['encoder_version']} | {row['kind_version']} | "
                f"{row['pointer_version']} | {row['shrink_mass']:.3f} | {row['grow_mass']:.3f} | "
                f"{row['expected_immediate_cost_delta']:+.3f} | {row['top_rule']} |"
            )

    policy = result["replay"]["policy"]
    lines.extend(
        [
            "",
            "## Retained replay policy",
            "",
            f"- Cross-entropy: {policy['cross_entropy']:.4f}; KL(target || model): {policy['kl_target_to_model']:.4f}",
            f"- Top-1 agreement: {policy['top1_agreement']:.3%}",
            f"- Model/target entropy: {policy['model_entropy']:.3f} / {policy['target_entropy']:.3f}",
            f"- Model/target STOP mass: {policy['model_stop_mass']:.4f} / {policy['target_stop_mass']:.4f}",
            f"- Pointer logit saturation: {policy['logit_saturation_fraction']:.3%}",
            "",
            "### Policy interventions",
            "",
            "| Removed input | JS divergence | Top-1 flips | CE delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, metrics in policy["interventions"].items():
        lines.append(
            f"| {name} | {metrics['mean_js_divergence']:.5f} | "
            f"{metrics['top1_flip_rate']:.3%} | {metrics['cross_entropy_delta']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "### Pointer glimpse heads",
            "",
            "| Head | Entropy | Effective actions | STOP mass | Top kind |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for head in policy["glimpse_heads"]:
        top = head["top_kinds"][0]
        lines.append(
            f"| {head['head']} | {head['entropy']:.3f} | {head['effective_actions']:.1f} | "
            f"{head['stop_mass']:.4f} | {top['name']} ({top['mass_per_row']:.3f}) |"
        )

    value = result["replay"]["value"]
    channels = result["value_weight_channels"]
    lines.extend(
        [
            "",
            "## Retained replay value",
            "",
            f"- MSE: {value['mse']:.4f}; sign accuracy: {value['sign_accuracy']:.3%}",
            f"- Reward ties: {value['reward_tie_fraction']:.3%}; positive labels: {value['positive_label_fraction']:.3%}",
            f"- Mean prediction for +1/-1 labels: {value['positive_label_prediction_mean']:+.3f} / {value['negative_label_prediction_mean']:+.3f}",
            f"- Correlation with terminal reward margin: {value['reward_margin_correlation']:+.3f}",
            f"- Swap antisymmetry mean absolute error: {value['swap_antisymmetry_error_mean_abs']:.3f}",
            f"- Self-pair |V(g,g)|: {value['self_pair_value_mean_abs']:.3f}",
            f"- Identical self/opponent readouts: {value['self_opponent_readout_identical_fraction']:.3%}",
            f"- Rows whose exact graph pair has conflicting labels: {value['rows_in_conflicting_graph_pairs_fraction']:.3%}",
            f"- Opponent share of input-gradient attribution: {value['opponent_attribution_fraction']:.3%}",
            f"- First-layer sum/difference channel norm ratio: {channels['sum_to_difference_norm_ratio']:.3f}",
            f"- First-layer self vs -opponent weight cosine: {channels['self_vs_negative_opponent_cosine']:+.3f}",
            "",
            "## Graph-pair-grouped readout probes",
            "",
            "| Representation | Initial | Final |",
            "|---|---:|---:|",
        ]
    )
    initial_probes = result["representation_probes"]["initial"]
    final_probes = result["representation_probes"]["final"]
    for name in (
        "self_node_count",
        "self_edge_count",
        "opponent_node_count",
        "opponent_edge_count",
    ):
        lines.append(
            f"| {name} R2 | "
            f"{initial_probes['structure'][name]['r2']:+.3f} | "
            f"{final_probes['structure'][name]['r2']:+.3f} |"
        )
    if value["reward_tie_fraction"] >= 0.99:
        lines.extend(
            [
                "",
                "Value-label linear probes are omitted: tie labels are coin flips, and a random row split "
                "would leak repeated graph pairs across train and test.",
            ]
        )

    lines.extend(
        [
            "",
            "## Parameter movement",
            "",
            "| Group | Parameters | Relative update |",
            "|---|---:|---:|",
        ]
    )
    for name, metrics in result["parameter_changes"].items():
        lines.append(
            f"| {name} | {metrics['parameters']} | {metrics['relative_update']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "The replay results describe only the final retained window, not the full training distribution. "
            "Fixed-root candidate deltas are one-step measured costs and do not include downstream search value. "
            "Applied states in the fixed value sweep retain the root clock, so their cost correlation is a "
            "same-clock counterfactual rather than on-trajectory value calibration. "
            "Input-removal interventions are causal sensitivity checks but create out-of-distribution head activations.",
            "",
        ]
    )
    return "\n".join(lines)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    output_dir = (args.output_dir or run_dir / "mechinterp").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = [int(value) for value in args.checkpoints.split(",")]
    patch_versions = [int(value) for value in args.patch_checkpoints.split(",")]
    if len(patch_versions) != 2:
        raise ValueError("--patch-checkpoints must contain exactly two versions")
    meta = json.loads((args.probe_dir / "meta.json").read_text(encoding="utf-8"))
    expose_position = args.position_features == "on"

    checkpoint_results = []
    models = {}
    manifests = {}
    for version in versions:
        model, manifest = load_model(checkpoint_dir, version, args.device)
        models[version] = model
        manifests[version] = manifest
        checkpoint_results.append(
            {
                "version": version,
                "training_step": manifest.training_step,
                "model_version": manifest.model_version.hex(),
                "fixed_probe": fixed_probe(
                    model,
                    manifest.feature_schema,
                    args.probe_dir,
                    meta,
                    args.device,
                    expose_position,
                ),
            }
        )
        print(f"fixed probe version={version} step={manifest.training_step}", flush=True)

    initial = models[versions[0]]
    final = models[versions[-1]]
    if initial.arch.value_input != "pair" or final.arch.policy_head != "pointer":
        raise ValueError("this tool requires pair value and pointer policy heads")

    with replay_client(
        args.graphzero_bin.resolve(), run_dir / "replay", args.batch
    ) as (client, ack):
        replay_batches = sample_replay(
            client,
            args.samples,
            args.batch,
            args.window_rows,
            args.seed,
        )
    print(f"sampled replay rows={args.samples}", flush=True)

    replay = replay_analysis(final, replay_batches, args.device, args.seed)
    initial_self, initial_opponent = readouts_for_model(initial, replay_batches, args.device)
    final_readouts = replay.pop("readouts")
    representation = {
        "initial": representation_probes(
            initial_self,
            initial_opponent,
            final_readouts["value_target"],
            final_readouts["learner_reward"],
            final_readouts["opponent_reward"],
            final_readouts["self_node_count"],
            final_readouts["self_edge_count"],
            final_readouts["opponent_node_count"],
            final_readouts["opponent_edge_count"],
            final_readouts["graph_pair_groups"],
            args.seed,
        ),
        "final": representation_probes(
            final_readouts["self"],
            final_readouts["opponent"],
            final_readouts["value_target"],
            final_readouts["learner_reward"],
            final_readouts["opponent_reward"],
            final_readouts["self_node_count"],
            final_readouts["self_edge_count"],
            final_readouts["opponent_node_count"],
            final_readouts["opponent_edge_count"],
            final_readouts["graph_pair_groups"],
            args.seed,
        ),
    }

    result = {
        "run_dir": str(run_dir),
        "position_features": args.position_features,
        "replay_ack": {
            "produced_rows": ack.produced_rows,
            "episodes": ack.episodes,
            "best_cost": ack.best_cost,
        },
        "checkpoints": checkpoint_results,
        "policy_gate_events": policy_gate_events(run_dir, manifests),
        "replay": replay,
        "representation_probes": representation,
        "value_weight_channels": value_weight_channels(final),
        "parameter_changes": parameter_changes(initial, final),
        "cross_checkpoint_policy_patching": cross_checkpoint_policy_patching(
            models,
            args.probe_dir,
            meta,
            args.device,
            expose_position,
            early_version=patch_versions[0],
            late_version=patch_versions[1],
        ),
    }
    label_kinds(result, kind_names(args.probe_dir, meta))
    markdown = markdown_report(result)
    result = json_ready(result)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {output_dir / 'report.json'}", flush=True)
    print(f"wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
