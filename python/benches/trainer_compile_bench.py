from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from gz.checkpoints import DirectorySource  # noqa: E402
from gz.checkpoints.weights import load_state_dict  # noqa: E402
from gz.model.exphormer import ArchConfig, BatchStager, build_model  # noqa: E402
from gz.trainer.data import TrainingBatch, TrainingStager  # noqa: E402
from gz.trainer.loop import LoopConfig, TrainerLoop  # noqa: E402
from gz.trainer.publish import EmaWeights  # noqa: E402
from gz.trainer.sampler import SampleClient  # noqa: E402


def main() -> int:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("trainer compile benchmark requires CUDA")
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    resolved = DirectorySource(args.checkpoint_dir).resolve_latest()
    schema = resolved.manifest.feature_schema
    arch = ArchConfig.from_dict(resolved.manifest.arch_config)
    model = build_model(schema, arch).to(device)
    model.load_state_dict(load_state_dict(resolved.weights_path))

    next_batches = batch_source(args, resolved.manifest.feature_schema_hash, schema, device)
    loop = TrainerLoop(
        model,
        LoopConfig(
            lr=3e-4,
            warmup_steps=0,
            total_steps=args.warmup + args.steps,
            lr_schedule="constant",
            value_weight=1.0,
            grad_clip=1.0,
            weight_decay=1e-4,
            optimizer=args.optimizer,
            adamw_lr=3e-4,
            run_seed=args.seed,
            value_mirror=False,
            compile_model=args.compile_model,
            compile_mode=args.compile_mode,
        ),
    )
    ema = EmaWeights(model, decay=0.0)

    torch.cuda.synchronize(device)
    warmup_started = time.perf_counter()
    for index in range(args.warmup):
        policy_batch, value_batch = next_batches(index)
        loop.train_step(
            policy_batch,
            value_batch,
            with_metrics=False,
        )
        if not args.skip_ema_update:
            ema.update(model)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    staging_seconds = 0.0
    for index in range(args.steps):
        staging_started = time.perf_counter()
        policy_batch, value_batch = next_batches(index)
        staging_seconds += time.perf_counter() - staging_started
        loop.train_step(
            policy_batch,
            value_batch,
            with_metrics=False,
        )
        if not args.skip_ema_update:
            ema.update(model)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    counters = torch._dynamo.utils.counters
    result = {
        "batch": args.batch,
        "balanced_padded_edges": args.balance_padded_edges,
        "balanced_padded_subjects": args.balance_padded_subjects,
        "compile_mode": args.compile_mode if args.compile_model else "eager",
        "compiled": args.compile_model,
        "device": str(device),
        "includes_ema": not args.skip_ema_update,
        "skip_ema_update": args.skip_ema_update,
        "input": (
            "replay-cached"
            if args.reuse_staged
            else "replay-staged"
            if args.sample_socket
            else "synthetic-device"
        ),
        "matmul_precision": args.matmul_precision,
        "optimizer": args.optimizer,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "staging_ms": 1000.0 * staging_seconds / args.steps,
        "steady_step_ms": 1000.0 * elapsed / args.steps,
        "steady_steps_per_s": args.steps / elapsed,
        "steps": args.steps,
        "unique_graphs": counters["stats"]["unique_graphs"],
        "value_batch": args.value_batch,
        "warmup_seconds": warmup_seconds,
        "warmup_steps": args.warmup,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--value-batch", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-socket", type=Path)
    parser.add_argument("--reuse-staged", action="store_true")
    parser.add_argument("--balance-padded-edges", action="store_true")
    parser.add_argument("--balance-padded-subjects", action="store_true")
    parser.add_argument("--window-rows", type=int, default=30000)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--optimizer", choices=("adamw", "muon_mixed"), default="adamw")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument("--skip-ema-update", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    args = parser.parse_args()
    if args.batch < 1 or args.value_batch < 1:
        parser.error("batch sizes must be positive")
    if args.warmup < 1 or args.steps < 1:
        parser.error("warmup and measured steps must be positive")
    if args.window_rows < 1:
        parser.error("window rows must be positive")
    if args.reuse_staged and args.sample_socket is None:
        parser.error("--reuse-staged requires --sample-socket")
    if args.balance_padded_edges and args.sample_socket is None:
        parser.error("--balance-padded-edges requires --sample-socket")
    if args.balance_padded_subjects and args.sample_socket is None:
        parser.error("--balance-padded-subjects requires --sample-socket")
    return args


def batch_source(args, schema_hash: object, schema: object, device: object):
    if args.sample_socket is None:
        policy_batches = tuple(
            make_batch(schema, args.batch, device, index) for index in range(2)
        )
        value_batches = tuple(
            make_batch(schema, args.value_batch, device, index + 2) for index in range(2)
        )
        return lambda index: (policy_batches[index % 2], value_batches[index % 2])

    sampler = SampleClient(args.sample_socket)
    sampler.wait_until_ready(1, policy_rows=True)
    if sampler.feature_schema_hash != schema_hash:
        raise RuntimeError("checkpoint and replay feature schemas differ")
    if max(args.batch, args.value_batch) > sampler.max_batch:
        raise RuntimeError("requested batch exceeds replay service capacity")
    policy_sample = sampler.sample(args.batch, args.window_rows, args.seed, kind="policy")
    value_sample = sampler.sample(
        args.value_batch,
        args.window_rows,
        args.seed + 1,
        kind="value",
    )
    sampler.close()
    if args.balance_padded_edges:
        balance_padded_edges(policy_sample.batch, schema.edge_type_count)
        balance_padded_edges(value_sample.batch, schema.edge_type_count)
    if args.balance_padded_subjects:
        balance_padded_subjects(policy_sample.batch)
        balance_padded_subjects(value_sample.batch)
    policy_stager = TrainingStager(schema, sampler.max_batch, device)
    value_stager = TrainingStager(schema, sampler.max_batch, device)

    if args.reuse_staged:
        staged = (
            policy_stager.copy(policy_sample.batch, policy_sample.targets),
            value_stager.copy(value_sample.batch, value_sample.targets),
        )
        return lambda _index: staged

    def next_batches(_index: int):
        return (
            policy_stager.copy(policy_sample.batch, policy_sample.targets),
            value_stager.copy(value_sample.batch, value_sample.targets),
        )

    return next_batches


def balance_padded_edges(batch: object, edge_type_count: int) -> None:
    for prefix in ("", "opponent_"):
        count = getattr(batch, f"{prefix}edge_count")
        src = getattr(batch, f"{prefix}edge_src")
        dst = getattr(batch, f"{prefix}edge_dst")
        edge_type = getattr(batch, f"{prefix}edge_type")
        edge_index = np.arange(src.shape[1], dtype=src.dtype).reshape(1, -1)
        valid = edge_index < count.reshape(-1, 1)
        dummy = edge_index % batch.dims.max_nodes
        np.copyto(src, dummy, where=~valid, casting="unsafe")
        np.copyto(dst, dummy, where=~valid, casting="unsafe")
        np.copyto(
            edge_type,
            edge_index % edge_type_count,
            where=~valid,
            casting="unsafe",
        )


def balance_padded_subjects(batch: object) -> None:
    subjects = batch.action_subjects
    slots = np.arange(subjects.shape[2], dtype=batch.subject_count.dtype).reshape(1, 1, -1)
    valid = slots < batch.subject_count.reshape(*batch.subject_count.shape, 1)
    dummy = np.arange(subjects.shape[1] * subjects.shape[2], dtype=subjects.dtype)
    dummy = (dummy % batch.dims.max_nodes).reshape(1, subjects.shape[1], subjects.shape[2])
    np.copyto(subjects, dummy, where=~valid, casting="unsafe")


def make_batch(schema: object, capacity: int, device: object, seed: int) -> TrainingBatch:
    import torch

    features = BatchStager(schema, capacity, device, pinned_staging=False).dummy()
    node_count = min(64, schema.max_nodes)
    edge_count = min(256, schema.max_edges)
    action_count = min(512, schema.max_actions)

    features.node_count.fill_(node_count)
    features.node_tokens.copy_(
        (torch.arange(schema.max_nodes, device=device) % (schema.node_vocab_size - 1) + 1)
        .to(features.node_tokens.dtype)
        .expand(capacity, -1)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    features.node_attrs.normal_(generator=generator)
    features.edge_count.fill_(edge_count)
    edge_index = torch.arange(schema.max_edges, device=device)
    features.edge_src.copy_((edge_index % node_count).to(features.edge_src.dtype).expand(capacity, -1))
    features.edge_dst.copy_(
        ((edge_index + 1) % node_count).to(features.edge_dst.dtype).expand(capacity, -1)
    )
    features.edge_type.copy_(
        (edge_index % schema.edge_type_count).to(features.edge_type.dtype).expand(capacity, -1)
    )
    features.action_count.fill_(action_count)
    features.action_kind.fill_(2)
    features.action_prior.zero_()
    features.subject_count.fill_(1)
    features.action_subjects.zero_()
    features.position.zero_()
    features.opponent_present.fill_(1)
    features.opponent_state_present.fill_(1)
    features.opponent_node_count.copy_(features.node_count)
    features.opponent_node_tokens.copy_(features.node_tokens)
    features.opponent_node_attrs.copy_(features.node_attrs)
    features.opponent_edge_count.copy_(features.edge_count)
    features.opponent_edge_src.copy_(features.edge_src)
    features.opponent_edge_dst.copy_(features.edge_dst)
    features.opponent_edge_type.copy_(features.edge_type)
    features.opponent_position.zero_()

    policy = torch.zeros((capacity, schema.max_actions), device=device)
    policy[:, 0] = 1.0
    value = torch.ones(capacity, device=device)
    value[1::2] = -1.0
    return TrainingBatch(
        features=features,
        policy=policy,
        value=value,
        value_valid=torch.ones(capacity, device=device),
        horizon_value=torch.zeros((capacity, 2), device=device),
        horizon_value_valid=torch.zeros(capacity, device=device),
        reward=-torch.ones(capacity, device=device),
        row_count=capacity,
    )


if __name__ == "__main__":
    raise SystemExit(main())
