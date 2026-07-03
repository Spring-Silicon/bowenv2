from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

from gz.codec import FeatureSchemaConfig
from gz.model.exphormer import ArchConfig, BatchStager, build_model


@dataclass(frozen=True, slots=True)
class Result:
    batch: int
    mode: str
    p50_ms: float
    p95_ms: float
    max_ms: float
    rows_per_s: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema = whittle_schema()
    arch = ArchConfig(dim=args.dim, layers=args.layers, heads=4, ffn_dim=args.dim * 4, dropout=0.0)
    results = []
    for batch in (64, 256):
        for mode, bf16, compile_model in (
            ("fp32-eager", False, False),
            ("bf16-eager", True, False),
            ("bf16-compile", True, True),
        ):
            results.append(run_one(torch, schema, arch, batch, device, mode, bf16, compile_model, args.warmup, args.iters))

    print("| batch | mode | p50 ms | p95 ms | max ms | rows/s |")
    print("|---:|---|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result.batch} | {result.mode} | {result.p50_ms:.3f} | "
            f"{result.p95_ms:.3f} | {result.max_ms:.3f} | {result.rows_per_s:.1f} |"
        )
    return 0


def run_one(
    torch: object,
    schema: FeatureSchemaConfig,
    arch: ArchConfig,
    batch: int,
    device: object,
    mode: str,
    bf16: bool,
    compile_model: bool,
    warmup: int,
    iters: int,
) -> Result:
    model = build_model(schema, arch).to(device).eval()
    runner = torch.compile(model, fullgraph=True, mode="reduce-overhead") if compile_model else model
    stager = BatchStager(schema, batch, device)
    tensors = stager.dummy()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast(torch, device, bf16):
                runner(tensors)
        sync(torch, device)
        for _ in range(iters):
            start = time.perf_counter()
            with autocast(torch, device, bf16):
                runner(tensors)
            sync(torch, device)
            times.append((time.perf_counter() - start) * 1000.0)
    p50 = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    return Result(
        batch=batch,
        mode=mode,
        p50_ms=p50,
        p95_ms=p95,
        max_ms=max(times),
        rows_per_s=batch / (p50 / 1000.0),
    )


def autocast(torch: object, device: object, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return _NullContext()


def sync(torch: object, device: object) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def whittle_schema() -> FeatureSchemaConfig:
    return FeatureSchemaConfig(
        name="whittle-bench",
        node_vocab_size=7,
        node_attr_dim=0,
        edge_type_count=3,
        action_kind_vocab_size=10,
        max_nodes=64,
        max_edges=448,
        max_actions=256,
        max_subjects=8,
        expander_degree=5,
        expander_seed=0,
    )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
