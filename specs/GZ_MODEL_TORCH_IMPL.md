# Torch Model Implementation Spec (Model Work Order B)

Status: implementation work order

Purpose: implement GZ_MODEL.md in `python/gz` — the graph-trunk layer
family with policy/value heads, `gz/checkpoints`, and the evaluator's
torch backend with warm-then-swap-shaped loading. After this work order,
`python -m gz.evaluator --backend torch --checkpoint-dir ...` serves a
real (randomly initialized until the trainer exists) model over the
proven transport.

EXECUTION ORDER: after GZ_FEATURES_EXPANDER_IMPL.md and after
GZ_TRAINING_DATA_IMPL.md (this work order consumes the expander fixture
and the `extra_args` spawn pass-through; the trainer work order follows
this one).

Authority: `GZ_MODEL.md` (architecture, kernels, heads, test contract),
`GZ_PYTHON.md` (layout, layering, torch rule, checkpoints — amended in
stage 2), `GZ_EVAL_PROTOCOL.md` (the checkpoint-backed handshake branch).
Contract wins; report conflicts.

Read before starting:

```text
specs/GZ_MODEL.md                 (the contract this implements)
specs/GZ_PYTHON.md                (checkpoints shape, torch optionality)
python/gz/evaluator/server.py     (handshake adopt branch being extended)
python/gz/codec/                  (BatchView, SchemaDims)
python/tests/fixtures/batch_expander.gzfb   (from work order A)
```

## Stage 0: Environment

This box has an RTX PRO 6000 (Blackwell, sm_120). Blackwell requires
recent CUDA 12.8 torch wheels:

```text
pip install --break-system-packages "torch>=2.7" safetensors
(use the cu128 index URL if the default wheel lacks sm_120 kernels)
verify before anything else, and paste the output into the stage commit:
python3 - <<'EOF'
import torch
print(torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda")
print((x @ x).sum().item())
print(torch.compile(lambda t: t * 2 + 1)(x).sum().item())
EOF
```

If the compile smoke fails on this GPU, stop and report — every later
stage assumes it works. Update pyproject's torch extra to `torch>=2.7`.
The GZ_PYTHON.md torch rules still hold: lazy module-local imports; core
tests never import torch; torch tests `pytest.importorskip` — but on this
box they must actually run, and the acceptance checklist requires it.

## Stage 1: Schema Config In Python

`gz/codec/schema.py` gains `FeatureSchemaConfig`: a frozen dataclass with
every Rust field (name, vocab sizes, attr dim, edge types, max dims,
expander degree/seed) plus `encode()`/`decode()` matching the SHELLO_ACK
serialization in GZ_TRAINING_DATA_IMPL.md byte for byte (one codec, used
by the sampler client later and the checkpoint manifest now). Python
never computes `FeatureSchemaHash`; the hash travels alongside the config
wherever the config goes. Tests: roundtrip, golden bytes literal.

## Stage 2: gz/checkpoints

Per GZ_PYTHON.md, with one amendment made explicitly in that spec: the
manifest embeds the FULL `feature_schema` config (not only its hash),
because the evaluator and trainer construct models from
`(FeatureSchemaConfig, ArchConfig)` and the manifest is where a
checkpoint binds both.

```text
gz/checkpoints/
  manifest.py    dataclass <-> manifest.json; fields per GZ_PYTHON.md
                 plus feature_schema (config dict) and
                 feature_schema_hash hex; strict validation on read
  publish.py     write version_N.tmp/ -> fsync weights + manifest ->
                 atomic rename -> atomic latest.json replace;
                 model_version derived via common.hashing (blake2b over
                 arch_config_hash, feature_schema_hash, weights hash)
  source.py      CheckpointSource ABC + DirectorySource: resolve latest
                 -> (manifest, weights path); verifies weights blake2b
                 and manifest hash-config consistency before returning
  weights.py     safetensors save/load, torch-lazy imports
```

Tests (no torch needed except weights.py's, which skip without it):
publish/resolve roundtrip in a tmpdir; weights tampering detected;
manifest field validation; latest.json replacement is atomic (old version
still resolvable mid-publish); model_version determinism and sensitivity.

## Stage 3: The Model

`gz/model/exphormer.py` implements GZ_MODEL.md exactly:

```text
ArchConfig frozen dataclass (name, dim, layers, heads, ffn_dim, dropout,
activation, aggregation, global_tokens) + canonical encoding +
arch_config_hash (blake2b via common.hashing)
registry.build(schema: FeatureSchemaConfig, arch: ArchConfig) -> module;
"gz-graph-v1" registered alongside "stub"
trunk: embeddings, in-model edge mirroring (type offset =
edge_type_count), segment-softmax attention AND gine aggregation behind
the aggregation switch, K global tokens via dense SDPA with key padding
masks, pre-norm residuals, multiplicative masking only
heads: subjects-gather policy head (zero pool for empty subjects),
tanh value head off the global readout
forward(batch: BatchView-shaped tensors) -> (values [B], logits [B, A])
the model consumes TENSORS, not BatchView: a thin
tensors_from_batch(view, device, pinned_staging) helper owns the
numpy-view -> pinned-buffer copy_ -> non_blocking H2D path (numpy views
from the receive buffer are read-only; torch.from_numpy would fail —
staging is mandatory, preallocated once per capacity)
```

The four binding property tests from GZ_MODEL.md, implemented exactly as
stated there (fp32/CPU where zero tolerance is specified):

```text
padding invariance, batch independence, mask correctness (including
out-of-range subjects pointing at padding), and the compile test:
torch.compile(fullgraph=True) succeeds, compiled == eager within bf16
tolerance, and varying row_count at fixed shapes triggers no recompile
(assert via torch._dynamo counters or compile-time callbacks)
```

Plus: both aggregations pass all four; STOP action logit is finite and
subject-pool-free; expander-typed edges from the work-order-A fixture
flow through end to end.

## Stage 4: Evaluator Torch Backend

`gz/evaluator/backends.py` gains `TorchBackend`:

```text
constructed from (CheckpointSource, device, compile flags); resolves
latest, builds registry model from the manifest's schema + arch config,
loads weights, moves to device, wraps in bf16 autocast +
torch.compile(fullgraph=True, mode per config), and WARMS: one
full-capacity dummy batch through the compiled path before serving
eval(BatchView) -> (model_version from the manifest, GZFO bytes via the
existing OutputEncoder); staging buffers preallocated at construction
```

`server.py` handshake gains the checkpoint-backed branch from
GZ_EVAL_PROTOCOL.md: a torch-backed server VALIDATES the client's
feature_schema_hash against the checkpoint's (ERROR code 3 on mismatch)
instead of adopting it; batch_capacity is still adopted (and bounds the
staging/warmup allocation — reject capacities beyond a configured max).

`__main__.py`: `--backend stub|torch`, `--checkpoint-dir`, `--device`
(default cuda if available), `--no-compile` escape hatch for debugging.

Hot-swap note: the warm-then-swap slot structure from GZ_PYTHON.md is
honored by construction (load+warm happens before serving starts), but
mid-run polling/swap stays out of scope — phase alternation loads latest
at startup.

Tests (torch required, run on this box): a randomly initialized model is
published to a tmpdir with `publish`, served by `serve()` on a thread,
and driven by the existing Python test client: handshake with the right
hash succeeds and HELLO_ACK carries the checkpoint's model_version;
wrong hash gets ERROR 3; an eval on the expander fixture returns finite
values/logits with correct shapes; two identical evals return identical
bytes (inference determinism on one device with compile fixed).

## Stage 5: Bench

`python/benches/model_bench.py`: builds max-size synthetic batches,
reports p50/p95/max per-batch latency and rows/s for B in {64, 256} at
Whittle dims, compile on vs off, eager fp32 vs bf16+compile. Run on the
GPU; paste the table into the commit message — these are the numbers the
trainer work order and any pipelining decision will be judged against.

## Stage 6: Docs And Final Verification

```text
GZ_PYTHON.md: manifest amendment (embedded feature_schema) applied;
gz/checkpoints marked implemented
GZ_MODEL.md: mark implemented-by pointer; note any contract conflicts
found rather than silently deviating
AGENTS.md: this spec listed
```

```bash
python3 -m pytest python/tests            # torch tests included, on GPU
grep -rn "import torch" python/gz | grep -v "exphormer\|weights\|backends"   # empty
cargo test --all                          # Rust untouched, still green
python3 python/benches/model_bench.py
python3 -m gz.evaluator --backend torch --checkpoint-dir /tmp/gz-ckpt-smoke \
  --socket /tmp/gz-torch.sock             # after publishing a random ckpt
```

Acceptance checklist:

```text
all four GZ_MODEL.md property tests pass, for both aggregations
compile test proves fullgraph + no recompile across row_count changes
checkpoint publish -> resolve -> serve roundtrip works with tamper and
hash-mismatch rejection
torch imports remain lazy and confined; core test path torch-free
the torch handshake branch validates instead of adopts, ERROR 3 exact
bench table in the commit message, GPU numbers, compile on/off
Rust workspace untouched
```

## Out Of Scope

```text
the trainer (next work order, needs this + GZ_TRAINING_DATA_IMPL)
mid-run checkpoint hot swap and polling
Rust CLI --evaluator torch wiring (rides with the trainer work order)
FlexAttention, buckets > 1, gine/global ablation runs
opponent conditioning
```
