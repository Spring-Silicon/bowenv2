# GraphZero Codebase Outline

GraphZero is a Rust selfplay/search pipeline with a Python evaluator and
trainer. The production path is generated-root, two-player symmetric Whittle
selfplay with one shared `gz-graph-v2` Exphormer policy/value model.

## Runtime Flow

```text
Whittle root generator
  -> symmetric Gumbel MCTS workers
  -> batched feature extraction
  -> Python torch evaluator
  -> terminal GraphEngine::measure for both players
  -> paired replay append
  -> replay sample service
  -> Python trainer
  -> atomic checkpoint publication
  -> evaluator hot-swap
```

Both player perspectives are written atomically. Runtime rewards enter replay
only after `GraphEngine::measure` returns a terminal scalar. The trainer and
evaluator share the same model definition and checkpoint manifest code.

## Rust Workspace

### `gz-engine`

Dependency-light engine boundary. It owns `GraphEngine`, batch traits, portable
graph/action identities, engine options, measurement results, hashes, and
contract-test helpers. It has no dependency on search, replay, Python, or a
concrete engine.

### `gz-engine-whittle`

Whittle boolean graph implementation. It owns graph/candidate arenas, rule
enumeration and application, generated roots, graph measurement, compaction,
serialization, and the Whittle feature extractor.

### `gz-search`

Search algorithms over `GraphEngine` handles:

- greedy search;
- beam search;
- Gumbel MCTS, including async task polling and symmetric selfplay;
- PUCT MCTS, sharing the generic MCTS task/tree machinery.

STOP is a search action appended after engine candidates. Search never embeds
Whittle-specific candidate semantics. Symmetric search alternates player
perspective, backs values up with the corresponding sign, and optionally
promotes the selected subtree for tree reuse.

### `gz-eval`

Engine-independent policy/value request and result types. An eval request owns
portable action metadata plus position features. Symmetric requests can attach
the other player's graph as explicit board state.

### `gz-features`

Validated feature rows, schema hashing, fixed-layout batch collation, training
target encoding, bf16 conversion, and output decoding. The live model consumes
one joint board containing the current and other player graphs.

### `gz-eval-service`

Framed Unix-socket evaluator protocol and process backend. It supports bounded
batch submission, pipelined completion, explicit model generations, and model
release after episode leases drain.

### `gz-eval-whittle`

Whittle measurement-based evaluator used by tests and local search examples.

### `gz-orchestrator`

Serial, batched, and threaded execution drivers. The threaded production path
owns lane worker pools, bounded evaluator queues, model-version leases,
admission shaping, wave batching, handle release, feature extraction, and the
replay sink.

### `gz-measurer`

Projects completed measured symmetric games into paired replay artifacts. It
computes the sign outcome, applies the rewrite-count tiebreak, derives horizon
auxiliary targets, records measurement diversity, and admits both perspectives
atomically.

### `gz-replay`

RocksDB replay storage. It validates portable episode/row records, writes rows
and indexes atomically, enforces whole-episode retention, samples a bounded
recent row window with batched MultiGet, persists produced/consumed counters,
and exposes symmetric telemetry.

### `gz-cli`

The `graphzero` binary provides:

- `selfplay` for generated-root symmetric selfplay;
- `replay-init` for schema initialization;
- `replay-serve` for trainer sampling;
- `distill-generate` for reducing-uniform policy data generation.

## Python Package

### `gz.model`

The only live architecture is `gz-graph-v2`: joint-board Exphormer trunk,
pointer policy head, and tanh scalar value head. Optional V8/V32/terminal-score
auxiliary heads share the trunk. Serialized architecture fields that are fixed
at runtime remain in checkpoint metadata for compatibility.

### `gz.evaluator`

Stages fixed-layout batches, executes compiled torch inference, serves framed
requests, polls checkpoint pointers, and keeps in-flight batches on their
requested model generation.

### `gz.trainer`

Loads one-layer inherited TOML configs, supervises Rust/Python children,
prefetches replay batches, trains policy/value/optional auxiliary heads,
publishes bounded actor checkpoints, logs metrics, and supports checkpointed
resume. AdamW and torch Muon optimizers are retained.

### `gz.checkpoints`, `gz.codec`, and `gz.proto`

Shared checkpoint manifests and atomic publication, Rust-compatible feature and
target codecs, and framed protocol helpers.

## Configuration

Canonical files:

```text
configs/bases/whittle-generated-exphormer-v2-symmetric-selfplay.toml
configs/whittle-generated-exphormer-v2-symmetric-selfplay.toml
configs/bases/distill-generated-reducing-uniform-gz.toml
configs/distill-generated-reducing-uniform-100k.toml
```

An experiment config may `extends` exactly one base. Bases define reusable
runtime/model policy; leaf configs define run identity, paths, duration, and
the intended ablation.

## Verification

```bash
cargo fmt --all --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
PYTHONPATH=.:python uv run --project python pytest -q python/tests
```
