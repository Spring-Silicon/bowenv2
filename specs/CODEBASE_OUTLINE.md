# GraphZero Codebase Outline

Status: draft

Purpose: define the initial Rust workspace for an async selfplay/search
pipeline built around a modular `GraphEngine`. The first implementation does
not need an actual compiler backend. It should prove the engine/search/eval/
measure/replay architecture with a Whittle-backed test engine first.

## Decisions

```text
Name: graphzero
Crate names: gz-*
Initial real test engine: WhittleTestEngine, not ../graphs
Foundation crate: gz-engine, not gz-core
Measurement: GraphEngine::measure
Features: separate FeatureExtractor<E>
Candidates: fully engine-owned abstraction
Engine trait: sync GraphEngine, async EngineServer wrapper
Learner: Python trainer sidecar later; no Rust learner crate initially
Neural eval: EvalClient boundary; Python evaluator sidecar allowed
Replay storage: RocksDB + compact binary row encoding
Actor model staleness: leave vague for now
Replay insert: only after the row graph has a MeasureResult
Compiler backend: future work
```

## Design Rules

```text
1. Search depends on GraphEngine, not concrete game/compiler state.
2. Search stores E::Graph and E::Candidate handles only.
3. Candidate semantics are engine-owned; search never assumes site/family shape.
4. Feature extraction is separate from the engine.
5. Measurement is part of GraphEngine.
6. Rows enter replay only after measurement succeeds or fails explicitly.
7. Replay stores portable graph/action contexts, not process-local handles.
8. Whittle is the first concrete adapter; add a fake adapter later only when it
   materially simplifies search/orchestration tests.
9. GraphEngine is sync; async scheduling, queues, timeouts, and batching live
   in EngineServer/orchestrator layers.
10. Rust selfplay/search never imports Python or PyTorch. Neural inference is
    accessed through EvalClient.
11. Python trainer never owns replay storage format directly. It samples through
    a Rust replay sampling boundary.
12. Engineer for maximum measured performance subject to correctness. Avoid
    convenience boundaries or abstractions that add avoidable hot-path overhead.
```

## Workspace

```text
graphzero/
  Cargo.toml
  crates/
    gz-engine/
    gz-engine-whittle/
    gz-features/
    gz-search/
    gz-eval/
    gz-replay/
    gz-orchestrator/
    gz-cli/
  python/
    evaluator/
    trainer/
  configs/
  specs/
  tests/
  tools/
```

No learner crate initially. Training can be added after the async data pipeline
is working and measured replay rows exist.

## Crates

### `gz-engine`

Foundational crate for engine traits and engine-boundary types.

`GZ_ENGINE.md` owns the detailed crate contract.

`GraphEngine` is intentionally sync. It defines deterministic graph operations,
not scheduling. Async behavior comes from an `EngineServer` wrapper owned by the
orchestrator.

Owns:

```rust
GraphEngine
BatchGraphEngine
GraphHash
CandidateHash
ActionSetHash
MeasureConfigHash
SearchConfigHash
EngineId
EngineVersion
ModelVersion
PortableGraphId
ReplayGraphContext
PortableCandidateRef
PortableSearchActionRef
SearchStepRef
CandidateOptions
MeasureOptions
CandidateInfo
ApplyResult
MeasureResult
MeasureSummary
LatencyStats
EngineError
EngineResult
```

Does not own:

```text
runtime actor ids
episode ids
row ids
replay storage schema
metrics backend
concrete engine implementation
async runtime choice
```

Rules:

```text
no torch
no Python
no concrete engine implementation
no async runtime choice unless unavoidable
```

```rust
pub trait GraphEngine {
    type Graph;
    type Candidate;

    fn root(&self) -> Self::Graph;
    fn hash(&self, graph: Self::Graph) -> EngineResult<GraphHash>;
    fn candidates(&mut self, graph: Self::Graph, opts: CandidateOptions, out: &mut Vec<Self::Candidate>) -> EngineResult<()>;
    fn candidate_info(&self, graph: Self::Graph, candidate: Self::Candidate) -> EngineResult<CandidateInfo>;
    fn apply(&mut self, graph: Self::Graph, candidate: Self::Candidate) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>>;
    fn measure(&mut self, graph: Self::Graph, opts: MeasureOptions) -> EngineResult<MeasureResult<Self::Graph>>;
}

pub trait BatchGraphEngine: GraphEngine {
    fn candidates_batch(...);
    fn apply_batch(...);
    fn measure_batch(...);
}
```

This crate owns engine-neutral interfaces, result types, options, hashes,
portable graph/candidate/action refs, errors, and contract tests. Small
cross-pipeline identity newtypes like `ModelVersion` and `SearchConfigHash`
live here only to avoid dependency cycles. Runtime/orchestration ids live in
`gz-orchestrator`. Replay-specific ids and schemas live in `gz-replay`.

The async wrapper shape belongs outside this crate:

```rust
SearchActor
  -> async EngineClient request
  -> EngineServer task
  -> sync GraphEngine call
  -> async reply
```

Reasons:

```text
Whittle engine stays simple
no async-trait or boxed future overhead in engine contracts
batching/backpressure stay centralized
process boundaries can be added later behind EngineClient
```

### `gz-engine-whittle`

Adapter over existing Whittle game/search environment for the first non-fake
engine.

`GZ_ENGINE_WHITTLE.md` owns the detailed crate contract.

Purpose:

```text
test the abstract engine/search/eval/measure/replay pipeline on a real domain
without waiting for a compiler backend
match existing Whittle native rewrite semantics without Python in the hot path
```

Responsibilities:

```text
map Whittle state -> E::Graph
map Whittle move/action -> E::Candidate
generate random-walked Whittle training roots
enumerate legal candidates through Whittle
apply Whittle candidates
measure Whittle states through Whittle domain logic
provide CandidateInfo for policy/replay
export/import Whittle graph artifacts for diagnostics and replay workflows
```

This adapter exists to validate the architecture, not to become graphzero's
final domain.

### `gz-features`

Generic feature extraction, separate from the engine.

```rust
pub trait FeatureExtractor<E: GraphEngine> {
    type StateFeatures;
    type CandidateFeatures;
    type Batch;

    fn state_features(&mut self, engine: &E, graph: E::Graph) -> Result<Self::StateFeatures>;

    fn candidate_features(
        &mut self,
        engine: &E,
        graph: E::Graph,
        candidates: &[E::Candidate],
    ) -> Result<Self::CandidateFeatures>;

    fn collate(&mut self, rows: &[FeatureRow<E>]) -> Result<Self::Batch>;
}
```

Owns:

```text
FeatureSchemaHash
state feature cache by PortableGraphId + FeatureSchemaHash
candidate feature cache by PortableCandidateRef + FeatureSchemaHash
action-history encoding
batch padding
shape metadata
feature schema metadata exported to evaluator/trainer
```

### `gz-search`

Search algorithms over `GraphEngine`.

```text
random rollout
greedy policy rollout
beam search
Gumbel MCTS
async/tree-parallel MCTS later
```

Forbidden deps:

```text
gz-engine-whittle
future compiler engine adapters
replay storage
training code
```

Search sees candidates only through:

```rust
engine.candidates(...)
engine.candidate_info(...)
engine.apply(...)
```

Search does not care whether a candidate is a site-level rewrite, family-level
rewrite, Whittle move, or future compiler transform.

### `gz-eval`

Policy/value evaluator used by search.

`GZ_EVAL.md` owns the detailed crate contract.

The first boundary is blocking and batch-first so serial Gumbel-MCTS can call it
directly. Async batching and Python-backed inference can be added later without
changing the action-aligned policy/value output shape.

```rust
pub trait Evaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()>;
}

pub struct EvalOutput {
    pub model_version: ModelVersion,
    pub policy_logits: Vec<f32>,
    pub value: f32,
}
```

Initial adapters:

```text
RandomValueEvaluator
RecordedEvaluator
PythonProcessEvaluator once full-scale Exphormer is used
```

Owns:

```text
action-aligned eval request/output records
blocking evaluator trait
output validation
cheap deterministic evaluators
model version tag
```

Does not own search trees, candidate enumeration, terminal measurement, replay,
or training. No learner yet.

### `gz-replay`

Durable measured-row store.

Storage decision:

```text
RocksDB
compact binary encoding with bincode or postcard
column families for rows, episodes, indexes, metadata
```

Reason:

```text
fast append
fast random reads
ordered keys for windows/ranges
durable enough for long runs
lower overhead than SQLite for high-volume row blobs
```

Rows enter replay only after `GraphEngine::measure` returns a `MeasureResult`
for the row graph.

```rust
pub struct ReplayRow {
    pub root: ReplayGraphContext,
    pub measured_graph: ReplayGraphContext,
    pub action_history: Vec<PortableSearchActionRef>,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<(PortableSearchActionRef, f32)>,
    pub value_target: Option<f32>,
    pub reward_target: Option<f32>,
    pub measurement: MeasureSummary,
    pub model_version: Option<ModelVersion>,
    pub search_config_hash: SearchConfigHash,
}
```

`ReplayRow` is a training/sampling record by default. It is not guaranteed to
reconstruct engine-local `E::Graph` or `E::Candidate` handles. Workflows that
need resume, deterministic replay, remeasure, or debug export must store an
episode trace and use an adapter-owned resolver/artifact path.

```rust
pub struct EpisodeTrace {
    pub root: ReplayGraphContext,
    pub steps: Vec<SearchStepRef>,
    pub final_graph: ReplayGraphContext,
}
```

Owns:

```text
append-only episode log
finalized measured rows
window/range indexes
sampling indexes
schema versioning
ReplaySampleService for Python trainer batches
```

Python training should not read RocksDB directly. Rust owns replay schema,
indexes, ratio control, and sampling policy. Python sees batches returned by a
stable sampling API.

### `gz-orchestrator`

Async supervisor and backpressure controller.

Owns:

```text
actor pool
eval service lifecycle
engine server lifecycle
measure queue
replay writer
ratio/backpressure controller
metrics
shutdown/restart
```

No learner-specific controller initially. Keep ratio language generic:

```rust
produced_rows = measured_rows_inserted_into_replay
consumed_rows = rows_read_by_downstream_consumer
sample_ratio = consumed_rows / produced_rows
```

The first downstream consumer may be a mock consumer.

## Python Training And Eval Boundary

The long-term full-scale Exphormer setup has four concurrent roles:

```text
Rust selfplay/orchestrator process
  runs many MCTS/search actors
  calls EngineServer for candidates/apply/measure
  calls EvalClient for leaf evals
  appends measured rows to replay

Python evaluator process
  loads current PyTorch/Exphormer checkpoint
  receives batched leaf eval requests
  returns policy/value/reward with ModelVersion
  hot-swaps model versions after trainer publishes checkpoints

Python trainer process
  requests batches from Rust ReplaySampleService
  runs optimizer steps
  publishes versioned checkpoints

Replay/checkpoint storage
  stores measured rows, sampling indexes, and model artifacts
```

Hard boundary:

```text
gz-engine: no Python, no torch
gz-search: no Python, no torch
gz-replay: owns storage schema; Python samples through service API
gz-eval: default build has no Python or torch; a future adapter may implement
PythonProcessEvaluator behind the Evaluator boundary
```

Leaf eval flow:

```text
SearchActor[0..N]
  -> EvalClient.evaluate(...)
  -> Rust eval batcher queue
  -> Python evaluator batch request
  -> PyTorch/Exphormer forward
  -> batch response routed back to actors
```

Do not call Python one leaf at a time. The Rust side owns batching,
backpressure, request ids, timeouts, and response routing.

Checkpoint flow:

```text
trainer writes checkpoints/run_id/version_N.tmp/
trainer fsyncs weights and manifest
trainer atomically renames to checkpoints/run_id/version_N/
trainer atomically updates checkpoints/run_id/latest.json
evaluator loads and warms version_N
evaluator swaps new requests to version_N
in-flight evals finish on their original ModelVersion
```

Replay rows store the `ModelVersion` used to produce search policy/value data.
Feature batches and model checkpoints must agree on:

```text
EngineVersion
ActionSetHash
FeatureSchemaHash
ModelVersion
```

If those tags disagree, the evaluator or trainer must fail fast.

### `gz-cli`

Human entry points.

```bash
graphzero smoke-async --engine fake
graphzero smoke-async --engine whittle
graphzero probe-actions --engine whittle
graphzero apply-path --engine whittle --path "..."
graphzero measure --engine whittle --state ...
graphzero rollout --engine whittle --mode random
```

Future compiler commands can be added after a compiler engine exists.

## Dependency Direction

```text
gz-engine
  <- gz-engine-whittle
  <- gz-features
  <- gz-search
  <- gz-eval
  <- gz-replay
  <- gz-orchestrator
      <- gz-cli
```

Forbidden dependencies:

```text
gz-search -> gz-engine-whittle
gz-search -> gz-replay
gz-engine -> gz-search
gz-engine -> gz-replay
gz-engine -> gz-engine-whittle
```

## First Vertical Slice

```text
WhittleTestEngine
RandomValueEvaluator
WhittleFeatureExtractor
InMemoryReplay or temp RocksDB
Orchestrator
smoke-async --engine whittle
```

Acceptance:

```text
actors generate Whittle episodes
engine measures selected Whittle graphs
only measured rows enter replay
mock consumer can read rows
backpressure can throttle actors/consumer
all queues drain on shutdown
```

## Second Vertical Slice

```text
RocksDB replay
recorded evaluator
rollout/search smoke
```

Acceptance:

```text
enumerate Whittle candidates
apply Whittle candidates deterministically
measure selected Whittle graphs through GraphEngine::measure
write measured rows to RocksDB
run search without depending on Whittle concrete types
```

## Deferred

```text
actual compiler engine
../graphs adapter
Qwen runtime correctness
learner/training loop
model export/promotion
actor model staleness policy
```

## Remaining Design Questions

1. Should RocksDB keys be episode-major or graph-hash-major first?
2. Should feature caches live inside `FeatureExtractor` only, or can
   `Orchestrator` own shared cross-actor feature caches?
3. What should the fake engine model: tree game, DAG rewrite game, or tiny
   Whittle-like domain?
