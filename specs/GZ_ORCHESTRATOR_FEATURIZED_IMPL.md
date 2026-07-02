# Featurized Eval Path Implementation Spec (Work Order C)

Status: implementation work order

Purpose: wire the proven eval transport into selfplay. Feature extraction
runs on lanes at park time, a featurized batcher collates rows and drives a
`FeatureEvalBackend`, and `graphzero selfplay --evaluator process-stub`
routes every leaf eval through the Python evaluator. The acceptance test is
the end-to-end oracle: selfplay through the socket equals selfplay through
the in-process stub, episodes field for field.

This work order also folds in the three findings from the work order A
review (stage 1), because this is the work order that makes them matter.

Authority: `GZ_ORCHESTRATOR.md`, `GZ_FEATURES.md`, `GZ_EVAL_PROTOCOL.md`,
and `GZ_EVAL_SERVICE.md` own the contracts. Contract wins on conflict;
report conflicts.

Read before starting:

```text
specs/GZ_EVAL_SERVICE.md                   (the run_featurized sketch)
crates/gz-orchestrator/src/pool.rs         (park site being extended)
crates/gz-orchestrator/src/lanes.rs        (driver being extended)
crates/gz-eval-service/src/backend.rs      (FeatureEvalBackend)
crates/gz-eval-service/src/process.rs      (ProcessBackend; finding 1 fix)
crates/gz-features/src/collator.rs         (FeatureCollator, BatchHeader-ish
                                            internals for finding 1)
crates/gz-engine-whittle/src/features.rs   (WhittleFeatureExtractor)
```

## Hard Constraints

```text
Stage order below; every stage ends with cargo fmt / cargo test --all /
cargo clippy --all-targets --all-features -- -D warnings and
python3 -m pytest python/tests. Commit per stage; stage 0 commits any
dirty tree.
gz-search, gz-engine, gz-eval, gz-replay untouched; gumbel goldens
untouched. gz-features may only gain the finding-1 validation helper.
Existing entry points run / run_with_replay and their tests are untouched.
The lane -> batcher message for the featurized path carries no engine
generics (same structural rule as EvalJob and ReplayJob).
Only portable data crosses threads: FeatureRow is portable by contract;
Reference values and engine handles stay on their lanes.
All channels bounded, capacities justified in comments, std threading
only.
Dependency amendments (stage 2): gz-orchestrator gains default deps on
gz-features and gz-eval-service; gz-cli gains gz-features and
gz-eval-service. Amend GZ_ORCHESTRATOR.md's dependency contract (same
precedent as gz-replay).
Fail-fast everywhere, matching the existing drivers; extraction errors
abort the run.
```

## Stage 0: Commit

Commit any dirty tree.

## Stage 1: Work Order A Findings

In `gz-features`:

```rust
/// Validates per-row action counts against an encoded batch without
/// copying sections: parses the header, then compares expected against
/// the action_count section read in place.
pub fn validate_batch_action_counts(
    bytes: &[u8],
    expected: &[u32],
) -> FeatureResult<()>;
```

Bounds-checked, allocation-free, reusing the existing offset arithmetic.
`ProcessBackend::eval` switches to it (the full `FeatureBatchView::parse`
copy per eval goes away; `StubBackend` keeps the full parse — it needs the
sections). Add a gz-features unit test (mismatch at each position, length
mismatch, count > max_actions) and keep the existing eval-service tests
green unchanged.

In `gz-eval-service` conformance:

```text
extend the equivalence test to several deterministic batches: at least
four, covering row_count = 1, a partial batch, a full batch, and varied
node/action counts from seeded arithmetic
add the kill-order lifecycle test: spawn + connect, capture the child
pid, drop the EvaluatorProcess while the backend is alive; assert drop
returns promptly, /proc/<pid> no longer exists (unix-only crate), and the
backend's next call fails with an Io or Protocol error
```

## Stage 2: Pool Hook And Featurized Batcher

Pool (`pool.rs`):

```text
ParkedEval gains row: Option<FeatureRow> and action_count: u32
(request.actions.len(), cast-checked).
drive gains an optional extractor parameter:
  extractor: Option<&mut dyn FeatureExtractor<E>>
When present, the park step maps request.position (EvalPositionContext ->
PositionFeatures) and calls extract(engine, work.graph, &work.candidates,
position) while both are still in hand; the row is stored in the parked
slot and carried into ParkedEval. Extraction errors propagate (fail-fast).
When absent, behavior is byte-identical to today; existing callers pass
None and existing tests must not change.
```

Messages and batcher (`lanes.rs`):

```rust
struct FeaturizedEvalJob {
    lane: usize,
    slot: usize,
    token: WorkToken,
    row: FeatureRow,
    action_count: u32,
}
```

```text
run_featurized_batcher(backend: B, collator: FeatureCollator, intake,
reply_txs, config): same size/deadline collection loop as run_batcher;
collate_into with a reused byte buffer and reused row/count scratch;
backend.eval(bytes, action_counts); convert each RowOutput plus the
batch's model_version into EvalOutput (gz-eval constructor, validated at
task resume as today); route replies by (lane, slot, token). Single
in-flight batch, v1.
Backend errors are fail-fast: return the error, drop reply senders,
lanes unwind exactly like the existing batcher.
batch_sizes recorded as today.
```

Config invariants, validated before any thread spawns:

```text
collator batch capacity == config.max_batch
every lane extractor reports the same FeatureSchemaHash ->
internal("feature schema mismatch") otherwise
every lane engine reports the same EngineIdentity ->
internal("engine identity mismatch") otherwise (also fixes a latent gap:
the existing drivers never checked this)
```

## Stage 3: Entry Points

Mirroring the existing run / run_with_replay convention:

```rust
pub struct FeaturizedRuntime<X, B> {
    pub extractors: Vec<X>,     // one per lane
    pub backend: B,
}

impl<E, V> ThreadedGumbelOrchestrator<E, V> ... {
    pub fn run_featurized<R, X, B>(
        self, root_sources, context, featurized: FeaturizedRuntime<X, B>,
    ) -> EngineResult<ThreadedRun<...>>;

    pub fn run_featurized_with_replay<R, X, B, P>(
        self, root_sources, context, featurized: FeaturizedRuntime<X, B>,
        replay: ReplayRuntime<'_, P>,
    ) -> EngineResult<ThreadedReplayRun<...>>;
}
```

where `X: FeatureExtractor<E> + Send`, `B: FeatureEvalBackend + Send`.
The four run methods share their lane/batcher internals; no copy-pasted
lane loops. `V: Evaluator` is unused on the featurized paths — restructure
generics however reads best (e.g., the featurized methods ignore
`self.evaluator`), but do not break the existing constructors.

Note for callers using `ProcessBackend`: the `EvaluatorProcess` must
outlive the run; the caller holds it and passes the connected backend in.

Stage 3 tests (`tests/featurized.rs`, Whittle + WhittleFeatureExtractor +
StubBackend):

```text
featurized selfplay completes; two identical runs produce field-equal
ThreadedRun results (determinism)
featurized + replay: rows land in a temp store exactly as in the existing
replay integration tests (store validation is the oracle)
schema mismatch across lanes rejected; capacity mismatch rejected;
lane-count mismatches rejected
extraction failure aborts the run (extractor stub erroring on a known
graph, via a tiny wrapping extractor in the test)
```

## Stage 4: The End-To-End Oracle

`tests/featurized_process.rs` (python3 + numpy required; fail loudly,
never skip, same pattern as conformance):

```text
run featurized selfplay with StubBackend; run it again from identical
config/seeds with a spawned Python evaluator via ProcessBackend; assert
the ThreadedRun episodes are field-equal across the two runs.
This passes only if extraction, collation, transport, stub arithmetic,
output decoding, and reply routing are all identical — but transport and
stub are already sealed by work order A, so a failure here indicts the
orchestrator wiring specifically.
also assert every episode's steps carry model_version == the stub
constant (proving model_version flows from the backend into search
results and thus into future replay rows).
```

## Stage 5: CLI

`graphzero selfplay --evaluator random|stub|process-stub` (default
`random`, existing behavior untouched):

```text
random        existing RandomValueEvaluator path
stub          run_featurized(_with_replay) with StubBackend
process-stub  spawns the evaluator (working_dir python/ resolved relative
              to the binary's cwd with a --python-dir override; socket in
              a temp dir), connects, runs, and reports the child's exit
extractors: WhittleFeatureExtractor::new(engine) per lane
summary line gains evaluator=<kind> and, for featurized runs, the model
version
smoke test for --evaluator stub via the existing run() unit-test pattern;
process-stub is covered by a manual command in final verification, not a
unit test (CI-side python spawning stays in the Rust integration tests)
```

## Stage 6: Load Generator

`crates/gz-eval-service/examples/eval_load.rs`:

```text
args: --backend stub|process [--python-dir PATH] --batches N --batch-size B
builds deterministic synthetic FeatureRows (seeded arithmetic, no rand),
collates once per batch with reused buffers, drives the backend, and
prints: batches, rows/s, p50/p95/max per-batch latency (microseconds)
run once against both backends; paste both outputs into the commit
message (the process numbers are the baseline the torch backend and
pipelining work will be measured against)
```

## Stage 7: Docs And Final Verification

```text
GZ_ORCHESTRATOR.md: dependency contract amendment (gz-features,
gz-eval-service default); Role list gains the featurized eval path.
GZ_EVAL_SERVICE.md: mark the orchestrator section implemented, pointing
here.
CODEBASE_OUTLINE.md: gz-cli section gains the --evaluator flag.
AGENTS.md: this spec in the list.
```

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
python3 -m pytest python/tests
target/debug/graphzero selfplay --replay-dir /tmp/gz-feat-smoke \
  --episodes 8 --evaluator process-stub
cargo run -p gz-eval-service --example eval_load -- --backend process \
  --batches 50 --batch-size 64
```

Acceptance checklist:

```text
the stage-4 oracle passes: socket selfplay == in-process selfplay
run / run_with_replay behavior and tests unchanged; goldens untouched
FeaturizedEvalJob carries no engine generics; FeatureRow is the only
feature payload crossing threads
ProcessBackend::eval no longer full-parses batches (finding 1); kill-order
and multi-batch conformance tests exist (findings 2, 3)
schema-hash, capacity, and engine-identity invariants validated before
threads spawn
collation reuses buffers across batches; no per-batch allocation beyond
what FeatureRow Vecs force
CLI process-stub smoke run works end to end
load generator numbers pasted into the commit message
```

## Out Of Scope

```text
torch backend, checkpoints, hot swap (next work orders)
batcher pipelining (protocol and batch_id are ready for it)
evaluator restart policy; multiple sequential connections
opponent trajectory registration (job 2)
shape buckets; feature cache tuning
trainer
```
