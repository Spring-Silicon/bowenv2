# Trainer Rust Prerequisites Implementation Spec (Trainer Work Order 1)

Status: implementation work order

Purpose: implement GZ_TRAINER.md's four Rust prerequisites — the
self-average reference, the in-process sample service, unbounded selfplay,
and the torch-evaluator CLI wiring. After this work order the Rust side is
complete for concurrent training; work orders 2 (evaluator hot-swap) and
3 (trainer + supervisor) are Python-only.

Authority: `GZ_TRAINER.md` (the decisions), `GZ_REPLAY.md` (outcome rules
being amended), `GZ_ORCHESTRATOR.md`. Contract wins; report conflicts.

Read before starting:

```text
specs/GZ_TRAINER.md                      (Rust Prerequisites section)
crates/gz-orchestrator/src/reference.rs  (provider trait being extended)
crates/gz-orchestrator/src/lanes.rs      (observe hook call site)
crates/gz-cli/src/serve.rs               (loop being made in-process)
crates/gz-cli/src/selfplay.rs            (new flags)
crates/gz-replay/src/records.rs          (SelfAverage kind)
```

## Hard Constraints

```text
Every stage ends with cargo fmt / cargo test --all /
cargo clippy --all-targets --all-features -- -D warnings and
python3 -m pytest python/tests. Commit per stage; stage 0 commits the
dirty tree (GZ_TRAINER.md, AGENTS.md).
gz-search, gz-features, gz-eval-service, python/ untouched.
No replay SCHEMA_VERSION bump: ReplayReferenceKind::SelfAverage is
appended as the LAST enum variant, which keeps every existing store's
postcard bytes decoding identically. State this in a comment on the enum.
Determinism: per-lane EMA updates happen on the lane thread in episode
completion order — deterministic given the run config, no shared state.
```

## Stage 0: Commit

Commit the dirty tree.

## Stage 1: Self-Average Reference

The provider needs to observe completed episodes, which the trait cannot
do today. Extend it:

```rust
pub trait ReferenceProvider<E: GraphEngine> {
    fn reference(&mut self, engine: &mut E, root: E::Graph)
        -> EngineResult<Option<Reference<E::Graph>>>;

    /// Called by the driver for every replay-eligible completed episode,
    /// with the learner's final measured reward. Default: no-op.
    fn observe(&mut self, learner_reward: f32) {}
}
```

The replay lanes (both the plain and featurized variants) call
`provider.observe(learner_reward)` for every episode that projected
successfully, immediately after projection, on the lane thread.

`Reference<G>` changes: `final_graph` becomes `Option<ReplayGraphContext>`
and `steps` may be empty (self-average has neither). Existing providers
wrap their values in `Some`; projection already maps into the Option-al
`ReplayReference.final_graph`. Adjust the existing tests mechanically.

```rust
pub struct SelfAverageProvider {
    decay: f32,          // default 0.99, validated in (0, 1)
    ema: Option<f64>,
}
```

```text
reference(): ema None -> Ok(None) (first episodes are unlabeled);
otherwise Some(Reference { kind: SelfAverage, final_reward: ema as f32,
final_graph: None, steps: [], search_config_hash: None,
model_version: None }). Never touches the engine.
observe(r): ema = r on first call; else decay * ema + (1 - decay) * r.
f64 accumulator to avoid drift.
one provider per lane (existing structure) -> one EMA per lane, per run.
```

`ReplayReferenceKind::SelfAverage` appended (last variant); GZ_REPLAY.md
outcome rules gain: "SelfAverage: reward EMA of the learner's own recent
episode rewards on that lane; adaptive; unlabeled until the EMA seeds."

CLI: `--reference self-average` and `--reference-ema-decay D`.

Tests (gz-orchestrator + gz-cli):

```text
first completed episode per lane is unlabeled (value_target None), later
episodes labeled; EMA arithmetic pinned by literals
labels flip sign as episode rewards cross the EMA (scripted rewards)
observe is called only for projected (eligible) episodes
a run with --reference self-average produces a store whose labeled rows
validate (store admission is the oracle, as always)
existing provider tests pass with the Option-al final_graph
```

## Stage 2: In-Process Sample Service

Refactor `gz-cli/src/serve.rs` so the accept-and-serve loop is a library
function over `&ReplayStore` (the binary subcommand keeps working
unchanged), then:

```text
graphzero selfplay --serve-socket PATH
  valid ONLY with --episodes 0 (validated at parse; the serving thread
  never joins — the process ends by signal, and the store's WriteBatch
  atomicity makes that safe; put that sentence in the code comment)
  the selfplay process wraps its ReplayStore in Arc, spawns one detached
  serving thread before the run starts, and serves sequential clients on
  PATH for the process lifetime
```

Concurrency is already sound: `append_episode` and `sample_rows` are
`&self` and internally serialized; the backpressure gate reads the same
counters the sampler advances, which makes the ratio control live.

Tests:

```text
concurrent produce/sample: start a threaded selfplay run on a test thread
with --serve-socket semantics (call the library pieces directly), sample
from a client while episodes are still appending; samples succeed,
produced_rows advances between acks, consumed_rows advances with samples
--serve-socket without --episodes 0 is rejected with a usage error
backpressure engages: tiny max_row_backlog, no sampling -> production
stalls at the cap; start sampling -> production resumes (extends the
existing backpressure test to the live-consumer case)
```

## Stage 3: Unbounded Selfplay

```text
--episodes 0 = unbounded: an UnboundedRoots RootSource (never returns
None) per lane. The run only ends by signal. Summary printing is
unreachable in this mode and that is fine; periodic stats are a deferred
item in GZ_TRAINER.md.
Unit-test the root source; the unbounded path itself is exercised by the
stage-2 concurrency test (bounded wall-clock, killed by the test).
```

## Stage 4: Torch Evaluator Wiring

```text
graphzero selfplay --evaluator torch --checkpoint-dir DIR
  [--eval-device DEV]
spawns the evaluator child with extra_args = ["--backend", "torch",
"--checkpoint-dir", DIR, "--device", DEV]; DEV defaults to cuda:0.
--checkpoint-dir required with --evaluator torch, rejected otherwise.
```

Tests: argument construction unit test (the spawned command line is
inspectable); the end-to-end torch run is work order 3's supervisor test —
do not attempt it here (it needs a published checkpoint, which needs the
trainer).

## Stage 5: Docs And Final Verification

```text
GZ_REPLAY.md outcome-rules amendment (stage 1); AGENTS.md lists this spec;
CODEBASE_OUTLINE gz-cli section gains the new flags.
```

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
python3 -m pytest python/tests
```

Acceptance checklist:

```text
self-average labels work end to end through store admission; first
episodes unlabeled; enum appended last with the compatibility comment
sample service runs inside the selfplay process; concurrent
produce/sample test passes; live backpressure test passes
--episodes 0, --serve-socket, --evaluator torch, --checkpoint-dir,
--eval-device, --reference self-average, --reference-ema-decay all parse
and validate; usage message updated
no schema/version bumps anywhere; python tests untouched and green
```

## Out Of Scope

```text
evaluator hot-swap (work order 2)
the trainer and supervisor (work order 3)
periodic selfplay stats; graceful SIGTERM handling
per-root EMA keying or persistence
```
