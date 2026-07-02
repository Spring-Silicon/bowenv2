# gz-eval-service Spec

Status: draft

Purpose: define the process-backed evaluator — the wire protocol between the
Rust orchestrator and a Python evaluator process, the Python evaluator
serving a deterministic stub model, the featurized eval path inside the
orchestrator, and the synthetic load generator. After this work order, the
entire neural-eval transport is built and verified end to end with zero ML:
`graphzero selfplay` can route every leaf eval through a Python process and
produce episodes identical to an in-process run.

## Architecture Overview

```text
lane thread                    batcher thread                 Python process
───────────                    ──────────────                 ──────────────
worker parks on Eval
  extractor.extract()          collect FeaturizedEvalJobs
  (engine in hand)      ──>    collator.collate_into()
FeatureRow crosses             GZFB batch bytes        ──UDS──> frombuffer/view
(portable, no handles)                                          stub model
                               decode_outputs()        <─UDS──  GZFO output bytes
                               EvalOutput per row               + ModelVersion
                               resume (worker, token)
```

The decisions this spec fixes (review before implementation):

```text
1. New crate gz-eval-service owns the protocol: framing, handshake, the
   client, process spawn/lifecycle, and the stub model's Rust reference
   implementation. It depends on gz-features and gz-engine only.

2. The evaluator is a child process of the orchestrator. Spawned, connected
   once with connect-retry plus handshake on that same connection, killed on
   drop if still running. PING is a live-connection health check, not a
   readiness probe, because the v1 Python server accepts one client and exits
   after that connection closes.

3. The payload IS the gz-features encoding. EVAL frames carry GZFB batch
   bytes verbatim; results carry GZFO bytes verbatim. The service layer
   adds only framing, batch ids, and the handshake. Rust and Python cannot
   disagree about tensor layout without FeatureSchemaHash failing the
   handshake.

4. The stub model is a bit-exact cross-language function over the encoded
   batch (integer arithmetic + power-of-two divisions only), implemented
   once in Rust and once in numpy. It powers the equality oracle: selfplay
   through the socket must equal selfplay through the in-process stub,
   field for field. This extends the project's oracle discipline across
   the language boundary and catches encoding/alignment/routing bugs with
   zero ML noise.

5. The featurized path is a new backend boundary in the orchestrator, not
   a replacement: run()/run_with_replay() over portable Evaluators stay
   untouched. A FeatureEvalBackend trait abstracts "bytes in, outputs out"
   so the in-process stub and the socket client are interchangeable.

6. Single in-flight batch per connection in v1. The protocol carries
   batch_id from day one so pipelining (2-3 in flight) is a client change
   later, not a protocol change. ModelVersion rides every EVAL_RESULT so
   checkpoint hot-swap also lands later without protocol change.

Deliberately deferred: the trainer, checkpoint loading/hot-swap, torch,
batcher pipelining, evaluator restart policy, shared-memory transport,
opponent trajectory registration (job 2).
```

## Role

`gz-eval-service` answers:

```text
How do feature batches reach an out-of-process model and how do outputs
come back, versioned and verified?
```

It owns:

```text
the wire protocol: frame format, types, handshake, error frames
the blocking client (connect, handshake, eval round trip, ping)
evaluator child-process spawn, connect retry, and kill-on-drop
the stub model reference implementation (Rust)
the FeatureEvalBackend trait and its two v1 backends
```

It does not own:

```text
feature schema or encoding (gz-features)
where extraction/collation run (gz-orchestrator)
the Python implementation's internals (python/evaluator, specced below)
models, torch, checkpoints, training
retry/restart policy (fail-fast v1)
```

## Dependency Contract

`gz-eval-service` (Rust):

```text
allowed: std, gz-engine, gz-features
forbidden: tokio, torch/Python bindings, serde, gz-search, gz-replay,
gz-orchestrator, engine adapters
transport is std::os::unix::net; this crate is unix-only, which is
acceptable and documented
```

`python/evaluator`:

```text
allowed: Python stdlib, numpy
forbidden in this work order: torch, any framework, any pip-only dependency
(numpy and pytest are installed as system packages)
```

`gz-orchestrator` integration is work order C. This crate exposes
`RowOutput` values and `ModelVersion`; conversion into `EvalOutput` belongs
to the orchestrator-side featurized path.

## Wire Protocol

Superseded by `GZ_EVAL_PROTOCOL.md`. That spec owns frame types, framing,
handshake validation/adoption rules, error codes, and protocol constants.

## Stub Model

Superseded by `GZ_EVAL_PROTOCOL.md`. The Rust reference implementation
operates on `FeatureBatchView` and is used by the in-process backend and by
conformance-test expectations.

## Rust API

```rust
pub trait FeatureEvalBackend {
    fn eval(
        &mut self,
        batch_bytes: &[u8],
        action_counts: &[u32],
    ) -> ServiceResult<BackendOutputs>;
}

pub struct BackendOutputs {
    pub model_version: ModelVersion,
    pub rows: Vec<RowOutput>,          // gz-features RowOutput
}

pub struct StubBackend { /* collator-compatible; pure Rust */ }

pub struct ProcessBackend { /* owns the connection; single in-flight */ }

pub struct EvaluatorProcess { /* child + socket path; kill on drop */ }

impl EvaluatorProcess {
    pub fn spawn(config: EvaluatorProcessConfig) -> ServiceResult<Self>;
    pub fn connect(&mut self, hello: &Hello) -> ServiceResult<ProcessBackend>;
}

impl ProcessBackend {
    pub fn connect_stream(
        stream: UnixStream,
        hello: &Hello,
        io_timeout: Duration,
    ) -> ServiceResult<Self>;
    pub fn ping(&mut self) -> ServiceResult<()>;
    pub fn model_version(&self) -> ModelVersion;
}

pub struct EvaluatorProcessConfig {
    pub python: PathBuf,           // default "python3"
    pub module: String,            // default "gz.evaluator"
    pub working_dir: PathBuf,      // caller resolves; no default guessing
    pub socket_path: PathBuf,
    pub ready_timeout: Duration,   // default 10s
    pub io_timeout: Duration,      // default 30s
}
```

Rules:

```text
spawn starts the child with --socket <path>, inherits stdout/stderr, and
does not probe readiness.
connect retries UnixStream::connect on the configured socket path until
ready_timeout, then sends HELLO on that same connection and requires
HELLO_ACK.
Drop for EvaluatorProcess kills and reaps the child; no orphans.
ProcessBackend::eval writes one EVAL frame and blocks for its EVAL_RESULT;
ERROR frames or connection loss map to ServiceError::Backend and are
fail-fast for the caller.
ServiceError is small: Handshake, Protocol, Backend, Io(bounded message).
```

## Python Evaluator

Superseded by `GZ_PYTHON.md` and `GZ_PYTHON_FRAMEWORK_IMPL.md`. The Python
implementation now lives under the `gz` package, with protocol code in
`gz.proto`, batch parsing in `gz.codec`, the stub in `gz.model`, and serving
in `gz.evaluator`.

## Orchestrator Featurized Path

Work order C scope; not implemented by Work Order A.

The existing portable-Evaluator paths (`run`, `run_with_replay`) are
untouched. New alongside them:

```text
pool: the Parked payload gains an optional FeatureRow. drive() accepts an
optional extractor (&mut dyn FeatureExtractor<E>); when present, the park
step extracts while graph and candidates are still in hand and stores the
row; extraction errors are fail-fast. action_count for output decoding is
request.actions.len(), retained alongside.

lane -> batcher message: FeaturizedEvalJob { lane, slot, token, row,
action_count } — portable, no engine generics (same structural rule as
EvalJob).

featurized batcher: reuses the existing size/deadline collection logic;
collator.collate_into -> backend.eval -> RowOutput per row ->
EvalOutput { model_version, policy_logits, value } -> replies routed by
(lane, slot, token). Single in-flight v1.

entry point:
  pub fn run_featurized<R, X, B>(
      self,
      root_sources: Vec<R>,
      context: GumbelEpisodeContext,
      extractors: Vec<X>,               // one per lane
      backend: B,
      replay: Option<ReplayRuntime<'_, P>>-shaped optional replay support
  ) -> EngineResult<ThreadedRun<...>>
  where X: FeatureExtractor<E> + Send, B: FeatureEvalBackend + Send;
signature details are implementation-shaped, but replay must compose (the
CLI wants featurized eval + replay in one run) rather than adding a third
and fourth method later.

batch_capacity = the batcher's max_batch; the collator and handshake use
the same value.
```

CLI: `graphzero selfplay --evaluator random|stub|process-stub`. `random`
is the existing default; `stub` uses StubBackend in-process; `process-stub`
spawns the Python evaluator (flags `--python`, `--socket-dir` optional).

## Test Strategy

gz-eval-service unit (no Python):

```text
frame codec roundtrip for every frame type; oversized/undersized frames
rejected
handshake against a tiny in-Rust test server: accept, and each mismatch
field -> Handshake error
stub reference: golden literals for hand-built batches; padded rows all
zero
StubBackend output shapes/truncation via decode_outputs
EvaluatorProcess: spawn failure (bad binary path) errors within timeout;
drop kills the child (no zombie: waitpid reaps)
```

Python (pytest): codec fixture parse + stub literals, as above.

Cross-language conformance (Rust integration test, requires python3+numpy;
fail loudly with an install message if spawn fails, do not skip silently):

```text
spawn the real evaluator; handshake succeeds; PING works
for several seeded synthetic batches: ProcessBackend output ==
StubBackend output, bit-identical
handshake with a corrupted schema hash is rejected with the schema error
code
```

Orchestrator integration:

```text
featurized selfplay with StubBackend on Whittle: completes, deterministic
across two identical runs
the oracle: featurized selfplay through the Python process == featurized
selfplay through StubBackend, episodes field-equal (this is the
acceptance test of the entire work order)
featurized + replay: rows land in the store as in the existing replay
integration tests
CLI smoke: --evaluator stub and --evaluator process-stub both run
```

## Implementation Plan

1. Prerequisite: gz-features implemented, reviewed, committed. Commit any
   dirty tree.
2. `crates/gz-eval-service`: frame codec, Hello types, protocol constants,
   stub reference implementation, StubBackend, FeatureEvalBackend. Unit
   tests including the in-Rust test server.
3. `python/evaluator`: codec, stub, server, __main__, pytest suite, and
   the Rust helper that (re)generates the committed fixture.
4. EvaluatorProcess spawn/readiness/drop-kill + ProcessBackend + the
   cross-language conformance tests.
5. Orchestrator featurized path: pool park hook, FeaturizedEvalJob,
   featurized batcher over FeatureEvalBackend, run_featurized with
   optional replay. StubBackend integration tests.
6. CLI --evaluator flag; the Python end-to-end equality test; load
   generator example (`examples/eval_load.rs`: seeded synthetic rows,
   N batches through a chosen backend, prints p50/p95 latency and
   rows/s).
7. Docs: GZ_ORCHESTRATOR.md dependency amendment; CODEBASE_OUTLINE
   python/evaluator note; AGENTS.md lists this spec. Full verification:
   fmt, test --all, clippy -D warnings, pytest, and a manual
   `graphzero selfplay --evaluator process-stub --episodes 8`.
```

Every stage compiles and passes `cargo test --all` before the next; gz-search,
gz-engine, gz-eval, gz-replay, and the goldens are untouched throughout.

## Deferred

```text
trainer, checkpoint manifest/loading/hot-swap (protocol already carries
ModelVersion)
batcher pipelining (protocol already carries batch_id)
evaluator restart/backoff policy
torch, the real Exphormer model, GPU placement
shared-memory transport
opponent trajectory registration and resolution (job 2)
multiple concurrent client connections
```
