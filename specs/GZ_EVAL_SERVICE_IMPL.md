# gz-eval-service Implementation Spec (Work Order A)

Status: implementation work order

Purpose: implement the Rust side of the eval process boundary — the frame
codec, the stub reference implementation, the `FeatureEvalBackend` trait,
evaluator process spawn/lifecycle, the blocking `ProcessBackend` client,
and the cross-language conformance tests against the existing Python
evaluator. After this work order, a Rust caller can hand encoded feature
batches to either an in-process stub or the real Python process and get
bit-identical results, with every mismatch case rejected by exact error
code.

The orchestrator featurized path, the CLI flag, and the load generator are
work order C, not this one.

Authority: `GZ_EVAL_PROTOCOL.md` owns the wire contract and stub formulas;
`GZ_FEATURES.md` owns the payload encoding; `GZ_PYTHON.md` owns the Python
side. If this document disagrees with a contract, the contract wins;
report the conflict.

Read before starting:

```text
specs/GZ_EVAL_PROTOCOL.md            (frames, handshake, error codes, stub)
specs/GZ_EVAL_SERVICE.md             (crate role; re-sliced in stage 0)
crates/gz-features/src/collator.rs   (FeatureBatchView, FeatureCollator,
                                      RowOutput, decode_outputs)
python/gz/evaluator/server.py        (the server being spoken to; NOTE:
                                      it accepts exactly ONE client, then
                                      the process exits)
python/gz/proto/frames.py            (the Python framing this must match)
```

## Design Corrections To The Old Sketch

Two changes from GZ_EVAL_SERVICE.md's original draft, both forced by
grounding against the implemented Python server:

```text
1. No separate readiness probe. The server accepts one client and exits
   after that connection closes, so "retry connect + PING until ready,
   then connect for real" would consume the only slot. Spawn and connect
   are one flow: connect(hello) retries the UDS connect until
   ready_timeout, then performs the handshake on that same connection,
   which becomes the backend. PING is a health check on the live
   connection, not a readiness probe.
2. No gz-eval dependency. BackendOutputs is ModelVersion (gz-engine) plus
   RowOutput rows (gz-features); converting to EvalOutput is work order
   C's orchestrator-side concern. Dependencies: std, gz-engine,
   gz-features only.
```

## Hard Constraints

```text
Stage order below; every stage ends with cargo fmt, cargo test --all,
cargo clippy --all-targets --all-features -- -D warnings, and
python3 -m pytest python/tests. Commit per stage; stage 0 commits the
current dirty tree first.
New crate gz-eval-service: deps std, gz-engine, gz-features. Forbidden:
tokio, serde, rand, gz-eval, gz-search, gz-replay, gz-orchestrator,
engine adapters. Transport is std::os::unix::net; the crate is unix-only
and says so in lib.rs docs.
No changes to gz-search, gz-engine, gz-eval, gz-replay, gz-orchestrator,
or python/ (one narrow gz-features exception in stage 3, below).
Protocol constants: PROTOCOL_VERSION = 1 defined here;
ENCODING_VERSION is re-exported from gz-features, never redefined.
Buffers are reused: one read buffer and one write buffer per connection,
growing monotonically; one write_all per frame.
Conformance tests fail loudly with an actionable message when python3 or
numpy is missing or spawn fails. They must never silently skip.
std::time::Instant is allowed in this crate (retry deadlines); wall-clock
dates are not.
```

## Stage 0: Commit And Re-Slice

Commit the dirty tree. Then re-slice `specs/GZ_EVAL_SERVICE.md`:

```text
delete the wire protocol, stub model, and Python evaluator sections,
replacing each with a one-line pointer (GZ_EVAL_PROTOCOL.md, GZ_PYTHON.md,
GZ_PYTHON_FRAMEWORK_IMPL.md)
update the dependency contract: gz-eval removed per the correction above
update the process-lifecycle sketch to the single connect flow
mark the orchestrator featurized path section "work order C scope"
add GZ_EVAL_SERVICE_IMPL.md to AGENTS.md's spec list
```

## Stage 1: Crate Skeleton

```text
crates/gz-eval-service/
  Cargo.toml        gz-engine, gz-features (path deps, workspace version)
  src/
    lib.rs          #![forbid(unsafe_code)], unix-only note, re-exports
    error.rs
    frames.rs
    hello.rs
    stub.rs
    backend.rs
    process.rs
  tests/
    common/mod.rs   in-Rust scripted test server (stage 4)
    frames.rs
    stub.rs
    process.rs
    conformance.rs
```

```rust
pub type ServiceResult<T> = Result<T, ServiceError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceError {
    Handshake(String),          // version/field mismatches, bad HELLO_ACK
    Protocol(String),           // framing violations, batch_id mismatch
    Backend { code: u32, message: String },   // server ERROR frames
    Io(String),                 // connect/read/write/spawn failures
}
```

Messages bounded to 512 bytes at construction. Keep the enum this small.

## Stage 2: Frames And Handshake Types

`frames.rs` implements GZ_EVAL_PROTOCOL.md exactly, mirroring
`python/gz/proto/frames.py`:

```rust
pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME: usize = 256 * 1024 * 1024;
pub const FRAME_HELLO: u8 = 1;  // ... through FRAME_ERROR = 7

pub fn read_frame<'a>(
    stream: &mut UnixStream,
    buf: &'a mut Vec<u8>,
) -> ServiceResult<(u8, &'a [u8])>;    // payload excludes the type byte

pub fn write_frame(
    stream: &mut UnixStream,
    buf: &mut Vec<u8>,
    frame_type: u8,
    parts: &[&[u8]],
) -> ServiceResult<()>;                 // assemble into buf, one write_all
```

Rules: zero/oversized body length and unknown frame types are
`Protocol` errors; reads use `read_exact` into the reused buffer; EOF
mid-frame is an `Io` error.

`hello.rs`: `Hello` and `HelloAck` structs using gz-engine id types
(`EngineId`, `EngineVersion`, `ModelVersion`) and
`gz_features::FeatureSchemaHash`, with `encode(&self, &mut Vec<u8>)` /
`decode(&[u8])` in the exact field order of the contract. `decode_error`
for ERROR payloads (code, bounded utf8 message; invalid utf8 is replaced,
not fatal).

Unit tests: round trip every frame type over a `UnixStream::pair`;
truncated/oversized/unknown-type rejection; Hello field order pinned by a
byte-literal test (so a field reorder cannot pass).

## Stage 3: Stub Reference And StubBackend

`stub.rs`:

```rust
pub const STUB_MODEL_VERSION: ModelVersion = /* gz-stub-v1 zero-padded */;

pub fn stub_row_outputs(view: &FeatureBatchView) -> Vec<RowOutput>;
```

The GZ_EVAL_PROTOCOL.md formulas, computed in u64 with wrapping ops,
`((raw as i64 - K) as f32) / K as f32` for the exact power-of-two
divisions. Each `RowOutput.policy_logits` is truncated to the row's true
action count; only rows below `row_count` are returned.

`backend.rs`:

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
    pub rows: Vec<RowOutput>,
}

pub struct StubBackend;   // parse -> validate counts -> stub_row_outputs
```

`action_counts` is the caller's per-row truth (work order C passes the
request action lengths); both backends validate it against the batch
header (`len == row_count`, each count <= max_actions) with a `Protocol`
error on mismatch.

gz-features exception: `ProcessBackend` must decode GZFO bytes without a
`FeatureCollator` (it has the schema hash, not the schema config). If
`decode_outputs` currently requires collator state, lift its header-driven
core into a public free function
`gz_features::decode_outputs(bytes: &[u8], action_counts: &[u32]) ->
FeatureResult<Vec<RowOutput>>` and delegate the method to it. No behavior
change; gz-features tests unchanged.

Unit tests: golden literals for a hand-built batch (assert-print-paste),
padded rows and padded action slots excluded from outputs, formula match
against a second scalar implementation in the test for seeded synthetic
count arrays (the same double-implementation trick as the Python side).

## Stage 4: Process Lifecycle And ProcessBackend

`process.rs`:

```rust
pub struct EvaluatorProcessConfig {
    pub python: PathBuf,          // default "python3"
    pub module: String,           // default "gz.evaluator"
    pub working_dir: PathBuf,     // caller resolves; no default guessing
    pub socket_path: PathBuf,
    pub ready_timeout: Duration,  // default 10s
    pub io_timeout: Duration,     // default 30s, read+write on the stream
}

pub struct EvaluatorProcess { /* Child; kill-and-reap on Drop */ }

impl EvaluatorProcess {
    pub fn spawn(config: EvaluatorProcessConfig) -> ServiceResult<Self>;
    pub fn connect(&mut self, hello: &Hello) -> ServiceResult<ProcessBackend>;
}

pub struct ProcessBackend { /* stream + buffers + batch_id counter */ }

impl ProcessBackend {
    pub fn ping(&mut self) -> ServiceResult<()>;
    pub fn model_version(&self) -> ModelVersion;   // from HELLO_ACK
}

impl FeatureEvalBackend for ProcessBackend { ... }
```

Rules:

```text
spawn: python -m <module> --socket <path>, current_dir = working_dir,
stderr inherited, stdout inherited. Spawn failure is Io with the command
line in the message.
connect: retry UnixStream::connect on NotFound/ConnectionRefused with a
short sleep (10ms) until ready_timeout; then set io_timeout on the
stream, send HELLO, require HELLO_ACK (an ERROR frame here maps to
Handshake with the server's message; anything else is Protocol). Records
the acked model_version. One connect per process, matching the one-client
server; a second connect call is an error.
eval: single in-flight. Send EVAL with a monotonically increasing
batch_id; block for EVAL_RESULT; batch_id mismatch is Protocol; ERROR
maps to Backend { code }; decode GZFO via the gz-features free function;
model_version is taken from the frame (it can change under hot swap
later, so it is per-result, not cached).
ping: nonce round trip; wrong nonce is Protocol.
Drop for EvaluatorProcess: if the child is still running, kill; always
wait (reap). Dropping ProcessBackend first closes the stream, which makes
the server exit normally — both orders leave no zombie.
```

The in-Rust scripted test server (`tests/common/mod.rs`) binds a UDS in a
temp dir, runs one scripted exchange on a thread (accept -> respond with a
configured frame list), and lets unit tests drive ProcessBackend without
Python: HELLO_ACK happy path, ERROR at handshake, ERROR at eval with each
code, wrong batch_id, wrong nonce, connection drop mid-eval, oversized
frame. Socket paths come from std::env::temp_dir() + process id + a
counter (no wall clock).

## Stage 5: Cross-Language Conformance

`tests/conformance.rs`, running the real Python evaluator. Resolve
`working_dir` as `env!("CARGO_MANIFEST_DIR")/../../python`; if spawn or
connect fails, panic with: what failed, and that the test requires
python3 + numpy (`python3 -c "import numpy"`). Never skip.

```text
handshake: HELLO_ACK protocol_version == 1 and model_version ==
STUB_MODEL_VERSION; ping round trip works
bit-exact equivalence: for several deterministic synthetic batches
(FeatureRows built by a small in-test arithmetic generator over fixed
seeds — splitmix-style, no rand — collated by a real FeatureCollator
under a real FeatureSchema, including partial batches with
row_count < capacity and rows with zero-candidate STOP-only action
lists): ProcessBackend output equals StubBackend output with f32
compared via to_bits, and model_version matches the constant
adopted-schema enforcement: after a successful EVAL, send a batch
collated under a different FeatureSchema -> Backend code 3; fresh
process, HELLO then a batch with a different capacity -> Backend code 4
handshake rejection: hand-crafted HELLO with wrong protocol_version ->
ERROR code 1 (Handshake); wrong encoding_version -> code 2
lifecycle: dropping the backend lets the child exit on its own within a
timeout (reaped, exit status 0); dropping EvaluatorProcess with a live
connection kills and reaps — assert no zombie via the wait result
spawn failure: a nonexistent python binary errors within ready_timeout
```

The equivalence loop is the acceptance test of the work order: after it
passes, Rust and Python cannot disagree about framing, GZFB parsing,
GZFO encoding, stub arithmetic, or routing without a test failing.

## Stage 6: Docs

```text
CODEBASE_OUTLINE.md: add gz-eval-service to the workspace tree and a short
crate section pointing at GZ_EVAL_SERVICE.md + GZ_EVAL_PROTOCOL.md
AGENTS.md: spec list entry added in stage 0; verify
workspace Cargo.toml member added in stage 1; verify
```

## Final Verification

```bash
cargo fmt --all -- --check
cargo test --all           # includes conformance; python3+numpy required
cargo clippy --all-targets --all-features -- -D warnings
python3 -m pytest python/tests
```

Acceptance checklist:

```text
conformance equivalence passes bit-for-bit, including partial batches
every server error code is produced and mapped by exactly one test
no readiness PING probe exists; connect-retry-then-handshake is the flow
no zombies: both drop orders reap the child
gz-features change limited to the decode_outputs free-function lift (if
needed at all); its existing tests unchanged
buffers reused per connection; one write_all per frame
ENCODING_VERSION has exactly one Rust definition (gz-features)
python/ untouched; gz-search goldens untouched
```

## Out Of Scope

```text
orchestrator featurized path, run_featurized, FeaturizedEvalJob (work
order C)
CLI --evaluator flag and the load generator (work order C)
pipelining (batch_id is already on the wire for it)
evaluator restart/backoff policy; multiple sequential connections
torch backend, checkpoints, hot swap
shared-memory transport
```
