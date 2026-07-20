# Python Framework Implementation Spec

Status: implementation work order

Purpose: build the `python/gz` framework — packaging, the layered core
(`common`, `proto`, `codec`, `model/stub`, `checkpoints`), and the
stub-serving evaluator — tested end to end in pytest against Rust-generated
fixtures. After this work order the Python half exists as a real, layered,
efficient package; the Rust `gz-eval-service` crate and the orchestrator
integration land in later work orders against it.

Authority: `GZ_PYTHON.md` owns the structure/layering contract,
`GZ_FEATURES.md` owns the batch/output encoding, and the protocol contract
is extracted into `GZ_EVAL_PROTOCOL.md` in stage 1. If this document
disagrees with a contract, the contract wins; report the conflict.

Read before starting:

```text
specs/GZ_PYTHON.md            (layout, layering, torch rule, checkpoints)
specs/GZ_FEATURES.md          (GZFB/GZFO layout, token conventions)
specs/GZ_EVAL_SERVICE.md      (protocol + stub sections being extracted)
crates/gz-features/src/       (the Rust encoder the fixtures come from)
```

## Design Principles (binding)

Clean, human-readable code and extreme efficiency, made concrete:

```text
READABILITY
type hints on every public function and dataclass field
dataclasses for records; small modules; no class hierarchies where a
function does the job
one-line docstrings only where the name does not already say it; comments
only for constraints the code cannot express (same comment rule as Rust)
code is ruff-format-compatible at default settings (4-space, 88 cols,
double quotes) so adopting ruff later is a no-op diff
thin __main__.py files over testable functions, exit codes for errors

EFFICIENCY (hot path = frame read -> batch parse -> stub -> output encode)
zero-copy everywhere: numpy frombuffer views into one receive buffer;
sections are views, never .copy()
no per-row Python loops on the hot path; the stub is fully vectorized
steady-state serving allocates nothing per request beyond what numpy
forces: the receive buffer, output buffer, and frame header buffer are
preallocated and reused, growing only monotonically
one syscall per frame write (header + body joined into one buffer before
sendall); frame reads use recv_into loops, never recv-and-concat
no logging, string formatting, or exception construction on the hot path;
startup, swap, and error events only, to stderr
benchmarked, not asserted: the codec bench script (stage 8) prints
rows/s and MB/s; its output is pasted into the commit message
```

## Hard Constraints

```text
Stage order below; every stage ends with python3 -m pytest python/tests
passing plus cargo fmt / cargo test --all / cargo clippy --all-targets
--all-features -- -D warnings (Rust is touched only in stage 7's fixture
generator; its suite must stay green throughout).
Stage 0 commits the current dirty tree first. Commit per stage.
stdlib + numpy only. No torch, no pip-only packages, nowhere — enforced
by acceptance check, not convention. No blake3 (Python hashes are stdlib
blake2b per GZ_PYTHON.md).
gz/trainer is NOT created in this work order; no empty placeholder
modules anywhere. Every module shipped has a consumer and tests.
Python never encodes GZFB (only Rust produces batches); Python decodes
GZFB and encodes GZFO. Do not write a Python GZFB encoder, including for
tests — fixtures come from Rust.
The import layering table in GZ_PYTHON.md is binding; a test asserts it
(stage 2).
Tests must not require installing the package: python/tests/conftest.py
puts python/ on sys.path. (System pip is PEP-668 externally managed;
pip install -e python/ is for deployments, documented, not required.)
```

## Stage 0: Commit

Commit the dirty working tree (gz-cli, lanes.rs, docs) before starting.

## Stage 1: Extract GZ_EVAL_PROTOCOL.md

Create `specs/GZ_EVAL_PROTOCOL.md` as the language-neutral contract, moved
(not rewritten) from GZ_EVAL_SERVICE.md:

```text
the frame format and types (HELLO, HELLO_ACK, EVAL, EVAL_RESULT, PING,
PONG, ERROR), field-by-field, with the 256 MiB cap and framing rules
the handshake validation rules, with one clarification made explicit:
which HELLO fields a stub server VALIDATES (protocol_version,
encoding_version against constants) versus ADOPTS from the first client
and then enforces on every EVAL (feature_schema_hash, batch_capacity)
versus RECORDS only (engine tags); a checkpoint-backed server later
validates schema hash against its checkpoint instead of adopting
the ERROR codes (1 protocol, 2 encoding, 3 schema, 4 capacity,
5 malformed)
the stub model formulas and the stub ModelVersion constant, verbatim
PROTOCOL_VERSION = 1 for this original work order; GZ_EVAL_PROTOCOL.md v2
supersedes it with episode model leases and MODEL_RELEASE
```

GZ_EVAL_SERVICE.md keeps its Rust-crate and orchestrator sections and gains
supersession pointers: protocol/stub sections -> GZ_EVAL_PROTOCOL.md,
Python sections -> GZ_PYTHON.md. Add GZ_EVAL_PROTOCOL.md to AGENTS.md's
spec list.

## Stage 2: Package Skeleton

```text
python/pyproject.toml     distribution "graphzero", packages gz*,
                          requires-python >= 3.12, dependencies numpy>=1.26,
                          [project.optional-dependencies] torch = torch,
                          safetensors
python/gz/__init__.py     __version__ = "0.1.0", nothing else
python/gz/common/ proto/ codec/ model/ evaluator/
                          packages with __init__.py re-exporting their
                          public names (gz/checkpoints has no consumer
                          until the trainer/torch work orders and is not
                          created here)
python/tests/conftest.py  sys.path insertion
python/tests/test_layering.py
```

`test_layering.py` walks `python/gz` with `ast`, collects import statements
per subpackage, and asserts the GZ_PYTHON.md layering table (including:
torch appears nowhere; evaluator is imported by nothing but itself). This
is the cheap structural gate that keeps the framework honest as it grows.

## Stage 3: gz/common

```text
tags.py     fixed-width binary ids: EngineId(16), EngineVersion(16),
            ModelVersion(16), ActionSetHash(32), FeatureSchemaHash(32).
            One small frozen class parameterized by width, or five thin
            frozen dataclasses — whichever reads better. bytes in/out,
            lowercase-hex str round trip, equality/hash, length-validated
            construction. Mirrors gz-engine conventions.
hashing.py  model_version(arch_config_hash, feature_schema_hash,
            weights_hash) -> ModelVersion: first 16 bytes of blake2b over
            a domain prefix ("gz-model-version-v1") and the length-
            delimited inputs. file_blake2b(path) -> 32-byte hex for
            weights files, streaming reads.
log.py      setup(name) -> stderr logger, single-line format. Nothing else.
```

Tests: hex round trips, wrong-length rejection, model_version determinism
and sensitivity to each input, file hash against a known literal.

## Stage 4: gz/proto

Implements GZ_EVAL_PROTOCOL.md. Stdlib only.

```text
frames.py   frame type constants, PROTOCOL_VERSION, ENCODING_VERSION
            (imported value must equal gz-features' constant; asserted by
            fixture tests), MAX_FRAME = 256 MiB
            read_frame(sock, buf) -> (frame_type, memoryview): recv_into
            loop into the caller's growable bytearray; returns a view, no
            copy; raises ProtocolError on cap/eof violations
            write_frame(sock, frame_type, *parts): joins header + parts
            into one buffer, single sendall
hello.py    Hello / HelloAck frozen dataclasses using common.tags; encode
            to bytes / decode from view, field-exact per the contract
errors.py   ProtocolError with the numeric code; error-frame encode/decode
```

Tests over `socket.socketpair()`: round trip every frame type; truncated,
oversized, and unknown-type frames raise with the right codes; Hello
field-for-field round trip.

## Stage 5: gz/codec

Implements the Python side of GZ_FEATURES.md's encoding. numpy only.

```text
schema.py   SchemaDims: the header-carried dims (N, E, A, S, D, capacity,
            row_count) plus FeatureSchemaHash. Python does NOT compute
            schema hashes; it compares the 32 bytes it was handed.
batch.py    BatchView.parse(buf: memoryview) -> BatchView:
            validates magic "GZFB", encoding version, header sanity, and
            total length arithmetic BEFORE creating views; then exposes
            one numpy view per section (node_count, node_tokens,
            node_attrs (absent when D=0), edge_*, action_*, subject_*,
            position), each with the exact dtype/shape from the spec,
            all zero-copy into buf. Section offsets are computed
            arithmetically with the 4-byte alignment rule; there is no
            scanning.
outputs.py  OutputEncoder(capacity, max_actions): owns one reusable
            bytearray; encode(values: ndarray[B], logits: ndarray[B, A],
            row_count) -> memoryview of the GZFO bytes (magic, version,
            row_count, A, then the two sections), written in place.
```

Tests (fixtures arrive in stage 7; stage-5 tests use hand-built byte
strings for small synthetic headers/sections assembled with struct.pack in
the test file — assembling test bytes by hand is not "encoding GZFB in
Python"; it is how a decoder is tested against the paper spec):

```text
header validation: bad magic / version / dims / length each rejected
offset arithmetic: a synthetic 2-row batch with known bytes yields views
whose values match the hand-computed literals, section by section,
including the alignment padding and the D=0 absent-attrs case
zero-copy: mutating the underlying buffer is visible through the views
OutputEncoder: exact bytes for a known 2-row output; buffer reuse across
calls yields identical bytes with no growth
```

## Stage 6: gz/model + gz/evaluator

```text
model/registry.py  ARCHS: dict name -> build(schema_dims, arch_config);
                   registers "stub". build raises KeyError with the known
                   names on miss.
model/stub.py      stub(batch: BatchView) -> (values, logits): the
                   GZ_EVAL_PROTOCOL.md formulas, fully vectorized in
                   uint64 numpy arithmetic (wraparound semantics), then
                   masked: logits beyond each row's action_count zeroed,
                   rows beyond row_count zeroed. Returns f32 arrays.
evaluator/backends.py  StubBackend: holds an OutputEncoder; eval(view) ->
                   (model_version, gzfo memoryview). The stub ModelVersion
                   constant from the protocol spec.
evaluator/server.py    serve(socket_path, backend, *, ready_event=None):
                   bind UDS, listen(1), accept one client, handshake per
                   the contract (validate/adopt/record split), then the
                   EVAL loop: validate each batch header against the
                   adopted schema hash + capacity, backend eval, one
                   EVAL_RESULT per EVAL in order; PING -> PONG; any
                   violation -> ERROR frame with the right code, close,
                   return. Single-threaded, blocking, one client, then
                   returns (the Rust side owns respawn policy later).
evaluator/__main__.py  argparse --socket PATH; log startup line; serve
                   with StubBackend; exit 1 on error.
```

The stub must be internally cross-checked: `tests/test_stub.py` implements
the formulas a second time as scalar pure-Python integer arithmetic and
compares against the vectorized implementation on the fixture batch and on
randomized (seeded) count arrays. Two independent implementations agreeing
is the pre-Rust-conformance oracle.

## Stage 7: Rust-Generated Fixtures

Add `crates/gz-features/examples/gen_python_fixtures.rs`: hand-builds the
rows below with the real `FeatureCollator` and writes
`python/tests/fixtures/batch_attr1.gzfb` and `batch_attr0.gzfb`. Fixtures
are committed; the example is the only way to regenerate them.

Primary fixture (schema "gz-fixture-v1", node_vocab 7, attr_dim 1,
edge_types 2, action_kind_vocab 12, max_nodes 8, max_edges 4,
max_actions 6, max_subjects 2, batch_capacity 4, row_count 3):

```text
row 0: tokens [1,2,3], attrs [0.5,-1.0,2.0],
       edges (0->2,t0) (1->2,t1),
       actions [(kind 4, prior 0.25, subj [2]), STOP],
       position (0, 0, 1.0, 0.125)
row 1: tokens [6], attrs [1.5], no edges, actions [STOP],
       position (1, 2, 0.75, 0.125)
row 2: tokens [1,1,4,5,2], attrs [0.0,0.25,0.5,0.75,1.0],
       edges (0->2,t0) (1->2,t1) (2->4,t0) (3->4,t1),
       actions [(2, -0.5, [0,1]), (3, 0.0, []), (4, 1.0, [4]),
                (5, 0.125, [2,3]), (6, -1.0, [0]), STOP],
       position (3, 1, 0.5, 0.25)
```

Secondary fixture: attr_dim 0, one row (tokens [1,2], one edge, one
candidate + STOP) — exists to cover the absent-attrs-section path.

Python fixture tests transcribe expectations from the row table above
(literals in the test, not recomputed), assert every section of both
fixtures, and assert the encoding version in the header equals
`gz.proto.frames.ENCODING_VERSION`. Stub-on-fixture expectations are
captured with the assert-print-paste procedure and cross-checked by the
scalar reference implementation.

`tests/test_server.py` runs the full protocol against `serve()` on a real
UDS in a temp dir (server in a test thread, ready_event for startup): a
minimal test client performs HELLO -> EVAL(fixture bytes) -> EVAL_RESULT
(stub values match), PING/PONG, then each mismatch case (bad protocol
version, bad encoding version, changed schema hash after adoption, wrong
capacity, malformed frame) against fresh serve() instances, asserting the
specific ERROR codes.

## Stage 8: Bench

`python/benches/codec_bench.py` (plain script, not pytest): builds one
max-size synthetic batch in memory (bytes assembled once), then loops
BatchView.parse -> stub -> OutputEncoder.encode for N iterations; prints
batch bytes, rows/s, MB/s, and per-batch microseconds for B in {64, 256}.
Run it; paste the output into the stage's commit message. No pass/fail
thresholds — the numbers are for humans and for regressions to be visible
in history.

## Stage 9: Docs

```text
AGENTS.md: add this spec (GZ_EVAL_PROTOCOL.md was added in stage 1).
CODEBASE_OUTLINE.md: the python/ tree gains benches/; note that
python/gz is implemented through the evaluator (stub serving) with
trainer pending.
GZ_PYTHON.md: no changes expected; report any contradiction found while
implementing instead of editing it silently.
```

## Final Verification

```bash
python3 -m pytest python/tests
grep -rn "import torch" python/           # must be empty
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
python3 python/benches/codec_bench.py
python3 -m gz.evaluator --socket /tmp/gz-eval-smoke.sock &  # starts, logs
```

Acceptance checklist:

```text
layering test passes and covers the torch ban
every module has a consumer; no gz/trainer, no placeholders
BatchView is zero-copy (mutation-visibility test proves it)
stub is vectorized and agrees with the scalar reference implementation
fixtures are Rust-generated, committed, and transcribed literals match
server enforces the validate/adopt/record handshake split with exact
error codes
bench output recorded in a commit message
Rust workspace untouched except the fixture example; gz-search goldens
untouched
```

## Out Of Scope

```text
gz/trainer and GZ_TRAINER.md
gz/checkpoints (manifest, publish, resolve, weight loading): no consumer
until the torch backend and trainer land; GZ_PYTHON.md fixes its shape
torch backend and the real model
the Rust gz-eval-service crate, ProcessBackend, spawn/conformance tests
orchestrator featurized path and CLI flag
pipelining, hot swap, multiple clients
ruff adoption
```
