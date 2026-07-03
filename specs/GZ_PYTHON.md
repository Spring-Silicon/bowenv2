# GraphZero Python Spec

Status: draft

Purpose: define the structure, layering, packaging, and conventions of the
Python half of GraphZero — one installable package serving two processes,
the evaluator and the trainer. This is a contract spec; work orders
reference it the way Rust work orders reference crate contracts.

This spec supersedes the standalone `python/evaluator` layout sketched in
`GZ_EVAL_SERVICE.md` and the `python/evaluator` + `python/trainer` sibling
directories in CODEBASE_OUTLINE. Those work orders will be re-sliced
against this structure.

## Core Decision

One package, not sibling programs. The evaluator and trainer share the
things that must never drift:

```text
the model definition: the evaluator loads checkpoints the trainer writes;
both must construct bit-identical networks
the batch codec: training batches from the replay sample service use the
same GZFB encoding as eval batches (train/serve parity is by construction)
the checkpoint manifest: one side writes it, the other reads it; a single
implementation is what keeps the format from forking
the version-tag checks: EngineVersion / ActionSetHash / FeatureSchemaHash /
ModelVersion agreement, checked identically everywhere
```

Two sibling packages force either duplication or a third "shared" package
invented later under pressure. One package with layered subpackages gives
the sharing without the drift.

## Layout

The import package is `gz`, matching the `gz-*` crate naming and avoiding
the `graphzero/python/graphzero` stutter. The pyproject distribution name
is `graphzero` (sklearn-style distribution/import split); nothing is ever
published, so the split costs nothing.

```text
python/
  pyproject.toml            distribution "graphzero", import package "gz"
  gz/
    common/                 version tags, config loading, logging setup
    proto/                  wire framing, frame types, HELLO/handshake,
                            error codes (GZ_EVAL_PROTOCOL.md is the contract)
    codec/                  GZFB parse / GZFO encode via numpy views,
                            FeatureSchema representation, section offsets
                            (GZ_FEATURES.md is the contract)
    model/
      registry.py           arch name -> constructor;
                            build(feature_schema, arch_config) -> model
      stub.py               deterministic stub model (numpy)
      exphormer.py          the real network (torch; later work order)
    checkpoints/            manifest schema, atomic publish, latest
                            resolution, checkpoint source (dir or URL)
    evaluator/
      __main__.py           argparse over a testable serve() function
      server.py             accept loop, dispatch, hot-swap slot
      backends.py           stub backend (numpy) and torch backend (later)
    trainer/                later work order; shape fixed here
      __main__.py
      sampler.py            replay sample-service client
      data.py               batch decode -> training tensors
      loop.py               optimizer steps, losses
      publish.py            checkpoint publication via checkpoints/
  tests/                    pytest; mirrors subpackages
    fixtures/               committed golden batch bytes and manifests
```

## Import Layering

Allowed imports, in dependency order (a row may import from rows above it
and from `common`):

```text
common     -> stdlib only
proto      -> stdlib only
codec      -> numpy
model      -> codec; stub.py numpy only; exphormer.py torch (lazy)
checkpoints-> stdlib; weight-format helpers import lazily
evaluator  -> proto, codec, model, checkpoints
trainer    -> proto, codec, model, checkpoints
```

Rules:

```text
Nothing imports evaluator or trainer except their own __main__.
No circular imports; the layering above is the whole graph.
__main__.py files are thin: parse args, call one testable function, map
errors to exit codes. Same convention as gz-cli.
```

## Torch Optionality (hard rule)

Everything required to serve the stub model — `common`, `proto`, `codec`,
`model/stub`, `checkpoints`, `evaluator` with the stub backend — must
import and run with stdlib + numpy alone.

```text
torch is imported only inside model/exphormer.py, the evaluator's torch
backend, and trainer/ — and only at the point of use (module-local lazy
import), never at package import time.
importing gz, gz.evaluator, or running the stub-backed evaluator on a
machine without torch must work.
pyproject declares torch under an extra: graphzero[torch]. Core install
depends on numpy only.
tests for core packages must not import torch; torch-dependent tests skip
cleanly (pytest.importorskip) when torch is absent.
```

Why: the eval-service work orders forbid torch; conformance tests must run
on any box; a broken CUDA install must not take down protocol tests or the
stub serving path.

## Model Identity

The rule that keeps evaluator and trainer building the same network:

```text
A model is constructible from (FeatureSchema, ArchConfig) and nothing
else. No model code path may read files, environment, or globals.
model/registry.py maps an arch name to a constructor; both processes call
registry.build(schema, arch_config) and load weights into the result.
ArchConfig is a plain declarative dict (layers, dims, heads, ...) with a
canonical encoding and a derived arch_config_hash.
```

## Checkpoints

`checkpoints/` owns both directions of the checkpoint lifecycle.

Manifest (`manifest.json` inside each version directory):

```text
manifest_version        format version, checked on read
model_version           16B hex; identifies these exact weights
arch                    { name, config }, plus arch_config_hash
feature_schema          full FeatureSchemaConfig object
feature_schema_hash     32B hex
engine_id, engine_version, action_set_hash    16B/16B/32B hex
training_step, run_id
weights                 { filename, bytes, blake2b-256 hex, format }
```

Rules:

```text
weights format is safetensors. torch.save/pickle is forbidden: it is
neither zero-copy nor safe to load, and safetensors loads without torch
present if needed.
model_version is derived, not minted: first 16 bytes of a domain-prefixed
blake2b (stdlib hashlib; these hashes never cross to Rust as anything but
opaque bytes) over (arch_config_hash, feature_schema_hash, weights hash).
Identical weights + config always yield the identical version; no clocks.
publish protocol (trainer side): write checkpoints/run_id/version_N.tmp/,
fsync weights and manifest, atomic rename to version_N/, then atomically
replace latest.json ({ version_dir, model_version }).
resolution (evaluator side): a CheckpointSource abstraction with two
implementations planned — local directory now, URL fetch later (the
distributed-trainer hedge). Consumers only see the abstraction.
hot swap (evaluator side): load and warm the new version off to the side
(including any compile/capture warmup), then swap the serving slot;
in-flight batches finish on the old version. A new checkpoint whose tags
disagree with the serving configuration is refused loudly on stderr and
the evaluator keeps serving the old version — fail fast on the artifact
without killing selfplay.
```

Implemented in `python/gz/checkpoints`: local directory source, strict
manifest parsing, safetensors save/load, atomic publish, latest
resolution, and hash-verified weights.

## Version Tags

The four-tag agreement rule (EngineVersion, ActionSetHash,
FeatureSchemaHash, ModelVersion) is enforced at every boundary the Python
half touches: the eval handshake, checkpoint load, and training batch
ingest. One implementation in `common/`, used everywhere. On disagreement:
refuse the artifact or connection with a specific error; never continue
with ambiguous semantics.

## Packaging And Environment

```text
requires-python >= 3.12 (the deployment box; Jetson measure workers are
Rust and do not run this package).
dependencies: numpy. extras: torch (graphzero[torch] pulls torch +
safetensors).
config files are TOML read with stdlib tomllib; no yaml dependency.
dev install: pip install -e python/. Entry points are module mains:
python -m gz.evaluator --socket PATH
python -m gz.trainer --config PATH
module mains resolve with python/ as the working directory (the Rust
spawner's default) or after installation; they do not resolve from the
repo root. pytest works from anywhere via tests/conftest.py.
logging: stderr only; nothing on the eval hot path. Startup, swaps, and
errors are one-line events.
```

## Trainer Shape (fixed now, built later)

The subpackage boundaries anticipate the distributed constraint so DDP is
a loop change, not a restructure:

```text
sampler.py is a per-rank client of the replay sample service; consumers
are anonymous and additive, so N ranks just sample independently.
data.py turns sampled GZFB batches into training tensors through the same
codec the evaluator uses.
loop.py owns optimizer steps and losses; it is the only DDP-aware module.
publish.py publishes on rank 0 only.
```

Trainer details (losses, schedules, staleness policy) belong to a future
GZ_TRAINER.md, not here.

## Testing

```text
pytest, rooted at python/tests, mirroring subpackages.
tests/fixtures holds committed binary fixtures (golden GZFB batch bytes,
a golden manifest); fixtures are regenerated by Rust test helpers and
committed, never built ad hoc in Python.
cross-language conformance tests (Rust spawning the real evaluator and
comparing against the Rust stub bit-for-bit) live on the Rust side in
gz-eval-service; the pytest suite covers Python-internal behavior.
core test paths never import torch; torch tests skip cleanly without it.
run: python3 -m pytest python/tests
```

## Tooling

```text
pytest is the gate now.
ruff (format + check) is the intended analog of cargo fmt + clippy; adopt
when it enters the environment, then it becomes part of the verification
step in work orders. Until then, match the style of existing files.
```

## Relationship To Other Specs

```text
GZ_EVAL_PROTOCOL.md   frames, handshake, error codes, stub model formulas
                      (language-neutral contract; proto/ and codec/
                      implement the Python side)
GZ_FEATURES.md        GZFB/GZFO encoding and FeatureSchemaHash (codec/
                      implements the Python side)
GZ_EVAL_SERVICE.md    the Rust service crate and orchestrator integration;
                      its Python sections are superseded by this spec
future GZ_TRAINER.md  training loop, losses, schedules, staleness
```

One Rust-side ripple: `EvaluatorProcessConfig` defaults become
`module = "gz.evaluator"`, `working_dir = python/`.

## Deferred

```text
the trainer implementation and GZ_TRAINER.md
torch backend, Exphormer model, compile/CUDA-graph warmup details
checkpoint URL source implementation
opponent trajectory registration in the evaluator (job 2; the server
gets a per-connection state slot for it, nothing more is reserved)
multi-connection serving, metrics endpoints
ruff adoption
```
