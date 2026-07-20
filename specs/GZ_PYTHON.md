# Python Package

## Scope

`python/gz` is one installable package shared by evaluator and trainer. Shared
model, codec, protocol, and checkpoint implementations prevent train/serve
drift.

## Layout

```text
gz.common       logging, hashing, shared tags
gz.proto        framed socket helpers and evaluator handshake
gz.codec        Rust-compatible feature/output/target views
gz.model        gz-graph-v2 model, registry, deterministic stub
gz.checkpoints  manifests, weight hashing, atomic publication, sources
gz.evaluator    stub/torch backends and Unix-socket server
gz.trainer      config, supervisor, sampler, losses, optimizer, publishing
```

Evaluator and trainer entry points are thin:

```bash
python -m gz.evaluator --socket PATH ...
python -m gz.trainer --config PATH
```

## Dependency Direction

```text
common -> stdlib
proto -> common
codec -> common + numpy
model -> codec
checkpoints -> common (+ lazy torch/safetensors weight helpers)
evaluator -> proto + codec + model + checkpoints
trainer -> proto + codec + model + checkpoints
```

Evaluator and trainer do not import one another. Torch is imported only by the
real model/backend/trainer paths; protocol and codec tests remain usable without
a CUDA runtime.

## Model Identity

A model is constructed only from `FeatureSchemaConfig` and `ArchConfig`. Both
are canonically encoded and hashed. Trainer and evaluator call the same model
builder and load strict state dicts. Runtime files, environment variables, or
process role never alter model topology.

## Checkpoints

Checkpoint manifests bind:

- exact model version and training step;
- architecture and architecture hash;
- feature schema and schema hash;
- engine/action-set identity;
- weight filename, byte length, and content hash.

Weights use safetensors. Publication writes and verifies weights/manifest in a
temporary directory, atomically renames the completed version, then atomically
replaces pointer JSON. Model version is derived from configuration and weight
content rather than time.

The evaluator polls a directory pointer, validates every tag, constructs and
warms a new model before adoption, and keeps leased older generations until all
episodes release them. A bad checkpoint is rejected while the current model
continues serving.

## Codecs

Python batch/target parsers mirror Rust section offsets and dtypes with NumPy
views. They validate magic, versions, schema hash, dimensions, lengths, and
counts before exposing arrays. `BatchStager` owns reusable pinned CPU and device
buffers; event-guarded ping-pong prevents reuse before asynchronous transfers
complete.

Reserved legacy wire sections are parsed only to preserve the current encoding
layout and are not staged into the live model.

## Testing

Pytest covers codecs, manifests/publication, evaluator framing and hot-swap,
model entry-point parity, trainer loss/optimizer/sampling behavior, config
validation, child supervision, and resume. Rust owns cross-language protocol
conformance tests that spawn the Python evaluator.

```bash
PYTHONPATH=.:python uv run --project python pytest -q python/tests
```
