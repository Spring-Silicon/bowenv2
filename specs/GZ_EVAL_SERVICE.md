# Evaluator Service

## Scope

`gz-eval-service` is the Unix process boundary between Rust feature batchers and
the Python model server. It owns framing, handshake validation, process
lifecycle, pipelined batch handles, model-generation selection, and explicit
generation release.

The language-neutral frame layout is specified in `GZ_EVAL_PROTOCOL.md`.

## Handshake

The client sends protocol/encoding versions, engine identity, action-set hash,
feature-schema hash, and capacity. The server validates these against its
backend/checkpoint before accepting eval traffic. A mismatch returns a bounded
error frame and closes the connection.

## Batch Lifecycle

Rust submits one fixed-layout GZFB batch with requested action counts and model
generation. Python validates and stages the complete frame, executes the model,
and returns flattened per-row logits/values plus the exact served model version
and active generation for future admissions.

`FeatureEvalBackend` separates submission from completion so batchers can keep
multiple requests in flight. Pending handles complete FIFO per process. Partial
socket reads are handled by exact-length frame helpers; malformed/truncated or
oversized frames are protocol errors.

## Hot Swap And Leases

The server polls an atomic checkpoint pointer. A valid new checkpoint is loaded
and warmed before it becomes active. Existing submitted batches name and finish
on their requested generation. Rust episode leases keep an older generation
resident until no live game needs it, then send `MODEL_RELEASE`.

Residency is bounded. The server does not load an unbounded chain of old models,
and the client never releases a generation with live episode or batch users.

## Process Backend

`EvaluatorProcess` creates a private Unix socket, spawns `python -m
gz.evaluator`, waits for readiness with a bounded timeout, and connects a
`ProcessBackend`. Startup failure, child death, socket failure, server error, or
protocol mismatch is returned to the orchestrator. Cleanup removes owned socket
paths and terminates only children it spawned.

## Stub Backend

The deterministic stub backend exists for protocol/conformance tests. Rust and
Python implementations must produce identical outputs for the same GZFB bytes.
It is not a training fallback.

## Correctness Requirements

- validated handshake before eval frames;
- exact frame lengths and bounded allocations;
- row/action output counts validated before routing;
- exact model version in every result;
- no mixed generation within one batch;
- no generation release while leased or in flight;
- child/server errors propagate without synthesized outputs;
- one process's hot-swap state is never assumed to match another process.
