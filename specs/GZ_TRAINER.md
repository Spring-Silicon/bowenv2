# Trainer

## Scope

`python/gz/trainer` supervises concurrent generated-root selfplay and GPU
training. It owns config resolution, child lifecycle, replay sampling, optimizer
steps, metrics, checkpoint publication, and resume. Search, measurement, and
replay validation remain in Rust.

## Processes

```text
graphzero selfplay
  -> evaluator process on eval_device
  -> RocksDB replay writer
  -> in-process sample service

Python trainer on device
  <- replay sample socket
  -> checkpoint directory
  -> evaluator checkpoint hot-swap
```

The driver creates the replay schema, publishes an initial model when needed,
starts selfplay, waits for `min_startup_rows`, and trains concurrently. Child
death is fail-fast. Shutdown stops prefetch, terminates children, closes metric
writers, and removes owned sockets. Replay durability comes from RocksDB atomic
appends, not orderly process exit.

## Configuration

TOML supports exactly one `extends` layer. The base owns reusable architecture
and runtime policy; the leaf owns run paths, identity, duration, and ablations.

The live architecture contract is fixed to `gz-graph-v2`, joint-board
Exphormer, pointer policy, tanh scalar value, and optional
`v8-v32-score` auxiliary heads. Retired architecture keys remain accepted only
at their fixed serialized values so current manifests can be loaded.

The selfplay contract is generated-root symmetric selfplay with a sign outcome,
length tiebreak, and no reference/arena pipeline. Legacy selfplay keys are
accepted only at the corresponding disabled values for existing run configs.

## Sampling

The sampler validates protocol, encoding, feature-schema, and capacity metadata
before accepting batches. Policy and value may use the same sampled batch or an
independent value sample/window. Prefetch uses a bounded background thread and
propagates errors to the trainer. Sample seeds are deterministic functions of
the configured data seed and training step.

Reuse telemetry is reported separately for policy and value windows. The reuse
gate can delay optimizer admission when consumed samples exceed configured data
production.

## Losses

The policy loss is soft-target cross entropy over legal actions. Padding is
masked; STOP is masked only for replay mode V1. The scalar value loss is MSE
between the tanh prediction and the symmetric target in `{-1, 0, +1}`.

Optional auxiliary losses supervise V8 and V32 horizon outcomes plus normalized
terminal score. Every target has an explicit valid mask. Zero-valid batches
produce a differentiable zero loss. `value_trunk_grad_scale` scales value-side
gradients entering the shared trunk without changing value-head gradients.

## Optimization

AdamW and `torch.optim.Muon` are supported. Muon receives eligible hidden
matrices; embeddings, biases, gains, and output readouts use AdamW. Constant and
cosine schedules, warmup, weight decay, gradient clipping, eager or
`torch.compile` execution, and optional trainer weight EMA are supported.

## Checkpoints

Publication is atomic: weights and manifest are completed before a pointer is
replaced. Manifests bind model version, training step, architecture, feature
schema, and state-dict identity. Ordinary actor snapshots have bounded
retention; latest, best/arena-compatible pointer files, permanent step pointers,
and resume state are preserved.

Resume restores model, optimizer, scheduler, EMA, and training step, then
continues the same W&B run when its run ID is present. A checkpoint with a
mismatched architecture/schema is rejected before training or serving.

## Metrics

The trainer logs policy/value/auxiliary losses, target coverage, sign accuracy,
gradient norms, parameter norms, learning rates, throughput, policy/value reuse,
replay production/consumption, evaluator fill, selfplay outcome/cost/rewrite
metrics, model versions, and publication events. JSONL remains local; W&B is an
optional mirror.

## Required Invariants

- no optimizer step before a complete validated sample;
- no value loss on invalid target rows;
- no publication of partially written weights or manifests;
- no model hot-swap within an episode lease;
- no silent child, prefetch, evaluator, replay, or metrics-writer failure;
- resume never resets optimizer/scheduler state while claiming continuation.
