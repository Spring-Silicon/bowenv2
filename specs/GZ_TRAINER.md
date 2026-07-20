# Trainer

## Scope

`python/gz/trainer` supervises concurrent generated-root selfplay and GPU
training. It owns config resolution, child lifecycle, replay sampling, optimizer
steps, metrics, checkpoint publication, and resume. Search, measurement, and
replay validation remain in Rust.

## Processes

```text
graphzero selfplay
  -> evaluator process reading the actor checkpoint source on eval_device
  -> RocksDB replay writer
  -> in-process sample service

Python trainer on device
  <- replay sample socket
  -> learner checkpoint directory
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
Exphormer, pointer policy, tanh scalar value, and optional `v8-v32-score` or
`v8-v32-score-soft-policy-v2` auxiliary heads. Retired architecture keys remain
accepted only at their fixed serialized values so current manifests can be
loaded.

The selfplay contract is generated-root symmetric selfplay with a sign outcome,
length tiebreak, and no reference/arena pipeline. Legacy selfplay keys are
accepted only at the corresponding disabled values for existing run configs.

The actor checkpoint source is independent from the learner checkpoint sink.
By default both use the learner checkpoint directory and `latest.json`,
preserving online checkpoint hot-swap. `actor_checkpoint_dir` selects a
read-only external source, `actor_checkpoint_pointer` selects its named pointer,
and `eval_poll_interval = 0` freezes the resolved model for the lifetime of each
selfplay evaluator. Actor and learner architectures may differ, but their
feature schema hashes must match.

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

The optional soft-policy head predicts the normalized `policy^(1/T)` target over
the same legal actions. It has an independent action embedding, projection,
glimpse, feed-forward path, normalization, and pointer key; only encoded graph
tensors are shared with the serving policy. Its loss weight and graph-trunk
gradient scale are independent, while the auxiliary head always receives the
full weighted loss gradient. Serving uses only the main policy output.
The retired `v8-v32-score-soft-policy` layout remains loadable for checkpoint
serving but is rejected by training configuration.

## Optimization

AdamW and `torch.optim.Muon` are supported. Muon receives eligible hidden
matrices; embeddings, biases, gains, and output readouts use AdamW. Constant and
cosine schedules, warmup, weight decay, gradient clipping, eager or
`torch.compile` execution, and optional trainer weight EMA are supported.

## Checkpoints

Publication is atomic: weights and manifest are completed before a pointer is
replaced. Manifests bind model version, training step, architecture, feature
schema, and state-dict identity. Ordinary learner snapshots have bounded
retention; latest, best/arena-compatible pointer files, permanent step pointers,
and resume state are preserved.

Resume restores model weights and training step, rebuilds optimizer and
scheduler state, and seeds a fresh EMA from the loaded model. It is therefore
an approximate continuation, although it continues the same W&B run when its
run ID is present. A checkpoint with a mismatched architecture/schema is
rejected before training or serving.

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
- no actor checkpoint whose feature schema differs from the learner/replay;
- no model hot-swap within an episode lease;
- no silent child, prefetch, evaluator, replay, or metrics-writer failure;
- resume is reported as approximate because optimizer/scheduler moments reset.
