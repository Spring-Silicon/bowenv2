# Evaluator Hot-Swap Implementation Spec (Trainer Work Order 2)

Status: implementation work order

Purpose: implement GZ_TRAINER.md's two Python prerequisites — mid-run
checkpoint hot-swap in the evaluator (the GZ_PYTHON.md contract) and the
model's pre-tanh value exposure for training. Python-only; independent of
work order 1.

Authority: `GZ_PYTHON.md` (the hot-swap contract), `GZ_MODEL.md`,
`GZ_TRAINER.md`. Contract wins; report conflicts.

Read before starting:

```text
specs/GZ_PYTHON.md                (Checkpoints: hot swap rules)
python/gz/evaluator/backends.py   (TorchBackend gaining the swap slot)
python/gz/evaluator/server.py     (swap check between EVAL frames)
python/gz/checkpoints/source.py   (latest resolution being polled)
python/gz/model/exphormer.py      (value head raw output)
```

## Hard Constraints

```text
Every stage ends with python3 -m pytest python/tests and the Rust suite
(cargo test --all) untouched and green. Commit per stage; stage 0 commits
any dirty tree.
The layering and torch-optionality rules of GZ_PYTHON.md hold; the stub
path and all core tests remain torch-free.
The serving hot path gains no per-eval work beyond one lock-free-ish slot
check between frames; loading and warming happen on the loader thread.
Swap failures never take down serving: a bad checkpoint is logged to
stderr once and skipped; the old model keeps serving (GZ_PYTHON rule).
```

## Stage 0: Commit

Commit any dirty tree.

## Stage 1: Pre-Tanh Value Exposure

The model's `forward` returns `(value_raw, logits)` — tanh moves out of
the model:

```text
exphormer.py: values head returns the raw scalar (no tanh in forward)
TorchBackend applies torch.tanh(values) before encoding outputs
the stub path is untouched (numpy stub already emits final values)
property tests updated mechanically; the compile/no-recompile test and
both-aggregation parameterization must still pass
GZ_MODEL.md heads section: one-line amendment (tanh applied by the
serving backend; the trainer consumes the raw scalar)
```

This keeps one compiled forward for both serving and training; the
trainer's logistic loss consumes `value_raw` directly.

## Stage 2: Hot-Swap In TorchBackend

```text
TorchBackend gains a loader thread started at handshake (it needs the
adopted capacity for warmup):
  every poll_interval (default 10.0s, constructor + --poll-interval flag):
    resolve latest; if model_version == serving version, sleep again
    if the manifest's feature_schema_hash != the serving schema hash, or
    arch build/load fails: stderr one line, remember the rejected
    model_version so it logs once, keep serving
    else: build model from the manifest, load weights, move to device,
    compile with the same flags, warm with the stager's dummy batch
    (warm 3x — reduce-overhead capture settles over the first calls;
    this also fixes the existing single-warm note from the last review),
    then publish (runner, model_version) into a lock-protected slot
serving loop: between EVAL frames, one cheap slot check; on a pending
swap, replace the runner and version. In-flight work is unaffected by
construction (single in-flight; the swap happens strictly between
batches). EVAL_RESULT already carries model_version per result, so the
swap is externally observable with zero extra plumbing.
shutdown: the loader thread is a daemon thread; the process exits without
joining it.
__main__: --poll-interval; 0 disables polling (static serving, the
current behavior).
```

Warm-up contention note (goes in a comment): warming compiles/captures on
the same GPU that is serving; evals during a warm are slower, workers
park, nothing breaks. Accepted.

Amendment (post-review): loader-thread warming is impossible under
reduce-overhead — CUDA graph capture is single-threaded per process, so
capture from the loader thread fails and every checkpoint gets rejected
(verified on the GPU; mode="default" avoids it but costs 3x on the hot
path: 2.95ms vs 0.97ms at B=64). The implemented design instead has the
loader publish the slot unwarmed after build/load/to(device), and
apply_pending_swap warms 3x on the serving thread before adopting. The
serving pause is near zero for same-arch checkpoints (inductor cache hit;
measured ~0s vs 13s cold) and workers park through it either way. The
warm-count test asserts warming happens at adoption, not in the loader,
and the primary swap test runs compiled on the GPU.

## Stage 3: Tests

All torch tests, on the GPU:

```text
swap happens: publish v0, serve on a thread, eval (result carries v0's
model_version); publish v1 with different weights; with poll_interval
tiny (0.05s), subsequent evals carry v1's version within a bounded wait;
outputs differ from v0's for the same batch (weights actually changed);
no eval errors across the swap
tag mismatch refused: publish a checkpoint under a different schema
config; evals keep carrying the old version; exactly one stderr line
(capsys/capfd), serving uninterrupted
broken checkpoint dir mid-poll (manifest deleted): ignored, serving
uninterrupted
poll_interval 0: no loader thread, behavior identical to today
warm counts: the loader warms 3x (assert via a counting wrapper or
monkeypatched runner)
```

## Stage 4: Docs And Final Verification

```text
GZ_MODEL.md heads amendment (stage 1); GZ_PYTHON.md hot-swap section
marked implemented; AGENTS.md lists this spec.
```

```bash
python3 -m pytest python/tests
cargo test --all
```

Acceptance checklist:

```text
forward returns raw values; tanh lives in the backend; all four property
tests and the compile test still pass for both aggregations
a mid-serving swap is proven by model_version changing in EVAL_RESULTs
with zero errors
rejected checkpoints log once and never interrupt serving
poll_interval 0 preserves exact current behavior
```

## Out Of Scope

```text
the trainer and supervisor (work order 3)
multi-connection serving; background-stream warmup isolation
checkpoint URL sources
```
