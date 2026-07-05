# Sample Reuse Ratio Control Implementation Spec

Status: implementation work order (deferred -- 8x reuse accepted for now)

Purpose: cap samples-per-row by pausing the trainer when it gets ahead
of production. Today nothing controls reuse: the trainer samples at
full speed regardless of production, so the ratio is an accident of
hardware and drifts mid-run (first-curve ran at ~8x, the 1024 run swung
with stop-collapse). Fresh-data-per-gradient-step becomes a controlled
experimental variable instead of a residue. Trainer-side only: the
producer side is deliberately not gated (see Out Of Scope).

Authority: `GZ_TRAINER.md` (the loop contract; amend its ratio-control
paragraph, which currently names only the backlog gate),
`GZ_EVAL_PROTOCOL.md` (sample protocol unchanged). Contract wins;
report conflicts.

Read before starting:

```text
python/gz/trainer/driver.py     the trainer loop; refresh() already
                                polls live produced/episode counters
python/gz/trainer/sampler.py    SampleClient.refresh -- the poll channel
```

## The Ratio And Its Gate

```text
r = (consumed_rows - anchor_c) / (produced_rows - anchor_p), CUMULATIVE
from a steady-state anchor (counters snapshotted when the trainer loop
starts). This makes the gate a synchronous quota -- "k gradient steps
per n new rows", the standard update-to-data control in practical
AlphaZero-style loops -- with a built-in startup grace period from the
bootstrap rows. Windowed (per-log-interval) reuse is REPORTED as a
metric but never used as the control signal: delta control oscillates
at these timescales.

One cap: while r > max_samples_per_row, the trainer pauses before its
next sample until production catches up.
```

Config surface:

```text
[trainer]  max_samples_per_row = 0.0   # 0 disables (today's behavior)
--replay-backlog stays as the producer-side absolute safety cap; the
two are independent and cannot deadlock against each other (one pauses
the consumer, the other the producer, on opposite inequalities).
```

## The Gate (Python, driver loop)

```text
Before each sample: if (consumed_next - anchor_c) / (produced -
anchor_p) > max_samples_per_row, poll
sampler.refresh() with a short sleep (0.2s) until the inequality
clears. consumed = (step + 1) * batch is exact trainer-side knowledge;
produced comes from the ack.

Exemptions and liveness:
  never gates until produced >= min_startup_rows (startup exemption)
  every poll iteration runs check_child(selfplay) and check_memory --
  a dead producer aborts the run instead of waiting forever (the 300s
  socket timeouts and reconnecting sampler make long idles safe)
Metrics: accumulate wait into the perf window; log perf/gate_wait_ms
per log interval (0 when unbound). The lr schedule becomes data-coupled
while gating -- that is the point, note it in GZ_TRAINER.md.
```

## Semantics Notes

```text
Gates shift timing only. Window contents are already timing-dependent
(the sample window anchors to live produced_rows), so no new class of
nondeterminism is introduced; per-step sampling stays seeded by
(run_seed, step).
Deadlock analysis: the gate waits on production with liveness checks
(dead producer -> abort); the backlog safety cap pauses the producer
on the opposite inequality, so both cannot bind at once.
Retention does not interact: both counters are monotonic and
unaffected by row deletion.
```

## Stages

```text
1. The gate: config, loop integration, gate_wait_ms metric.
   Tests: gate math unit tests (startup exemption, bound/unbound);
   integration -- tiny selfplay (1 lane, 1 worker) + max_samples_per_row
   small: gate engages, windowed ratio lands within 20% of the cap,
   gate_wait_ms > 0 in metrics; kill selfplay mid-gate -> run aborts
   naming selfplay (not a hang).
2. Docs: GZ_TRAINER.md ratio paragraph rewritten (backlog cap = safety,
   trainer quota = control); config docs; AGENTS.md lists this spec.
   Full-run configs gain max_samples_per_row = 12 as a commented-out
   suggestion (off by default -- turning control on is an experiment,
   not a default flip).
```

Acceptance checklist:

```text
with the gate off, behavior is byte-identical to today
the gate holds windowed reuse near the cap under starved production
no hangs: producer death aborts a gating trainer
perf/gate_wait_ms visible; all suites green
```

## Out Of Scope

```text
a producer-side ratio gate (pausing selfplay when reuse drops below a
floor) -- non-standard, and only earns its keep in the compiler regime
where every row carries a paid measurement and reuse < 1 wastes spend;
revisit alongside the compiler engine
PID/derivative controllers -- the cumulative quota is enough
dynamic retargeting of R during a run
gating the evaluator or publish cadence
reanalyze (MuZero-style target regeneration) -- the eventual
replacement for the R_high cap in the compiler regime, where stale
rows should be re-searched with the current net instead of rationed;
a separate future work order
```
