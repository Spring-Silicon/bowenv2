# Sample Reuse Ratio Control Implementation Spec

Status: implementation work order

Purpose: keep samples-per-row roughly constant by pausing whichever side
of the loop is ahead. Today nothing controls reuse in the direction that
matters: the trainer samples at full speed regardless of production, so
the ratio is an accident of hardware and drifts mid-run (first-curve ran
at ~8x, the 1024 run swung with stop-collapse). Fresh-data-per-gradient
-step becomes a controlled experimental variable instead of a residue.

Authority: `GZ_TRAINER.md` (the loop contract; amend its ratio-control
paragraph, which currently names only the backlog gate),
`GZ_EVAL_PROTOCOL.md` (sample protocol unchanged). Contract wins;
report conflicts.

Read before starting:

```text
python/gz/trainer/driver.py     the trainer loop; refresh() already
                                polls live produced/episode counters
python/gz/trainer/sampler.py    SampleClient.refresh -- the poll channel
crates/gz-orchestrator/src/lanes.rs  ReplayBackpressure -- the existing
                                admission gate (absolute backlog cap)
crates/gz-cli/src/selfplay.rs   --replay-backlog plumbing to copy
crates/gz-replay/src/store.rs   produced/consumed atomics (both live)
```

## The Ratio And Its Two Actuators

```text
r = consumed_rows / produced_rows, measured on deltas over a control
window (cumulative ratios are dominated by history and respond too
slowly to be a control signal).

One target band with hysteresis, R_low < R_high:
  r > R_high  ->  the trainer is grinding stale data: TRAINER pauses
                  before its next sample until production catches up.
  r < R_low   ->  selfplay outruns consumption: SELFPLAY admission
                  pauses (the existing backpressure machinery with a
                  ratio-derived, moving threshold).
The band gap prevents oscillation; both gates binding simultaneously
is impossible (r cannot be both above R_high and below R_low).
```

Config surface (one quantity, two watermarks):

```text
[trainer]  max_samples_per_row = 0.0   # R_high; 0 disables
[selfplay] min_samples_per_row = 0.0   # R_low;  0 disables
Validation: when both are set, min < max / 1.5 (an explicit hysteresis
gap); each works alone. --replay-backlog stays as the absolute safety
cap and composes (either gate may bind).
```

## Trainer Actuator (Python, driver loop)

```text
Before each sample: if (consumed + batch) / produced > R_high, poll
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

## Selfplay Actuator (Rust, admission gate)

```text
Generalize ReplayBackpressure: today it blocks admission while
produced - consumed > max_row_backlog. Add an optional ratio bound:
block admission while
  produced > consumed / R_low + slack_rows
with slack_rows defaulting to 50_000 so selfplay can build the initial
buffer, and a hard exemption while consumed == 0 (bootstrap and
startup would otherwise deadlock).

Both counters are already lock-free atomics on the shared store; the
gate polls them exactly as the backlog gate does (gate_poll 1ms).
CLI: --replay-min-reuse FLOAT (0 = off), plumbed like --replay-backlog;
driver passes [selfplay] min_samples_per_row through.
```

## Semantics Notes

```text
Gates shift timing only. Window contents are already timing-dependent
(the sample window anchors to live produced_rows), so no new class of
nondeterminism is introduced; per-step sampling stays seeded by
(run_seed, step).
Deadlock analysis: trainer gate waits on production with liveness
checks; selfplay gate waits on consumption and is released by trainer
progress or killed by the supervisor on trainer failure (existing
process-group kill). The two bands cannot bind at once.
Retention does not interact: both counters are monotonic and
unaffected by row deletion.
```

## Stages

```text
1. Trainer gate: config, loop integration, gate_wait_ms metric.
   Tests: gate math unit tests (startup exemption, bound/unbound);
   integration -- tiny selfplay (1 lane, 1 worker) + max_samples_per_row
   small: trainer gate engages, windowed ratio lands within 20% of the
   cap, gate_wait_ms > 0 in metrics; kill selfplay mid-gate -> run
   aborts naming selfplay (not a hang).
2. Selfplay gate: ReplayBackpressure ratio bound, CLI flag, driver
   plumb. Tests: unit math incl. consumed==0 exemption and slack;
   integration -- step_sleep large + min_samples_per_row: production
   stalls, windowed ratio stays >= R_low, and resumes when sampling
   resumes (mirror the existing live-backpressure test).
3. Docs: GZ_TRAINER.md ratio paragraph rewritten (backlog cap = safety,
   ratio band = control); config docs; AGENTS.md lists this spec.
   Full-run configs gain max_samples_per_row = 12, min_samples_per_row
   = 2 as commented-out suggestions (off by default -- turning control
   on is an experiment, not a default flip).
```

Acceptance checklist:

```text
with both gates off, behavior is byte-identical to today
trainer gate holds windowed reuse near R_high under starved production
selfplay gate holds windowed reuse near R_low under a slowed trainer
no hangs: producer death aborts a gating trainer; trainer death kills
the selfplay group (existing supervision)
perf/gate_wait_ms visible; all suites green
```

## Out Of Scope

```text
PID/derivative controllers -- watermark hysteresis is enough at these
timescales
dynamic retargeting of R during a run
gating the evaluator or publish cadence
```
