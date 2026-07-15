# Benchmark Ablations

These runs are controlled screens against the recorded benchmark recipes.
Each leaf extends one immutable base directly; inheritance stays one layer
deep.

Compare runs by completed actor cohort and optimizer step, never wall time.
The primary gauges are terminal mean/best cost, STOP rate, episode length,
learner win rate, samples per row, and measured-final repeat rate.

| Run | Baseline | Added treatment |
| --- | --- | --- |
| `cadence-01-s240` | archived effective `benchmark-steady-arena-r8-1m` | 32 actors, 8 updates per 32 episodes, publish every 8 with one-cohort lag |
| `throughput-02-policy-budget-10k` | current steady-arena `44x32` throughput recipe | policy-only normalized remaining budget |

Run one seed through 240 steps as a screen. Replicate a treatment only after it
changes the qualitative outcome; otherwise move to the next single treatment.
Cadence-01 starts from the archived run's exact `version_0` weights with a
fresh optimizer, avoiding initialization-order drift in the current model code.
Throughput-02 uses the baseline's 10,000-step horizon. The discarded 2,048-step
startup probe (`a7moiuxk`) showed roughly 269-second episode latency; at reuse 8,
it could have finished from less than one initial actor cohort and therefore
could not measure trained-policy self-play.
