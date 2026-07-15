# Pool32 causal ablations, 2026-07-10

All screens used the fixed seed-42 root, 44 lanes x 32 workers, 48
simulations, one evaluator, fresh replay, and the release GraphZero binary
whose SHA-256 begins `37617e9c`. Failed screens were stopped from rolling
metrics at step 1,000 or 1,500. Replay stores were pruned after each run;
metrics, logs, checkpoints, W&B runs, and queue summaries remain.

| Run | Result | Last step | Mean cost | Best cost |
|---|---:|---:|---:|---:|
| Exact historical positive control | pass | 2,491 | 92.3 | 85 |
| Seed 5 only | fail | 1,011 | 138.3 | 114 |
| Seed 5 + clip 3 | fail | 1,011 | 135.9 | 103 |
| Seed 5 + batch 512 | fail | 1,001 | 137.0 | 108 |
| Seed 5 + old optimizer, batch 256, clip 1 | fail | 1,021 | 134.6 | 101 |
| Seed 5 + clip 3 + batch 512, AdamW | fail | 1,001 | 137.0 | 108 |
| Seed 5 + old optimizer + clip 3, batch 256 | fail | 1,001 | 134.8 | 99 |
| Seed 5 + old optimizer + batch 512, clip 1 | pass | 2,491 | 95.6 | 83 |
| Seed 42 + old optimizer + batch 512, clip 1 | fail | 1,011 | 164.8 | 158 |
| Minimal trainer + Exphormer | fail screen | 1,561 | 120.6 | 102 |
| Minimal trainer + live sampled trajectory | pass | 2,491 | 101.2 | 87 |
| Minimal trainer + GraphZero encodings + SAGE | pass | 2,491 | 94.1 | 77 |
| Seed 17 + old optimizer + batch 512 | fail screen | 1,501 | 116.4 | 100 |
| Model seed 5 + data seed 42 | pass | 1,491 | 103.3 | 91 |
| Model seed 42 + data seed 5 | fail | 1,001 | 164.9 | 158 |
| AdamW + `2e-4` cosine | fail | 1,001 | 137.8 | 109 |
| Muon + `3e-4` constant | fail | 1,001 | 135.4 | 99 |
| Muon + `2e-4` constant | pass | 1,491 | 100.7 | 94 |
| Muon + `3e-4` cosine | fail | 1,001 | 133.2 | 100 |

## Conclusions

1. The full-throughput actor topology is not the cause: every passing control
   used the ordinary 44 x 32 topology.
2. Active-policy sampled trajectories are not the cause. With the verified
   trainer and legacy architecture they reached mean cost 101.2.
3. In the pool32/shared-value context, the minimal passing trainer factors are
   model seed 5, Muon, learning rate `2e-4`, and batch 512. Removing any one
   failed its paired screen. Cosine decay and clip 3 are not required.
4. The seed effect belongs to the Torch/model RNG branch, which controls model
   initialization and dropout. Model5/data42 passed; model42/data5 failed.
   Initial gated costs were 110 for model seed 5 and 157 for model seed 42.
5. Exphormer is the architecture factor in the controlled GraphZero bundle.
   The exact SAGE reciprocal reached mean 94.1; Exphormer remained near 120
   through step 1,561. GraphZero's other encodings/profile work with SAGE.

These are interaction results for this fixed graph and opponent/data recipe.
They do not imply that AdamW, learning rate `3e-4`, or batch 256 are
universally broken: earlier no-pool/independent-value runs succeeded with
these settings. Here they are interaction factors in the defined screen.
