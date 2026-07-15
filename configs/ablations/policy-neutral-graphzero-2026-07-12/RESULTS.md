# Neutral Policy Initialization on GraphZero Exphormer

## Control

All three runs extend `gated-pool32-s42-throughput.toml` and therefore use the
GraphZero-native four-layer Exphormer, pointer policy, pair/tanh value head,
fixed seed-42 root, gated 32-trajectory opponent pool, and 44 x 32 actor
topology. The runs hold data seed 42 fixed, set
`trainer.policy_init = "neutral"`, and vary only model seed 5, 17, or 42.

The horizon is 1,000 trainer steps. The matched default-initializer control is
`runs/gated-pool32-s42-10k`, evaluated at the same step-991 logging point.

## Results

`terminal mean` is the row-weighted cost in the sampled replay batch.
`episode EMA` is the replay service's EMA over newly completed self-play
episodes. Lower cost is better. `tail mean` averages the ten logged replay
means from steps 901 through 991.

| Initializer | Model seed | Step-991 terminal mean | Step-991 episode EMA | Tail mean | Best cost |
|---|---:|---:|---:|---:|---:|
| default | 42 | 141.05 | 140.45 | 140.61 | 116 |
| neutral | 5 | 119.94 | 104.25 | 134.83 | 99 |
| neutral | 17 | 168.68 | 135.30 | 169.29 | 101 |
| neutral | 42 | 103.30 | 102.51 | 107.12 | 92 |

All runs completed normally and published step 1,000. Their metrics and logs
are under `runs/policy-neutral-gz-s42-model{5,17,42}-1`.

## Verdict

Uniform policy initialization helps but is not sufficient on this setting.
It fixes the matched seed-42 failure and seed 5's newly completed episodes are
also strong by step 1,000. Seed 17 remains poor in replay and only reaches a
135.30 episode EMA despite finding a cost-101 episode. The intervention removes
initial policy-logit bias, but Exphormer training remains materially dependent
on the randomly initialized trunk and value pathway.
