# Uniform GraphZero Batch Scaling

## Control

All runs use the full GraphZero four-layer Exphormer recipe, neutral policy
initialization, fixed data/self-play seed 42, gated 32-trajectory opponent pool,
and `max_reuse = 8`. Batch-512 model seed 17 is reused from the independent
ablation; this queue adds model seeds 5 and 42, then tests batch 1024 on model
seed 17. Every result is measured at a 1,000-step trainer horizon.

Lower cost is better. `terminal mean` is row-weighted over the sampled replay
batch. `episode EMA` tracks newly completed self-play episodes. `tail mean`
averages the ten logged replay means from steps 901 through 991.

## Batch 512 Across Seeds

| Model seed | Step 501 | Step 751 | Step 991 | Episode EMA 991 | Tail mean | Best |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 127.10 | 106.61 | 108.76 | 104.06 | 107.64 | 100 |
| 17 | 123.93 | 103.89 | 99.98 | 102.53 | 99.91 | 89 |
| 42 | 121.19 | 107.23 | 124.07 | 105.69 | 118.69 | 90 |
| Mean | 124.07 | 105.91 | 110.94 | 104.09 | 108.75 | - |

For the same three seeds, batch 256 averaged `130.64` step-991 terminal mean,
`114.02` episode EMA, and `137.08` tail mean. More importantly, its episode-EMA
range was `102.51` to `135.30`; batch 512 narrows that range to `102.53` to
`105.69`. Batch 512 therefore removes the observed seed collapse in this
three-seed fixed-root screen.

## Batch 1024 on Seed 17

| Batch | Step 501 | Step 751 | Step 991 | Episode EMA 991 | Tail mean | Best | Wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 168.37 | 170.00 | 168.68 | 135.30 | 169.29 | 101 | 6.5 min |
| 512 | 123.93 | 103.89 | 99.98 | 102.53 | 99.91 | 89 | 10.8 min |
| 1024 | 159.86 | 101.98 | 101.16 | 101.10 | 101.67 | 79 | 17.5 min |

| Batch | Produced rows at step 991 | Samples per produced row |
|---:|---:|---:|
| 256 | 80,814 | 3.14 |
| 512 | 200,149 | 2.54 |
| 1024 | 317,485 | 3.20 |

Batch 1024 learns later than batch 512, reaches essentially the same final
population quality, and costs 1.63x as much wall time. Its material advantage
is best trajectory quality: cost 79 versus 89. Since batch size is coupled to
the reuse gate and actor wall time, this is a full pipeline batch/cadence result,
not an isolated gradient-variance measurement.

## Verdict

Batch 512 is the better default for this fixed-root GraphZero Exphormer setup:
it succeeds across all three tested model seeds and matches Whittle-path
population quality without batch 1024's additional cost. Batch 1024 is useful
only if best-found trajectory quality is worth substantially slower training.
