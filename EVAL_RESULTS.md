# Evaluation Results

## Symmetric 1M Model: Noise and Search Budget

Date: 2026-07-17

### Checkpoint

- Run: `sym-s42m5-l96w4-r0-20k-fixed-1`
- Training step: `20000`
- Checkpoint pointer: `latest.json` (`version_2500`)
- Model version: `d5184f603818026cd87fab8e3c8f1414`
- Evaluation set: `runs/datasets/whittle-eval-v1`

### Method

Each of the 20 graphs ran one symmetric game with the same pinned checkpoint
controlling both seats. Results therefore contain 20 games and 40 measured seat
trajectories. Aggregate `mean cost` is the mean across both seats; `best-of-two`
is the mean of the lower P1/P2 terminal cost on each graph.

Common settings:

- `max_steps = 96`
- `c_visit = 50.0`
- `c_scale = 1.0`
- `tree_reuse = false`
- `wave_batching = true`
- STOP enabled
- `no_backtrack = true`
- `length_tiebreak = true`
- evaluator batch capacity `128`
- seed `42`
- mean root cost `98.45`

Noise off used `gumbel_scale = 0.0` and disabled overlap. Noise on used the
training settings `gumbel_scale = 1.0` and `gumbel_noise_overlap = 0.5`.

### Aggregate Results

| Noise | Considered / simulations | Mean P1 | Mean P2 | Mean cost | Mean absolute reduction | Mean percent reduction | Best-of-two | Mean rewrites | Stops | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Off | 8 / 48 | 57.05 | 57.15 | **57.10** | 41.35 | 40.977% | 55.35 | 93.525 | 18 / 40 | 66.705s |
| On | 8 / 48 | 59.90 | 58.05 | 58.975 | 39.475 | 39.202% | 56.95 | 88.900 | 18 / 40 | 64.794s |
| Off | 32 / 128 | 54.55 | 55.10 | **54.825** | **43.625** | **42.729%** | **53.55** | 89.400 | 22 / 40 | 152.284s |
| On | 32 / 128 | 87.60 | 87.80 | 87.70 | 10.75 | 10.181% | 83.75 | 77.975 | 16 / 40 | 156.897s |

All four evaluations completed all 20 games with zero dropped episodes and
used the same model version.

### Per-Graph Terminal Costs

Each result is `P1 / P2`; lower is better.

| Graph | Root | 8/48 off | 8/48 on | 32/128 off | 32/128 on |
|---:|---:|---:|---:|---:|---:|
| 0 | 97 | 57 / 55 | 47 / 49 | 44 / 45 | 69 / 74 |
| 1 | 126 | 48 / 43 | 97 / 88 | 41 / 44 | 134 / 142 |
| 2 | 122 | 65 / 62 | 73 / 68 | 56 / 54 | 81 / 85 |
| 3 | 84 | 44 / 43 | 45 / 49 | 92 / 88 | 91 / 83 |
| 4 | 52 | 38 / 39 | 40 / 40 | 39 / 39 | 42 / 43 |
| 5 | 114 | 57 / 56 | 57 / 56 | 54 / 55 | 81 / 95 |
| 6 | 77 | 44 / 44 | 46 / 48 | 46 / 52 | 75 / 70 |
| 7 | 126 | 64 / 65 | 67 / 65 | 65 / 68 | 131 / 111 |
| 8 | 81 | 54 / 57 | 68 / 59 | 51 / 47 | 95 / 88 |
| 9 | 83 | 43 / 42 | 42 / 42 | 43 / 44 | 64 / 80 |
| 10 | 135 | 104 / 89 | 83 / 79 | 133 / 130 | 117 / 112 |
| 11 | 124 | 120 / 114 | 119 / 95 | 53 / 52 | 131 / 124 |
| 12 | 95 | 51 / 51 | 54 / 54 | 50 / 50 | 96 / 95 |
| 13 | 65 | 44 / 45 | 50 / 49 | 42 / 44 | 57 / 60 |
| 14 | 96 | 51 / 52 | 58 / 57 | 55 / 54 | 87 / 86 |
| 15 | 120 | 37 / 42 | 49 / 53 | 41 / 36 | 77 / 93 |
| 16 | 93 | 48 / 48 | 40 / 42 | 43 / 48 | 98 / 97 |
| 17 | 105 | 54 / 56 | 60 / 68 | 52 / 55 | 92 / 70 |
| 18 | 111 | 68 / 84 | 56 / 53 | 50 / 52 | 71 / 81 |
| 19 | 63 | 50 / 56 | 47 / 47 | 41 / 45 | 63 / 67 |

### Direct Observations

- With noise off, increasing the budget from `8/48` to `32/128` improved mean
  cost by `2.275` and mean percent reduction by `1.752` percentage points.
- At `8/48`, enabling training noise increased mean cost by `1.875`.
- At `32/128`, enabling the same noise increased mean cost by `32.875` and
  reduced mean percent reduction by `32.548` percentage points.
- The noisy `32/128` degradation is not exclusively explained by STOP. It had
  fewer stopped seats than deterministic `32/128` (`16` versus `22`), although
  several seats stopped after only 3 to 42 rewrites and its mean trajectory was
  shorter.
- Noise-enabled measurements are one fixed-seed evaluation, not a multi-seed
  estimate or confidence interval.

### Commands

```bash
cargo build --release -p gz-cli --example eval_symmetric_whittle_set

target/release/examples/eval_symmetric_whittle_set runs/datasets/whittle-eval-v1 runs/sym-s42m5-l96w4-r0-20k-fixed-1/checkpoints cuda:0 latest.json 8 48 0 -1 42
target/release/examples/eval_symmetric_whittle_set runs/datasets/whittle-eval-v1 runs/sym-s42m5-l96w4-r0-20k-fixed-1/checkpoints cuda:0 latest.json 8 48 1 0.5 42
target/release/examples/eval_symmetric_whittle_set runs/datasets/whittle-eval-v1 runs/sym-s42m5-l96w4-r0-20k-fixed-1/checkpoints cuda:0 latest.json 32 128 0 -1 42
target/release/examples/eval_symmetric_whittle_set runs/datasets/whittle-eval-v1 runs/sym-s42m5-l96w4-r0-20k-fixed-1/checkpoints cuda:0 latest.json 32 128 1 0.5 42
```

Harness: `crates/gz-cli/examples/eval_symmetric_whittle_set.rs`

Verification:

```bash
cargo clippy -p gz-cli --example eval_symmetric_whittle_set -- -D warnings
```
