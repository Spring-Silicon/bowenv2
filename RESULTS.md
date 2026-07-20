# Evaluation Results

## Symmetric Gumbel-MCTS: 8/48 versus 8/128

Measured on 2026-07-18 using the fixed 20-graph Whittle evaluation set at
`runs/datasets/whittle-eval-v1`.

### Checkpoint and protocol

- Run: `whittle-sym-gen-s42-m5-l96-w4-1m-reuse-on-vw300k-vtg0p1-20k-1`
- Checkpoint: `latest.json`, training step 20,000
- Model version: `e15098b3252f68dbe00e471827fd96a5`
- Policy and value: same checkpoint
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition, `cuda:0`
- Maximum episode steps: 96
- Considered actions: 8
- Simulations: 48 or 128
- Gumbel scale: 0
- Gumbel-noise overlap: disabled (`-1`)
- C scale: 1
- Tree reuse: off
- Wave batching: on
- Seed: 42

The two evaluations differ only in simulation count.

### Aggregate results

Lower cost is better.

| Metric | 8/48 | 8/128 | 8/128 delta |
|---|---:|---:|---:|
| Mean root cost | 98.450 | 98.450 | 0.000 |
| Mean P1 cost | 43.600 | 43.750 | +0.150 |
| Mean P2 cost | 43.700 | 42.800 | -0.900 |
| Mean seat cost | 43.650 | 43.275 | **-0.375** |
| Mean best-of-two cost | 43.100 | 42.350 | **-0.750** |
| Mean absolute reduction | 54.800 | 55.175 | +0.375 |
| Mean percent reduction | 53.490% | 53.914% | +0.424 pp |
| P1/P2 wins/ties | 6/7/7 | 6/8/6 | — |
| Stopped seats | 24/40 | 22/40 | -2 |
| Mean rewrites per seat | 90.100 | 89.075 | -1.025 |
| Evaluation rows | 175,694 | 455,771 | 2.594x |
| Evaluation batches | 3,723 | 11,073 | 2.974x |
| Mean batch size | 47.192 | 41.161 | -12.8% |
| Harness elapsed time | 62.151 s | 181.586 s | **2.922x** |

The 128-simulation search reduced mean seat cost by 0.375, or 0.86%, while
increasing elapsed time by 119.435 seconds, or 192.2%.

### Paired outcomes

Using each graph's mean across its two seats:

- 8/128 improved 8 graphs, tied 6, and regressed 6.
- The paired mean delta was -0.375 cost with standard deviation 1.919.
- An approximate paired 95% interval across these graphs is [-1.273, +0.523].

Using individual seats:

- 8/128 improved 15 seats, tied 15, and regressed 10.

Using the better result from the two seats:

- 8/128 improved 7 graphs, tied 11, and regressed 2.
- Mean best-of-two cost improved by 0.750.

The interval above describes variation across this fixed graph set. It is not
a multi-seed confidence interval.

### Per-graph costs

`Mean delta` is the 8/128 mean seat cost minus the 8/48 mean seat cost, so a
negative value favors 8/128.

| Graph | Root | 8/48 P1/P2 | 8/128 P1/P2 | Mean delta |
|---:|---:|---:|---:|---:|
| 0 | 97 | 41 / 42 | 41 / 43 | +0.5 |
| 1 | 126 | 39 / 39 | 39 / 39 | 0.0 |
| 2 | 122 | 49 / 49 | 53 / 49 | +2.0 |
| 3 | 84 | 38 / 36 | 33 / 33 | -4.0 |
| 4 | 52 | 38 / 37 | 37 / 38 | 0.0 |
| 5 | 114 | 51 / 50 | 51 / 53 | +1.5 |
| 6 | 77 | 39 / 39 | 39 / 39 | 0.0 |
| 7 | 126 | 61 / 62 | 62 / 60 | -0.5 |
| 8 | 81 | 49 / 52 | 50 / 49 | -1.0 |
| 9 | 83 | 39 / 39 | 39 / 39 | 0.0 |
| 10 | 135 | 41 / 41 | 41 / 41 | 0.0 |
| 11 | 124 | 51 / 50 | 49 / 50 | -1.0 |
| 12 | 95 | 37 / 38 | 38 / 40 | +1.5 |
| 13 | 65 | 45 / 44 | 44 / 41 | -2.0 |
| 14 | 96 | 53 / 52 | 49 / 47 | -4.5 |
| 15 | 120 | 36 / 36 | 36 / 36 | 0.0 |
| 16 | 93 | 36 / 41 | 47 / 36 | +3.0 |
| 17 | 105 | 52 / 49 | 49 / 47 | -2.5 |
| 18 | 111 | 42 / 43 | 40 / 41 | -2.0 |
| 19 | 63 | 35 / 35 | 38 / 35 | +1.5 |

### Interpretation

For mean seat cost, 8/128 provides a small and graph-dependent improvement for
nearly three times the elapsed time. The 20-graph result does not establish a
reliable improvement in this metric. At this checkpoint, 8/48 is the stronger
compute-quality tradeoff.

The best-of-two metric favors 8/128 more consistently. That result is relevant
only when the deployed procedure actually runs both seats or two equivalent
rollouts and selects the better terminal graph.

### Reproduction

```bash
cargo build --release -p gz-cli --example eval_symmetric_whittle_set

target/release/examples/eval_symmetric_whittle_set \
  runs/datasets/whittle-eval-v1 \
  runs/sym1m-s42m5-l96w4-r1-vw300k-vtg0p1-20k-1/checkpoints \
  cuda:0 latest.json 8 48 0 -1 42

target/release/examples/eval_symmetric_whittle_set \
  runs/datasets/whittle-eval-v1 \
  runs/sym1m-s42m5-l96w4-r1-vw300k-vtg0p1-20k-1/checkpoints \
  cuda:0 latest.json 8 128 0 -1 42
```

Harness: `crates/gz-cli/examples/eval_symmetric_whittle_set.rs`.
