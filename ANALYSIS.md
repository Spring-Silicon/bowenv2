# 1M Auxiliary-Head Model: 20-Graph STOP Timing Analysis

Date: 2026-07-19

## Question

When the final 1M checkpoint is evaluated on the 20-graph set with a 96-rewrite
limit, when does each trajectory first reach its best measured cost, and what
does it do afterward?

The target checkpoint is from:

`whittle-sym-gen-s42-m5-l96-w4-1m-reuse-on-vw300k-vtg0p1-c4s48-aux-v8-v32-score-clip10-20k-1`

The saved evaluation used:

- checkpoint `step_20000.json`, model version
  `a425d7634080f77acb3464577c755a9e`;
- `runs/datasets/whittle-eval-v1`;
- 20 cases, 96 maximum rewrites, 4 considered actions, and 48 simulations per
  move;
- Gumbel scale `0`, seed `42`, tree reuse on, wave batching on, and maximum
  evaluator batch 32.

In this document, step 0 is the root and step N means that N candidate rewrites
have been applied. Selecting STOP is not counted as a rewrite.

## Reproducibility Caveat

The original evaluator saved terminal summaries but deleted its temporary
replay store. Its log therefore cannot recover intermediate graph states. The
original saved result was:

| Metric | Original evaluation |
|---|---:|
| Mean root cost | 98.450 |
| Mean seat cost | 42.225 |
| Mean best-of-two cost | 41.700 |
| Mean absolute reduction | 56.225 |
| Mean percent reduction | 55.021% |
| STOP selections | 22/40 |
| Mean rewrites per seat | 79.050 |

I reran the exact nominal configuration while retaining replay, then replayed
every selected action and called `GraphEngine::measure` on every intermediate
graph offline. The retained rerun produced mean best-of-two cost 41.550, mean
seat cost 41.825, 21/40 STOP selections, and 82.300 mean rewrites.

The rerun was not bit-reproducible despite a fixed seed and zero Gumbel scale.
Several individual trajectories differed from the saved evaluation. The timing
results below are exact for the retained rerun, not for the deleted original
trajectories. The cause of this evaluation nondeterminism has not been
diagnosed.

## Main Finding

The model is not finding a better graph and then destroying it. It is reaching
its final minimum and continuing to make cost-neutral rewrites.

- All 40 seat trajectories ended at their own lowest observed cost.
- Across all 40 seats, 38 reached that cost before their trajectory ended.
- They made 777 rewrites after first reaching their final minimum: 19.425 per
  seat on average.
- All 777 post-minimum rewrites were cost-neutral. There were zero measured
  improvements and zero measured regressions after the minimum was first
  reached.
- Looking only at the earliest seat to attain each graph's best-of-two cost,
  18/20 graphs reached their best before that seat ended.
- Those winning trajectories made 245 additional neutral rewrites, or 12.250
  per graph, after the best cost was already present.
- The winning seat eventually selected STOP on 17/20 graphs. Those 17 seats
  still made 149 neutral rewrites after first reaching the best cost, an
  average delay of 8.765 rewrites before STOP.
- The other three winning seats hit the 96-rewrite limit after 27, 49, and 20
  neutral post-best rewrites.

This is evidence of delayed plateau stopping, not failure to preserve the best
graph.

## Per-Graph Timing

`First best` is the earliest rewrite step at which either seat reached the
lowest cost observed for that graph. `Seat end` is the selected seat's final
rewrite count. `After best` is their difference.

| Graph | Best cost | First best | Seat | Seat end | After best | Ended by STOP |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 41 | 52 | 2 | 78 | 26 | yes |
| 1 | 39 | 69 | 1 | 96 | 27 | no |
| 2 | 43 | 87 | 1 | 87 | 0 | yes |
| 3 | 33 | 59 | 1 | 59 | 0 | yes |
| 4 | 36 | 21 | 2 | 28 | 7 | yes |
| 5 | 51 | 64 | 2 | 72 | 8 | yes |
| 6 | 39 | 39 | 1 | 55 | 16 | yes |
| 7 | 57 | 80 | 1 | 88 | 8 | yes |
| 8 | 44 | 51 | 1 | 61 | 10 | yes |
| 9 | 39 | 38 | 1 | 43 | 5 | yes |
| 10 | 41 | 72 | 2 | 79 | 7 | yes |
| 11 | 47 | 68 | 2 | 69 | 1 | yes |
| 12 | 37 | 71 | 1 | 76 | 5 | yes |
| 13 | 41 | 58 | 2 | 68 | 10 | yes |
| 14 | 47 | 55 | 2 | 80 | 25 | yes |
| 15 | 36 | 63 | 2 | 67 | 4 | yes |
| 16 | 42 | 47 | 1 | 96 | 49 | no |
| 17 | 46 | 76 | 1 | 96 | 20 | no |
| 18 | 37 | 62 | 1 | 66 | 4 | yes |
| 19 | 35 | 48 | 2 | 61 | 13 | yes |

Across graphs, the first-best step had mean 59, median 62, and range 21-87.
Every realized best was therefore present before step 96, although one did not
appear until step 87.

## On-Trajectory Budget Snapshots

The following table truncates the retained 96-step trajectories at each depth.
If a seat had already selected STOP, its terminal graph is held fixed. This is
a hindsight view of the same trajectories, not a new evaluation configured
with a smaller maximum budget.

| Rewrite step | Mean best-of-two | Gap from terminal | Graphs already terminal-best |
|---:|---:|---:|---:|
| 16 | 74.850 | +33.300 | 0/20 |
| 24 | 64.600 | +23.050 | 1/20 |
| 32 | 56.450 | +14.900 | 1/20 |
| 48 | 46.700 | +5.150 | 5/20 |
| 64 | 42.500 | +0.950 | 13/20 |
| 80 | 41.650 | +0.100 | 19/20 |
| 96 | 41.550 | +0.000 | 20/20 |

Most of the realized result was present by step 64, and nearly all of it was
present by step 80. Step 48 was still materially worse. A real evaluation with
`max_steps = 64` or `80` may follow different trajectories because remaining
budget is a model input and changing the configured horizon changes every
position's features. These snapshots cannot replace actual shorter-budget
evaluations.

## Interpretation

The current training signal gives only indirect pressure to stop on a measured
cost plateau:

- `GraphEngine::measure` supplies terminal reward; search does not measure each
  intermediate graph during normal operation.
- Symmetric value targets are relative game outcomes (`z` and `-z`), including
  the rewrite-count tiebreak. They do not directly label the absolute measured
  utility of STOP at every state.
- With STOP enabled, policy training imitates the MCTS target for STOP like any
  other legal action. It does not receive a separate label saying that the
  current measured cost has stopped improving.
- The auxiliary terminal-score head regularizes the learned representation,
  but the current evaluator serves the policy and main value used by search;
  the score head is not directly used to rank STOP against continuation.

The model does have the remaining-budget feature and the terminal
rewrite-count tiebreak, so it is not completely missing a stopping incentive.
The trajectory data shows that this indirect incentive is insufficiently
calibrated: it often recognizes STOP eventually, but only after a long sequence
of rewrites that leave measured cost unchanged.

This diagnosis does not prove that every neutral rewrite was irrational when
selected. A neutral rewrite can be a setup move whose expected downstream
value is positive. It shows, in hindsight, that the selected continuation did
not realize any further benefit on these trajectories.

## Next Diagnostic

The next evaluation should retain replay and log, per move:

- measured cost for offline analysis;
- STOP policy prior and target mass;
- whether STOP entered the considered-action set;
- STOP visit count and visit fraction;
- `Q(STOP) - max Q(continue)`;
- selected action and whether the following rewrite changed measured cost.

Comparing those fields at checkpoints 13k, 17k, 20k, and 30k would distinguish
three failure modes: the policy never proposes STOP, search considers STOP but
assigns it a poor value, or search values STOP correctly but sequential
halving still selects continuation. The zero-noise evaluation path should also
be made reproducible before interpreting small checkpoint differences.

## Related Training Continuation

After this analysis, the same 1M run was resumed from step 20,000 to step
30,000 with constant AdamW learning rate reduced from `2e-4` to `2e-5`.
Optimizer moments restart under the current resume implementation. The first
verified resumed records at steps 20,001, 20,011, and 20,021 all reported
`lr = 2e-5`; by step 20,021 throughput had returned to 3.859 steps/s after
initial compilation, gradient norm was 1.927, and clipping was inactive.
