# Neutral Policy Initialization

## Question

Does zero-initializing the final pointer scoring projection reduce fixed-root
training sensitivity to the model seed?

## Control

The three runs use the same fixed seed-42 root, data seed 42, gated-policy
trajectory pool, architecture, optimizer, replay cadence, and 1,000 training
steps. Only `trainer.model_seed` changes between 5, 17, and 42. All runs use
`trainer.policy_init = "neutral"`.

At checkpoint version 0, all three models emit exactly zero for every one of
the root's 772 valid action logits. The resulting entropy is `log(772)` and
STOP probability is `1 / 772` for every model seed.

## Results

Lower terminal cost is better. Tail mean is the mean of the ten logged
terminal-cost means from steps 901 through 991.

| Model seed | Step 1 mean | Step 501 mean | Step 751 mean | Step 991 mean | Tail mean | Best observed |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 151.53 | 155.54 | 100.61 | 88.98 | 88.40 | 75 |
| 17 | 116.04 | 142.82 | 103.57 | 101.51 | 101.51 | 85 |
| 42 | 111.12 | 142.54 | 111.42 | 92.33 | 92.46 | 76 |

Matched runs with the default initializer are available for model seeds 5 and
42:

| Model seed | Initializer | Step 991 mean | Tail mean | Best observed through step 991 |
|---:|---|---:|---:|---:|
| 5 | default | 117.17 | 120.38 | 91 |
| 5 | neutral | 88.98 | 88.40 | 75 |
| 42 | default | 164.76 | 164.72 | 158 |
| 42 | neutral | 92.33 | 92.46 | 76 |

The matched default runs are `runs/pool32-causal-model5-data42-1` and
`runs/pool32-causal-seed42-old-optimizer-batch512-1`. The neutral run metrics
are under `runs/policy-neutral-s42-model{5,17,42}-1`.

## Verdict

The initializer passed this short experiment: all three seeds escaped the bad
fixed-root basin by 1,000 steps, and the previously collapsing seed 42 improved
its tail mean from 164.72 to 92.46. It does not make startup seed-independent:
the step-1 means still range from 111.12 to 151.53 because the random value
head continues to influence MCTS before it is trained. Longer runs are still
required to establish final solution quality.
