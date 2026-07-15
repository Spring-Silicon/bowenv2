# Policy Opponent Sampling Modes

Status:

```text
sampled-trajectory  implemented
sampled-tree        implemented
```

Authority for sampled-tree mode:

```text
Policy-Based Self-Competition for Planning Problems
https://arxiv.org/pdf/2306.04403

Official reference implementation:
../policy-based-self-competition/gumbel_mcts.py
../policy-based-self-competition/gaz_ptp/experience_worker.py
```

## Configuration

```toml
[arch]
value_input = "pair"

[selfplay]
reference = "policy"
root_mode = "fixed"
policy_opponent_mode = "sampled-trajectory"
reference_gamma = 0.0
reference_trajectory_pool = 0
```

The complete enum is:

```text
greedy-trajectory    existing policy-reference behavior
sampled-trajectory  fresh active-policy rollout per learner episode
sampled-tree        paper-style two-player tree against the gated incumbent
```

When `policy_opponent_mode` is absent, all legacy behavior is unchanged,
including `reference_trajectory_pool`.

`sampled-trajectory` requires:

```text
reference = "policy"
root_mode = "fixed"
value_input = "pair"
reference_trajectory_pool = 0
reference_gamma = 0
a featurized evaluator
```

`reference_gamma` is rejected because this mode does not select an accepted
or latest historical checkpoint. It samples whichever active model the
evaluator serves.

## Sampled Trajectory Semantics

Each learner episode consumes one root and has two phases under one episode
ID:

```text
SampleReference
    run a categorical policy rollout from the root
    use the evaluator's ordinary active model on every policy request
    measure the terminal reference graph
    materialize a unique, one-use trajectory reference

RunLearner
    run the ordinary Gumbel MCTS episode from the same root
    pair every learner eval/replay row with the materialized trajectory
    measure the learner terminal graph
    append learner rows only
```

There is no global trajectory pool or fill barrier. A worker starts the
learner phase as soon as its own reference prelude completes. Reference-only
work does not increment learner episode or replay-row counters.

### Active Model Contract

Reference eval requests use the same active evaluator path as learner evals.
Checkpoint publication may hot-swap the active model between any two rollout
steps. This is intentional.

```text
step 0 may use checkpoint t
step 1 may use checkpoint t
step 2 may use checkpoint t+1
```

The resulting graph sequence and measured reward are fixed before learner
search begins, so the learner still receives one immutable opponent
trajectory.

The rollout task observes the model version returned for every policy
decision. The trajectory-level replay `model_version` is:

```text
Some(version)  every played reference step used that version
None           the rollout crossed a hot-swap, or has no policy step
```

The implementation never attributes a mixed rollout to one checkpoint. There
are no model-version leases, targeted eval frames, historical model slots, or
version-homogeneous batching in this mode.

The gated-policy challenge and `reference_gamma` identity selection are not
used by sampled-trajectory. `ReplayReferenceKind::Gumbel` identifies the
result as an active-policy rollout.

### Categorical Action Selection

At each reference state:

```text
1. Enumerate engine rewrite candidates and append STOP.
2. Evaluate the active policy once.
3. Draw one unit-scale Gumbel value per action.
4. Rank actions by policy_logit + Gumbel.
5. Try the fixed ranking until an action passes apply/no-backtrack masks.
6. Use STOP when it wins the ranking or every permitted rewrite fails.
```

Unit-scale Gumbel top-1 is exactly categorical sampling from softmax(policy
logits). This is direct policy sampling, not one-simulation MCTS: no child
value evaluation, sequential halving, overlap tempering, or tree reuse occurs.

The rollout inherits:

```text
max_steps
candidate options and deterministic candidate order
position feature export
no_backtrack
mask_stop, unless reference_mask_stop overrides it
measure options
```

When `mask_stop` is true, STOP is excluded while an applicable rewrite remains
and becomes the final fallback. Rejected apply results and their graph handles
are released before the next ranked action is tried.

Reference randomness is derived from the run/search seed, episode ID, and a
sampled-trajectory domain salt. It does not depend on evaluator completion
order and is separate from the learner episode's Gumbel stream.

### Pairing and Replay

The trajectory stores the root state, every resulting state, and its measured
terminal reward. Learner alignment remains:

```text
learner time t + search depth d -> reference row min(t + d, last_row)
```

STOP re-evaluation therefore sees the terminal opponent state when the
learner search reaches the opponent horizon.

Replay contains one row per actual learner move:

```text
policy target   existing learner Gumbel improved policy
value target    learner terminal result versus reference terminal result
opponent state  aligned sampled trajectory state
trajectory ID   unique and nonzero; never reused by another episode
```

Both terminal rewards come only from `GraphEngine::measure`. A failed or
invalid reference measure aborts that root before learner search and appends
no rows.

Replay backpressure blocks new reference preludes but never interrupts a
prelude or paired learner phase already admitted.

## Sampled Tree

`sampled-tree` is not an alias for sampled-trajectory. It requires:

```text
reference = "gated-policy"
root_mode = "fixed", or "generated" with a nonempty reference arena
value_input = "pair"
value_batch > 0
value_mirror = false
reference_gamma = 0
reference_trajectory_pool = 0
tree_reuse = false
length_tiebreak = false
eval_processes = 1
a torch evaluator and best.json incumbent checkpoint pointer
```

Each game follows the paper's two-player tree:

```text
two independent graph copies start at the same root
learner role is sampled between player 1 and player 2
learner turns use Gumbel MCTS
reference actions are sampled inside learner simulations
actual reference play uses greedy policy actions
both terminal graphs are measured
policy replay contains learner turns
value replay contains both player perspectives
```

The learner is assigned to player 1 or player 2 from the episode seed. Player 1
acts first. Learner search evaluations route to the current model and carry the
opponent graph directly as the pair input. Incumbent chance-node and actual-play
evaluations route to the evaluator following `best.json`. Chance-node actions
sample from the incumbent categorical policy; actual incumbent actions are
greedy. A checkpoint promotion may become visible between requests: existing
tree nodes retain their evaluated policy while newly expanded nodes use the
newly served incumbent.

STOP freezes only the actor that selected it. The other actor continues until
it selects STOP or reaches its own step budget. Both final graphs are measured.
Player 1 wins exact reward ties.

Replay atomically stores two records per completed game while incrementing the
completed-game counter once. The store persists a separate produced-policy-row
counter so startup and reuse gates do not count incumbent-only value rows as
new policy data. A replay directory is stamped `sampled-tree-v1` and cannot be
reopened in a standard mode, or vice versa. For turn index `t`, value examples
are:

```text
player 1  (p1_t, p2_t,     z)
player 2  (p2_t, p1_{t+1}, -z)
```

Frozen actors clamp to their terminal state. Learner rows carry the Gumbel
policy target. Incumbent rows carry an all-zero policy target and are excluded
from policy samples. Value samples draw from both records. A reference model
version is stored only when every actual step for that actor used one version.

## Acceptance Tests

Sampled-trajectory must verify:

```text
one root produces exactly one reference prelude and one learner episode
the prelude never enters replay or learner counters
each episode receives a unique nonzero trajectory ID
categorical ranking skips rejected and no-backtrack actions without re-drawing
reference measurement finishes before learner admission
mid-rollout model-version changes are accepted
mixed rollouts store no false trajectory-level model attribution
learner evals and replay rows use the same aligned opponent states
STOP re-evaluation clamps to the terminal opponent row
all success and failure paths release every created engine handle once
legacy configs without policy_opponent_mode remain behavior-identical
```

Sampled-tree must additionally verify:

```text
both learner roles preserve player-1-first alternation
current and incumbent eval work never crosses model routes
in-tree incumbent actions are categorical while actual actions are greedy
STOP freezes one actor and the other continues
player 1 receives +1 on exact ties and player 2 receives -1
replay pair append is atomic and counts as one completed game
policy samples contain only learner rows; value samples contain both players
replay pair states use (p1_t,p2_t) and (p2_t,p1_{t+1}) alignment
all success and failure paths release every created engine handle once
```
