# PUCT MCTS

## Scope

`PuctMcts` is the retained single-agent AlphaZero-style PUCT implementation in
`gz-search`. It shares node storage, async work, episode progression, STOP,
measurement, and tree reuse with Gumbel MCTS. It does not implement symmetric
selfplay, root Dirichlet noise, or intra-root parallel simulations.

## Configuration

`PuctMctsConfig` controls:

- episode step and simulation budgets;
- `c_puct` exploration strength;
- root action temperature and temperature move count;
- tree reuse;
- position-feature export;
- STOP masking and no-backtrack;
- engine candidate and measure options.

All semantic fields participate in `puct_search_config_hash`.

## Expansion

The shared root task expands a graph once, preserving engine candidate order,
and appends STOP. It requests one policy/value evaluation and stores validated
logits, softmax priors, scalar value, model version, and portable action
metadata. A node is terminal when no unmasked rewrite can continue.

## Selection

For each unmasked action `a` at node `s`, PUCT selects the maximum score:

```text
Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
```

Unvisited actions use the node value through the shared Q-completion rule.
Ties use deterministic action order. Selecting STOP backs up the node value and
does not call the engine. Selecting a candidate emits `Apply`; rejected actions
are masked and selection continues.

## Backup

PUCT is single-agent, so values are not negated between tree levels. Each
traversed edge increments its visit count and value sum, then recomputes Q.
Within-simulation context repetition terminates that descent without creating a
cycle. `no_backtrack` additionally masks children matching an earlier episode
root.

## Root Target And Action

After the configured fresh simulation count, the policy target is normalized
root visits over legal actions. Action selection uses visit counts with the
configured temperature during early moves and argmax afterward. Returned root
records preserve exact action order, selected index, portable action identity,
root value, search value, Q maximum, model version, and work statistics.

## Episode Completion

The episode task repeats root search until STOP or `max_steps`, then emits one
terminal `Measure` and returns only after the result is resumed. It records one
step per selected action and tracks every created graph/candidate for explicit
release by the caller.

## Tree Reuse

With reuse enabled, the selected candidate's reachable subtree is compacted and
promoted. Cached nodes and visit/Q ledgers are retained, while the next move
receives a full fresh simulation budget relative to the carried visit baseline.
STOP cannot promote a child. A missing or mismatched selected child falls back
to a fresh tree.

## Tests

PUCT tests cover deterministic scoring/election, STOP, rejection, no-backtrack,
task token invariants, final measurement, tree reuse, config hashing, and handle
release. Shared MCTS behavior is additionally covered by the Gumbel task and
release suites.
