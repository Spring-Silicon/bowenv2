# Gumbel MCTS

## Scope

`GumbelMcts` provides single-agent Gumbel search and the production symmetric
selfplay search. Both use deterministic engine candidate order, search-owned
STOP, validated policy/value evaluations, sequential-halving root allocation,
and explicit handle ownership.

## Configuration

`GumbelMctsConfig` includes:

- episode `max_steps` and fresh simulations per move;
- maximum root actions considered;
- deterministic base/noise seeds;
- Gumbel scale and optional overlap calibration;
- completed-Q constants `c_visit` and `c_scale`;
- action-selection temperature schedule;
- tree reuse, STOP masking, no-backtrack, and position export;
- candidate and measure options;
- `SingleAgent` or `SymmetricSelfplay` value semantics.

Semantic options are included in the search hash. Episode IDs contribute a
separate mixed noise seed so concurrent games from the same root remain
deterministic but explore differently.

## Root Expansion

The engine enumerates candidates once. Search appends STOP and requests one
policy/value evaluation for the node. The evaluator output must contain one
finite logit per action and one finite scalar value. Softmax logits become node
priors.

`mask_stop` masks STOP whenever a rewrite exists but leaves it available at a
STOP-only node. `no_backtrack` masks candidates whose applied context matches
the current or an earlier episode root. If all rewrites are masked, STOP is
unmasked so selection always has a legal action.

## Root Schedule

At a fresh root, deterministic Gumbel samples perturb policy logits. The top
`min(max_considered, action_count)` actions enter sequential halving. Each round
allocates visits to surviving actions and eliminates by the Gumbel completed-Q
score until the configured simulation budget is exhausted.

When `gumbel_noise_overlap >= 0`, a deterministic bisection chooses a per-root
noise scale whose noisy argmax has the configured overlap with the prior's
top-considered set. Negative disables calibration and uses `gumbel_scale`
directly.

The policy target is normalized root visits in full legal-action order.
Training-move action selection samples that target at configured temperature;
later moves use visit argmax.

## Descent And Backup

Selection emits `Apply` for an unexpanded candidate edge. Rejected candidates
are masked and selection resumes. Accepted children are expanded/evaluated at
most once. A repeated context within one simulation terminates descent rather
than constructing a cycle.

Single-agent search backs the same scalar value through every edge. Symmetric
search alternates players and negates the backed value whenever node perspective
changes. Terminal symmetric values are exact win/draw/loss values after both
final graphs are measured.

## Symmetric Board

The symmetric task stores both player graphs, rewrite counts, active/stopped
state, and player to move. An eval contains the acting graph/actions plus
`EvalOpponentWork` for the other graph. The orchestrator extracts both feature
graphs into one joint-board row; there is no trajectory ID or reference policy.

STOP retires only the acting player. A player with no candidate takes an
untrained forced pass. The game ends when both players are stopped, blocked, or
at horizon. Each final graph receives one `GraphEngine::measure`; reward wins,
then fewer rewrites breaks equal rewards when enabled.

## Tree Reuse

On a candidate move, search may compact and promote the selected branch. The
complete state must match the next live root: both graphs, player to move,
rewrite counts, and active/stopped status. A forced-pass ambiguity or mismatch
falls back to a fresh tree.

Promoted nodes keep cached expansions, evaluator outputs, visits, value sums,
and Q values. Every move still runs the configured number of new simulations,
counted relative to the carried root visit baseline. Policy targets and action
selection use carried plus fresh statistics.

## Async Task Contract

Root and episode tasks expose only `Expand`, `Eval`, `Apply`, and `Measure`
work. Each outstanding token is resumed exactly once. Pending work survives a
poll boundary; abort/drop returns all speculative handles. Final episode results
include created handles for caller release, selected action records, root stats,
terminal measurements, and the search hash.

## Required Tests

- deterministic root schedules and config hashes;
- policy/action/index alignment;
- STOP-only, STOP-masked, rejection, cycle, and no-backtrack cases;
- token mismatch, duplicate resume, and poll-after-done errors;
- symmetric sign backup, passes, STOP, ties, and paired measurements;
- tree-reuse decision/statistics parity and state-match fallback;
- exact graph/candidate release on success, abort, and errors.
