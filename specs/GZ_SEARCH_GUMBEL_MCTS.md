# gz-search Serial Gumbel-MCTS Spec

Status: draft

Purpose: specify the first GraphZero Gumbel-MCTS implementation. This version
matches the WhittleZero search semantics where they matter, but it is strictly
serial: no async driver, no wave search, no in-flight leaves, and no virtual
visits.

The implementation should prove the tree math, STOP semantics, eval request
shape, and opponent-trajectory conditioning before adding orchestrator-driven
parallelism.

## Source Check

The math below is based on:

```text
Policy Improvement by Planning with Gumbel, Danihelka et al., ICLR 2022
https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf
DeepMind Mctx gumbel_muzero_policy
https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py
https://github.com/google-deepmind/mctx/blob/main/mctx/_src/action_selection.py
https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py
https://github.com/google-deepmind/mctx/blob/main/mctx/_src/seq_halving.py
../whittlezero/native/engine/mcts.cpp
../whittlezero/native/mcts_math.h
```

Confirmed points:

```text
Root sampling uses one Gumbel value per legal action.
Root considered actions are the top-m actions by gumbel + logits.
Root simulations use Sequential Halving over the considered set.
Root scores compare gumbel + logits + sigma(completed_q).
sigma(q) = (c_visit + max_child_visits) * c_scale * q.
Unvisited action Q is completed with a mixed value estimate.
The improved policy is softmax(logits + sigma(completed_q)).
Non-root action selection is argmax(pi_prime - N(a) / (1 + sum_b N(b))).
The final root action is selected from considered actions by visit count, with
root score as the tie-break.
```

WhittleZero uses already-scaled value outputs directly in the sigma transform.
Mctx's default q-transform additionally rescales values. GraphZero v1 should
follow WhittleZero: values are assumed to be on the evaluator's search scale,
initially `[-1, 1]`, and no min/max rescale is applied inside search.

## Scope

This spec includes:

```text
serial root Gumbel-MCTS
serial episode loop that runs root search once per selected transition
tree arena layout
root sequential halving
completed-Q policy target
STOP as a search action
opponent trajectory indexing
eval request context required for leaf evaluation
final measurement boundary
```

This spec excludes:

```text
async worker driver
wave MCTS
virtual visits
parallel leaf expansion
cross-worker eval batching
feature extraction
replay storage
neural/Python evaluator implementation
identical-tree PTP
pairwise utility target generation
```

## Role

Serial Gumbel-MCTS answers:

```text
Given an engine graph, ordered legal search actions, and a policy/value
evaluator, which search action should this worker take next?
```

It owns:

```text
worker-local tree arena
root Gumbel samples
sequential halving schedule
per-edge visit/value stats
policy target over root actions
selected action record
episode selected path
```

It does not own:

```text
opponent rollout generation
opponent feature encoding
eval batching across workers
runtime queues
model execution
replay insertion
engine measurement implementation
```

## Public API Draft

```rust
pub struct GumbelMcts {
    config: GumbelMctsConfig,
}

pub struct GumbelMctsConfig {
    pub max_steps: usize,
    pub simulations: NonZeroUsize,
    pub max_considered_actions: NonZeroUsize,
    pub seed: u64,
    pub gumbel_scale: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    pub temperature_moves: usize,
    pub tree_reuse: bool,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

pub struct GumbelRootResult<G, C> {
    pub root: G,
    pub selected_after: G,
    pub selected_after_context: ReplayGraphContext,
    pub selected_action: SearchAction<C>,
    pub selected_action_ref: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub selected_action_index: usize,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub considered_action_indices: Vec<usize>,
    pub policy_target: Vec<f32>,
    pub root_value: f32,
    pub root_search_value: f32,
    pub root_q_max: f32,
    pub model_version: ModelVersion,
    pub stats: GumbelRootStats,
}

pub struct GumbelRootStats {
    pub simulations: usize,
    pub expanded_nodes: usize,
    pub eval_count: usize,
    pub carried_nodes: usize,
    pub carried_root_visits: u32,
}

pub struct GumbelEpisode<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<GumbelStep<G, C>>,
    pub root_stats: Vec<GumbelRootStats>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: GumbelStopReason,
    pub search_config_hash: SearchConfigHash,
}
```

The first implementation may keep the concrete fields narrower if tests do not
need every statistic. Do not add replay schemas in `gz-search`.

Search context:

```rust
pub struct GumbelSearchContext {
    pub root_step: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
    pub selection_temperature: f32,
    pub opponent: Option<GumbelOpponentContext>,
}

pub struct GumbelEpisodeContext {
    pub opponent: Option<GumbelOpponentContext>,
}

pub struct GumbelOpponentContext {
    pub trajectory_id: u64,
    pub row_count: u32,
}
```

Opponent comparison is always same-index: learner step `t` compares against
opponent row `t`, clamped to the end of the opponent trajectory.

Config rules:

```text
max_steps may be zero for episode runs
simulations is NonZeroUsize
max_considered_actions is NonZeroUsize
gumbel_scale must be finite and non-negative
c_visit must be finite and non-negative
c_scale must be finite and non-negative
candidate_options are passed unchanged to engine.candidates()
measure_options are used only for final episode measurement
tree_reuse carries the selected child subtree to the next episode step
```

There is no user-facing max-depth budget. Like WhittleZero, each simulation
descends through already-expanded tree nodes until it reaches a new leaf, STOP,
an engine rejection/mask event, a no-legal-action state, or a path cycle guard.
Depth is still tracked for eval context and opponent-row alignment; it is not a
search budget.

Tree reuse:

```text
When tree_reuse is true, an episode carries the selected non-STOP child
subtree from step t into the root search for step t+1. The carried root is
not expanded or evaluated again. The new root task keeps the carried node
payloads and visit/value statistics, but uses the new GumbelSearchContext
for all future eval requests. Root Gumbel noise is regenerated from
(seed, root_step), so reuse remains deterministic.

Carried policy/value outputs may have been produced with the previous
position context or an older model version. This staleness is accepted
within one episode; there is no cross-episode reuse or staleness-triggered
re-evaluation.

With tree_reuse disabled, root scheduling keeps the original exact
eligibility rule: visits == target_visits. With tree_reuse enabled,
eligibility is visits <= target_visits, and schedule slots with no eligible
action are skipped without consuming a simulation. GumbelRootStats reports
carried_nodes and carried_root_visits for seeded roots; both are zero for
fresh roots.
```

Execution methods:

```rust
impl GumbelMcts {
    pub fn new(config: GumbelMctsConfig) -> Self;

    pub fn search_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: GumbelSearchContext,
    ) -> EngineResult<GumbelRootResult<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: Evaluator;

    pub fn run<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: Evaluator;
}
```

`search_root` does not measure. `run` measures the final graph before returning
an episode, because replay admission still requires `GraphEngine::measure`.

`search_root` and `run` accept `gz_eval::EngineEvaluator<E>`. Plain
`gz_eval::Evaluator` implementations work through the blanket adapter in
`gz-eval`; engine-specific evaluators implement the engine-aware trait in their
own eval crate.

```rust
pub trait EngineEvaluator<E: GraphEngine> {
    fn evaluate(
        &mut self,
        engine: &mut E,
        input: EngineEvalRequest<'_, E>,
    ) -> EngineResult<EvalOutput>;
}
```

`gz-search` owns no evaluator implementations. A measurement-backed Whittle
diagnostic evaluator belongs in `gz-eval-whittle`; future neural evaluators
belong in eval/domain crates and will be called through the orchestrator.

## Tree Layout

Use an arena of compact node indexes:

```rust
struct Node<G, C> {
    graph: G,
    context: ReplayGraphContext,
    candidates: Vec<C>,
    eval_actions: Vec<EvalAction>,
    action_refs: Vec<PortableSearchActionRef>,
    summaries: Vec<Option<SearchCandidateSummary>>,
    logits: Vec<f32>,
    priors: Vec<f32>,
    value: f32,
    model_version: ModelVersion,
    children: Vec<Option<NodeIndex>>,
    visits: Vec<u32>,
    value_sum: Vec<f32>,
    q: Vec<f32>,
}
```

Rules:

```text
STOP is implicit at action index candidates.len().
action_refs.len() == candidates.len() + 1.
logits.len() == priors.len() == visits.len() == q.len() == action_refs.len().
children has one entry per action; STOP children remain None.
Search stores E::Graph and E::Candidate handles only.
Search never stores graph bodies.
```

`priors` are `softmax(logits)` over the action list. Evaluator logits are
finite by contract. If softmax underflows to zero for every action, fall back to
uniform priors.

## Eval Requests

Expansion builds exactly one `EvalRequest` per newly expanded node:

```text
enumerate engine candidates
build candidate action refs from candidate_info
append STOP action ref
build EvalRequest with graph context, action list, and search position context
call evaluator
store logits and value on the node
```

The root node must be expanded and evaluated before any simulations run.

A leaf node is evaluated when a simulation reaches a graph/context not already
present in the tree. The serial v1 may call `evaluate_one`; using
`evaluate_batch` with one request is also fine. Do not add async or queues.

## Serial Algorithm

Root search:

```text
expand root
if root has no legal action, select STOP
sample one Gumbel value per root action
considered = top max_considered_actions by logits + gumbel
schedule = considered_visit_sequence(considered.len(), simulations)

for target_visit in schedule:
    eligible = considered actions with root.visits[action] == target_visit
    if eligible is empty, this is a bug
    forced_root_action = argmax root_score(action) over eligible
    path, leaf = descend from root with forced_root_action
    expand leaf if needed
    backup leaf value along path

policy_target = improved_policy(root)
selected_action = best considered action by visits, then root_score
if selection_temperature > 0, sample from visit counts instead
return root result
```

The implementation is serial. Each loop iteration fully descends, expands, and
backs up before the next iteration starts.

No virtual visits are needed in v1 because there are no concurrent in-flight
paths.

## Sequential Halving Schedule

Use the WhittleZero/Mctx schedule:

```text
if max_considered <= 1:
    sequence = 0, 1, 2, ... simulations - 1
else:
    log2max = ceil(log2(max_considered))
    visits = [0; max_considered]
    considered = max_considered
    while sequence.len < simulations:
        extra = max(1, simulations / (log2max * considered))
        repeat extra times:
            append visits[0..considered] to sequence
            increment visits[0..considered]
        considered = max(2, considered / 2)
    truncate sequence to simulations
```

The root scheduler uses the sequence value as the required current visit count.
At each target, only considered actions with exactly that visit count are
eligible.

## Q Math

For each node action `a`:

```text
N(a) = visits[a]
W(a) = value_sum[a]
Q(a) = W(a) / N(a), if N(a) > 0
```

For unvisited actions, `Q(a)` is undefined until completed.

Mixed value:

```text
if sum_a N(a) == 0:
    v_mix = node.value
else:
    visited_prior_mass = sum priors[a] where N(a) > 0
    if visited_prior_mass <= 0:
        v_mix = node.value
    else:
        visited_q = sum priors[a] * Q(a) where N(a) > 0
        visited_q = visited_q / visited_prior_mass
        v_mix = (node.value + sum_a N(a) * visited_q) / (1 + sum_a N(a))
```

Completed Q:

```text
completed_q(a) = Q(a) if N(a) > 0, else v_mix
```

Sigma:

```text
sigma_q(a) = (c_visit + max_b N(b)) * c_scale * completed_q(a)
```

Root score:

```text
root_score(a) = root_gumbel(a) + logits(a) + sigma_q(a)
```

Improved policy:

```text
pi_prime = softmax(logits + sigma_q)
```

Non-root action:

```text
select argmax_a pi_prime(a) - N(a) / (1 + sum_b N(b))
```

Final root action:

```text
select considered action with maximum N(a)
break ties by root_score(a)
if selection_temperature > 0:
    sample action with probability proportional to N(a)^(1 / selection_temperature)
    fall back to the best-count action if all counts are zero
```

Search value:

```text
root_search_value = visit-weighted mean Q over visited root actions
root_q_max = max Q over root actions, or root.value if no action was visited
```

## Descent And Backup

Descent:

```text
start at root
first action is forced_root_action
after the root, use non-root action selection
Candidate action:
    apply candidate
    create/find child node for ApplyResult.after
STOP action:
    terminate the path at the current graph
```

If `GraphEngine::apply` rejects a candidate during search, mark that edge
illegal for this root search and restart selection from the same node. A
rejected candidate must not receive a visit.

Backup:

```text
leaf_value = expanded leaf node value, or current node value for terminal paths
for each edge on the path:
    visits[action] += 1
    value_sum[action] += leaf_value
    q[action] = value_sum[action] / visits[action]
```

Reward-backup variants are deferred. The v1 backup is pure leaf value backup.

## STOP

STOP is a normal search action and must be scored by the evaluator.

Rules:

```text
STOP is appended after engine candidates.
STOP receives a policy logit from eval.
STOP can be sampled into the root considered set.
STOP can be selected at root or non-root nodes.
STOP never calls GraphEngine::apply.
STOP terminates the selected episode at the current graph.
For selected STOP, after == before in the episode step.
```

If STOP is chosen inside a simulation, the backup value is the current graph's
value under the appropriate eval position context. With no opponent trajectory,
that is the current node value. With an opponent trajectory, see terminal STOP
alignment below.

## Opponent Trajectory Context

GraphZero should copy the WhittleZero idea, not the concrete Whittle storage.

The opponent trajectory is eval-side context:

```text
opponent rollout generation belongs outside GumbelMcts
opponent feature encoding belongs outside GumbelMcts
GumbelMcts stores only a small trajectory id/count in request context
tree nodes do not store opponent graph handles
tree nodes do not store opponent feature rows
```

Eval context:

```rust
pub struct EvalPositionContext {
    pub root_step: u32,
    pub leaf_depth: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
    pub opponent: Option<EvalOpponentContext>,
}

pub struct EvalOpponentContext {
    pub trajectory_id: u64,
    pub row_count: u32,
}
```

Indexing rule:

```text
opponent_row = min(root_step + leaf_depth, row_count - 1)
```

`root_step` is the learner episode step.

Terminal STOP alignment:

```text
if STOP is selected in a simulation and opponent row_count > 0:
    effective_leaf_depth = max(actual_depth, row_count - 1 - root_step)
else:
    effective_leaf_depth = actual_depth
```

This evaluates STOP against the final opponent row, matching the WhittleZero
behavior where a learner that stops is compared against the opponent completing
its trajectory.

Efficiency plan:

```text
1. Opponent rollout creates graph states once per episode.
2. Feature/eval layer encodes those states once into a contiguous trajectory
   table.
3. The evaluator registers or receives that table and returns a trajectory id
   plus row_count.
4. Each EvalRequest carries root_step and leaf_depth, not a copied opponent row.
5. Serial evaluators resolve the row directly.
6. Future orchestrator batchers gather rows by trajectory id and row index while
   collating batches.
7. Eval cache keys include trajectory id, resolved opponent row index, budget
   context, graph context, action refs, and model version.
```

This keeps search tree memory proportional to visited learner nodes, not to
`visited_nodes * opponent_feature_dim`.

## Episode Loop

`GumbelMcts::run` repeats root search:

```text
current = root
for step in 0..max_steps:
    root_context.root_step = step
    root_context.selection_temperature = 1.0 if step < temperature_moves else 0.0
    result = search_root(current)
    record result as a GumbelStep
    if result.selected_action is STOP:
        stop with SelectedStop
    current = result.selected_after
measure current
return GumbelEpisode
```

The selected candidate is not applied again in the episode loop. Root search has
already applied every visited candidate edge and returns `selected_after`.

## Output Rules

Root output must include:

```text
selected action ref
ordered policy target over actions
considered action indices
root value
root search value
root q max
candidate count
action count including STOP
search config hash
model version from root eval
```

`stats.simulations` is the number of simulations actually completed. It can be
less than the requested count only if all currently selectable root actions are
masked by engine rejection.

Policy target:

```text
policy_target = improved_policy(root)
Gumbel noise does not enter policy_target
unconsidered actions may receive nonzero target mass through completed Q
```

This is deliberate. The Gumbel noise affects exploration and the selected
action, not the completed-Q policy target.

## SearchConfigHash

Add a Gumbel hash:

```text
hash("gz-search-gumbel-mcts-v1",
     max_steps,
     simulations,
     max_considered_actions,
     seed,
     gumbel_scale bits,
     c_visit bits,
     c_scale bits,
     temperature_moves,
     candidate_options.max_candidates,
     candidate_options.deterministic_order,
     measure_options.config_hash,
     measure_options.samples,
     measure_options.timeout_ms,
     measure_options.deterministic)
```

The hash must not include:

```text
root graph hash
engine-local handles
opponent trajectory id
model version
```

Those are recorded separately in graph/eval/replay context.

## Validation Tests

Required math tests:

```text
considered_visit_sequence matches WhittleZero/Mctx examples
Gumbel scale 0 with uniform logits gives deterministic considered order
mixed_value uses raw node value before any action is visited
completed_q fills unvisited actions with mixed value
improved_policy ignores root Gumbel noise
non-root selector follows pi_prime - N/(1+sumN)
final action selects max visits and then root score
```

Required search tests:

```text
root request appends STOP
STOP can be selected at root
STOP is never passed to apply
serial simulations produce the same result independent of Vec reserve capacity
same seed/config/root/evaluator gives identical selected action and policy
different seed can change root Gumbel considered set
zero simulations is rejected by config validation
max_considered_actions is clamped to legal action count
selected candidate is applied during root search and not applied again by the
episode loop
final graph is measured before episode return
```

Required opponent-context tests:

```text
without opponent context, leaf_depth is actual tree depth
with opponent context, opponent row is root_step + leaf_depth clamped to last
root_step is the learner episode step
STOP uses terminal STOP alignment to resolve the last opponent row
EvalRequest does not copy opponent feature rows
```

## Implementation Plan

1. Add math helpers in `gz-search::gumbel`:
   `considered_visit_sequence`, softmax, mixed value, completed Q,
   improved policy, non-root selector, root score, Gumbel sampler.
2. Add config validation and `SearchConfigHash` support.
3. Add a compact tree arena for one root search.
4. Implement node expansion into `EvalRequest` with STOP last.
5. Implement serial root search with one leaf expansion per simulation.
6. Implement episode loop around repeated root searches.
7. Add opponent position fields to eval request records before enabling
   trajectory-conditioned search.
8. Add focused math and search tests using the existing search test engine.
9. Run `cargo fmt`, `cargo test -p gz-search`, `cargo test --all`, and clippy.

## Deferred

```text
wave MCTS
virtual visits
poll/resume task refactor (protocol owned by GZ_ORCHESTRATOR.md)
orchestrator eval queues
feature-backed evaluator
reward backup
root-relative value transform
EMA value centering
Gumbel noise overlap scaling
no-backtrack masking
hindsight policy targets
pairwise value item generation
identical-tree PTP
```
