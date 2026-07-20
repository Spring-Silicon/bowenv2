# Search

## Boundary

`gz-search` implements search over `GraphEngine` without depending on Whittle,
replay, orchestration, Python, or a concrete evaluator service. Search stores
only engine graph/candidate handles and portable identities. Candidate order and
semantics remain engine-owned.

The retained algorithms are:

- `GreedySearch`;
- `BeamSearch`;
- `GumbelMcts`;
- `PuctMcts`.

## Actions

Every node enumerates engine candidates in deterministic engine order and then
appends one search-owned STOP action. STOP is never passed to
`GraphEngine::apply`. Portable action references bind candidates and STOP to the
node's `ReplayGraphContext`.

Evaluator policy logits, legal-action lists, policy targets, and selected-action
indexes use this exact order. Search validates evaluator output size and finite
values before using it.

## Async Work Protocol

MCTS tasks expose a poll/resume state machine:

```text
Expand  -> enumerate candidates and portable metadata
Eval    -> evaluate policy/value for one expanded node
Apply   -> apply one engine candidate
Measure -> measure the completed terminal graph
Done    -> return the episode plus all owned handles
```

Every work item carries a unique token. Resume rejects stale, duplicate, or
wrong-kind results. A task reports `Blocked` only when outstanding work exists.
Abort/error paths return every created graph and candidate through
`take_all_handles`; normal progress exposes no-longer-needed handles through
`take_releasable`.

## Shared MCTS Core

Gumbel and PUCT share private node storage and generic root/episode task
machinery in `mcts/`. The shared core owns:

- one eval per expanded node;
- child handle ownership and subtree compaction;
- visit, value-sum, Q, prior, and mask ledgers;
- no-backtrack and within-simulation cycle handling;
- STOP handling and terminal measurement;
- tree-reuse promotion and final handle partitioning.

Algorithm strategy modules own root scoring, visit allocation, policy targets,
and final action election. `mcts` is private and is not a general game library.

## Gumbel MCTS

Single-agent Gumbel MCTS uses sequential halving over a considered root set.
Symmetric selfplay uses a dedicated alternating-player root task, negating
backups when perspective changes and attaching the other player's graph as the
second board. See `GZ_SEARCH_GUMBEL_MCTS.md` and `GZ_SYMMETRIC_SELFPLAY.md`.

## PUCT MCTS

PUCT uses the shared single-agent task with the PUCT action score and
visit-derived policy target. See `GZ_SEARCH_PUCT_MCTS.md`.

## Greedy And Beam

Greedy and beam are synchronous measurement-based kernels. Greedy picks the
best measured successor or STOP at each step. Beam retains graph-distinct top
states per layer. Both return measured episodes and release all speculative
handles according to the engine contract.

## Tree Reuse

When enabled, MCTS promotes the selected child's reachable subtree. The promoted
root keeps cached expansions/evals and carried visit/Q ledgers; each move still
receives the configured number of fresh simulations measured relative to the
carried root baseline. Reuse falls back to a fresh tree when the promoted state
does not exactly match the next root state.

## Hashing

Search config hashes include every option that changes action generation,
selection, backup, or measurement semantics. Position-feature export is
excluded intentionally because it changes model inputs but not the search
algorithm's internal budget. Existing hash namespace byte strings are retained
when public names are cleaned up.

## Correctness Requirements

- deterministic candidate enumeration and apply for a fixed engine state;
- STOP last at every evaluator/search boundary;
- no action applied under a context other than the one that enumerated it;
- one terminal `GraphEngine::measure` before an episode is complete;
- no graph/candidate handle released while a live node or returned episode
  references it;
- all created handles released exactly once on completion, abort, and error;
- selected index, portable action, legal actions, and policy target remain
  aligned end-to-end.
