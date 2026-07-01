# Serial Orchestrator Implementation Spec

Status: implementation work order

Purpose: implement the serial slice of `GZ_ORCHESTRATOR.md`: the poll/resume
work protocol in `gz-search`, the Gumbel task state machines, the
compatibility wrappers, and the `gz-orchestrator` crate with the serial
driver.

Authority: `GZ_ORCHESTRATOR.md` owns the design contract. This document is
the ordered work plan. If they disagree, `GZ_ORCHESTRATOR.md` wins; report
the conflict instead of improvising.

Read before starting:

```text
specs/GZ_ORCHESTRATOR.md        (design contract being implemented)
specs/GZ_SEARCH_GUMBEL_MCTS.md  (tree math that must not change)
specs/GZ_SEARCH.md              (crate rules for gz-search)
crates/gz-search/src/gumbel.rs  (the kernel being refactored)
crates/gz-search/src/support.rs (existing helpers)
crates/gz-eval/src/types.rs     (EvalRequest/EvalOutput/EngineEvaluator)
```

## Hard Constraints

```text
Work in the stage order below. Each stage must compile and pass
`cargo test --all` before the next stage starts.
Stage 0 goldens are captured BEFORE any kernel change and must pass
unchanged at every later stage. Bit-identical f32 output is required; the
refactor reorganizes control flow, it must not reorder arithmetic or RNG
consumption.
No new dependencies anywhere. No tokio, no async, no threads, no serde.
Do not modify gz-engine, gz-eval, gz-eval-whittle, or gz-engine-whittle.
Do not modify greedy, beam, or random search.
Public signatures of GumbelMcts::new/config/search_config_hash/
search_root/run/run_from_root must not change. Adding fields to
GumbelRootResult is allowed where this spec says so.
All new code: #![forbid(unsafe_code)] crates, no thread_local, no global
mutable state, no wall-clock or OS RNG.
Every stage ends with: cargo fmt, cargo test --all,
cargo clippy --all-targets --all-features -- -D warnings.
```

## Current State

```text
workspace members: gz-engine, gz-engine-whittle, gz-search, gz-eval,
gz-eval-whittle. gz-orchestrator does not exist yet.

gz-search/src/gumbel.rs: run-to-completion kernel.
  GumbelMcts::search_root(engine, evaluator, root, context) expands the
  root (Tree::expand: engine.candidates + candidate_info per candidate +
  one evaluator.evaluate call), samples root gumbels, runs the sequential
  halving schedule via Tree::select_leaf (engine.apply during descent,
  Tree::expand at new leaves, Tree::stop_value for STOP backup), then
  selects and returns GumbelRootResult.
  GumbelMcts::run loops search_root per step and calls engine.measure on
  the final graph.
  Tree::stop_value issues a SECOND eval of an already-expanded node when
  opponent alignment moves the effective leaf depth. This is a real
  suspension point; do not miss it.
  GumbelRng consumption order: root gumbels first (sample_root_gumbels),
  then possibly sample_count_action at selection. Nothing else consumes
  RNG. Preserve this order exactly.

gz-search/src/support.rs: candidate_info (validating), graph_context,
graph_context_from_hash, step_ref, internal(message) -> EngineError.

gz-search/tests/: common/mod.rs is a scripted test engine supporting
rejected applies (.rejected(graph, candidate)); tests/gumbel.rs has a
scripted evaluator builder (.row(graph, logits, value)) and covers STOP,
rejection masking, opponent alignment, temperature, and seeds.

gz-eval: Evaluator, EngineEvaluator<E>, EngineEvalRequest { graph,
candidates, request, measure_options }, EvalRequest::with_position,
EvalOutput::validate_for, eval_error_to_engine_error.
gz-eval-whittle: WhittleMeasureEvaluator (unit struct, EngineEvaluator
for WhittleEngine).
```

## Stage 0: Golden Fixtures

Add `crates/gz-search/tests/gumbel_goldens.rs` (with `mod common;`) before
touching any kernel code.

Implement a deterministic fingerprint of episode output:

```text
fingerprint = blake3 over a canonical byte encoding, rendered as hex
encoding rules: little-endian fixed-width integers, u64 length prefixes for
sequences, f32 encoded via to_bits, hashes/ids via as_bytes, enums as a u8
discriminant
```

Fields to encode, in order, for a `GumbelEpisode`:

```text
search_config_hash
stop_reason
root_context, final_context
final_measure.graph_hash, final_measure.measured, final_measure.valid,
final_measure.scalar_reward (Option tag + bits)
steps.len()
per step: step_ref (before/action/after), selected_rank,
engine_candidate_count, action_count, policy_target,
considered_action_indices, root_value, root_search_value, root_q_max,
model_version
```

For a `GumbelRootResult` (used by root-level goldens): encode
selected_action_ref, selected_action_index, engine_candidate_count,
action_count, considered_action_indices, policy_target, root_value,
root_search_value, root_q_max, model_version, stats.simulations,
stats.expanded_nodes, stats.eval_count.

Golden cases (reuse the scripted engine/evaluator patterns from
`tests/gumbel.rs`):

```text
G1 multi-step episode, no opponent, temperature 0, gumbel_scale > 0
G2 temperature_moves > 0 (exercises sample_count_action RNG path)
G3 opponent context that forces the terminal STOP re-eval
   (row_count large enough that effective_depth > depth)
G4 a rejected candidate that gets masked during simulation
G5 max_steps = 0 (measure-only episode)
G6 single search_root call fingerprinted as GumbelRootResult
```

Capture procedure:

```text
write each test asserting fingerprint == "TODO"
run once; the assertion failure message must print the actual hex
paste actuals into the constants; re-run to green
commit this stage before starting stage 1
```

These tests are the equivalence oracle. They must never be edited again
during this work order except to add cases.

## Stage 1: Work Protocol Types

New module `crates/gz-search/src/work.rs`, re-exported from `lib.rs`.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WorkToken(u64);

impl WorkToken {
    #[must_use] pub const fn value(self) -> u64;
}

#[derive(Debug)]
pub enum SearchPoll<G, C, R> {
    Work(SearchWork<G, C>),
    Blocked,
    Done(R),
}

#[derive(Debug)]
#[non_exhaustive]
pub enum SearchWork<G, C> {
    Expand(ExpandWork<G>),
    Apply(ApplyWork<G, C>),
    Measure(MeasureWork<G>),
    Eval(EvalWork<G, C>),
}

#[derive(Clone, Copy, Debug)]
pub struct ExpandWork<G> {
    pub token: WorkToken,
    pub graph: G,
    pub options: CandidateOptions,
}

#[derive(Clone, Copy, Debug)]
pub struct ApplyWork<G, C> {
    pub token: WorkToken,
    pub graph: G,
    pub candidate: C,
}

#[derive(Clone, Copy, Debug)]
pub struct MeasureWork<G> {
    pub token: WorkToken,
    pub graph: G,
    pub options: MeasureOptions,
}

#[derive(Clone, Debug)]
pub struct EvalWork<G, C> {
    pub token: WorkToken,
    pub graph: G,
    pub candidates: Vec<C>,
    pub request: EvalRequest,
    pub measure_options: MeasureOptions,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum SearchWorkResult<G, C> {
    Expand(ExpandResult<C>),
    Apply(ApplyResult<G, C>),
    Measure(MeasureResult<G>),
    Eval(EvalOutput),
}

#[derive(Clone, Debug)]
pub struct ExpandResult<C> {
    pub graph_hash: GraphHash,
    pub candidates: Vec<ExpandedCandidate<C>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ExpandedCandidate<C> {
    pub candidate: C,
    pub candidate_hash: CandidateHash,
    pub kind: CandidateKindId,
    pub tags: CandidateTags,
    pub static_prior: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EngineIdentity {
    pub engine_id: EngineId,
    pub engine_version: EngineVersion,
    pub action_set_hash: ActionSetHash,
}

impl EngineIdentity {
    #[must_use]
    pub fn from_engine<E: GraphEngine>(engine: &E) -> Self;

    #[must_use]
    pub fn context(&self, graph_hash: GraphHash) -> ReplayGraphContext;
}
```

Rules:

```text
No new error type. Protocol violations are EngineError::Internal via
support::internal with these exact stable messages:
  "unknown work token"
  "mismatched work result"
  "resume without pending work"
  "poll after done"
EngineIdentity::context composes PortableGraphId + action_set_hash exactly
like support::graph_context_from_hash does today.
```

Unit tests: `EngineIdentity::from_engine`/`context` against the shared test
engine matches `support::graph_context` output for the same graph.

## Stage 2: GumbelRootTask

Refactor the root search in `gumbel.rs` into a task. If the file grows past
~1500 lines, split the task machinery into `src/gumbel_task.rs`; keep
modules flat otherwise.

Public API:

```rust
pub struct GumbelRootTask<G, C> { /* private */ }

impl<G: Copy, C: Copy> GumbelRootTask<G, C> {
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelSearchContext,
    ) -> Self;

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelRootResult<G, C>>>;

    pub fn resume(
        &mut self,
        token: WorkToken,
        result: SearchWorkResult<G, C>,
    ) -> EngineResult<()>;

    #[must_use]
    pub fn root_context(&self) -> Option<ReplayGraphContext>;
}
```

Add `pub root_context: ReplayGraphContext` to `GumbelRootResult`. Do not
add it to the stage-0 fingerprints.

Token bookkeeping (serial width 1):

```text
monotonically increasing u64 counter per task, starting at 0
at most one pending token; pending state records token + expected result
kind + the retained EvalRequest when the pending work is Eval
poll with a pending token returns Blocked
resume must match pending token and kind, else the stage-1 errors
Eval results are validated with EvalOutput::validate_for against the
retained request, mapped through eval_error_to_engine_error
after Done, poll returns the "poll after done" error
```

Two-phase node creation (replaces `Tree::expand`):

```text
phase A: emit ExpandWork { graph, options: config.candidate_options }
phase B: on ExpandResult, build (task-side, from ExpandedCandidate rows):
  PortableCandidateRef::new(node_context, candidate_hash)
  eval_actions (EvalAction::candidate rows, then EvalAction::stop(context))
  action_refs (candidate refs, then stop ref)
  summaries (SearchCandidateSummary rows, then None)
  EvalRequest::with_position(context, actions, position(depth))
  then emit EvalWork { graph, candidates, request, measure_options }
phase C: on EvalOutput, finalize the Node exactly as Tree::expand does
  today (softmax priors, zeroed visit/q arrays), increment eval_count
```

Node context rules:

```text
root node context = identity.context(ExpandResult.graph_hash)
child node context = identity.context(ApplyResult.after_hash), computed
when the Apply result arrives (mirrors graph_context_from_hash today)
if a child ExpandResult.graph_hash disagrees with the context already
derived from after_hash, return EngineError::Internal (engine contract
violation), do not silently prefer either
```

Descent state machine (replaces `Tree::select_leaf`). Persisted per
in-flight simulation: current node index, depth, path (Vec<Edge>), seen
context set, forced root action Option, plus the suspension stage:

```text
select action (runnable):
  forced root action on the first step, select_nonroot after
  STOP action -> push edge; if stop_value needs no re-eval (no opponent,
    or effective_depth == depth) back up node.value immediately and start
    the next schedule entry; otherwise emit EvalWork built exactly like
    stop_value today (cloned eval_actions, position(effective_depth)) and
    suspend
  candidate action -> emit ApplyWork and suspend

on ApplyResult:
  rejected -> mask_action (logit = -inf, prior = 0), re-enter select action
    on the same node in the same poll; no visit, no token wasted on retry
  accepted, child exists -> cycle guard via seen set (repeat context backs
    up child value immediately); else descend and re-enter select action
  accepted, no child -> emit ExpandWork for the child and suspend

on child ExpandResult -> build request, emit EvalWork, suspend
on child EvalOutput  -> finalize node, link child, back up leaf value,
  start next schedule entry
on STOP EvalOutput   -> increment eval_count, back up output.value, start
  next schedule entry
```

Everything else is unchanged and must not be touched: gumbel sampling,
considered_actions, considered_visit_sequence, root_scores, improved
policy, backup arithmetic, mixed/completed Q, selection (including
temperature sampling), stats fields, RNG implementation and consumption
order.

Selection runs when the schedule is exhausted (or no eligible action
remains) and produces `Done(GumbelRootResult)` with `root_context` filled.
No engine or eval work is needed for selection.

## Stage 3: search_root Wrapper

Reimplement `GumbelMcts::search_root` as an inline driver over
`GumbelRootTask`, preserving its exact signature:

```text
create task with EngineIdentity::from_engine(engine)
loop on poll:
  Work(Expand w)  -> service inline (below), resume
  Work(Apply w)   -> engine.apply(w.graph, w.candidate), resume
  Work(Measure w) -> engine.measure(w.graph, w.options), resume
  Work(Eval w)    -> evaluator.evaluate(engine, EngineEvalRequest {
                       graph: w.graph, candidates: &w.candidates,
                       request: &w.request,
                       measure_options: w.measure_options }), resume
  Blocked         -> EngineError::Internal "serial driver blocked"
  Done(result)    -> return result
```

Inline Expand servicing (shared `pub(crate)` helper in gz-search so stage
4's wrapper reuses it):

```text
engine.candidates(graph, options, &mut buf)
graph_hash = engine.hash(graph)
per candidate: support::candidate_info (validating), project to
ExpandedCandidate
return ExpandResult
```

Engine call ordering within servicing must match the old kernel
(candidates, then per-candidate candidate_info; hash where the old code
hashed). The scripted test engine is order-sensitive.

Acceptance for this stage: every existing gz-search test and every stage-0
golden passes unchanged.

## Stage 4: GumbelEpisodeTask And run Wrapper

```rust
pub struct GumbelEpisodeTask<G, C> { /* private */ }

impl<G: Copy, C: Copy> GumbelEpisodeTask<G, C> {
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelEpisodeContext,
    ) -> Self;

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelEpisode<G, C>>>;

    pub fn resume(
        &mut self,
        token: WorkToken,
        result: SearchWorkResult<G, C>,
    ) -> EngineResult<()>;
}
```

Behavior, mirroring the current `run` exactly:

```text
per step: build GumbelSearchContext (root_step, budget_fraction,
budget_step, selection_temperature, opponent) with the same formulas as
today; run an inner GumbelRootTask by forwarding its Work items and
resumes (tokens seen by the driver are the episode task's own; keep an
internal mapping or share the counter — driver must never see duplicate
live tokens)
record GumbelStep from each GumbelRootResult via support::step_ref
episode root_context comes from the first root task's root_context()
on STOP or after max_steps: emit MeasureWork { final graph,
config.measure_options }, then Done(GumbelEpisode)
max_steps == 0: no root task runs; emit Measure immediately; derive
root/final context as identity.context(MeasureResult.graph_hash)
```

Reimplement `GumbelMcts::run` as an inline driver over `GumbelEpisodeTask`
(same drive loop as stage 3). `run_from_root` stays a one-liner.

Acceptance: all existing tests and all stage-0 goldens pass unchanged.

New task-level tests in gz-search (drive tasks by hand, no wrapper):

```text
root task first emits Expand for the root, then Eval for the root
resume with unknown token is rejected
resume with mismatched result variant is rejected
resume of an eval with wrong policy_logits length is rejected
double resume of a token is rejected
poll after Done is rejected
poll while pending returns Blocked
dropping a task with an outstanding token is safe (no panic on drop)
rejected ApplyResult masks the action and the next poll emits another
  Apply (or handles STOP) without consuming a resume
opponent alignment case emits the terminal STOP Eval with the adjusted
  position context
episode task emits Measure exactly once, after the final root search
episode task completes through STOP and through max steps
hand-driven task output fingerprint equals the stage-0 golden for the
  same config (G1 at minimum)
```

## Stage 5: gz-orchestrator Crate

Add `crates/gz-orchestrator` and register it in the workspace `members`.

```text
crates/gz-orchestrator/
  Cargo.toml        deps: gz-engine, gz-eval, gz-search (path deps,
                    workspace version/edition like the other crates)
                    dev-deps: gz-engine-whittle, gz-eval-whittle
  src/
    lib.rs          #![forbid(unsafe_code)], re-exports
    ids.rs          WorkerId(u64), EpisodeId(u64): Clone Copy Debug Eq
                    PartialEq Hash, const new(u64), value()
    serial.rs       SerialGumbelOrchestrator
```

```rust
pub struct SerialGumbelOrchestrator<E, V> {
    worker_id: WorkerId,
    next_episode_id: u64,
    engine: E,
    evaluator: V,
    search: GumbelMcts,
}

pub struct SerialEpisode<G, C> {
    pub worker_id: WorkerId,
    pub episode_id: EpisodeId,
    pub episode: GumbelEpisode<G, C>,
}

impl<E, V> SerialGumbelOrchestrator<E, V>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
{
    pub fn new(worker_id: WorkerId, engine: E, evaluator: V, search: GumbelMcts) -> Self;

    pub fn run_from_root(
        &mut self,
        context: GumbelEpisodeContext,
    ) -> EngineResult<SerialEpisode<E::Graph, E::Candidate>>;

    pub fn run(
        &mut self,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<SerialEpisode<E::Graph, E::Candidate>>;
}
```

The serial driver in `serial.rs` writes its own explicit poll/service/
resume match (the stage-3 loop shape). It constructs `GumbelEpisodeTask`
directly; it does not call `GumbelMcts::run`. Expand servicing here mirrors
the stage-3 helper (duplicating ~15 lines is acceptable; do not make the
gz-search helper public for this).

Episode ids increment per completed episode, starting at 0. Errors do not
consume an id.

## Stage 6: Orchestrator Tests

`crates/gz-orchestrator/tests/serial.rs`:

```text
drives an episode to completion with a small local scripted engine and a
  gz-eval RandomValueEvaluator (through the blanket EngineEvaluator
  adapter), or with dev-dep test helpers if a local engine is too heavy —
  prefer the smallest engine that works
episode ids increment across runs
result equals GumbelMcts::run on a fresh identical engine/evaluator
  (field-by-field or via a fingerprint helper duplicated in the test)
```

`crates/gz-orchestrator/tests/whittle.rs` (integration):

```text
WhittleEngine + WhittleMeasureEvaluator, constructed the same way existing
  gz-eval-whittle tests construct them
run_from_root completes; stop reason is SelectedStop or MaxSteps
episode.final_measure.measured && valid
episode matches GumbelMcts::run with WhittleMeasureEvaluator on a fresh
  identical engine (determinism across the two drivers)
```

## Final Verification

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
```

Acceptance checklist:

```text
stage-0 goldens unchanged and passing
all pre-existing tests passing without modification
gz-search has no new dependencies; still no async, no adapter deps
gz-orchestrator default build depends only on gz-engine/gz-eval/gz-search
GumbelMcts public entry points behave identically (goldens prove it)
protocol errors use the exact stable messages from stage 1
no code path in tasks reads an engine or evaluator directly
```

## Out Of Scope

```text
wave math, virtual visits, multiple outstanding tokens in the Gumbel task
async runtime, threads, channels, queues, batching
queued or process engine lanes
replay integration, ratio control, metrics, shutdown
trajectory table leases (serial evaluators resolve opponent rows directly)
changes to greedy/beam/random search
```
