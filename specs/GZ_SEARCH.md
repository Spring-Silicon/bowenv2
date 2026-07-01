# gz-search Spec

Status: draft

Purpose: define the search crate for GraphZero. The first implementations are
deterministic measured greedy rollout, deterministic measured beam search,
random rollout, and serial Gumbel-MCTS. The crate must stay shaped for its main
future use: many parallel Gumbel-MCTS selfplay workers driven by an async
orchestrator.

`gz-search` exists to prove that search can drive an engine without knowing the
engine's graph body, candidate structure, or domain semantics, while keeping
search state cheap enough to run many workers in one process.

## Role

`gz-search` answers:

```text
Given a GraphEngine, how does a worker choose and record graph transitions?
```

It owns:

```text
worker-local search state
algorithm-neutral episode and step records
greedy rollout configuration
greedy rollout execution
beam search configuration
beam search execution
random rollout execution
serial Gumbel-MCTS execution
search stop reasons
search config hashing
private test engines for search tests
```

It does not own:

```text
concrete engine adapters
Whittle-specific logic
feature extraction
neural evaluation
replay storage
async scheduling
bounded queues
actor ids
episode ids
Python or torch integration
```

## Dependency Contract

Allowed:

```text
std
gz-engine
gz-eval for policy/value search algorithms
blake3 if gz-search computes SearchConfigHash
```

Allowed behind explicit features:

```text
serde support for search config and output records
```

Forbidden:

```text
tokio or any async runtime
async-trait
rocksdb
torch/Python bindings
gz-engine-whittle
future concrete engine adapters
gz-features
gz-replay
gz-orchestrator
training code
```

The crate must be synchronous. Async execution, queueing, timeouts,
backpressure, and actor supervision belong to `gz-orchestrator`.

`gz-search` may depend on the default, dependency-light `gz-eval` policy/value
traits and records. It must not depend on Python-backed evaluators, torch,
replay storage, or async eval services.

## Worker Architecture

The long-term target is many concurrent selfplay workers, with Gumbel-MCTS as
the main search algorithm. `gz-search` must therefore provide search kernels
and records that are easy for an async process to drive, but it must not own
the async process.

Boundary:

```text
gz-search:
  owns deterministic search state machines and worker-local scratch
  stores E::Graph and E::Candidate handles
  emits portable transition records and final measured episode data

gz-orchestrator:
  owns async tasks
  owns bounded queues
  batches engine/eval/measure work across workers
  owns cancellation, shutdown, metrics, and replay insertion
```

Rules:

```text
No global mutable search state.
No thread-local RNG.
No unbounded queues.
No runtime handles in search structs.
No graph bodies in search nodes.
No concrete engine adapter types.
Each worker owns its tree/path state, scratch buffers, and RNG state.
All cross-worker batching belongs outside gz-search.
```

The first greedy, beam, random, and serial Gumbel-MCTS implementations may be
run-to-completion over a mutable `GraphEngine`. Later async/wave Gumbel-MCTS
should be able to use a step/state-machine driver so the orchestrator can batch
leaf expansion, evaluation, apply, and measurement requests across many workers.

Do not add the full state-machine API for greedy alone. Do keep shared records,
hashing, scratch ownership, and dependency boundaries compatible with that
future API.

## Gumbel-MCTS Direction

Gumbel-MCTS is the expected primary algorithm once feature extraction and eval
exist. Greedy, beam, and random code must not make that path harder.

The serial Gumbel-MCTS contract is specified in `GZ_SEARCH_GUMBEL_MCTS.md`.
That spec defines the root sequential-halving math, completed-Q policy target,
STOP semantics, episode loop, and opponent-trajectory eval context. The serial
implementation follows that spec without async or wave search.

Gumbel-MCTS search state needs:

```text
worker-local tree arena
node arrays keyed by compact node indexes
E::Graph handles on expanded nodes
SearchAction<E::Candidate> handles or PortableSearchActionRef values on edges
visit counts
prior logits or probabilities
value estimates
gumbel noise sampled from worker-local RNG
pending expansion/evaluation markers
selected action path records
```

The async process around it will likely need to batch:

```text
candidate enumeration
candidate application
feature extraction
neural evaluation
terminal measurement
replay row writing
```

Rules for `gz-search` now:

```text
Design records around selected transitions, not greedy-only concepts.
Keep search state worker-local and movable between async tasks if needed.
Make scratch ownership explicit rather than hidden in globals.
Keep engine and candidate handles opaque.
Keep portable refs in outputs so replay/orchestration does not need local
handles.
Do not bake synchronous run-to-completion as the only possible search shape.
```

Gumbel-MCTS details owned by `GZ_SEARCH_GUMBEL_MCTS.md`:

```text
tree policy formula
root gumbel sampling
sequential halving schedule
value backup rules
opponent trajectory indexing
serial search/eval integration
```

The poll/resume work protocol that hosts async and wave execution is owned
by `GZ_ORCHESTRATOR.md`; the protocol types and Gumbel task state machines
live in `gz-search` per that spec. Wave tree math (virtual visits, in-flight
bookkeeping, halving barriers) remains deferred until the serial task
implementation matches the run-to-completion goldens.

## Implemented Algorithms

The initial algorithms are measured greedy rollout, measured beam search, and
measured random rollout.

Greedy rollout at each step:

```text
measure the current graph
enumerate legal engine candidates
append STOP as a search action
apply each candidate action
measure each accepted successor graph
score STOP as the current graph reward
choose the best search action
stop when STOP is selected
```

This is intentionally simple. It validates the engine/search boundary and
episode recording before adding evaluators, features, tree search, or replay
storage.

Beam search at each depth:

```text
expand every non-stopped graph in the active beam
enumerate legal engine candidates for each expanded graph
append STOP as a search action for each expanded graph
apply each candidate action
measure each accepted successor graph
score STOP as the current graph reward
rank candidate successors and STOP actions together
keep the top beam_width graph-distinct entries
carry selected STOP entries without expanding them again
return the best retained path when max_depth is reached or all entries stopped
```

Beam search is still a synchronous run-to-completion kernel. It exists to test
multi-path episode selection without introducing neural evaluation, async tree
stepping, or a generic search trait.

Random rollout at each step:

```text
measure the current graph
enumerate legal engine candidates
append STOP as a search action
uniformly sample one action from candidates + STOP using the configured seed
apply and measure only the selected candidate when a candidate is selected
stop when STOP is selected, the selected candidate is rejected, an unscored
graph is reached, or max_steps is reached
```

Random rollout is intentionally small. It exists as a cheap baseline and
smoke-test path through the same engine-owned candidates, STOP insertion, apply,
measure, and episode recording used by real search.

Planned later algorithms:

```text
policy rollout
async/wave Gumbel-MCTS
tree-parallel async search
```

Do not add a generic `SearchAlgorithm` trait for greedy alone. Add the trait or
state-machine interface when Gumbel-MCTS or another second real algorithm makes
the shared boundary concrete.

## Search Actions

All search algorithms select `SearchAction`, not raw `E::Candidate`.

```rust
pub enum SearchAction<C> {
    Candidate(C),
    Stop,
}
```

Rules:

```text
GraphEngine enumerates only engine candidates.
gz-search wraps engine candidates as SearchAction::Candidate.
gz-search appends SearchAction::Stop to the legal action list.
Every algorithm scores/selects from legal SearchAction values.
Only Candidate actions can call GraphEngine::apply().
STOP is selected through the same action-selection path as candidates.
```

## Public API

```rust
pub struct GreedySearch {
    config: GreedySearchConfig,
}

#[derive(Clone, Debug)]
pub struct GreedySearchConfig {
    pub max_steps: usize,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

pub type GreedyEpisode<G, C> = SearchEpisode<G, C, GreedyStopReason>;

pub struct BeamSearch {
    config: BeamSearchConfig,
}

#[derive(Clone, Debug)]
pub struct BeamSearchConfig {
    pub max_depth: usize,
    pub beam_width: NonZeroUsize,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

pub struct BeamEpisode<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<SearchStep<G, C>>,
    pub layers: Vec<BeamLayer<G>>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: BeamStopReason,
    pub search_config_hash: SearchConfigHash,
}

pub struct BeamLayer<G> {
    pub depth: usize,
    pub entries: Vec<BeamEntrySummary<G>>,
}

pub struct BeamEntrySummary<G> {
    pub graph: G,
    pub context: ReplayGraphContext,
    pub measure: MeasureSummary,
    pub reward: f32,
    pub stopped: bool,
    pub carried: bool,
    pub parent_index: Option<usize>,
    pub selected_action: Option<PortableSearchActionRef>,
    pub selected_rank: Option<usize>,
    pub engine_candidate_count: Option<usize>,
    pub action_count: Option<usize>,
}

pub struct SearchCandidateSummary {
    pub kind: CandidateKindId,
    pub tags: CandidateTags,
    pub static_prior: f32,
}

pub struct SearchEpisode<G, C, S> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<SearchStep<G, C>>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: S,
    pub search_config_hash: SearchConfigHash,
}

pub struct SearchStep<G, C> {
    pub before: G,
    pub after: G,
    pub action: SearchAction<C>,
    pub step_ref: SearchStepRef,
    pub selected_action: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub selected_measure: MeasureSummary,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub selected_rank: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GreedyStopReason {
    MaxSteps,
    SelectedStop,
    UnscoredCurrentGraph,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BeamStopReason {
    MaxDepth,
    SelectedStop,
    UnscoredRoot,
}
```

Execution methods:

```rust
impl GreedySearch {
    pub fn new(config: GreedySearchConfig) -> Self;

    pub fn run_from_root<E: GraphEngine>(
        &self,
        engine: &mut E,
    ) -> EngineResult<GreedyEpisode<E::Graph, E::Candidate>>;

    pub fn run<E: GraphEngine>(
        &self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<GreedyEpisode<E::Graph, E::Candidate>>;
}

impl BeamSearch {
    pub fn new(config: BeamSearchConfig) -> Self;

    pub fn run_from_root<E: GraphEngine>(
        &self,
        engine: &mut E,
    ) -> EngineResult<BeamEpisode<E::Graph, E::Candidate>>;

    pub fn run<E: GraphEngine>(
        &self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<BeamEpisode<E::Graph, E::Candidate>>;
}
```

Rules:

```text
run_from_root uses engine.root().
run starts from the supplied graph.
max_steps = 0 is valid and returns a measured zero-step episode.
candidate_options are passed unchanged to engine.candidates().
measure_options are passed unchanged to engine.measure().
GreedySearch and BeamSearch must not call export_graph().
GreedySearch and BeamSearch must not inspect graph or candidate handles.
```

`SearchEpisode` and `SearchStep` are algorithm-neutral selected-transition
records. `BeamEpisode.layers` records the retained beam frontier at each depth.
`GumbelRootResult` and `GumbelEpisode` add algorithm-specific root statistics,
visit-derived values, and policy targets alongside these records.

`SearchAction::Candidate` wraps an engine-owned candidate handle.
`SearchAction::Stop` is search-owned control flow and must never be passed to
`GraphEngine::apply()`.

## Config Rules

`GreedySearchConfig` rules:

```text
max_steps may be zero
candidate_options are accepted as-is
measure_options must already satisfy MeasureOptions invariants
```

No adapter-specific settings belong in `GreedySearchConfig`.

`BeamSearchConfig` rules:

```text
max_depth may be zero
beam_width must be nonzero
candidate_options are accepted as-is
measure_options must already satisfy MeasureOptions invariants
```

No adapter-specific settings belong in `BeamSearchConfig`.

`CandidateOptions.max_candidates` limits only engine candidates. `STOP` is
still appended by `gz-search` and remains legal.

When `max_candidates` is set, the algorithm is greedy over the projected engine
candidate set plus `STOP`, not over every legal engine candidate.

## SearchConfigHash

`gz-search` computes `SearchConfigHash` from the exact search settings that can
change the episode path.

Initial derivation:

```text
hash("gz-search-greedy-v1",
     max_steps,
     candidate_options.max_candidates,
     candidate_options.deterministic_order,
     measure_options.config_hash,
     measure_options.samples,
     measure_options.timeout_ms,
     measure_options.deterministic)

hash("gz-search-beam-v1",
     max_depth,
     beam_width,
     candidate_options.max_candidates,
     candidate_options.deterministic_order,
     measure_options.config_hash,
     measure_options.samples,
     measure_options.timeout_ms,
     measure_options.deterministic)
```

Rules:

```text
use a domain prefix
use fixed-width or length-delimited fields
do not hash debug strings
do not include engine-local handles
same config bytes must produce the same SearchConfigHash across platforms
changing any setting that can change selected steps must change the hash
```

The hash identifies search behavior, not engine behavior. Engine identity,
engine version, action-set hash, and graph hash remain in the portable graph
and candidate references from `gz-engine`.

## Portable Contexts

Search output must contain portable references for each selected transition.

For a graph `g`:

```text
PortableGraphId = hash(g) + engine_id + engine_version
ReplayGraphContext = PortableGraphId + engine.action_set_hash()
```

For a candidate `c` on graph `g`:

```text
PortableCandidateRef = ReplayGraphContext(g) + candidate_info(g, c).candidate_hash
PortableSearchActionRef::Candidate(PortableCandidateRef)
```

For `STOP` on graph `g`:

```text
PortableSearchActionRef::Stop = ReplayGraphContext(g)
```

Rules:

```text
SearchStep.step_ref.before must equal the before graph context.
SearchStep.step_ref.action must equal the selected action ref.
SearchStep.step_ref.after must equal the accepted after graph context.
Candidate action refs come from candidate_info().
STOP action refs are created by gz-search.
STOP action refs use the current graph context and have no CandidateHash.
Search output may retain E::Graph and E::Candidate handles for in-process use.
Replay and checkpoints must use portable refs, not local handles.
```

## STOP

`gz-search` does not ask the engine for a STOP candidate.

STOP is represented as `SearchAction::Stop` and
`PortableSearchActionRef::Stop`, not by `E::Candidate`.

Reasons:

```text
GraphEngine candidates are engine-owned rewrite/action candidates.
Whittle does not expose STOP as a rewrite candidate.
Future compiler engines should not need fake engine actions for episode control.
Every search algorithm needs a way to select STOP.
Future policy targets and Gumbel-MCTS children need STOP to align with legal
search actions.
```

Rules:

```text
STOP is appended by gz-search after engine candidates.
STOP is always legal when a graph can be scored.
STOP does not count against CandidateOptions.max_candidates.
Selecting STOP terminates the episode at the current graph.
For STOP, after == before.
STOP is not passed to GraphEngine::apply().
STOP appears in action histories as PortableSearchActionRef::Stop.
```

## Greedy Scoring

The initial greedy objective is scalar reward returned by
`GraphEngine::measure`.

A measurement is scoreable when:

```text
measured == true
valid == true
scalar_reward == Some(finite f32)
```

Candidate scoring:

```text
append STOP to the legal action list
score STOP as the current graph reward
apply each candidate action
skip it if ApplyResult.rejected is Some(...)
measure ApplyResult.after
skip it if the measurement is not scoreable
candidate score = measurement.scalar_reward
```

Current graph scoring:

```text
measure current graph
if the current graph is not scoreable, stop with UnscoredCurrentGraph
```

Selection:

```text
select the candidate action with the highest scalar reward that strictly
improves on the current graph
if no candidate strictly improves reward, select STOP
if STOP is selected, record a STOP step and stop with SelectedStop
```

Candidate tie-break order:

```text
1. higher scalar_reward
2. higher CandidateInfo.static_prior
3. lower CandidateHash lexicographically
```

STOP beats every candidate whose score is less than or equal to the current
graph score. The candidate tie-break applies only among strictly improving
candidates. It must be deterministic and independent of candidate vector
position except where every stable key is equal.

## Beam Scoring

Beam search uses the same scoreable measurement definition as greedy search.

Candidate scoring:

```text
append STOP to each expanded graph's legal action list
score STOP as that graph's current reward
apply each candidate action
skip it if ApplyResult.rejected is Some(...)
measure ApplyResult.after
skip it if the measurement is not scoreable
candidate score = measurement.scalar_reward
```

Selection:

```text
rank all candidate successors and STOP actions from the expanded beam
retain at most beam_width graph-distinct entries
carry retained STOP entries forward without expansion
stop early when every retained entry is STOP
select the highest-ranked retained path as the final episode
```

Beam tie-break order:

```text
1. higher scalar_reward
2. STOP before continuing candidates
3. higher CandidateInfo.static_prior
4. lower PortableSearchActionRef lexicographically
5. shorter path depth
```

`beam_width = 1` should match greedy when greedy's strict-improvement rule and
beam's STOP tie-break produce the same one-step choice. Wider beams may keep
lower immediate rewards that can win at later depths.

## Measurement Rules

`measure()` may be called on any graph. The implemented searches measure:

```text
the current graph/root before expansion
accepted successor graphs during one-step lookahead
the final graph before returning
```

It may reuse a measurement already produced in the same run for the same graph
and `MeasureOptions`. It should not add a cross-run measurement cache in
`gz-search`.

When STOP is selected, `selected_measure` is the current graph measurement and
`final_measure` may reuse that same measurement.

The final episode is replay-eligible only when `final_measure` is measured and
valid. `gz-search` does not write replay rows, but it must return enough data
for `gz-replay` or `gz-orchestrator` to enforce measured-before-replay.

Measurement failures are data when returned as `MeasureResult.failure`.
`EngineError` from `measure()` aborts the run.

## Apply Rules

For each candidate action:

```text
candidate_info(before, candidate) must be read before recording a portable ref
apply(before, candidate) produces the successor considered by search scoring
rejected apply results are skipped
stale candidates or unknown candidates are EngineError and abort the run
```

For STOP:

```text
do not call candidate_info()
do not call apply()
after = before
step_ref.after = step_ref.before
selected_measure is the current graph measurement
```

`ApplyResult.changed == false` is allowed. Such candidates must still beat the
current scalar reward to be selected. Otherwise STOP is selected.

## Episode Output

`SearchEpisode` is an in-process search result, not a replay row.

It may store:

```text
E::Graph handles
E::Candidate handles
portable graph contexts
portable search action refs
measure summaries
search config hash
stop reason
```

It must not store:

```text
full graph bodies
graph artifacts
adapter-specific candidate structs
RocksDB keys
actor ids
Python tensors
```

Replay projection is a later crate concern. The selected-transition output is
sufficient to record the final selected path for greedy, beam, and random.
Gumbel-MCTS exposes policy targets explicitly in its own result type.

Whether to store one replay row per step or only the final graph belongs to
`gz-replay` / `gz-orchestrator`, not `gz-search`.

Gumbel-MCTS policy targets are explicit data in the Gumbel-MCTS result, not
inferred from greedy or beam fields.

## Error Handling

The run-to-completion APIs return `EngineResult<GreedyEpisode<_, _>>` and
`EngineResult<BeamEpisode<_, _>>`.

Rules:

```text
EngineError from engine calls aborts the run.
unscored measurements stop the episode; they are not EngineError.
no engine candidates selects STOP when the current graph is scoreable.
all candidates rejected selects STOP when the current graph is scoreable.
SearchStepRef construction failure is an internal bug and maps to
EngineError::Internal.
```

Add a `SearchError` type only when the crate has a real non-engine failure that
cannot be represented as config validation or stop reason.

## Determinism Invariants

For a fixed engine config/version, root graph, search config, and deterministic
measurement:

```text
run returns the same step count
run returns the same selected PortableSearchActionRef sequence
run returns the same final GraphHash
run returns the same stop reason
run returns the same SearchConfigHash
selected ranks are relative to engine deterministic candidate order followed by STOP
```

If `CandidateOptions.deterministic_order == false`, the engine may return a
different candidate order. Search selection must still be deterministic when
the measured scores and stable tie-break keys uniquely determine a winner.

## Performance Rules

Initial implementation should be simple and allocation-conscious:

```text
reuse one Vec<E::Candidate> for candidate enumeration
reuse candidate scratch buffers inside one run where possible
store portable search action refs, not CandidateInfo display strings or metadata
do not call export_graph in the search loop
do not allocate graph artifacts
do not add parallelism or async wrappers
do not cache measurements in gz-search initially
```

Search may call `measure()` many times. That is acceptable for the first
measured implementations because it is explicit algorithm behavior, not hidden
replay admission. If this is too expensive for a future compiler engine, add a
different algorithm or scoring boundary after measurement shows the cost.

## Crate Shape

```text
crates/gz-search/
  Cargo.toml
  src/
    lib.rs
    beam.rs
    episode.rs
    greedy.rs
    hash.rs
    scratch.rs
    support.rs
  tests/
    beam.rs
    common/
    greedy.rs
```

Keep modules flat. Do not add `manager`, `runner`, or `strategy` modules. Add
algorithm-specific modules only when they hold real state or logic.

## Test Strategy

Use a private deterministic test engine inside `gz-search` tests.

Do not add `gz-engine-fake` just to test this crate unless the private test
engine becomes large enough to justify a reusable adapter.

Required tests:

```text
zero-step run measures root and returns MaxSteps
STOP is appended after engine candidates
CandidateOptions.max_candidates does not remove STOP
no engine candidates selects STOP
unscored current graph returns UnscoredCurrentGraph
all rejected candidates selects STOP
all unscored successors selects STOP
best strict improvement is selected
no strict improvement selects STOP
tie-break uses static_prior then CandidateHash
STOP beats equal-reward candidates
STOP step has after == before and does not call apply()
step_ref action contexts are valid
selected action refs match candidate_info output or STOP
search_config_hash changes when path-affecting config changes
beam_width = 1 matches greedy on the same one-step choices
wider beam keeps lower immediate rewards that can win later
retained STOP entries are carried without expansion
beam layers record the retained frontier at each depth
beam search_config_hash changes when beam-affecting config changes
```

Do not duplicate `gz-engine` contract tests in `gz-search`. The search tests
only need enough engine behavior to prove search control flow and portable
recording.

## Implementation Plan

1. Add `crates/gz-search` to the workspace with dependency on `gz-engine`.
2. Implement `SearchAction`, `SearchEpisode`, `SearchStep`, and portable
   context helpers.
3. Implement minimal worker-local scratch for candidate vectors and candidate
   scoring.
4. Implement `GreedySearchConfig`, `GreedySearch::new`, and
   `SearchConfigHash` derivation.
5. Implement zero-step `run()` and `run_from_root()`.
6. Implement candidate enumeration and selected search action ref recording.
7. Implement STOP action injection.
8. Implement measured successor scoring and deterministic tie-breaks.
9. Implement stop reasons for selected STOP, unscored current graph, and max
   steps.
10. Add focused tests with a private deterministic test engine.
11. Implement `BeamSearchConfig`, `BeamSearch::new`, and beam
    `SearchConfigHash` derivation.
12. Implement measured beam expansion, STOP carrying, graph-distinct beam
    retention, and selected path reconstruction.
13. Add focused beam tests using the shared private deterministic test engine.
14. Run `cargo fmt`, `cargo test --all`, and
   `cargo clippy --all-targets --all-features -- -D warnings`.

## Deferred

```text
policy/evaluator scoring
feature extraction
batched search
async search actor
replay writer integration
RocksDB storage
Whittle-specific tests
compiler-specific tests
```
