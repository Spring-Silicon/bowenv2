# gz-search PUCT-MCTS and Shared Policy/Value MCTS Spec

Status: implemented

Purpose: add pure policy/value PUCT-MCTS to `gz-search` while making it use
the same tree, task, episode, work, context, measurement, and handle-lifecycle
abstractions as the existing Gumbel-MCTS implementation. Gumbel-MCTS and
PUCT-MCTS should differ only where their search algorithms differ.

This is an implementation spec. It defines the common internal boundary, the
PUCT math, the public PUCT surface, the Gumbel compatibility requirement, and
the order in which the refactor and new algorithm must land.

## Decision

GraphZero will have one private policy/value MCTS engine with two statically
dispatched root-search strategies:

```text
                         Gumbel strategy
                        /
shared policy/value MCTS core
                        \
                         PUCT strategy
```

The shared core owns:

```text
tree and node storage
candidate expansion
STOP insertion
policy/value eval requests
apply and rejection handling
cycle and episode-history masking
edge visit/value/Q ledgers
value backup
tree compaction and reuse
root and episode poll/resume state machines
position and opponent alignment
final terminal measurement
engine-handle ownership and release
```

The strategies own only:

```text
simulation scheduling
tree-action selection
root policy-target construction
final root-action election
algorithm-specific root metadata
```

There must not be two copies of the root/episode state machines. There must not
be a runtime algorithm branch or trait object in the search hot loop. Rust
generics must monomorphize the shared core for each strategy.

The current public Gumbel API and behavior remain compatible. PUCT exposes a
parallel public API over the same private machinery.

## Meaning of Pure PUCT

PUCT in this spec is policy/value MCTS. It is not the measured optimizer in
`../graphs/src/graphs/search/mcts.py`.

The `../graphs` implementation was reviewed because it uses a PUCT-shaped
selection expression, but its search semantics are different:

```text
it uses rule-authored static candidate priors
it runtime-measures every accepted child
it backs measured child reward through the tree
it returns the globally best measured node
```

GraphZero PUCT instead:

```text
gets action logits and a leaf value from EngineEvaluator<E>
softmaxes logits into action priors
backs evaluator leaf values through the tree
constructs the root policy target from visit counts
selects the next root action from visit counts
always measures the final episode graph through GraphEngine::measure
```

`CandidateInfo::static_prior` remains evaluator input metadata. The PUCT
kernel does not combine it with evaluator logits. An evaluator may use static
priors, ignore them, or combine them with learned scores.

## Leaf Evaluator Boundary

`EngineEvaluator<E>` is the leaf-evaluation extension point for both Gumbel
and PUCT. The shared MCTS core depends only on its validated `EvalOutput`; it
must not assume that the output came from a neural model.

The existing engine-aware input is sufficient:

```text
&mut E
evaluated E::Graph handle
ordered E::Candidate handles
EvalRequest with ordered action metadata and STOP
MeasureOptions
```

This permits all of these implementations without changing either MCTS
algorithm or `SearchWork`:

```text
learned policy/value model
static-prior or heuristic policy/value evaluator
measurement-backed evaluator that calls E::measure for the leaf value
CompilerEngine evaluator that calls CompilerEngine::measure
CompilerEngine evaluator that internally composes lowering and benchmarking
```

PUCT requires a non-negative action prior `P(s,a)`; it does not inherently
require logits. The shared `EvalOutput` uses one logit per candidate plus STOP
because that representation is also consumed by Gumbel-MCTS and is convenient
for learned policies. Equal finite logits, normally all `0.0`, softmax to a
uniform PUCT prior and are a valid policy for a measurement-backed evaluator.

`E::measure` alone still does not produce a complete shared `EvalOutput`: the
adapter must add the policy representation and an evaluator `ModelVersion`. It
maps the finite measured reward to `EvalOutput::value` and obtains logits from
candidate static priors, a uniform constant, heuristics, or a learned policy.
This composition belongs inside the concrete evaluator, not inside MCTS.

For a compiler backend, search must not gain `lower` or `benchmark` methods.
An `EngineEvaluator<CompilerEngine>` may call concrete compiler methods because
it receives `&mut CompilerEngine`; search continues to depend only on
`GraphEngine`. Prefer `CompilerEngine::measure` when lowering and benchmarking
are the backend's measurement semantics. A specialized compiler leaf evaluator
may compose them differently when its search estimate intentionally differs
from terminal measurement.

Leaf measurement and terminal measurement remain distinct protocol events. A
leaf evaluator may call `E::measure`, but it returns only `EvalOutput` to the
tree. The episode must still emit final `MeasureWork` and receive a terminal
`MeasureResult` before replay eligibility. An engine-owned exact-key cache may
deduplicate repeated work; search must not skip the terminal measurement based
on a leaf value.

## Scope

This spec includes:

```text
private shared policy/value MCTS tree storage
private generic root and episode task state machines
Gumbel strategy extraction with behavior parity
pure PUCT root search
pure PUCT episode search
PUCT tree reuse using the existing shifted-subtree abstraction
PUCT search config hashing
serial run-to-completion wrappers
poll/resume execution through SearchWork
shared conformance tests and PUCT algorithm tests
```

This spec excludes:

```text
parallel simulations within one root
virtual loss or virtual visits
wave search
Dirichlet root noise
new replay schemas
trainer or model changes
CLI selection of PUCT
orchestrator admission of PUCT episodes
PUCT sampled-tree or categorical opponent modes
compiler-specific cost scheduling
CUDA worker queues
```

Orchestrator, replay, trainer, and CLI integration require a follow-up spec
after `gz-search` behavior is stable. The public and task APIs in this spec
must make that integration mechanical.

## Non-Negotiable Invariants

The implementation must preserve these GraphZero invariants:

```text
search is generic over GraphEngine
search stores E::Graph and E::Candidate handles only
candidate semantics and deterministic order remain engine-owned
STOP is appended by search and is never passed to GraphEngine::apply
all evaluator outputs are validated before entering the tree
one expanded node receives at most one ordinary policy/value evaluation
GraphEngine::measure owns terminal measurement
an episode is returned only after its final measure completes
all created handles are either retained by the result/tree or released once
all task work is represented by the existing bounded-orchestrator-friendly
SearchWork protocol
```

PUCT is single-player search. A backed-up value is not negated between tree
levels. Competitive sampled-tree semantics remain Gumbel-specific and outside
the shared standard episode task.

## Module Layout

Target layout:

```text
crates/gz-search/src/
  mcts/
    mod.rs
    math.rs
    strategy.rs
    tree.rs
    types.rs
    driver.rs
    task/
      mod.rs
      root.rs
      episode.rs
      state.rs

  gumbel/
    mod.rs
    strategy.rs
    schedule.rs
    categorical.rs
    sampled_tree/
      mod.rs
      root.rs
      episode.rs

  puct/
    mod.rs
    strategy.rs
    task.rs
    types.rs
```

`mcts` is private to `gz-search`. It is not a new public framework or a
general-purpose game library.

Ownership:

```text
mcts/tree.rs
  common arena node storage
  graph/candidate/action metadata
  mask state
  visit/value/Q ledgers
  backup
  subtree compaction
  position context construction

mcts/task/root.rs
  expand -> eval -> select -> apply -> descend/expand -> backup loop
  token validation
  rejection, STOP, no-backtrack, and cycle handling

mcts/task/episode.rs
  repeated root searches
  selected-step recording
  shifted-subtree reuse
  final measurement
  handle partitioning and release batches

mcts/driver.rs
  serial SearchWork servicing for GraphEngine + EngineEvaluator

mcts/math.rs
  common softmax, deterministic RNG, root action sampling, and budget math

gumbel/strategy.rs
  Gumbel root state and strategy implementation

gumbel/schedule.rs
  root Gumbel noise, considered sets, and sequential-halving schedule

puct/strategy.rs
  PUCT root state, PUCT action score, visit target, and PUCT root election
```

`gumbel/categorical.rs` and `gumbel/sampled_tree/` remain specialized paths.
They may reuse common node/math helpers where natural, but this spec does not
require rewriting them around the generic standard episode task.

## Shared Internal Configuration

Public algorithm configs convert to a private common config:

```rust
struct MctsConfig {
    max_steps: usize,
    simulations: NonZeroUsize,
    seed: u64,
    temperature_moves: usize,
    tree_reuse: bool,
    export_position: bool,
    mask_stop: bool,
    no_backtrack: bool,
    candidate_options: CandidateOptions,
    measure_options: MeasureOptions,
}
```

Algorithm-specific configuration is not placed in `MctsConfig`:

```rust
struct GumbelStrategyConfig {
    max_considered_actions: NonZeroUsize,
    gumbel_scale: f32,
    gumbel_noise_overlap: f32,
    c_visit: f32,
    c_scale: f32,
}

struct PuctStrategyConfig {
    c_puct: f32,
}
```

The public `GumbelMctsConfig` keeps its current flat shape. Conversion to
private common and strategy configs is internal and must not require caller
changes.

## Strategy Boundary

The generic root task is parameterized by one private strategy trait. The
exact method names may change during implementation, but the responsibility
boundary must match this shape:

```rust
trait MctsStrategy {
    type RootState;
    type RootMetadata;

    fn start_root<G, C>(
        &self,
        tree: &MctsTree<G, C>,
        context: MctsSearchContext,
    ) -> Self::RootState;

    fn start_simulation<G, C>(
        &self,
        state: &mut Self::RootState,
        tree: &MctsTree<G, C>,
    ) -> Option<RootSelection>;

    fn select_nonroot<G, C>(
        &self,
        state: &Self::RootState,
        tree: &MctsTree<G, C>,
        node: usize,
    ) -> usize;

    fn complete_simulation(&self, state: &mut Self::RootState);

    fn finish_root<G, C>(
        &self,
        state: Self::RootState,
        tree: &MctsTree<G, C>,
    ) -> StrategyRootResult<Self::RootMetadata>;
}
```

The common root task, not the strategy, owns:

```text
descent paths
per-descent seen contexts
ApplyWork construction
ExpandWork construction
EvalWork construction
pending work tokens
child attachment
STOP and rejection behavior
value backup calls
simulation completion after backup
```

The strategy must never call the engine or evaluator directly.

## Shared Tree Layout

Use one compact arena for both algorithms:

```rust
struct MctsNode<G, C> {
    graph: G,
    context: ReplayGraphContext,
    candidates: Vec<C>,
    eval_actions: Vec<EvalAction>,
    candidate_hashes: Vec<CandidateHash>,
    summaries: Vec<Option<SearchCandidateSummary>>,
    logits: Vec<f32>,
    priors: Vec<f32>,
    value: f32,
    model_version: ModelVersion,
    children: Vec<Option<usize>>,
    visits: Vec<u32>,
    value_sum: Vec<f32>,
    q: Vec<f32>,
    masked: Vec<bool>,
}
```

Rules:

```text
STOP is action index candidates.len()
every per-action vector has candidates.len() + 1 entries
STOP children remain None
priors are softmax(logits) before search-side masking
unvisited Q is 0
masked actions are ignored by both strategies
masking does not destroy evaluator logits or priors
```

Use an explicit `masked` vector instead of overwriting logits with negative
infinity. This lets both strategies share masking and lets STOP be restored
when all rewrite actions become unavailable.

The common backup operation is:

```text
for each edge (node, action) on the descent path:
    visits[node][action] += 1
    value_sum[node][action] += leaf_value
    q[node][action] = value_sum[node][action] / visits[node][action]
```

Values must be finite. Search does not clamp, normalize, negate, or rescale
them. Evaluator value scale and `c_puct` must be configured consistently.

## Public PUCT API

### Search and Config

```rust
pub struct PuctMcts {
    config: PuctMctsConfig,
    search_config_hash: SearchConfigHash,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuctMctsConfig {
    pub max_steps: usize,
    pub simulations: NonZeroUsize,
    pub c_puct: f32,
    pub seed: u64,
    pub temperature_moves: usize,
    pub tree_reuse: bool,
    pub export_position: bool,
    pub mask_stop: bool,
    pub no_backtrack: bool,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}
```

Validation:

```text
simulations is nonzero by type
c_puct is finite and non-negative
max_steps may be zero
temperature_moves may exceed max_steps without error
```

`PuctMcts` methods mirror `GumbelMcts`:

```rust
impl PuctMcts {
    pub fn new(config: PuctMctsConfig) -> Self;
    pub const fn config(&self) -> PuctMctsConfig;
    pub const fn search_config_hash(&self) -> SearchConfigHash;
    pub fn root_budget(&self, step: usize) -> (f32, f32);

    pub fn search_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: PuctSearchContext,
    ) -> EngineResult<PuctRootResult<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>;

    pub fn run_from_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
    ) -> EngineResult<PuctEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>;

    pub fn run<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: PuctEpisodeContext,
    ) -> EngineResult<PuctEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>;
}
```

### Contexts

PUCT contexts mirror the standard Gumbel contexts:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PuctEpisodeContext {
    pub opponent: Option<MctsOpponentContext>,
    pub noise_seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsOpponentContext {
    pub trajectory_id: u64,
    pub row_count: u32,
    pub final_reward: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuctSearchContext {
    pub root_step: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
    pub selection_temperature: f32,
    pub opponent: Option<MctsOpponentContext>,
    pub noise_seed: u64,
    pub export_position: bool,
}
```

`noise_seed` is retained as the cross-algorithm episode randomness field. For
PUCT it affects temperature-based root-action sampling only. PUCT adds no
search noise.

`MctsOpponentContext` becomes the neutral canonical public record currently
expressed as `GumbelOpponentContext`. Preserve `GumbelOpponentContext` as a
public alias and export `PuctOpponentContext` as the same alias. Existing
Gumbel callers must continue to compile.

### Task Types

```rust
pub struct PuctRootTask<G, C>;
pub struct PuctEpisodeTask<G, C>;
```

They mirror the Gumbel task contract:

```text
PuctRootTask:
  new
  poll
  resume
  root_context

PuctEpisodeTask:
  new
  poll
  resume
  step_index
  take_releasable
  take_all_handles
  track_owned_root
```

Both tasks emit the existing `SearchWork<G, C>` variants and consume the
existing `SearchWorkResult<G, C>` variants. No PUCT-specific work protocol is
allowed.

### Results

PUCT result records mirror the standard Gumbel result shape so orchestrator
projection can later be generalized without changing replay rows:

```rust
pub struct PuctRootResult<G, C> {
    pub root: G,
    pub root_context: ReplayGraphContext,
    pub selected_after: G,
    pub selected_after_context: ReplayGraphContext,
    pub selected_action: SearchAction<C>,
    pub selected_action_ref: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub selected_action_index: usize,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub considered_action_indices: Vec<usize>,
    pub policy_target: Vec<f32>,
    pub root_value: f32,
    pub root_search_value: f32,
    pub root_q_max: f32,
    pub model_version: ModelVersion,
    pub stats: PuctRootStats,
}

pub struct PuctRootStats {
    pub simulations: usize,
    pub expanded_nodes: usize,
    pub eval_count: usize,
    pub portable_contexts: usize,
    pub carried_nodes: usize,
    pub carried_root_visits: u32,
}
```

For PUCT, `considered_action_indices` contains every final unmasked root action.
The field is retained for structural parity even though PUCT has no Gumbel
considered subset.

```rust
pub struct PuctEpisode<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<PuctStep<G, C>>,
    pub root_stats: Vec<PuctRootStats>,
    pub created_graphs: Vec<G>,
    pub created_candidates: Vec<C>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: PuctStopReason,
    pub search_config_hash: SearchConfigHash,
}
```

`PuctStep` mirrors `GumbelStep`:

```rust
pub struct PuctStep<G, C> {
    pub before: G,
    pub after: G,
    pub action: SearchAction<C>,
    pub step_ref: SearchStepRef,
    pub selected_action: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub selected_rank: usize,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<f32>,
    pub considered_action_indices: Vec<usize>,
    pub root_value: f32,
    pub root_search_value: f32,
    pub root_q_max: f32,
    pub model_version: ModelVersion,
}
```

`PuctStopReason` has:

```rust
pub enum PuctStopReason {
    MaxSteps,
    SelectedStop,
}
```

The Gumbel-only `competitive` episode extension is not added to PUCT.

## PUCT Algorithm

### Node Expansion

Expanding a graph:

```text
request engine candidates in deterministic engine order
create one EvalAction per candidate
append STOP as the last EvalAction
evaluate the graph once
validate output length and finiteness
softmax policy_logits into priors
store evaluator value and model version
initialize children=None, visits=0, value_sum=0, q=0, masked=false
apply configured STOP masking
```

The root expansion/evaluation initializes the tree and does not consume a
simulation. Every configured simulation performs at least one root action
selection unless only masked/invalid rewrite actions remain, in which case
STOP is restored and selected.

### Selection Score

At every node, including the root, PUCT selects the unmasked action with the
largest score:

```text
N(s) = sum_b N(s,b)

PUCT(s,a) = Q(s,a)
          + c_puct * P(s,a) * sqrt(max(N(s), 1)) / (1 + N(s,a))
```

Definitions:

```text
P(s,a) is the evaluator softmax prior
N(s,a) is the edge visit count, including carried visits after tree reuse
Q(s,a) is value_sum / visits, or 0 when unvisited
```

Selection ties use the lower stable action index. Candidate enumeration is
deterministic, and STOP is last, so this rule is reproducible without hashing
or allocation.

There is no min/max Q normalization, completed-Q transform, value negation,
static-prior multiplication, Gumbel noise, or sequential halving in PUCT.

### Descent and Leaf Evaluation

After selecting an action:

```text
STOP:
  append the STOP edge to the path
  back up the current node value
  complete the simulation

unexpanded candidate child:
  emit ApplyWork
  on accepted apply, attach the edge path
  expand and evaluate the child
  attach the child node
  back up the child evaluator value
  complete the simulation

expanded candidate child:
  append the edge to the path
  descend into the child
  continue PUCT selection

rejected candidate:
  mask the action
  retry selection without consuming a simulation

within-descent repeated context:
  append the edge
  back up the repeated child node's cached value
  complete the simulation
```

Like Gumbel-MCTS, PUCT has no user-facing simulation depth limit. A simulation
ends at a new leaf, STOP, a repeated context, or a fallback caused by masks.
The simulation count bounds total new tree nodes.

### STOP Re-evaluation for Opponent Alignment

The common task preserves current Gumbel opponent-alignment behavior. If STOP
at the current search depth must be valued against a later clamped opponent
row, the task emits the same secondary eval request before backup. PUCT and
Gumbel differ only in how STOP was selected, not in how the aligned value is
obtained.

### Simulation Completion

Each successful backup consumes exactly one simulation. These do not consume
a simulation:

```text
root initialization
rejected apply results
no_backtrack masks
action masking and retry
```

`PuctRootStats.simulations` must equal `config.simulations.get()` for a
completed root search. Search-level STOP guarantees a fallback when all
rewrite actions become unavailable.

## Root Target and Election

### Fresh Root

For a fresh root:

```text
delta_N(a) = N(root,a)
```

### Reused Root

At the start of every root search, snapshot carried visits:

```text
baseline_N(a) = N(root,a)
```

After the fresh simulation budget:

```text
delta_N(a) = N(root,a) - baseline_N(a)
```

Carried visits and Q values participate in PUCT selection. Root training
targets and root action sampling use `delta_N`, so each episode step records
the work performed for that decision rather than replaying the previous
root's visit mass.

### Policy Target

```text
total = sum_a delta_N(a)
policy_target[a] = delta_N(a) / total
```

Masked actions receive zero. If no rewrite receives a visit because every
rewrite was rejected or masked, the target is one-hot on STOP.

### Deterministic Election

At selection temperature zero, select the final unmasked action by:

```text
largest delta_N
then largest Q
then largest prior
then lower stable action index
```

### Temperature Election

For `selection_temperature > 0`, sample among final unmasked actions with:

```text
weight(a) = delta_N(a) ** (1 / selection_temperature)
```

Use the same deterministic worker-local RNG and root seed derivation as the
standard Gumbel episode action sampler. If weights are empty, zero, or
non-finite, fall back to deterministic election.

`temperature_moves` sets episode root selection temperature to `1.0` before
that step count and `0.0` afterward, matching Gumbel episode behavior.

### Root Values

PUCT result fields mean:

```text
root_value:
  evaluator value stored on the root node

root_search_value:
  visit-delta-weighted mean root Q over actions with delta_N > 0;
  root_value if there are no such actions

root_q_max:
  maximum finite Q among final unmasked root actions with at least one total
  visit; root_value if none exist
```

## STOP and Mask Semantics

STOP is a normal search action for policy, PUCT selection, visits, Q backup,
policy targets, and root election. It is not an engine candidate.

When `mask_stop` is false:

```text
STOP is unmasked from node initialization
```

When `mask_stop` is true:

```text
STOP is masked while at least one unmasked rewrite action exists
STOP-only nodes keep STOP unmasked
if every rewrite becomes masked or rejected, STOP is restored
```

Restoration uses the evaluator's original STOP logit/prior stored on the node.
Search does not fabricate a rewrite prior. A forced STOP fallback produces a
one-hot STOP policy target if no simulation could visit another action.

## no_backtrack and Cycles

Common task behavior:

```text
per-descent seen set:
  prevents an infinite cycle inside one simulation

episode visited-root set:
  when no_backtrack is enabled, masks an applied child equal to the current
  root or any earlier committed episode root
```

An action masked by `no_backtrack` does not consume a simulation. If every
rewrite revisits history, STOP is restored and the root target collapses to
STOP.

## Tree Reuse

PUCT uses the same shifted-subtree abstraction as Gumbel:

```text
after selecting a non-STOP root action, compact its child subtree
make the selected child node the next root
carry graph/candidate/eval payloads
carry children and edge visit/value/Q ledgers
release handles outside the retained subtree
do not reuse across episodes
```

At the next root:

```text
skip root expand/eval
use the new MctsSearchContext for future leaf evals
retain the carried root evaluator value/model version
snapshot carried root visits as baseline_N
run the full configured number of new simulations
```

This matches current Gumbel v8 reuse mechanics: reuse carries cached evidence,
but every move receives a fresh simulation budget counted relative to the
baseline.

PUCT selection uses total carried-plus-new visits and Q. Policy target and
root election use new visit deltas as defined above.

## Episode Semantics

The shared standard episode task owns:

```text
root search at the current graph
selected transition recording
episode position/budget context
temperature schedule
no_backtrack history
tree reuse
handle partitioning
STOP/max_steps termination
final GraphEngine::measure
```

PUCT episode loop:

```text
current = root

for step in 0..max_steps:
    run one PUCT root search
    record PuctStep
    if selected action is STOP:
        terminate with SelectedStop
    current = selected child
    optionally shift/reuse selected subtree

if max_steps is reached:
    terminate with MaxSteps

measure current through GraphEngine::measure
return PuctEpisode
```

`max_steps == 0` skips root search, measures the supplied root, and returns
`MaxSteps`.

Runtime reward and replay eligibility come only from the final episode
`MeasureResult`. A leaf value may be derived from measurement, but it remains a
search estimate and never substitutes for terminal measurement.

## Work Protocol

The common tasks use only:

```text
SearchWork::Expand
SearchWork::Apply
SearchWork::Eval
SearchWork::Measure
```

PUCT introduces no new work variant.

The root task may have one outstanding token. Poll while work is outstanding
returns `SearchPoll::Blocked`. Resume validates token and result variant before
mutating search state. Poll after completion and double resume are errors,
matching Gumbel task behavior.

The serial PUCT driver uses the common serial work service with
`GraphEngine + EngineEvaluator<E>`. The orchestrator can later drive the same
task without changes to the PUCT algorithm.

## Handle Ownership and Release

The shared episode task retains current Gumbel ownership rules:

```text
ExpandResult candidates are tracked as created candidate handles
ApplyResult.after is tracked as a created graph handle
selected path graphs survive until episode completion
shifted subtree handles survive into the next root
nonselected/noncarried handles become releasable immediately after a move
take_releasable drains only currently dead handles
take_all_handles drains every handle still owned by the task
```

PUCT public task methods use the same handle batch representation. Introduce
canonical public `MctsHandleBatch<G, C>` storage and preserve
`GumbelHandleBatch` and `PuctHandleBatch` as public aliases.

No error, STOP, rejection, cancellation, tree-reuse, or final-success path may
leak or double-release a handle.

## Search Config Hash

Add:

```rust
pub fn puct_search_config_hash(
    max_steps: usize,
    simulations: usize,
    c_puct: f32,
    seed: u64,
    temperature_moves: usize,
    tree_reuse: bool,
    mask_stop: bool,
    no_backtrack: bool,
    candidate_options: CandidateOptions,
    measure_options: MeasureOptions,
) -> SearchConfigHash;
```

Hash domain:

```text
gz-search-puct-mcts-v1
```

Hash fields in the signature order using the existing canonical encoders.
Mirror current Gumbel behavior by excluding `export_position`: it changes
exported evaluator position inputs/cache identity, not the search algorithm's
internal position accounting.

Every behavior-affecting PUCT config field must have a hash-change test.

The Gumbel hash domains and outputs must not change merely because code moves
into `mcts/`.

## Gumbel Compatibility

Extracting the shared core is not authorization to change Gumbel behavior.

Required parity:

```text
same root requests and evaluator requests
same RNG seeding
same root Gumbel samples
same considered sets
same sequential-halving schedule
same action masks
same selected actions
same policy targets
same root statistics
same tree-reuse behavior
same episode steps and terminal measures
same config hashes
same handle-release timing
```

All existing Gumbel unit tests, integration tests, task tests, release tests,
tree-reuse tests, and goldens must pass unchanged after the extraction.

If the shared `masked` representation exposes an existing Gumbel masking bug,
fixing that bug requires a separate explicit behavior change and hash/version
decision. Do not silently alter a golden during structural extraction.

## Tests

### Shared MCTS Conformance

Run the same conformance cases through Gumbel and PUCT strategy wrappers where
the algorithm choice should not matter:

```text
root emits Expand then Eval
eval actions contain candidates followed by STOP
wrong token is rejected
wrong result variant is rejected
poll while pending blocks
poll after done fails
node evaluator output is validated
engine-aware leaf evaluators may call E::measure while servicing EvalWork
STOP never emits ApplyWork
rejected actions are masked and retried
no_backtrack actions are masked and retried
within-descent cycles terminate
STOP re-evaluation uses aligned opponent position
tree reuse carries only the selected subtree
full fresh simulation budget runs after reuse
releasable handles appear before final measure
final measurement occurs before episode completion
leaf measurement does not suppress the final MeasureWork
all drop/error/success paths release exactly once
```

### PUCT Math

Focused PUCT tests:

```text
score matches the specified formula
equal logits produce a valid uniform prior
higher prior wins with equal Q and visits
higher Q can dominate a higher prior
lower visit count receives the larger exploration bonus
masked actions are never selected
stable action index resolves exact score ties
one newly expanded leaf consumes one simulation
one expanded node is evaluated once
backup updates every path edge exactly once
single-player backup does not negate values
```

### PUCT Root Target and Election

```text
fresh-root policy target is normalized visit counts
reused-root policy target uses visit deltas
carried visits affect selection but not the new target mass
zero-temperature election uses delta visits, Q, prior, then index
positive-temperature election is deterministic for the same seed
different noise_seed values can change sampled root actions
all-rejected rewrites restore STOP
STOP fallback yields one-hot STOP target
considered_action_indices contains every final unmasked action
root_search_value and root_q_max match their definitions
```

### PUCT Episode

```text
zero-step episode measures root and returns MaxSteps
selected STOP measures the unchanged graph
selected rewrite advances current graph
max_steps terminates after the configured committed moves
episode records legal actions and visit policy targets
search config hash is attached to the episode
tree reuse preserves decisions on a stable fixture
tree reuse reduces repeated expand/eval work on a stable fixture
```

### Backend Independence

PUCT tests use the private deterministic test engine/evaluator in
`crates/gz-search/tests/common/`. Do not add a Whittle dependency to
`gz-search`.

## Performance Guard

This work changes the current Gumbel hot path. Measure before extraction and
after Gumbel parity is restored.

Required report:

```text
baseline measurement
shared-core extraction measurement
absolute and percent delta
benchmark command
hardware/build profile
```

Use the existing release-mode serial Gumbel benchmark as the first guard:

```bash
cargo run --release -p gz-orchestrator --example serial_gumbel_bench
```

Also add or extend a PUCT benchmark only after the correctness suite passes.
Do not claim that generic extraction is zero-cost without measurement.

Hot-path rules:

```text
no Box<dyn MctsStrategy>
no per-selection hash maps
no JSON or strings
no cloning graph bodies
no extra allocation in PUCT score selection
reuse node vectors and descent buffers where current Gumbel code does
```

## Implementation Order

### Phase 0: Baseline

1. Run `cargo test -p gz-search`.
2. Run the relevant orchestrator Gumbel/task/release tests.
3. Record the serial Gumbel release benchmark.
4. Do not edit goldens during baseline collection.

### Phase 1: Shared Tree

1. Add private `mcts/tree.rs` and `mcts/types.rs`.
2. Move common node, edge, backup, action-ref, position, and compaction code.
3. Keep a Gumbel strategy wrapper over the shared tree.
4. Run all Gumbel tests and goldens.

### Phase 2: Shared Root Task

1. Add the private strategy trait.
2. Move common root poll/resume, expansion, eval, apply, STOP, mask, and backup
   behavior into `mcts/task/root.rs`.
3. Move Gumbel run-state/election behavior into `gumbel/strategy.rs` and
   `gumbel/schedule.rs`.
4. Preserve the public `GumbelRootTask` wrapper.
5. Run task, Gumbel, golden, tree-reuse, and release tests.

### Phase 3: Shared Episode Task

1. Move standard root repetition, context construction, tree reuse, final
   measure, and handle partitioning into `mcts/task/episode.rs`.
2. Preserve the public `GumbelEpisodeTask` wrapper.
3. Leave categorical and sampled-tree behavior unchanged.
4. Run all `gz-search` and relevant `gz-orchestrator` tests.

### Phase 4: PUCT Strategy

1. Add `PuctMctsConfig` and validation.
2. Add `PuctStrategy` and PUCT score selection.
3. Add visit-delta tracking, policy target, and root election.
4. Add PUCT public root/task/result types.
5. Add config hashing.
6. Add focused PUCT tests.

### Phase 5: PUCT Episode

1. Add the public PUCT episode wrapper and serial methods.
2. Enable shared tree reuse.
3. Add final measurement and handle-lifecycle tests.
4. Run the complete `gz-search` suite.

### Phase 6: Verification

Run:

```bash
cargo fmt --check
cargo clippy -p gz-search --all-targets --all-features
cargo test -p gz-search
cargo test -p gz-orchestrator
cargo run --release -p gz-orchestrator --example serial_gumbel_bench
```

Report Gumbel benchmark before/after numbers. PUCT orchestrator integration is
not part of this phase.

## Acceptance Criteria

The work described by this spec is complete when:

```text
PUCT root and episode searches are public from gz-search
PUCT accepts any EngineEvaluator<E>
PUCT uses evaluator priors and values exactly as specified
measurement-backed leaf evaluation requires no MCTS or SearchWork variant
compiler leaf evaluation may lower and benchmark behind EngineEvaluator
PUCT policy targets come from fresh visit deltas
PUCT and Gumbel use one shared private tree and standard task engine
the only strategy differences are scheduling, selection, target, and election
all existing Gumbel public APIs still compile
all existing Gumbel tests and goldens pass unchanged
all shared and PUCT tests pass
terminal measurement remains GraphEngine::measure
handle release is exact on success and failure paths
gz-search remains independent of Whittle and future compiler backends
the Gumbel performance before/after delta is measured and reported
```

Follow-up integration work may then make the orchestrator select Gumbel or
PUCT without changing either algorithm's tree/task implementation.
