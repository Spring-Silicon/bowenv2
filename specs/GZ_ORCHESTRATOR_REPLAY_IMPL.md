# Replay Integration Implementation Spec

Status: implementation work order

Purpose: close the first vertical slice. Implement modular reference
providers, episode-to-row projection, a replay sink in the threaded driver,
a minimal admission backpressure gate, and a thin `gz-cli` selfplay command.
After this work order, `graphzero selfplay` runs multi-worker Whittle
selfplay end to end: episodes -> outcome labels -> measured rows -> RocksDB
replay -> sampled by a consumer, with backpressure throttling admission.

Authority: `GZ_ORCHESTRATOR.md` and `GZ_REPLAY.md` own the design contracts.
This document is the ordered work plan and includes two explicit contract
amendments (stage 1 and stage 6). If anything else disagrees, the contracts
win; report the conflict instead of improvising.

Read before starting:

```text
specs/GZ_REPLAY.md                        (row/label model, projection contract)
specs/GZ_ORCHESTRATOR.md                  (driver design)
crates/gz-orchestrator/src/lanes.rs       (threaded driver being extended)
crates/gz-orchestrator/src/pool.rs        (worker pool being extended)
crates/gz-replay/src/records.rs           (schema being extended)
crates/gz-search/src/greedy.rs, beam.rs, random.rs (reference kernels)
```

## Design Summary

"Opponent/reference" is two jobs. This work order implements job 1 only:

```text
job 1 (now): label reference. A competing optimization run on the same root
produces one measured final reward; the episode label is
sign(learner_reward - reference_reward).
job 2 (later): eval conditioning. The reference trajectory conditions the
neural value head. The Reference record carries the trajectory now so job 2
is a consumer change, not a schema change — but nothing registers or uses
trajectories in this work order.
```

References are modular behind one trait. v1 providers: root baseline,
greedy, beam, random — all synchronous on the lane, using the lane's engine.
A future Gumbel/frozen-checkpoint provider is an ordinary episode run by the
worker pool; the `Reference` record shape is deliberately a projection of an
episode so that lands later as a scheduling change, not a boundary change.

Cost rule to preserve in doc comments: the measured greedy/beam/random
kernels measure candidate successors while searching. They are valid
reference providers only for cheap-measure engines (Whittle). Compiler-regime
references must measure once (policy rollout, Gumbel) — the trait must not
assume measurement is cheap.

## Hard Constraints

```text
Stage order below; every stage compiles and passes cargo test --all before
the next starts.
Stage 0 commits the current working tree first. Commit per stage after.
gz-search and gz-engine must not change. gz-search goldens must pass
untouched at every stage.
gz-eval, gz-eval-whittle, gz-engine-whittle must not change.
New dependency edges allowed: gz-orchestrator -> gz-replay (default, stage 6
amends the contract), gz-cli -> workspace crates. Nothing else. No clap, no
anyhow, no rand: gz-cli parses args with std only.
Threading stays std-only. All channels bounded. Only portable data crosses
threads: projected replay records are fully portable; Reference values stay
on their lane.
References are computed after admission and before the pool drives any
task (job 2 ordering, free now).
Fail-fast error policy throughout, matching the threaded driver.
Every stage ends with: cargo fmt, cargo test --all,
cargo clippy --all-targets --all-features -- -D warnings.
```

## Stage 0: Commit

The working tree holds the uncommitted gz-replay crate, the legal_actions
change, and spec/docs edits. Commit it (one commit is fine) before starting.

## Stage 1: ReplayReference Schema Extension (gz-replay)

A ±1 label is meaningless without knowing what it was measured against, and
mixed reference pools per run are expected later. Extend the schema now,
while nothing durable exists.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ReplayReferenceKind {
    RootBaseline,
    Greedy,
    Beam,
    Random,
    Gumbel,
}

pub struct ReplayReference {
    pub kind: ReplayReferenceKind,
    pub reward: f32,
    pub final_graph: Option<ReplayGraphContext>,
    pub trajectory_id: Option<u64>,
    pub search_config_hash: Option<SearchConfigHash>,
    pub model_version: Option<ModelVersion>,
}
```

Rules:

```text
OpponentFinal is removed; Gumbel replaces it (unused until the neural
opponent exists).
RootBaseline has search_config_hash = None; algorithmic references carry
their kernel's SearchConfigHash.
Bump SCHEMA_VERSION to 2. The postcard layout changes; a v1 store must fail
to open with SchemaMismatch (existing test already proves the mechanism).
Update GZ_REPLAY.md's records section and note the version bump.
Extend record validation only where cheap: reward finite (exists), nothing
kind-specific.
```

## Stage 2: Reference Providers (gz-orchestrator)

New module `crates/gz-orchestrator/src/reference.rs`, public.

```rust
pub trait ReferenceProvider<E: GraphEngine> {
    fn reference(
        &mut self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<Option<Reference<E::Graph>>>;
}

pub struct Reference<G> {
    pub kind: ReplayReferenceKind,
    pub final_reward: f32,
    pub final_graph: ReplayGraphContext,
    pub steps: Vec<ReferenceStep<G>>,
    pub search_config_hash: Option<SearchConfigHash>,
    pub model_version: Option<ModelVersion>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReferenceStep<G> {
    pub graph: G,
    pub context: ReplayGraphContext,
}
```

Rules:

```text
Ok(None) means no usable reference (final measurement unscoreable): the
episode proceeds and its rows get value_target = None. EngineError aborts.
final_reward comes from a measurement that is measured && valid with a
finite scalar reward — the same scoreable rule gz-search uses.
steps[0] is the root state; steps.last() is the final state. steps.len() is
the future row_count for job 2. Graph handles in steps are lane-local and
must never leave the lane.
Providers are deterministic for a fixed config, engine config, and root.
model_version is None for all v1 providers.
```

Providers:

```text
RootBaselineProvider { measure_options }
  measures the root once; steps = [root]; final_reward = root reward;
  kind RootBaseline; search_config_hash None
GreedyReferenceProvider { search: GreedySearch }
  runs GreedySearch::run(engine, root); steps from the episode's selected
  path (root, then each step's after state, using the recorded step
  contexts); final_reward from episode.final_measure; hash from
  episode.search_config_hash; kind Greedy
BeamReferenceProvider { search: BeamSearch }     same shape, kind Beam
RandomReferenceProvider { search: RandomSearch } same shape, kind Random
```

These call the engine and kernels directly — they are driver-side code, not
search tasks. No extra measure calls beyond what the kernels already do.

Stage 2 tests (`tests/reference.rs`, Whittle dev-deps):

```text
root baseline reward equals engine.measure(root) reward; steps == [root]
greedy provider matches a direct GreedySearch::run on a fresh identical
engine: same final reward, same step contexts, same config hash
same for beam and random
provider is deterministic across two calls on fresh engines
unscoreable final measurement returns Ok(None)   (scripted local engine or
whittle config that cannot score, whichever is simpler)
```

## Stage 3: Projection (gz-orchestrator)

New module `crates/gz-orchestrator/src/project.rs`, public.

```rust
pub fn project_episode<G, C>(
    episode: &GumbelEpisode<G, C>,
    reference: Option<&Reference<G>>,
) -> Option<(ReplayEpisodeRecord, Vec<ReplayRow>)>;
```

Rules:

```text
Returns None iff the episode is not replay-eligible (final measure not
measured/valid/finite). Callers count these as dropped.
learner_reward = episode final scalar reward.
value_target = sign(learner_reward - reference.final_reward) in
{-1.0, 0.0, +1.0}; None when reference is None.
reward_target = Some(learner_reward) on every row.
One ReplayRow per GumbelStep: state = step_ref.before, selected_action,
legal_actions/policy_target cloned from the step, action_history
accumulated from prior selected actions, model_version = step's version,
final_measure = MeasureSummary::from(&episode.final_measure),
search_config_hash = episode's.
ReplayReference is built from Reference: kind, reward, final_graph,
search_config_hash, model_version; trajectory_id = None in this work order.
Projection performs no validation of its own beyond eligibility; the store's
validation is the oracle. A projection bug must surface as a store
rejection, not silent data.
```

Stage 3 tests (`tests/project.rs`):

```text
projected episode appends successfully to a temp ReplayStore (the store's
strict cross-validation passing IS the correctness assertion)
label matches the sign rule for win, loss, and tie references
reference = None yields value_target = None rows that still append
ineligible episode projects to None
row count equals step count; a STOP-terminated episode's last row is the
STOP decision state
```

Build test episodes by running the serial orchestrator on Whittle rather
than hand-building GumbelEpisode values.

## Stage 4: Sink, Gate, And Threaded Integration (gz-orchestrator)

### Pool change

`WorkerPool::admit` returns the admissions so lanes can attach references:

```rust
pub(crate) fn admit<E, R>(...) -> EngineResult<(Vec<(EpisodeId, G)>, bool)>
```

(admitted pairs in admission order, plus the existing roots_exhausted flag).
Existing callers adapt trivially.

### Threaded entry point

```rust
pub struct ReplayRuntime<'a, P> {
    pub store: &'a ReplayStore,
    pub providers: Vec<P>,               // one per lane
    pub backpressure: Option<ReplayBackpressure>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayBackpressure {
    pub max_row_backlog: NonZeroU64,
    pub gate_poll: Duration,             // default 1ms
}

impl<E, V> ThreadedGumbelOrchestrator<E, V> {
    pub fn run_with_replay<R, P>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
        replay: ReplayRuntime<'_, P>,
    ) -> EngineResult<ThreadedReplayRun<E::Graph, E::Candidate>>
    where
        R: RootSource<E> + Send,
        P: ReferenceProvider<E> + Send;
}

pub struct ThreadedReplayRun<G, C> {
    pub run: ThreadedRun<G, C>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,           // ineligible or reference-less-and-configured-to-drop? No: ineligible only
}
```

`providers.len()` must equal lanes (`internal("lane count mismatch")`).
`run` (without replay) keeps its exact current behavior and tests.

### Lane loop changes

```text
admit -> for each admitted (episode_id, root), in order:
  provider.reference(engine, root) -> store in a lane-local
  HashMap<EpisodeId, Option<Reference<G>>>
  (references exist before any task is driven)
drive -> for each completed OrchestratedEpisode:
  take the reference out of the map
  project_episode(&episode, reference.as_ref())
    Some -> send (ReplayEpisodeRecord, Vec<ReplayRow>) to the sink channel
    None -> count dropped
admission gate (only when backpressure is configured):
  before admitting, read store counters; if
  produced_rows - consumed_rows > max_row_backlog, skip admission this
  iteration; if the pool is also completely idle, sleep gate_poll before
  re-checking so a gated-idle lane does not busy-spin
```

Gate rules:

```text
The gate limits admission only; in-flight episodes always complete, so the
backlog can overshoot by up to (total workers x rows per episode). Document
this bound where the gate is implemented.
gate_poll sleeping is a throttled-idle path, not a hot path; this is the
one permitted sleep in the drivers.
```

### Sink

```text
one sink thread inside the existing scope, owning nothing but &ReplayStore
bounded channel lanes -> sink, capacity = lanes * workers_per_lane
message: (ReplayEpisodeRecord, Vec<ReplayRow>) — fully portable, no engine
generics (same structural rule as EvalJob)
sink appends via store.append_episode; counts episodes_appended
sink errors are fail-fast: a ReplayError maps to
internal("replay sink failed") and aborts the run (dropping the sink
receiver unblocks lanes into the same error, mirroring the eval batcher)
termination: lanes drop their sink senders when done; sink exits on
disconnect; scope joins everything
```

Stage 4 tests (`tests/replay_integration.rs`, Whittle + RandomValueEvaluator):

```text
two lanes with RootBaselineProvider: every eligible episode lands in the
store; episodes_appended + episodes_dropped == total episodes;
counters().produced_rows == sum of appended row counts
store contents are deterministic: two identical runs produce stores whose
episodes (fetched by id) are equal
greedy reference end to end: labels are only -1/0/+1, and at least one
non-None label exists
backpressure: tiny max_row_backlog, consumer thread sampling on an
interval; the run completes, and produced - consumed stayed <= backlog +
the documented overshoot bound at every consumer observation
sink failure aborts: drop/close the store dir mid-run is awkward — instead
inject failure by filling value_target with an invalid value via a
misbehaving provider? Not reachable through the public API; skip fault
injection here and rely on the unit-level mapping test:
internal("replay sink failed") is produced when append_episode errors
(test the mapping function directly)
```

## Stage 5: gz-cli

New crate `crates/gz-cli`, binary name `graphzero`, added to the workspace.

```text
crates/gz-cli/
  Cargo.toml    deps: gz-engine, gz-engine-whittle, gz-eval, gz-search,
                gz-orchestrator, gz-replay (binaries may compose concrete
                crates; the library crates stay engine-neutral)
  src/
    main.rs     thin: parse args, call run, print summary, exit code
    selfplay.rs the command implementation as a testable function
```

Command:

```bash
graphzero selfplay --replay-dir PATH [--episodes N] [--lanes L]
  [--workers-per-lane W] [--reference root|greedy|beam|random|none]
  [--seed S] [--max-steps M] [--simulations K] [--max-batch B]
```

Rules:

```text
std-only flag parsing (--flag value pairs); unknown flags are errors with a
usage message; no clap until commands multiply.
Defaults: episodes 16, lanes 2, workers-per-lane 8, reference root, seed 0,
max-steps 8, simulations 8, max-batch = workers total.
Engine is Whittle (generator roots from the seed, one generator per lane
with distinct derived seeds); evaluator is RandomValueEvaluator until the
neural path exists.
reference none runs without labels (value_target = None rows).
Prints a short summary: episodes appended/dropped, rows produced, win/loss/
tie tally from the labels, eval batch count and mean size, store counters.
Non-zero exit on any error, message to stderr.
selfplay.rs exposes run(config) -> Result<Summary, ...> so the integration
test drives it without spawning a process.
```

Stage 5 test: one smoke test calling the run function with a temp replay
dir and tiny settings; asserts the summary matches the store's counters.

## Stage 6: Contract Amendments And Docs

```text
GZ_ORCHESTRATOR.md: move gz-replay from "allowed later behind explicit
features" to default allowed, with one sentence of reason (the replay sink
is core, not optional). Update the Role list: replay sink driving and
ratio/backpressure gating are no longer "later".
GZ_REPLAY.md: already updated in stage 1; verify the projection contract
section matches what stage 3 built.
CODEBASE_OUTLINE.md: add gz-cli to the crate list section with the selfplay
command; workspace members += gz-cli (done in stage 5).
AGENTS.md: no new specs to list beyond this file.
```

## Final Verification

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
target/debug/graphzero selfplay --replay-dir /tmp/gz-smoke --episodes 8
```

Acceptance checklist:

```text
gz-search and gz-engine untouched; goldens pass unchanged
first-vertical-slice list from CODEBASE_OUTLINE holds: workers generate
episodes, only measured episodes enter replay, a consumer samples rows,
backpressure throttles admission, everything drains on completion
labels in the store are only -1/0/+1 and every labeled episode's reference
identity (kind + config hash) is recorded
Reference values never cross a lane boundary; sink messages have no engine
generics
existing threaded run() behavior and tests unchanged
the one sleep in production code is the gated-idle poll, documented
schema version is 2 and a v1 store fails to open
```

## Out Of Scope

```text
eval conditioning on reference trajectories (job 2): no trajectory tables,
no eval-backend leases, no EvalPositionContext changes
Gumbel/frozen-checkpoint reference provider
per-episode GumbelEpisodeContext
mixed per-episode reference pools / curricula
a real ratio controller (the gate is a fixed backlog cap)
retention/eviction in replay
Python trainer service
additional CLI commands
```
