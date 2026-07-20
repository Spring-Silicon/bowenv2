# Orchestrator

## Scope

`gz-orchestrator` executes search tasks against engines and evaluators. It owns
worker slots, bounded queues, batching, model leases, feature extraction,
admission shaping, replay submission, and handle release. It does not own search
semantics, model code, measurement semantics, or replay validation.

## Drivers

`SerialGumbelOrchestrator` services one task synchronously.
`BatchedGumbelOrchestrator` parks several single-agent tasks and evaluates their
requests in batches. `ThreadedGumbelOrchestrator` runs one engine per lane and is
the production symmetric/replay path.

Every admitted episode receives a deterministic noise seed derived from its
lane-scoped episode ID. Caller-supplied episode context is not accepted because
the orchestrator owns this identity.

## Worker Pool

Each fixed slot is exactly one of idle, running, or parked on one or more evals.
Polling services engine-local Expand/Apply/Measure work immediately. Eval work
is parked with portable request data and an optional extracted row. Resume
validates slot/token ownership before returning output to search.

Slots carry whether admission reserved evaluator pressure. Reservations are
consumed exactly once when the first eval is submitted or released if the task
finishes/errors first.

## Threaded Pipeline

Each lane owns its engine, root source, feature extractor, worker pool, episode
IDs, and created engine handles. Batcher threads own collators and evaluator
backends. Channels are bounded by worker/evaluator capacity.

```text
lane: admit -> poll engine work -> extract row -> send eval -> resume
batcher: group same model generation -> collate -> submit -> route replies
sink: measured symmetric game -> project labels -> atomic paired append
```

Multiple evaluator processes stripe lanes by `lane % process_count`. Each
process may have a different active generation temporarily; an episode leases
the generation current on its assigned route at admission. Batches never mix
generations.

## Symmetric Feature Rows

For each eval the lane extracts the acting graph with legal actions and the
other graph without actions, then attaches the latter as second-board state.
Replay projection later re-extracts both player traces with the same canonical
orientation. Feature extraction failures release task and projection handles
before propagating the error.

## Admission

Admission is work-conserving and may be shaped by evaluator capacity feedback.
It accounts for symmetric work per game, current outstanding/reserved evals,
observed episode work, and wave batching. A fixed stagger remains available but
the adaptive shaper does not impose a hard episode-rate ceiling.

Replay backpressure closes only new admission when
`produced_rows - consumed_rows` exceeds the configured backlog. In-flight games
finish. A gated idle lane sleeps for the configured poll interval instead of
spinning.

## Model Leases

At admission, a featurized replay episode acquires the current evaluator model
generation. Every eval job names that generation. A hot swap can be published
for newer admissions while old games continue on their lease. The batcher
releases an old model only after all episode leases and submitted batches drain.

## Measurement And Replay

Symmetric search measures both terminal graphs before completion. The lane
builds portable artifacts and sends one game to `ReplayMeasurer`. The sink
computes outcomes and appends both perspectives atomically. Projection or append
failure is fatal and cannot be counted as produced replay.

## Handle Ownership

The lane is the sole owner of engine handles created by its tasks and feature
projection. Search returns releasable batches during execution and a complete
created-handle ledger at termination. Error paths call `take_all_handles`.
Release may contain duplicate/deduplicated IDs because engine refcounts, not
handle numeric uniqueness, define ownership.

No handle crosses lane threads; only portable rows/requests do.

## Correctness Requirements

- no slot admitted twice or resumed under the wrong token;
- no unbounded channel in eval/replay paths;
- no mixed model generation inside an episode or eval batch;
- no replay row before terminal measurement;
- no dropped RocksDB/evaluator/feature error;
- no graph/candidate handle leak on normal, abort, channel, or backend errors;
- generated root ownership released once after task completion.
