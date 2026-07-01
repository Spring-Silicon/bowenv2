# gz-eval Spec

Status: draft

Purpose: define the evaluation crate used by search when an algorithm needs a
policy over legal search actions and a value estimate for a graph. The first
implementation should be cheap and deterministic. Later implementations can be
neural, batched, and backed by a Python process, without changing the search
tree's policy/value contract.

`gz-eval` exists because Gumbel-MCTS needs leaf values and root/action priors,
but those estimates are not engine measurements. `GraphEngine::measure` remains
the source of replay-eligible runtime reward.

## Role

`gz-eval` answers:

```text
Given a graph and an ordered legal action list, what policy logits and value
estimate should search use?
```

It owns:

```text
policy/value evaluator traits
eval request and output records
eval output validation
cheap deterministic evaluator implementations
model/evaluator version tagging
future adapters for feature-backed or process-backed evaluators
```

It does not own:

```text
search trees
candidate enumeration
STOP insertion
candidate application
terminal measurement
replay storage
training
async actor scheduling
unbounded queues
Python or torch in the default crate build
```

## Dependency Contract

Default allowed:

```text
std
gz-engine
blake3 if needed for deterministic model/evaluator version ids
```

Allowed later behind explicit features or adapter crates:

```text
gz-features for feature-backed evaluators
tokio or another runtime for a process/client evaluator
serde for request/output serialization
Python process protocol clients
```

Forbidden in the default crate:

```text
torch/Python bindings
rocksdb
gz-engine-whittle
future concrete compiler adapters
gz-replay
gz-orchestrator
trainer code
```

`gz-eval` must be usable by a serial, synchronous search implementation. The
core evaluator trait is therefore blocking. A blocking call may wait on a model,
process, or queue in future adapters, but the first implementation should not
add async runtime requirements.

## Evaluation Contract

An eval request is action-aligned.

```text
request.context is the portable identity of that graph
request.actions is the exact legal action order search will use
policy_logits[i] scores request.actions[i]
value estimates the graph's expected scalar outcome for search backup
```

Search owns legal action construction:

```text
enumerate GraphEngine candidates
wrap them as candidate actions
append STOP as the final search action
pass that ordered action list to gz-eval
```

`gz-eval` must never invent actions. It may score only the action list supplied
by search.

## Public API Draft

The first API should be batch-first with a single-row convenience wrapper.

```rust
pub trait Evaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()>;

    fn evaluate_one(
        &mut self,
        request: &EvalRequest,
    ) -> EvalResult<EvalOutput> {
        // wrapper around evaluate_batch
    }
}
```

The production-facing evaluator API does not receive `&E` or `E::Graph`.
Cross-worker eval batching cannot assume a shared `GraphEngine`, and engine
local graph handles are worker-local. Feature-backed evaluators should consume
portable contexts and feature rows once `gz-features` exists.

Request records:

```rust
pub struct EvalRequest {
    pub context: ReplayGraphContext,
    pub actions: Vec<EvalAction>,
}

pub struct EvalAction {
    pub action_ref: PortableSearchActionRef,
    pub metadata: EvalActionMetadata,
}

pub enum EvalActionMetadata {
    Candidate {
        kind: CandidateKindId,
        tags: CandidateTags,
        static_prior: f32,
    },
    Stop,
}
```

Output records:

```rust
pub struct EvalOutput {
    pub model_version: ModelVersion,
    pub policy_logits: Vec<f32>,
    pub value: f32,
}
```

Errors:

```rust
pub type EvalResult<T> = Result<T, EvalError>;

pub enum EvalError {
    InvalidRequest,
    InvalidOutput,
    BackendUnavailable,
    Internal,
}
```

Keep `EvalError` small. Do not mirror every possible backend error until a real
backend needs it.

## Request Rules

Every request must satisfy:

```text
actions is non-empty
actions contains exactly one STOP action
STOP is last for search-generated requests
every action_ref context matches request.context
every action_ref variant matches its EvalActionMetadata variant
candidate static_prior values are finite
```

STOP action metadata:

```text
metadata = EvalActionMetadata::Stop
action_ref = PortableSearchActionRef::stop(request.context)
```

Candidate action metadata:

```text
metadata carries CandidateInfo kind/tags/static_prior
action_ref is the PortableCandidateRef built from CandidateInfo
```

`EvalRequest` does not store `E::Candidate`. Search keeps local candidate
handles and maps eval outputs by action index.

## Output Rules

Every output must satisfy:

```text
one EvalOutput per EvalRequest
policy_logits.len() == request.actions.len()
all policy logits are finite
value is finite
model_version identifies the evaluator/model that produced the row
```

Policy logits are unnormalized. Search chooses how to apply softmax, Gumbel
noise, PUCT, sequential halving, or another selection rule.

Illegal actions are excluded from the request rather than masked with
non-finite logits. Do not use NaN or infinity as masking values.

Value is a scalar estimate in the same orientation as `MeasureResult` reward:
higher is better. It is not required to be measured, calibrated, or
replay-eligible.

## Measurement Boundary

`gz-eval` predictions are not measurements.

Rules:

```text
Evaluator::evaluate_batch has no GraphEngine access and must not measure.
Search may back up EvalOutput.value inside MCTS.
Replay rows still require GraphEngine::measure for the recorded graph.
Measured reward and eval value must remain separate fields.
```

This prevents cheap/random/neural values from being mistaken for runtime reward.

## Blocking And Batching

The core API is blocking and batch-first.

Rules:

```text
evaluate_batch may block the current worker
evaluate_batch writes outputs in request order
evaluate_batch clears out before appending outputs
evaluate_one is a convenience wrapper only
no async runtime is required for the default crate
no unbounded channels in eval paths
```

Serial Gumbel-MCTS can call `evaluate_one` or small `evaluate_batch` directly.
The future async orchestrator can batch leaf requests across many workers and
call the same evaluator boundary or a process-backed adapter.

Do not call Python one leaf at a time. A Python-backed evaluator must batch
requests before crossing the process boundary.

## First Evaluator

Implement `RandomValueEvaluator` first.

Behavior:

```text
policy_logits = vec![0.0; action_count]
value = deterministic pseudo-random value derived from seed + graph context
model_version = stable hash of evaluator name, seed, and value range
```

Config:

```rust
pub struct RandomValueEvaluatorConfig {
    pub seed: u64,
    pub value_min: f32,
    pub value_max: f32,
}
```

Default config:

```text
seed = 0
value_min = -1.0
value_max = 1.0
```

Design rules:

```text
batch grouping must not change outputs
request order must not change a row's value
the evaluator must not keep stateful RNG progress
the evaluator ignores graph bodies and candidate semantics
uniform policy is represented by zero logits, not random logits
```

Uniform policy plus root Gumbel noise is enough to exercise Gumbel-MCTS action
selection. Add non-uniform random logits later only if a test or benchmark needs
them.

## Future Evaluators

Likely later implementations:

```text
ConstantEvaluator for deterministic unit tests
RecordedEvaluator for golden policy/value fixtures
FeatureEvaluator once gz-features exists
ProcessEvaluator or PythonProcessEvaluator for neural inference
```

Feature-backed evaluators must make feature schema/version compatibility
explicit:

```text
EngineVersion
ActionSetHash
FeatureSchemaHash
ModelVersion
```

If these tags disagree, the evaluator must fail fast rather than producing
policy/value outputs with ambiguous semantics.

## Cache Rules

Do not add an eval cache in the first implementation.

If a future evaluator caches outputs, the key must include at least:

```text
PortableGraphId
ActionSetHash
ordered PortableSearchActionRef list
EngineVersion
ModelVersion
FeatureSchemaHash when features are used
evaluator config hash when no model version uniquely identifies config
```

Never cache by engine-local graph handle alone.

## Crate Shape

Initial layout:

```text
crates/gz-eval/
  Cargo.toml
  src/
    error.rs
    lib.rs
    random.rs
    types.rs
  tests/
    random.rs
    validation.rs
```

Keep modules flat. Do not add client/server/process modules until there is a
real backend.

## Test Strategy

Use small local test graph handles and portable refs. Do not depend on
`gz-engine-whittle` for `gz-eval` unit tests.

Required tests:

```text
request validation accepts candidates followed by STOP
request validation rejects empty action lists
request validation rejects missing STOP
request validation rejects duplicate STOP
request validation rejects action context mismatches
request validation rejects action kind mismatches
output validation rejects wrong policy_logits length
output validation rejects NaN/Inf logits
output validation rejects NaN/Inf value
RandomValueEvaluator returns one output per request
RandomValueEvaluator policy logits are uniform zeros
RandomValueEvaluator value is within configured range
RandomValueEvaluator is deterministic for the same graph/config
RandomValueEvaluator output is independent of batch order
RandomValueEvaluator model_version changes when config changes
```

## Implementation Plan

1. Add `crates/gz-eval` to the workspace with dependencies on `gz-engine` and
   `blake3`.
2. Implement `EvalAction`, `EvalActionMetadata`, `EvalRequest`, `EvalOutput`,
   `Evaluator`, `EvalError`, and validation helpers.
3. Implement `RandomValueEvaluatorConfig` validation.
4. Implement `RandomValueEvaluator` with uniform logits and context-derived
   deterministic pseudo-random values.
5. Add focused validation and random evaluator tests.
6. Run `cargo fmt`, `cargo test --all`, and
   `cargo clippy --all-targets --all-features -- -D warnings`.

## Deferred

```text
gz-features integration
feature schema hashes
recorded fixture evaluator
process-backed evaluator
Python protocol
async eval batching service
model checkpoint reload
reward head
eval output cache
```
