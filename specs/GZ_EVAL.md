# Evaluation Boundary

## Scope

`gz-eval` defines engine-independent policy/value requests and results. Search
builds requests from portable graph/action identities; orchestrators either
serve them through an engine evaluator or extract fixed-layout features for the
Python service.

## Request

An `EvalRequest` contains:

- one `ReplayGraphContext`;
- actions in deterministic search order;
- root step, leaf depth, budget fraction, and budget step.

Each candidate action carries its portable candidate reference, kind, tags, and
finite static prior. STOP carries only its state context and must occur exactly
once as the last action. Request validation rejects empty actions, context
mismatch, metadata/action mismatch, duplicate/missing/misordered STOP, and
non-finite priors.

Symmetric search attaches the other player's live graph separately as
`EvalOpponentWork`. It is execution work, not part of the portable request. The
orchestrator extracts it into the feature row's second board. No trajectory or
reference-policy context exists.

## Result

`EvalOutput` contains:

- exact `ModelVersion` used;
- one finite policy logit per requested action;
- one finite scalar value.

Batch validation checks row count, action count, and finite values before any
output enters search. The scalar value is from the request's acting-player
perspective and already has its serving activation applied exactly once.

## Evaluator Interfaces

`Evaluator` evaluates portable requests in batches. `EngineEvaluator<E>` is a
synchronous convenience boundary that may inspect engine graphs/candidates.
`RandomValueEvaluator` and `WhittleMeasureEvaluator` support deterministic
tests/examples; production uses featurized process evaluation.

## Position Export

Search always tracks real internal steps and budgets. With position export
disabled, request position fields are zeroed intentionally so the model cannot
infer the clock. This does not alter search seeding, horizon enforcement, or
search config identity.

## Correctness Requirements

- action count/order unchanged from request through logits;
- one model version for every eval in an episode lease;
- no model output used before validation;
- symmetric other-board state corresponds to the same live search node;
- model value perspective matches search backup perspective;
- evaluator errors propagate; no fallback output is synthesized.
