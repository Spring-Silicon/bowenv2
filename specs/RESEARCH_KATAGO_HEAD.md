# KataGo-Style Auxiliary Value And Score Heads Research Spec

Status: implemented behind the `v8-v32-score` model layout and trainer task
weights; the controlled experiments have not been run

Purpose: test whether multiple auxiliary prediction heads improve GraphZero's
shared representation and main terminal-outcome value head. Two auxiliary
value targets are formed from future MCTS root values; a third auxiliary head
predicts the actor's actual terminal Whittle graph size. The motivating failure
mode is high-variance supervision at early symmetric-selfplay states: every row
currently receives the final measured result even when much of that result was
determined by later Gumbel exploration. The auxiliary tasks provide local and
graded supervision for implicit regularization and representation learning
without changing the search objective or requiring intermediate graph
measurements.

Provenance: KataGo's short-term value and score targets, used since its g170
run. This spec adopts the exponentially averaged future-MCTS-value idea, not
KataGo's Go-specific horizons, score semantics, or search implementation.

Authority: `GZ_SYMMETRIC_SELFPLAY.md` owns the symmetric game and perspective
contract; `GZ_REPLAY.md` owns measured-before-replay and durable row semantics;
`GZ_MODEL.md` and `GZ_TRAINER.md` own the model and loss contracts. Those
contracts win if this research proposal conflicts with them.

Read before implementing:

```text
crates/gz-search/src/gumbel/types.rs       GumbelStep::root_search_value
crates/gz-search/src/gumbel/symmetric/    canonical player perspectives
crates/gz-orchestrator/src/{project,lanes}.rs
                                            episode -> measurer projection
crates/gz-measurer/src/{project,service}.rs
                                            terminal labels and symmetric game
crates/gz-replay/src/records.rs           durable row schema
python/gz/codec/targets.py                target wire layout
python/gz/model/exphormer.py              value tower and serving output
python/gz/trainer/{data,diagnostics,loop,driver}.py
                                            staging, losses, telemetry, config
```

## Research Hypothesis

The main value target remains the measured terminal result `z`. It is the
right optimization objective, but it is a high-variance label for early rows:
two similar states can receive opposite labels because of exploration and
mistakes many decisions later.

At every played decision, MCTS already produces a search-improved root value.
Exponential averages of the current and future root values should provide
lower-variance targets at two distinct timescales. They are biased toward the
current model and search, but more local than the terminal result.

The terminal win/loss target also discards magnitude: terminal Whittle graphs
of 40 and 100 nodes can receive the same sign label. Predicting the actor's
actual terminal compact node count supplies a dense, measured target about
optimization progress. Training the shared graph representation on the final
result, two future-value horizons, and terminal size may make the main head
learn sooner and generalize better.

This is an auxiliary-learning hypothesis. Lower auxiliary loss alone is not a
success; the experiment succeeds only if the unchanged main head or downstream
search improves.

## Non-Negotiable Runtime Contract

No auxiliary target may call `GraphEngine::measure`, lower a graph, benchmark a
graph, or launch another MCTS search.

```text
normal selfplay:
  search s_0; retain root_search_value[0]
  search s_1; retain root_search_value[1]
  ...
  perform the mode's existing terminal measurement(s)
  derive the terminal result z

target projection:
  make one backward arithmetic pass over the retained values
  derive terminal graph size from the existing terminal scalar reward
  append rows only after the existing measurement contract is satisfied
```

The target pass is `O(heads * episode_rows)` scalar arithmetic. It neither
changes nor increases the engine measurement count. If a future configuration
chooses an expensive leaf evaluator, these targets reuse the evaluations that
normal search already performed; they never request additional evaluations.

## Head Contract

The intended architecture has four scalar predictions:

```text
V_final     existing main head; predicts the final measured result
V8          new auxiliary head; mean future offset of 8 actor decisions
V32         new auxiliary head; mean future offset of 32 actor decisions
S_final     new auxiliary head; predicts the actor's terminal graph size
```

`V_final` is the only prediction consumed by the evaluator and search:

```text
leaf evaluation -> V_final -> normal alternating-perspective MCTS backup

V8       -- training only
V32      -- training only
S_final  -- training only
```

The auxiliary outputs are not averaged, voted, or otherwise ensembled into a
leaf value. Their targets have different semantics, so blending them would
change the meaning of the backed-up value and create a moving feedback loop in
which search is partly driven by heads trained to predict that search. Changing
this rule requires a separate search experiment and spec.

The evaluator result protocol remains scalar and unchanged. Serving need not
execute the auxiliary output projections. The trainer obtains all four
predictions from one shared graph-trunk pass; it must not run the graph trunk
once per head.

## Target Definition

For one canonical actor trace with `T` stored decisions, define:

```text
m_t = GumbelStep::root_search_value at decision t, in that actor's perspective
z   = the trace's measured terminal value target
H   = nominal mean future offset in actor decisions
lambda_H = H / (H + 1)
```

The finite-episode target is:

```text
y[T] = z
y[t] = (1 - lambda_H) * m_t + lambda_H * y[t + 1]
       for t = T-1, ..., 0
```

Equivalently:

```text
y[t] = (1 - lambda_H) * sum(
         lambda_H^k * m[t + k], k = 0 .. T-t-1
       )
       + lambda_H^(T-t) * z
```

The terminal tail makes the weights sum to one and anchors every truncated
trajectory to the real outcome. The proposed horizons are:

```text
V8:   H = 8,  lambda = 8/9   ~= 0.8888889
V32:  H = 32, lambda = 32/33 ~= 0.9696970
```

For the untruncated geometric weights, the expected future offset is
`lambda / (1 - lambda) = H`. `H` counts decisions in one stored actor trace,
not global alternating plies. This matches symmetric replay, which projects
one trace per canonical player. At the current per-player `max_steps = 96`, V8
is local while V32 covers a materially longer portion of the rewrite sequence
without duplicating the final-outcome target.

Target invariants:

```text
m_t and z must be finite and in [-1, 1]
y_t is finite and in [-1, 1] without clipping
lambda = 0 would reproduce the current root search value
if every m_t equals z, every y_t equals z
negating every m_t and z negates every y_t
```

Out-of-range values are producer errors; projection must reject them rather
than clamp them and hide a search bug.

## Symmetric Perspective Semantics

Compute targets independently for the two projected actor traces:

```text
P1 trace: its own canonical root-search values, terminal boundary z
P2 trace: its own canonical root-search values, terminal boundary -z
```

Never average raw P1 and P2 values together. Each `root_search_value` is
already expressed from the actor-to-move's canonical perspective. The target
pass preserves that perspective exactly.

The two traces need not produce exact pointwise negatives: they contain
different decision states and independent search evidence. The required
antisymmetry property is algebraic: if a trace and its terminal target are
explicitly mirrored and negated, its derived targets must negate exactly.

With tree reuse enabled, `m_t` is the root search value produced from the
aggregate carried-plus-new statistics used by that decision. That is the
search belief the method intends to capture. Reuse makes nearby targets more
correlated and possibly staler, so reuse-on and reuse-off diagnostics must be
reported separately.

## Terminal Score Semantics

For this Whittle research mode, `S_final` predicts the canonical actor's own
terminal compact graph node count:

```text
c_actor = -final_measure.scalar_reward
```

The Whittle adapter defines `scalar_reward = -(compact node count)`, so this is
the actual measured ending size, not a proxy and not the P1-P2 margin. P1 rows
target P1's final graph size; P2 rows target P2's final graph size. Unlike the
competitive value target, the score target is positive and is not negated when
the player perspective changes.

The semantic prediction is in node-count units. Optimize it on a bounded scale:

```text
score_scale = FeatureSchemaConfig.max_nodes
score_target_normalized = c_actor / score_scale
score_prediction_normalized = sigmoid(S_final_raw)
predicted_terminal_nodes = score_scale * score_prediction_normalized
```

Projection must verify that `c_actor` is finite, integral, and in
`[0, score_scale]`. Training uses MSE on the normalized values; diagnostics
report MAE and bias after decoding back to nodes. Normalization is purely for
loss conditioning and does not change what the head predicts.

This score definition is intentionally Whittle-specific at the research
configuration boundary. `gz-search` and replay do not learn node-count
semantics. A future compiler experiment must explicitly define its own terminal
score transform and scale before enabling this head.

## Data Flow And Storage

Current search already retains `root_search_value`; the orchestrator currently
drops it when constructing `CompletedEpisodeStep`. The implementation path is:

```text
GumbelStep.root_search_value
  -> CompletedEpisodeStep.root_search_value
  -> measurer projection while the complete actor trace and z are available
  -> ReplayRow.horizon_value_targets
  -> GZFT target response
  -> TrainingBatch

existing terminal MeasureResult.scalar_reward
  -> ReplayRow.reward_target
  -> existing GZFT reward target
  -> S_final target transform in the trainer
```

Projection, not `gz-search` and not the Python trainer, owns horizon-target
construction:

- Search owns the estimate but knows no terminal measurement.
- The measurer projection has the complete ordered trace and the admitted
  terminal target.
- Replay sampling is row-random and must not perform episode joins or future
  row reads in the training hot path.

Add a fixed-width optional row field:

```rust
pub horizon_value_targets: Option<[f32; 2]>; // [V8, V32]
```

Both targets are computed and stored together. All experiment arms consume
identical replay; loss weights alone determine whether a head trains. Rows
outside the initial symmetric mode carry `None`.

The stored replay twin mirrors the field. Opening a store with the new layout
requires the repository's normal replay schema/version boundary; silently
decoding old rows as zero targets is forbidden.

The target wire adds:

```text
horizon_value       f32[batch_capacity, 2]
horizon_value_valid u8[batch_capacity]
```

Padding and `None` rows have a zero validity mask. The two targets share one
mask because projection creates them atomically. The target encoding version
must advance. Under the current shared row/target `ENCODING_VERSION` contract,
that also creates a new `FeatureSchemaHash` and replay run boundary even though
the GZFR feature-row layout itself is unchanged. The evaluator's independent
GZFB/GZFO batch protocol does not change.

`S_final` requires no new replay or wire field: `reward_target` and the GZFT
`reward` section already carry the actor's measured terminal scalar reward on
every row. The trainer applies the declared Whittle transform `c = -reward` and
normalizes it by `max_nodes`. This reuse is valid only because row admission
already requires the terminal measurement.

Raw `root_search_value` is not stored durably after projection in this first
experiment. The two precomputed targets are sufficient for all declared
experiment arms, keep row sampling join-free, and avoid adding a research-only
scalar to every production row. Changing horizons therefore requires new
replay generation, but still never requires intermediate engine measurements.

## Model Shape

For the scalar/tanh value configuration used by symmetric selfplay, keep the
existing main value MLP and add independent small auxiliary MLPs over the same
trunk readout/value input:

```text
main_raw      = value_main(value_input)          # scalar
horizon_raw   = horizon_value(value_input)       # [V8, V32]
score_raw     = terminal_score(value_input)      # scalar
```

Apply the configured tanh value activation independently to `main_raw` and both
horizon outputs. Apply sigmoid only to `score_raw`. `value_only` and the normal
model forward retain their existing main scalar contract for serving/search. A
trainer-only path returns all four predictions from the already-computed trunk
readout; it must not encode the graph again for an auxiliary head.

The checkpoint manifest records the auxiliary-head layout. Every control and
treatment model in the research comparison contains all three auxiliary
scalar outputs; zero loss weights disable supervision in the control. This
matches parameter count and initialization while isolating the auxiliary
tasks.

Initial scope requires:

```text
training_mode = "symmetric-selfplay"
value_head = "scalar"
value_activation = "tanh"
value_reward = "sign"
```

HL-Gauss, logit/BCE, single-vanilla, opponent-reference modes, and non-Whittle
score transforms are not part of the first experiment.

## Loss Contract

The three value predictions use the existing masked tanh-value MSE. The score
head uses MSE on normalized terminal node count:

```text
L_tasks = w_final * mse(V_final, z)
        + w_v8    * mse(V8, y_H8)
        + w_v32   * mse(V32, y_H32)
        + w_score * mse(sigmoid(S_final_raw), terminal_nodes / max_nodes)

w_final + w_v8 + w_v32 + w_score = 1
```

The existing top-level `value_weight` multiplies `L_tasks` once. The existing
`value_trunk_grad_scale` scales the combined value gradient into the shared
graph trunk once, before the shared readout fans out to the heads; it must not
be applied independently per head. Keeping the weights normalized prevents the
experiment from silently increasing the total configured representation-loss
budget.

Starting research weights, not production defaults:

```text
                         final   V8    V32   score
main-only control:       [1.00, 0.00, 0.00, 0.00]
multi-horizon ablation:  [0.60, 0.20, 0.20, 0.00]
full auxiliary design:   [0.50, 0.20, 0.20, 0.10]
```

Loss weights belong to trainer configuration and checkpoint/run metadata.
Horizon definitions belong to the replay data contract; a trainer must reject
replay whose horizon metadata does not match the expected `[8, 32]` ordering.

## Implementation Stages

### Stage 1: Target production and replay

```text
carry root_search_value into CompletedEpisodeStep
derive V8 and V32 targets during measured symmetric projection
add the optional fixed-width targets to ReplayRow and its storage twin
advance replay and GZFT target encodings
stage targets and validity on the trainer device
verify existing reward_target decodes to the measured Whittle terminal size
```

This stage must demonstrate that the count and configuration hashes of engine
measurements are unchanged relative to an otherwise identical run.

### Stage 2: Heads and controlled multi-task experiment

```text
add V8, V32, and S_final auxiliary outputs over the shared trunk readout
keep evaluator/search emission main-only
add normalized per-head trainer weights
run the main-only control, multi-horizon ablation, and full auxiliary design
on identical replay with matched architecture and initialization
```

### Stage 3: Online search validation

Only if the full auxiliary design improves the declared offline primary
metrics:

```text
run matched online selfplay for the main-only and full auxiliary weights
compare fixed-budget search quality, terminal cost, and throughput
```

The multi-horizon ablation separates the effect of V8/V32 from the terminal
score head. Do not add a third auxiliary value horizon until these results show
that the two existing timescales add value.

## Test Contract

Pure target tests:

```text
closed-form expansion equals the backward recurrence
single-row trace has the expected root/terminal weighted target
constant trace remains constant
mirrored/negated trace produces exactly negated targets
empty trace produces no rows and does not fail
NaN, infinity, or values outside [-1, 1] are rejected
Whittle reward -73 produces terminal score 73
terminal score normalization roundtrips to node-count units within f32 tolerance
nonintegral, negative, or greater-than-max_nodes scores are rejected
```

Pipeline tests:

```text
orchestrator preserves root_search_value exactly into the artifact
symmetric P1/P2 projection uses z/-z terminal boundaries
replay storage and GZFT wire roundtrip both targets and validity
rows from unsupported modes carry invalid auxiliary targets
P1/P2 score targets use their own final graph sizes without sign negation
score supervision reuses reward_target rather than adding a second stored copy
measurement spy observes no additional measure calls
```

Model/trainer tests:

```text
trainer emits one shared trunk pass and four correctly shaped predictions
evaluator output remains one scalar and ignores auxiliary projections
changing only auxiliary projection parameters cannot change search output
zero auxiliary weights reproduce the main-only loss and gradients
configured head weights must be finite, nonnegative, and sum to one
score output decodes to graph-size units and its mask follows measured rows
the combined task loss receives value_trunk_grad_scale exactly once
```

## Experiment And Decision Rule

Use identical replay data, model architecture, parameter initialization,
optimizer settings, total value weight, trunk gradient scale, and sampling
schedule for all three arms. The only changed variable is the per-head loss
weighting.

Primary metrics:

```text
main-head loss on a fixed held-out set of measured symmetric games
main-head calibration/sign accuracy split by early, middle, and late depth
fixed 20-graph search evaluation at the same search budget
```

Secondary diagnostics:

```text
V8/V32 loss and correlation with z by depth
target variance by depth and horizon
terminal-score MAE/bias in nodes, split by current depth and final size
main/aux gradient norms entering the shared trunk
policy loss, selfplay terminal cost, and seat advantage
reuse-on versus reuse-off target correlation
training steps/s, evaluator positions/s, replay bytes/row
engine measurement count
```

### Online diagnostic telemetry

The dashboard receives at most 19 auxiliary-diagnostic metrics. Existing head
losses and terminal-score MAE/bias remain the primary learning curves and are
not duplicated. Diagnostics are computed only on trainer logging steps and
only for the `v8-v32-score` model layout. Signal and readout attribution require
at least one active auxiliary task; parameter updates are also emitted for the
main-only control using that layout. Non-logging steps retain the normal
training path without diagnostic backwards or parameter snapshots.

```text
target quality (8):
  v8_final_target_correlation
  v32_final_target_correlation
  v8_v32_target_correlation
  terminal_score_correlation
  early_v8_final_target_correlation
  early_v32_final_target_correlation
  early_v8_target_std
  early_v32_target_std

gradient interaction at the shared readout (5):
  effective_auxiliary_norm
  auxiliary_to_final_norm_ratio
  auxiliary_alignment_ratio
  final_auxiliary_cosine
  policy_auxiliary_cosine

optimizer effect (6):
  grad_clip_scale
  trunk_gradient_norm
  trunk_update_to_parameter
  value_final_update_to_parameter
  value_horizons_update_to_parameter
  terminal_score_update_to_parameter
```

The target correlations show whether the auxiliary labels add information or
mostly duplicate the final label. Early-horizon variance and correlation focus
on the high-noise region that motivated this experiment. Early means
`progress = 1 - position.budget_fraction < 1/3`. Correlations with an empty or
zero-variance input report zero. `terminal_score_correlation` compares the
decoded score prediction with measured final graph size; existing score MAE
and bias remain in node-count units.

Readout-gradient attribution differentiates each loss with respect to the
single shared graph readout. Effective norms include `value_weight`, per-head
loss weights, and the configured `value_trunk_grad_scale`. The alignment ratio
is `norm(sum(aux gradients)) / sum(norm(each aux gradient))`, so values near
zero expose cancellation. When policy and value use separate batches,
`policy_auxiliary_cosine` is omitted because the comparison is unavailable.

Parameter deltas are exact pre-step to post-step changes for the configured
optimizer, including adaptive scaling and weight decay. Ratios normalize each
update by that group's parameter norm, making the four differently-sized heads
comparable. Policy parameters are not snapshotted because the retained
questions concern trunk pressure and auxiliary-head update scale. The trunk
delta is the combined training update and cannot be attributed to one task.

Logging steps perform extra head-side backward work and snapshot trainable
parameters around the optimizer step. Match `log_interval` across experiment
arms. Do not compare their raw training throughput with a run that has a
different diagnostic cadence.

Promote the full V8/V32/score design only if it improves a predeclared primary
metric across the repeated seeds without regressing fixed-budget search
quality. Use the multi-horizon arm to attribute whether the score head adds to
or detracts from the value auxiliaries. Do not promote based on auxiliary loss,
training loss smoothness, or a single online selfplay curve. Report performance
and storage deltas from measurements; this spec makes no claim that the added
heads are free.

## Risks

```text
bootstrap bias: root_search_value depends on the current model and can
reinforce a bad estimate

Gumbel variance: a small search budget can make m_t noisy; exponential
averaging reduces but does not remove it

tree-reuse correlation: carried statistics make consecutive targets less
independent and can preserve stale model beliefs

trunk interference: auxiliary gradients can hurt policy or the main value
head even when their own losses fall

score-task dominance: terminal size is denser and may be easier than the
competitive target, causing the trunk to over-specialize to absolute size;
the normalized loss budget and per-head gradient metrics guard this risk

engine-specific scale: Whittle node count has a natural finite support, but a
compiler runtime score will require a separately justified transform and scale

moving targets: online replay contains targets produced by older pinned model
versions; that is intentional but must be diagnosed by replay age/version

storage/protocol cost: two f32 targets plus one validity byte increase target
traffic and durable rows; measure the actual delta
```

## Out Of Scope

```text
ensembling auxiliary heads into MCTS leaf values or backups
changing Gumbel-MCTS action selection, completed-Q targets, or tree reuse
additional GraphEngine::measure calls or per-step runtime/cost measurements
exponentially averaged short-horizon score targets
P1-P2 terminal score-margin prediction
a third auxiliary value horizon
non-symmetric training modes before the symmetric result is positive
retrofitting old replay that lacks root_search_value
```
