# GAZ Single Vanilla Mode

## Scope

`training_mode = "single-vanilla"` is a single-player expert-iteration mode.
It runs one learner trajectory, trains the policy on Gumbel-MCTS improvement
targets, and trains a single-state scalar value head on the learner episode's
measured terminal reward.

It does not create an opponent trajectory, historical incumbent evaluator,
arena challenger, pair-value input, or competitive win/loss label.

## Search Semantics

- Every played move starts a fresh Gumbel-MCTS tree. `tree_reuse` must be false.
- Completed Q values are min-max normalized with running bounds local to that
  move's tree before the Gumbel sigma transform.
- A child at the episode budget is evaluated once and marked terminal in the
  search tree. Later visits back up its predicted value without expanding it.
- STOP backs up the current node's predicted value. Search does not measure a
  STOP branch.
- The selected action and policy target retain the ordinary GraphZero Gumbel
  action indexing and projection path.

The STOP behavior is the deliberate predicted-STOP variant. It preserves one
runtime measurement per completed episode, but STOP comparisons inside search
are approximate rather than measured.

## Measurement And Replay

The orchestrator calls `GraphEngine::measure` once, after the played episode
selects STOP or reaches `max_steps`. Rows are admitted only after that result is
measured, valid, and has a finite `scalar_reward`.

`single-vanilla-v1` replay stores:

- no `ReplayReference`;
- `value_target = learner_reward = final_measure.scalar_reward` on the episode
  and every row;
- the usual MCTS policy target and portable action identities.

The distinct replay mode prevents raw rewards from mixing with competitive
sign, graded, or sampled-tree labels.

## Trainer Contract

Single Vanilla requires:

- `value_input = "single"`;
- `value_head = "scalar"`;
- `value_activation = "logit"`;
- `value_mirror = false`.

The trainer applies masked MSE directly to the raw scalar reward. It reports
value MAE and RMSE; competitive value-sign accuracy and learner win rate are
not emitted for this mode.

## Configuration Contract

All reference, trajectory-pool, incumbent-evaluator, and arena settings must be
disabled. STOP remains available, length tiebreaking is disabled, and the
competitive `value_reward` selector remains `"sign"` only as a neutral legacy
field; Single Vanilla does not use it to construct labels.

## Episode Length Schedule

Single Vanilla generated-root runs may optionally change `max_steps` at
learner update boundaries. `start_step = N` means the stage starts after `N`
optimizer updates have completed and supplies the replay batch for update
`N + 1`. STOP remains enabled, so `max_steps` is a horizon cap rather than a
forced episode length.

An explicit schedule is:

```toml
[selfplay]
max_steps = 20 # Must equal the final stage.

[episode_length_schedule]
mode = "explicit"

[[episode_length_schedule.stages]]
start_step = 0
max_steps = 5

[[episode_length_schedule.stages]]
start_step = 1000
max_steps = 10

[[episode_length_schedule.stages]]
start_step = 2000
max_steps = 20
```

Generated schedules use the same learner-step boundaries:

```toml
[episode_length_schedule]
mode = "linear" # Or "exponential" with factor = 2.0.
start = 5
increment = 5
interval_steps = 1000
maximum = 20
```

Each stage owns a separate replay store below `[paths].replay_dir`. The model,
optimizer, EMA, checkpoint directory, trainer step, and W&B run stay live
across transitions. Policy/value reuse gates reset at each stage, ordinary
sample seeds retain the absolute trainer step, and generated-root selfplay gets
a deterministic stage-specific seed so process restarts do not replay the same
root sequence.

The driver force-publishes the boundary model, discards old-stage prefetched
batches, stops old selfplay, records exact final replay counters, and starts the
new stage. `episode-length-schedule.json` persists the active stage and
completed-store counters for resume. Multi-stage scheduling currently requires
`training_mode = "single-vanilla"`, `root_mode = "generated"`, and
`publish_lag_blocks = 0`.
