# Symmetric Selfplay

## Scope

The production selfplay path is a two-player AlphaZero-style game over Whittle
rewrites. It uses adversarial MCTS and one shared joint-board policy/value model,
with no reference policy or arena. Replay V1 preserves the original
STOP-disabled mode; V2 enables learned per-player STOP actions.

## Game State

A game state is
`(p1_graph, p2_graph, player_to_move, p1_rewrites, p2_rewrites, active_players)`.
Both players begin from the same root graph. Search presents the state in the
current player's canonical perspective:

- current player's graph and remaining budget have board role 0;
- other player's graph and remaining budget have board role 1;
- the policy acts only on candidates from role 0;
- the scalar value is from the current player's perspective.

The Exphormer `state_input = "joint-board"` path concatenates both node/edge
sets, offsets role-1 edge endpoints, adds learned board-role embeddings, and
runs one trunk over the complete disconnected board.

## Turns And Termination

Players alternate decisions. `max_steps` is an independent successful-rewrite
budget for each player. A player with no legal candidate takes an untrained
forced pass and remains blocked. With STOP enabled, selecting STOP stores a
trained row, retires only that player, and leaves the graph unchanged; the
other player continues until it stops, blocks, or exhausts its budget. The game
ends as soon as neither player is active, including immediately after the
second player stops.

Both final graphs are measured through `GraphEngine::measure`. Higher measured
reward wins. If rewards are equal, fewer successful rewrites wins. Equal reward
and equal rewrite count is a draw.

## Search

Internal nodes alternate the player to move and negate backed-up values whenever
perspective changes. Forced passes also change perspective when control moves to
the other player. Terminal values are exactly `+1`, `0`, or `-1` in the node's
current-player perspective.

With `tree_reuse = true`, the selected branch's reachable subtree is compacted
and promoted only when its complete two-player board state matches the next
live root. Known stopped or horizon-exhausted players may be skipped during
that match; a branch containing an engine-discovered forced pass falls back to
a fresh tree. The promoted root retains its action visits and value sums. Each
new root still receives the full configured simulation budget, allocated from
per-run visit deltas, while action selection, Q completion, and the policy target
use the aggregate carried-plus-new statistics. Descendant structure, evaluator
outputs, and descendant statistics remain cached. `carried_nodes` and
`carried_root_visits` report the inherited subtree and root visits.

The model generation is leased when the game is admitted and remains pinned
until both players finish, so a game cannot mix checkpoint versions.

## STOP ABI

The feature/evaluator protocol appends STOP as its final action. With
`mask_stop = true` (replay V1), symmetric selfplay excludes that slot at every
semantic boundary:

- search truncates evaluator logits to engine candidates;
- selected actions, legal-action lists, and policy targets contain candidates
  only;
- replay validates that no symmetric legal action is STOP;
- trainer cross-entropy masks the action whose kind token is the reserved STOP
  token.

The model therefore receives no positive or negative policy gradient for STOP.

With `mask_stop = false` (replay V2), STOP is a normal final search action and
is retained in legal-action lists, policy targets, and trainer
cross-entropy. Its transition retires the current player without calling
`GraphEngine::apply`. Because the joint board topology alone does not encode a
retired player, V2 requires position features. Position features preserve the
player's actual rewrite count and remaining-budget fraction; a negative
`budget_step` marks a retired, blocked, or horizon-exhausted board.

## Replay And Labels

Each completed game projects two episode artifacts, one per canonical player
perspective, and appends them atomically. Neither artifact has a replay
reference. P1 rows receive the game result `z`; P2 rows receive `-z`. Draw rows
receive `0`. Policy and value train from the same rows, including the raw other
player graph needed by the joint-board trunk.

The replay store is stamped `symmetric-selfplay-v1` when STOP is masked and
`symmetric-selfplay-v2` when STOP is enabled. The modes cannot be mixed with
each other or with standard rows.

## Metrics

Each atomic game append updates persistent outcome, cost, and rewrite EMAs with
decay `0.99`. The trainer exposes them under `symmetric/`:

- `p1_win_rate_ema`, `p2_win_rate_ema`, `draw_rate_ema`,
  `decisive_rate_ema`, and `seat_advantage_ema` describe the measured outcome,
  including the rewrite-count tiebreak;
- `p1_terminal_cost_ema`, `p2_terminal_cost_ema`,
  `mean_terminal_cost_ema`, `best_of_two_terminal_cost_ema`,
  `terminal_cost_margin_ema`, and `terminal_cost_best` use terminal cost
  `-scalar_reward`; best-of-two is the EMA of each game's lower cost, while
  best is the all-time minimum across both players; STOP rows and forced passes
  are excluded;
- `p1_rewrites_ema`, `p2_rewrites_ema`, `game_rewrites_ema`, and
  `rewrite_margin_ema` count successful rewrites; forced passes add no row and
  no rewrite;
- `value_sign_accuracy_early_ema` and `value_sign_accuracy_late_ema` combine
  both player perspectives, exclude draws, and split at each player's step 40;
- `games_completed` and `game_latency_s` count and time complete two-player
  games.

Value-accuracy and latency EMAs are live-process telemetry and reset when the
replay producer reopens; the outcome, cost, and rewrite metrics survive reopen.

The older `selfplay/learner_win_rate_ema`, terminal-cost, and episode-length
gauges remain for compatibility but use the primary P1 record. Symmetric-mode
analysis must use the `symmetric/` metrics instead.

## Configuration Contract

The mode requires:

- the fixed `gz-graph-v2` joint-board Exphormer architecture;
- generated Whittle roots for production training;
- `length_tiebreak = true`; `tree_reuse` may be enabled or disabled;
- either `mask_stop = true` for V1 or `mask_stop = false` with
  `position_features = true` for V2;
- a featurized evaluator and a new topology-compatible checkpoint.

The initial implementation is available in
`configs/bases/whittle-generated-exphormer-v2-symmetric-selfplay.toml`.
