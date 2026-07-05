# Gated Policy Opponent Implementation Spec

Status: design for review

Purpose: add `reference = "gated-policy"` -- the historical-best policy
opponent. The plain policy opponent (implemented, GZ_OPPONENT_IMPL.md
Stage 3) re-bases the bar on every checkpoint: each swap's greedy
rollout REPLACES the reference, so a temporarily worse checkpoint
lowers the bar and win labels can cycle without the objective
improving. The gated variant keeps the bar at the best rollout any
published checkpoint has achieved: a new checkpoint's rollout is
played once and ACCEPTED only if it strictly beats the incumbent.
Labels then mean "search beat the strongest raw policy this run has
ever produced" -- a monotone, non-drifting bar.

Provenance: whittlezero's `arena_gate` (engine/whittle_arena.py,
GUMBEL_PTP_V5 spec section 3.3): the historical-best opponent theta_B
is replaced by the current policy theta only when theta beats theta_B
on a fixed arena set by summed raw objective -- "magnitude-aware, not
win rate", chosen there to prevent both cycling and better-at-the-
sign-game-worse-at-the-objective drift. Their default opponent rollout
is the greedy policy rollout; a deterministic tree-search rollout (the
Gumbel-AZ "greedy tree" opponent) is an opt-in of the same gate. In
our fixed-root regime the arena set is the single fixed root, so the
summed margin degenerates to a scalar comparison and theta_B's cached
arena vector degenerates to the stored reference scalar.

Authority: `GZ_REPLAY.md` (labeling contract), `GZ_OPPONENT_IMPL.md`
(the rollout machinery this builds on). Contract wins; report
conflicts.

Read before starting:

```text
crates/gz-orchestrator/src/reference.rs   PolicyReferenceProvider +
                                          rollout hooks (the machinery
                                          being parameterized)
crates/gz-orchestrator/src/lanes.rs       OpponentRollout (unchanged)
crates/gz-search/src/gumbel.rs            GumbelMcts::policy_rollout
crates/gz-replay/src/records.rs           ReplayReferenceKind
                                          (append-only)
../whittlezero/engine/whittle_arena.py    arena_gate -- the reference
                                          semantics
```

## Semantics

```text
State: best = Option<(reward, version, context, hash)>   the incumbent
       last_challenged = Option<ModelVersion>            dueness anchor
       pending = Option<ModelVersion>                    rollout in flight

Per checkpoint swap (new model_version on eval replies):
  rollout_due  <- latest != last_challenged (NOT != best.version: every
                  checkpoint is challenged exactly once, including ones
                  that lose)
  the lane plays ONE challenger rollout from the fixed root through
  the current evaluator -- identical machinery, config, and cost to
  the ungated policy opponent
  finish_rollout(outcome):
    last_challenged = pending version (measured or not: an unmeasured
    challenger retries, exactly like the ungated provider)
    ACCEPT  iff best is None or outcome.reward > best.reward
            -> best = (reward, version, context, hash)
    REJECT  otherwise -> best unchanged

reference() -> best scalar, kind = GatedPolicy, model_version =
  best.version (the INCUMBENT that set the bar, not the newest
  checkpoint -- rows are self-describing about which weights hold it)

The incumbent's rollout is never replayed: the greedy rollout is
deterministic per checkpoint on a fixed root, so the stored scalar IS
whittlezero's cached_best. Old checkpoints never need serving --
gating costs exactly one rollout + one measurement per swap, the same
as the ungated opponent. Strict inequality on accept: an exact tie
keeps the older incumbent (stable version attribution).
```

## Config Surface

```text
[selfplay] reference = "gated-policy"
CLI --reference gated-policy; requires --root-mode fixed (same
validation as policy).

reference_rollout_sims (new, default 1, applies to policy AND
gated-policy): simulations for the opponent rollout. 1 = greedy
argmax-policy rollout (today's behavior, whittlezero's default);
k > 1 = the deterministic tree-search opponent (Gumbel-AZ greedy
tree, whittlezero's rollout="tree") -- GumbelMcts::policy_rollout
keeps zero noise and temperature 0, max_considered stays 1 at sims 1
and follows the main config's max_considered when sims > 1.
Rejected when reference is any other kind, same style as the
torch-only flags.
```

## Implementation Shape

```text
PolicyReferenceProvider gains a gate mode instead of a second
provider -- the state machine differs only in finish_rollout and the
dueness anchor:
  enum PolicyGate { Latest, Best }
  Latest: today's behavior (last_challenged == best.version always)
  Best:   the semantics above
Lane-side OpponentRollout is untouched: it already drives hooks
blindly. CliReferenceProvider maps gated-policy to the Best mode.

ReplayReferenceKind: append GatedPolicy (append-only enum; postcard
variant indexes must not shift). GZ_REPLAY.md outcome rules gain the
kind: bar = historical-best greedy rollout, model_version = the
incumbent.

Gate observability: on every finish_rollout in Best mode, eprintln one
structured line from the provider's lane:
  event=policy_gate lane=N accepted=bool challenger=f best=f version=hex
Every lane gates independently over identical deterministic rollouts,
so lines are 32x duplicated -- log only from lane 0 or accept the
duplication (reviewer's choice; determinism makes both correct).
OPTIONAL follow-up stage (not required for acceptance): surface the
bar in the sample-service ack (protocol v5, +4 bytes f32
policy_bar_reward) so the trainer can chart selfplay/policy_bar in
wandb; the ratcheting bar is the run's single most interpretable
learning signal.
```

## Considerations (documented, by design)

```text
Monotone bar => saturation is possible: if the policy stops improving,
episodes trend toward uniform losses and the value head saturates at
-1 until search beats the bar again. This is whittlezero's intended
pressure (the win signal is reserved for genuine improvement); it is
NOT the self-average's balanced-labels regime. Runs that want balanced
labels keep reference = "self-average" or plain "policy".
Exact ties label 0.0 and are meaningful: the greedy rollout is
reachable by search.
Bounded burst runs (episodes <= lanes x workers) stay unlabeled, same
as plain policy -- labels bind at admission.
Per-lane gates are independent but converge to identical state
(deterministic rollout of the same checkpoint from the same root);
no cross-lane coordination is needed or wanted.
```

## Stages

```text
1. PolicyGate mode + reference_rollout_sims knob + CLI/config plumb +
   GatedPolicy kind. Tests: state-machine unit tests (accept, reject,
   reject-then-accept, unmeasured retry, last_challenged vs incumbent
   version separation, monotone bar invariant); integration with a
   version-switching evaluator where v2's rollout is WORSE (labels
   keep v1 as model_version) and v3's is better (labels flip to v3) --
   extend the existing SwitchingEvaluator harness with per-version
   value biases; CLI smoke at the 1023 shape.
2. Docs: GZ_REPLAY.md kind semantics; GZ_OPPONENT_IMPL.md gains a
   pointer; config docs list gated-policy + reference_rollout_sims.
3. (Optional, separate commit) ack protocol v5 policy_bar_reward +
   trainer wandb selfplay/policy_bar.
```

Acceptance checklist:

```text
gated-policy is a one-line config swap; all suites green
the bar never decreases within a run; rejected checkpoints leave
model_version untouched in new rows
every published checkpoint is challenged exactly once (measured), at
one rollout + one measurement per swap
sims knob: 1 reproduces today's greedy rollout bit-for-bit; k > 1
runs the deterministic tree opponent
```

## Out Of Scope

```text
multi-graph arena sets with summed-margin gating (the generated-root /
compiler regime; needs cached per-graph vectors and an arena-set
sampler with a seed disjoint from eval -- port whittlezero's
sample_arena_graphs rationale when a run needs it)
win-rate gates and Elo-style acceptance thresholds
serving historical checkpoints (the cached-scalar design exists
precisely to avoid it)
async gate scheduling (whittle_arena_async) -- per-swap rollouts are
already off the training path
```
