# Blake3 Volume Reduction Implementation Spec

Status: implementation work order

Purpose: cut selfplay CPU by computing FEWER blake3 digests, not
cheaper ones. blake3 is 15% of stub-selfplay CPU samples (19-21% in
production runs): every simulation apply canonicalizes the graph and
hashes ~1KB of canonical bytes (engine.rs insert()), for dedup AND the
portable GraphHash, whether or not that graph's portable identity is
ever needed.

Negative result to respect (2026-07-06, do not retry): swapping the
HASHER for already-hashed keys (identity/fold hasher for
GraphHash-keyed maps, FastHasher for id-keyed search maps) measured
NEUTRAL for engine maps and a 2.5% REGRESSION for search maps over 3x
stub-benchmark trials, despite DefaultHasher showing 6-11% of profile
samples. Profile share attributed cache-miss stalls and irreducible
byte-touching to the hash symbol. Per-call cost is not the target;
call volume is.

Authority: `GZ_ENGINE.md` (graph identity contract), `GZ_REPLAY.md`
(portable ids in stores are blake3 and must not change).

Read before starting:

```text
crates/gz-engine-whittle/src/engine.rs   insert(): canonicalize +
                                         hash per apply; by_hash dedup
crates/gz-search/src/gumbel/task/root.rs finalize_node -- every
                                         expanded node materializes a
                                         ReplayGraphContext for its
                                         EvalRequest
crates/gz-eval/src/types.rs              EvalRequest::with_position /
                                         validate_ref -- what the
                                         context is actually used for
```

## Stage 0 (required first): count before cutting

Instrument (debug counters, printed in the selfplay summary or a probe
binary): per episode, (a) graph inserts, (b) dedup hits, (c) portable
contexts actually materialized (eval requests + replay rows + reference
steps). The design below assumes (c) << (a). If the count shows every
insert's hash is consumed by an eval request, STOP and report -- the
win then requires relaxing EvalRequest's context (see Open Question)
and the spec needs a revision, not a heroic workaround.

## Design

```text
GraphArena::insert splits identity into two tiers:
  dedup identity   64-bit fingerprint of the canonical bytes (xxh3 or
                   a fold; engine-internal, never persisted) keying the
                   dedup map, with full canonical byte-compare on
                   fingerprint collision (canonical bytes are already
                   stored per graph record)
  portable identity  blake3 GraphHash, computed LAZILY on first
                   hash()/context request and memoized in the graph
                   record (Option<GraphHash>)

Everything persisted or crossing the process boundary keeps blake3:
replay rows, episode records, reference steps, checkpoint labels. Store
bytes must be identical before/after (same rows, same hashes).
```

## Open question the implementation must answer

EvalRequest currently carries a full ReplayGraphContext per expanded
node (root.rs finalize_node). If validate_ref/serving only use it for
shape checks and row bookkeeping that never leaves the process for
LEAF evals (only move-boundary rows persist), a transient context
(fingerprint-based, engine-local) can serve leaf evals and blake3
drops to move-boundary volume (~1/48th at 48 sims). If the wire or
validation genuinely needs the crypto hash per eval, say so in the
review and land only the lazy-memoization tier.

## Acceptance

```text
stub benchmark (192 episodes, 24x8, sims 32, seed 7): >= 3 trials
before/after on a quiet machine; accept only if user CPU drops >= 5%;
otherwise revert and report the counter data
episodes bit-identical at fixed seed (same rows, labels, hashes)
store bytes unchanged: portable ids remain blake3
full battery green
```

## Out Of Scope

```text
changing persisted hash algorithms or store schema
hasher swaps for existing maps (measured dead end, see above)
canonicalization (serialize_wav1) elimination -- separate follow-up if
the counters show it dominates after the blake3 cut
```
