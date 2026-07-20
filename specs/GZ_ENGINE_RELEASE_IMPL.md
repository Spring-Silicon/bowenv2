# Engine Arena Release Implementation Spec

Status: historical implementation record. Current ownership rules are in
`GZ_ENGINE.md`; commands and measurements below describe the retired fixed-root
pipeline and are not current CLI instructions.

Purpose: stop the selfplay memory leak that froze the box. The Whittle
engine retains every applied graph body, every enumerated candidate
body, and cache entries for both, forever. Measured at the 1024-action
config: ~5.3 MB retained per replay row (~100 KB per expansion is
candidate bodies alone), 300-900 MB/s at run throughput; box memory
went 6% to 100% in 26 minutes and the machine thrash-froze (no swap,
so the kernel evicts executable pages long before the OOM killer
fires). Every prior run leaked too, ~4x slower at 255-wide actions;
they ended or died of other causes first. This gates every long run.

Authority: `GZ_ENGINE.md` (contract change lands there), `GZ_ORCHESTRATOR.md`
(tasks stay pure state machines), `GZ_ENGINE_WHITTLE.md`.
Contract wins; report conflicts.

Read before starting:

```text
crates/gz-engine/src/            GraphEngine trait (release joins it)
crates/gz-engine-whittle/src/engine.rs
  GraphArena / CandidateArena    Vec arenas, no reuse -- become slabs
  Caches { candidates, transitions }  keyed by GraphHash; entries must
                                 die with their graphs
crates/gz-search/src/{gumbel,mcts}/ Gumbel tasks see every created
                                 handle (ApplyResult.after, ExpandResult
                                 candidates); episode result carries them
crates/gz-orchestrator/src/lanes.rs  lanes own engines; release after
                                 projection + append
crates/gz-cli/src/selfplay.rs    fixed-root mode: the source-owned root
                                 is never released
```

## Hard Constraints

```text
Every stage ends with cargo fmt --all -- --check, cargo clippy
--all-targets --all-features -- -D warnings, cargo test --all,
python3 -m pytest python/tests green. Commit per stage.
No behavioral change to search or labels: release happens strictly
after an episode's projection and append. All equality oracles and
goldens pass untouched.
Handle safety is the review focus: a released id must never be
dereferenced. Episodes own the handles they create exclusively (roots
come from the source and are excluded). Debug builds get generation
checks; release builds stay zero-overhead on the hot path.
The GraphEngine addition is default-no-op so non-Whittle engines and
every existing test compile unchanged.
```

## Stage 1: Contract

`GraphEngine` gains:

```rust
/// Frees engine resources for handles this caller owns. Using a
/// released handle afterwards is a contract violation; engines may
/// reuse the slots. Default: no-op (engines may retain forever).
fn release(
    &mut self,
    graphs: &[Self::Graph],
    candidates: &[Self::Candidate],
) -> EngineResult<()> {
    let _ = (graphs, candidates);
    Ok(())
}
```

GZ_ENGINE.md: contract text, ownership rule (creator owns; sources own
roots), and the explicit note that release is a lane-thread call, not a
SearchWork variant -- episodes are done when it runs.

## Stage 2: Whittle Slab Arenas

```text
GraphArena / CandidateArena become slab allocators: free list of slot
indexes; insert pops the free list before growing; release pushes.
Ids stay u32 slot indexes in release builds. Debug builds add a
generation counter per slot, checked on every dereference (panic on
stale handle) -- cfg(debug_assertions) only.
Cache invalidation: releasing a graph drops caches.candidates entry
for its hash and the transitions entries keyed by it; releasing a
candidate is covered by its parent graph's entry removal (candidate
ids only reach the cache through that entry). The fixed root's cache
entries survive because the root is never released.
The engine root graph (WhittleRoot) is never releasable: release of
the root id is an error.
Tests: slot reuse round-trip; arena len bounded across N
insert/release cycles; debug stale-handle panic; cache entries gone
after release; releasing the root errors.
```

## Stage 3: Episode Handle Tracking

```text
GumbelEpisodeTask accumulates created handles: every ApplyResult.after
graph (including rejected-then-masked applies' graphs if any were
created -- audit ApplyResult), every ExpandResult candidate. The root
passed in is NOT tracked. Tree reuse keeps handles within the episode;
at Done, the episode result gains created_graphs / created_candidates
(moved out, not cloned).
The final selected graphs per step and the episode's final graph are
episode-created and ARE released -- projection has already copied
portable contexts by then; nothing downstream holds engine handles.
Audit and document that claim in the work order review: replay records
hold ReplayGraphContext (portable), never Graph handles.
Tests: tracked counts equal expanded_nodes/eval counts from stats;
serial == threaded equality unchanged.
```

## Stage 4: Lane Release + CLI

```text
Replay lanes call engine.release(&episode.created_graphs,
&episode.created_candidates) after append succeeds (and also on
episode DROP paths -- unmeasured/invalid episodes must release too).
The serial driver releases after episode completion likewise.
CLI: no new flags; release is unconditional (the no-op default keeps
other engines unaffected).
The bounded-memory proof: rerun the leak probe from the post-mortem
(32 lanes x 8 workers, 1023 candidates, 48/8 sims, max-steps 128,
fixed root, stub evaluator): peak RSS at 192 episodes must be within
2x of peak RSS at 64 episodes (it was 5x before: 12.2 -> 60.5 GB).
Paste both numbers into the commit message.
```

## Stage 5: Docs

```text
GZ_ENGINE.md release contract (stage 1); GZ_ENGINE_WHITTLE.md slab +
generation notes; CODEBASE_OUTLINE design rule 6 gains the ownership
sentence; AGENTS.md lists this spec.
```

Acceptance checklist:

```text
leak probe flat: RSS(192 eps) < 2x RSS(64 eps) at the 1024 config
all equality oracles and goldens untouched
debug-build stale-handle dereference panics; release builds add no
hot-path cost (bench eval-rows/s within noise of baseline)
drop paths release; fixed root survives across episodes
```

## Implementation Review

Implemented in:

```text
351011e Add GraphEngine release contract
1f03694 Add Whittle arena release
c6986e3 Track Gumbel episode engine handles
62de3e7 Release episode engine handles
```

Result:

```text
Replay rows hold ReplayGraphContext, PortableSearchActionRef, and feature
bytes. No E::Graph handle enters replay storage.

Feature rows and replay rows are projected before release. Replay append is
acknowledged before release on replay paths.

Fixed roots are excluded from GumbelEpisode.created_graphs and are not released
by search episodes.

Whittle graph and candidate arenas use refcounted canonical caches. Equivalent
graphs or repeated candidate lists may share slots, but each returned handle
occurrence is an owned release reference.

Whittle transition cache stores GraphBody values, not graph ids. apply()
inserts/retains a fresh graph reference for each caller.
```

Leak proof:

```text
Command:
target/release/graphzero selfplay \
  --replay-dir /tmp/gz-release-probe-{64,192} \
  --episodes {64,192} \
  --lanes 32 \
  --workers-per-lane 8 \
  --reference root \
  --root-mode fixed \
  --evaluator stub \
  --seed 0 \
  --max-steps 128 \
  --simulations 48 \
  --max-considered 8 \
  --gumbel-scale 0 \
  --tree-reuse true \
  --max-candidates 1023 \
  --max-batch 256

RSS(64 episodes):  7,127,040 KB, wall 2:26.90, rows 8,192
RSS(192 episodes): 13,255,724 KB, wall 3:29.24, rows 24,576
Ratio: 1.86x
```

Verification:

```text
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
python3 -m pytest python/tests
```

All commands passed for the final implementation and documentation state.

Review amendments (post-implementation): the refcounted canonical
arenas are the right call -- content dedup means an episode's rewrite
cycle that reproduces existing content shares the slot, and per-holder
refcounts make release sound where the specced plain slab was not (the
first slab cut crashed with "unknown candidate" on exactly that
sharing). Two defects found and fixed in review:
1. Cache invalidation used full-map retain() scans per last-ref
   candidate release -- O(cache) per candidate, quadratic per episode.
   Invisible in the acceptance probe (fixed root + gumbel_scale 0 =
   identical episodes = tiny caches) but a livelock under generated
   roots with diverse in-flight episodes (a 96-episode probe ran >17
   minutes without finishing; both lane threads pinned in
   HashMap::retain). Candidate bodies carry their parent graph_hash
   and candidate_hash, so both invalidations are now keyed removals,
   and the transitions cache is nested by before-graph hash so graph
   release drops its entries O(1). Fixed shape: 73.8s vs 72.6s
   pre-release baseline (noise), identical episode outcomes.
2. release() hard-errored when the released list contained the engine
   root id. Rewrite cycles legitimately dedup episode-created graphs
   onto the root; that reference is owned and releasable. The guard is
   now on freeing the root's LAST reference only
   (GraphArena::release_protected).
Leak bound re-verified after both fixes: 6.3 GB peak at 64 episodes,
18.9 GB at 448 (decelerating: 48 MB/ep then 22 MB/ep -- plateau-shaped
residual from store caches, not a handle leak).
Known minor: generated-root mode leaks one source-owned root graph per
episode (~6 KB); sources own roots by contract and nothing releases
them. Acceptable rate; noted for the source-release follow-up.
Contract footnote for future engines: a rejected apply's `after` must
still be an owned reference (the search task releases it).
Second amendment (2026-07-05, found in production): candidates() used
to enumerate EVERY candidate (one owned reference each) and then
truncate to options.max_candidates -- the cut tail's references were
stranded. Any graph with more candidates than the mask leaked its
tail slots on every cache-miss expand (~14% of inserts at the 1023
mask, ~355K slots per episode, 206 GB RSS in 20 minutes at production
rate). Every acceptance probe had run gumbel_scale 0, where identical
episodes re-expand identical graphs and content dedup turns the
stranded tail into refcount bumps on existing slots -- structurally
invisible. Fix: the limit moves into enumeration, so tail candidates
never enter the arena, and truncated cache entries are marked so a
larger request re-enumerates (tests/release.rs pins both). Probe
discipline going forward: leak probes run with noise on (scale > 0)
so episodes explore unique subtrees.
Measured non-leak, for the record: peak candidate-arena occupancy is
one episode's full creation history (~3M slots at sims 48 /
max_steps 128) per concurrent episode, released only at episode end;
at 32x8 that is a large but bounded working set. Mid-episode release
of non-carried subtrees is the follow-up lever if it needs shrinking.
Third amendment (2026-07-05, same production run): the "plateau-shaped
residual from store caches" recorded above was neither plateau-shaped
nor store caches. Replay lanes retain every completed episode for the
run summary after clear()ing its trace Vecs -- and Vec::clear keeps
capacity. created_candidates alone reaches millions of ids per
episode, so every completed episode stranded ~20 MB of empty backing
buffer (29 MB/s at production rate, unbounded). The clear now drops
the buffers (fresh Vec::new); post-fix RSS is flat at the working-set
plateau. Residual husk retention is ~1 KB/episode; aggregating summary
counts instead of retaining episode husks is noted as cleanup.

## Out Of Scope

```text
cross-episode caches or transposition tables built on released slots
extractor cache redesign (already bounded)
compiler-engine specifics (the contract is the enabler; queued lanes
release identically)
```
