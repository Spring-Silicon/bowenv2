# Mid-Episode Subtree Release Implementation Spec

Status: implemented (2026-07-06); review notes in the landing commit

Purpose: bound the selfplay working set by the CARRIED search tree
instead of the episode's full creation history. Today every graph and
candidate created during an episode stays live until the episode ends
(`created_graphs`/`created_candidates` release in bulk after
projection): ~3M candidate slots per episode at sims 48 / max_steps
128, ~650 MB-1 GB per concurrent episode, ~160-190 GB at 32x8. That
working set is legitimate but it is the binding constraint on
concurrency: doubling workers doubles it, and the profiling work
showed selfplay is eval-latency-bound with 40 idle cores -- more
in-flight search is the cheapest selfplay throughput lever we cannot
currently afford to pull. Releasing dead subtrees at move boundaries
cuts the per-episode footprint by roughly the episode length (~10-20x
here), converting worker scaling from memory-gated to GPU-gated.

Authority: `GZ_ENGINE.md` (release contract), `GZ_SEARCH_GUMBEL_MCTS.md`
(search behavior is frozen: this work order changes memory timing,
never search results), `GZ_ENGINE_RELEASE_IMPL.md` (amendments carry
the measured numbers). Contract wins; report conflicts.

Read before starting:

```text
crates/gz-search/src/gumbel.rs      GumbelEpisodeTask.track_created_handles,
                                    compact_subtree / reused_child_task
                                    (the carried-set computation),
                                    GumbelEpisode.created_* ownership
crates/gz-orchestrator/src/lanes.rs run_lane_pipeline: pool.drive and
                                    release_episode_handles; the lane
                                    thread is where release runs
crates/gz-orchestrator/src/pool.rs  drive() -- where per-move drains hook
crates/gz-orchestrator/src/lanes.rs feature_rows_for_episode -- re-derives
                                    features from step graphs at episode
                                    end (the path-liveness constraint)
```

## The Liveness Contract

What must survive until episode end (projection + feature extraction
re-enumerate them):

```text
1. The selected path: every step's before/after graphs (episode.steps
   feed feature_rows_for_episode, which calls engine.candidates on
   step graphs) and the final graph (final measure).
2. The path graphs' candidate sets ARE re-enumerated at projection
   (cache hit or recompute), so their candidate ids created during
   search may be released mid-episode -- but see the simpler rule
   below.
3. The carried subtree under tree reuse: nodes kept by
   compact_subtree remain referenced by the next move's search --
   their graphs AND their candidate ids stay.
```

Everything else -- sibling subtrees discarded at move selection, which
is the bulk of the 48-sims-wide exploration -- is dead the moment the
root advances.

Releasable at the end of move k, by the simplest correct rule:

```text
dead(k) = created_during_move(k)  MINUS  carried_subtree_handles(k)
                                  MINUS  {selected_after graph of k}
```

Candidates of the selected-path graph: release them with dead(k)
(they are re-derivable; projection re-enumerates through the engine
cache or recomputes). Keeping the rule handle-set-based -- carried
set + one graph -- avoids any reasoning about which candidate
belongs to whom. Dedup aliasing is already safe: a dead sibling that
content-aliases a path graph only drops one refcount.

## Stages

```text
1. Per-move creation tracking (gz-search). GumbelEpisodeTask's
   created_graphs/created_candidates become per-move buffers: track
   into the current move's buffer; at move completion (root task Done,
   step recorded, subtree compacted) partition the buffer against the
   carried set + selected graph, move the dead handles into a
   `releasable` accumulator on the task, and roll the survivors into
   the next move's buffer (they release when THEY die or at episode
   end). GumbelEpisode keeps created_* for the epilogue: whatever is
   still live at episode end (final path, carried remnants).
   Tests: per-move partition unit tests on a scripted small search;
   the union of all mid-episode releases plus the episode-end set
   equals bit-for-bit the handle set the current code releases at the
   end (equality oracle across the whole episode, serial driver).

2. Lane drain (gz-orchestrator). pool.drive gains a post-poll drain:
   after each task poll, take_releasable() from the task and
   engine.release on the lane thread (release stays a lane-thread
   call, never SearchWork). The drain also runs on the drop/abort
   paths ahead of release_episode_handles, which stays as the
   epilogue for the remainder.
   Tests: arena_occupancy mid-episode stays O(carried tree + one
   move's churn) on a long fixed-root episode (assert peak live
   candidates < 4x one move's creation, vs ~len(episode)x today);
   occupancy returns to baseline at episode end, unchanged.

3. Acceptance and measurement. Byte-identical stores and episode
   outcomes vs the pre-change commit for fixed seeds (release timing
   must not change behavior; enumeration is content-deterministic, so
   a re-enumeration after a cache eviction returns identical
   candidates -- only cost shifts). Leak probe at the production
   shape WITH NOISE (gumbel_scale 1.0, the probe discipline): peak
   RSS at 32x8 / max_steps 128 drops >= 5x from the ~160-190 GB
   working set; paste before/after into the commit message.
   Then the payoff experiment, separate commit: workers_per_lane 8 ->
   16/24 with the freed memory; report episodes/s, rows/s, eval batch
   fill, and evaluator GPU utilization.
```

## Cost Notes (documented, accepted)

```text
Cache evictions increase: releasing a dead sibling's last ref evicts
its candidate-list cache entry, so a later move that re-explores the
same content pays one re-enumeration. At whittle costs this is noise
(enumeration is microseconds against a 2-6 ms eval round trip); in
the compiler regime re-enumeration is also cheap (measurement, not
enumeration, is the paid operation there).
Per-move partition cost: O(created-this-move) set membership against
the carried set -- the carried set is already materialized by
compact_subtree; reuse its hash set rather than rebuilding.
```

## Out Of Scope

```text
releasing path graphs before projection (would break
feature_rows_for_episode; revisit only if projection moves to
portable-only inputs)
changing tree reuse semantics, budgets, or the search config hash
per-simulation release granularity -- move boundaries are enough (the
tree between moves is exactly the carried set)
the workers/lanes scaling sweep itself beyond the single payoff
experiment (a config study, not code)
```
