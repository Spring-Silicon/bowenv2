# Replay Storage

## Scope

`gz-replay` stores portable measured episode records and training rows in
RocksDB. It does not own search, graph handles, feature extraction, or label
generation. `gz-measurer` creates symmetric labels and calls the paired append
API after both terminal graphs have been measured.

## Records

An episode record contains:

- portable root and final graph contexts;
- one `SearchStepRef` per stored row;
- terminal `MeasureSummary`;
- terminal reward, optional value target, and STOP status;
- search configuration hash and row count.

A row contains:

- portable state/root/action identities and action history;
- legal actions in evaluator order, with STOP last when enabled;
- the visit-derived policy target and selected action;
- sign value target and optional V8/V32 horizon targets;
- optional terminal-score target;
- terminal measurement, model version, search hash, and encoded feature row.

No process-local graph or candidate handle is durable. Stored legal candidate
actions omit their repeated graph context and retain only candidate hashes; the
row's state context reconstructs the portable action. Policy probabilities are
stored as bf16 bits because that is the precision consumed by the trainer.

The private legacy reference field in `ReplayOutcome` is retained solely to
decode existing schema-v8 symmetric stores. New appends reject non-empty legacy
references, and no reference type is exported by the crate.

## Data Modes

One store has exactly one mode:

```text
standard-v1                 distillation/single-episode rows
symmetric-selfplay-v1       paired rows with STOP masked
symmetric-selfplay-v2       paired rows with learned STOP
```

Modes cannot be mixed. A feature schema is also pinned on first use and must
match exactly on reopen.

## Symmetric Append

`append_episode_pair` validates both perspectives and commits them in one
RocksDB `WriteBatch`. The following invariants hold:

- both records and every row are measured and structurally valid;
- P2's value target is the negation of P1's;
- policy target length equals legal-action length;
- selected actions and action histories use the expected state contexts;
- row indexes, episode records, counters, and persistent symmetric metrics
  become visible atomically;
- a completed game advances the game counter once and row/episode sequences
  for both perspectives.

The append lock serializes writers. Sequence atomics are published only after
the RocksDB batch succeeds, so failed writes cannot expose missing rows.

## Sampling

Sampling is uniform with replacement over the most recent available rows:

```text
floor  = retained_floor
end    = produced_rows
width  = min(window_rows, end - floor)
start  = end - width
sample = start + uniform_u64(0, width)
```

Each request performs two batched MultiGet operations: global row sequence to
row key, then row key to serialized row. Missing or corrupt index/data entries
are errors; they are never silently skipped. `consumed_rows` advances only
after a complete sample succeeds, and concurrent sessions serialize its
persistent metadata update.

## Retention

Retention is row-bounded but deletes whole episodes. Once retained volume
exceeds the configured bound by 25%, append computes an episode-aligned floor.
Deletion uses a two-floor protocol:

1. publish the new retained floor after the current append;
2. on a later cycle, range-delete only below the previously published floor.

Lock-free samplers load the retained floor before `produced_rows`. Therefore an
in-flight sampler can only select rows at or above a floor whose data has not
yet been deleted. Sampling clamps its window to that floor.

## Counters And Metrics

Persistent counters include produced rows, consumed rows, completed games,
STOP games, episode sequence end, retention floors, and symmetric outcome/cost/
rewrite metrics. Runtime-only EMAs include episode latency and early/late
value-sign accuracy. The sample service exports these values to the trainer.

## Storage Compatibility

The current postcard schema version is 8. Existing legacy column-family names
must remain listed when RocksDB opens an old store, even though their indexes
are no longer read or written. Serialized private compatibility fields must not
be reordered. A deliberate incompatible record change requires a schema bump
and a fresh replay store.
