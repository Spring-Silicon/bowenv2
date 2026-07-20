# Features And Batch Encoding

## Scope

`gz-features` owns the portable representation between an engine-backed search
lane and Python model execution:

```text
GraphEngine handles
  -> FeatureExtractor<E>
  -> validated FeatureRow
  -> FeatureCollator
  -> fixed-layout GZFB bytes
  -> Python BatchView / BatchStager
```

It also owns encoded training targets and evaluator output decoding. Feature
extraction is engine-specific; schema, validation, collation, and wire formats
are engine-independent.

## Schema

`FeatureSchemaConfig` fixes vocabularies and maximum node, edge, action, subject,
and attribute dimensions, plus deterministic expander parameters. Its hash
binds replay rows, evaluator handshakes, and checkpoints. A store or checkpoint
cannot be reused under a different schema.

Some scalar-reference fields remain in the serialized schema/row layout only
for compatibility with existing schema-v8 replay stores. The live joint-board
model does not consume them.

## Feature Row

A row contains:

- node count, opcode tokens, and optional scalar attributes;
- typed directed edges;
- actions in evaluator order, each with kind, static prior, and subject nodes;
- root step, leaf depth, remaining-budget fraction, and budget step;
- optional other-player graph state and its position.

STOP is appended by search and encoded as the final action with the reserved
STOP kind token. The extractor never invents or reorders engine candidates.

Rows validate counts, index ranges, finite scalars, STOP placement, padding
constraints, and optional-board dimensions before crossing a thread or process
boundary.

## Whittle Extraction

`WhittleFeatureExtractor` maps arena graphs to opcode nodes and argument edges,
adds deterministic cached expander edges, and converts candidate metadata to
action kinds/subjects/static priors. The feature cache is valid only for graph
topology; action and position data remain per request.

For symmetric search the orchestrator extracts the acting graph with legal
actions and the other graph without actions, then attaches the latter as the
row's second board.

## GZFB Batch

`FeatureCollator` writes one capacity-shaped little-endian batch. Counts define
valid prefixes; padded sections are zero except subject indexes, which use the
reserved `0xffff` sentinel. Node/action indexes and kind tokens use bounded
integer widths, and floating feature sections use bf16 where specified.

The batch includes both board graphs and positions. Historical trajectory-ID
and scalar-reference slots remain zeroed reserved bytes so existing encoding
and schema compatibility are preserved; they have no runtime producer or model
consumer.

Python parses the buffer with zero-copy NumPy views, then `BatchStager` copies
only live model inputs into reusable pinned host/device tensors. Rust and Python
layout tests guard every section offset and total length.

## Targets

Training target encoding carries:

- per-action soft policy probabilities;
- selected-action index;
- main value target and valid mask;
- optional V8/V32 outcome targets and masks;
- optional terminal-score target and mask.

Target rows preserve the collated action count and order. Policy probabilities
round-trip through bf16 by design.

## Versioning

Row encoding, batch encoding, and output encoding have explicit versions.
Changing section order, width, shape, or meaning requires synchronized Rust and
Python changes, updated golden fixtures, and an encoding/schema migration.
