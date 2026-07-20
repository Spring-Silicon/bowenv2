# Model

## Runtime Contract

GraphZero has one live torch architecture: `gz-graph-v2`. It is a joint-board
Exphormer with a pointer policy head and one tanh scalar value head. The same
implementation in `python/gz/model/exphormer.py` is used by trainer and
evaluator.

The model consumes fixed-capacity `GraphBatchTensors` produced from the GZFB
wire format. Shapes depend only on feature schema and batch capacity, making
the forward compatible with `torch.compile` and CUDA graph capture.

## Joint Board

Each sample contains the acting player's graph and the other player's graph.
The model concatenates their valid nodes and edges into one disconnected graph,
offsets second-board edge endpoints, and adds a learned two-role embedding.
Both policy and value therefore condition on the complete game state through a
shared trunk.

Position input contains each board's remaining budget and budget step. A retired
or blocked board is marked by its position values before staging. Position
features enter the graph/global readout; they are not used to alter topology.

## Exphormer Trunk

Node input combines opcode token embeddings, optional scalar attributes, and
board-role embeddings. Rust supplies typed graph and deterministic expander
edges. Each layer performs sparse multi-head edge attention, global-token
exchange, and a feed-forward residual block with masking for padded nodes and
edges. Reverse edge directions are materialized in the model from forward wire
edges.

No graph construction or candidate enumeration occurs in Python.

## Policy Head

The pointer head builds one feature per legal action from:

- mean embedding of its subject nodes;
- action-kind embedding;
- static engine prior.

It scores each action against the acting-board graph readout. STOP is the final
action and has its own kind token with no subjects. Padding logits are masked.
The output order is identical to evaluator/search/replay legal-action order.

## Value Head

The scalar head reads the shared joint-board representation and returns
`tanh(value_raw)` in `[-1, 1]` from the acting player's perspective. Symmetric
search negates values when perspective changes. Training uses masked MSE against
the measured game outcome in `{-1, 0, +1}`.

## Auxiliary Heads

`auxiliary_heads = "v8-v32-score"` adds:

- V8 horizon outcome;
- V32 horizon outcome;
- normalized terminal score.

These heads share the trunk but have independent readouts and valid masks. They
are absent when `auxiliary_heads = "none"`.

## Architecture Identity

`ArchConfig` permits width, layer count, head count, FFN size, dropout, and the
auxiliary-head choice. Remaining serialized fields are fixed to the live
architecture. Keeping those keys in manifests allows exact validation and load
of current checkpoints without preserving alternate implementations.

A checkpoint manifest binds both `ArchConfig` and `FeatureSchemaConfig`. Any
topology, tensor-shape, vocabulary, or architecture mismatch is rejected before
weights are loaded.

## Entry Points

The model exposes:

```text
forward(batch)       -> value, policy logits
policy_logits(batch) -> policy logits
value_only(batch)    -> value
training_outputs     -> main plus enabled auxiliary outputs
```

Split and combined paths must agree in eval mode. Serving returns exactly one
tanh application; trainer loss consumes that bounded value directly.
