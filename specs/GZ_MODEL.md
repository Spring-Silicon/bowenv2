# gz Model Spec

Status: draft

Purpose: define the neural network contract for `python/gz/model` — a
single parameterized graph-trunk layer family with policy and value heads,
of which Exphormer-style sparse attention is the shipped default
configuration. This is a contract spec; the torch-backend work order
implements it.

Implemented by `python/gz/model/exphormer.py`,
`python/gz/checkpoints/`, and `python/gz/evaluator/backends.py` for the
initial torch inference path.

The design targets, in order: correctness of the first learning run
(minimize deltas against whittlezero's proven setup), inference latency at
compiler scale (~2000+ nodes), and `torch.compile(fullgraph=True)` +
CUDA-graph compatibility. The reference implementation in
`../whittlezero/model/exphormer.py` is the semantic ancestor; its runtime
edge construction, data-dependent shapes, and spectral expander search are
deliberately not inherited.

## Decisions

```text
1. One layer family, not one architecture. ArchConfig selects between
   softmax-attention aggregation (Exphormer semantics, the default) and
   GINE sum aggregation over the same edge tensors, and sets the number of
   latent global tokens. Ablations are config, not code.
2. Structure is data. Expander edges are emitted by the Rust extractor as
   a typed edge family (amending GZ_FEATURES.md); the model performs zero
   graph construction. Reverse edges are derived in-model by swapping the
   src/dst tensors (zero wire cost); global connectivity is dense masked
   attention against learned tokens (zero wire cost), not edges.
3. Everything is shape-static per bucket. No boolean-mask indexing, no
   .item(), no data-dependent shapes anywhere in forward. Padding flows
   through masked and wastes bounded FLOPs. v1 has exactly one bucket
   (Whittle: max_nodes = capacity).
4. Config splits by what it binds. Anything that changes wire bytes or
   their meaning (expander degree/seed, dims, vocab) is FeatureSchemaConfig
   and hashes into FeatureSchemaHash. Anything that changes only weights
   and compute (width, depth, aggregation, global tokens) is ArchConfig
   and hashes into the checkpoint manifest. A checkpoint binds both.
5. Determinism boundary: scatter-based aggregation uses CUDA atomics and
   is not bitwise reproducible across runs. Accepted for training; every
   bit-exact oracle in the project stops at the model boundary.
```

## Inputs

The model consumes a parsed GZFB batch (`gz.codec.BatchView`) directly;
sections and dtypes are owned by GZ_FEATURES.md. Derived masks, built once
per forward from counts (all static `[B, ...]` shapes):

```text
node_mask    [B, N]     j < node_count[b]
edge_mask    [B, E]     e < edge_count[b]
action_mask  [B, A]     a < action_count[b]
subject_mask [B, A, S]  s < subject_count[b, a]
```

Embedding:

```text
node state h0 [B, N, D] = token_embedding(node_tokens)
                        + attr_proj(node_attrs) when attr_dim > 0
global tokens g0 [B, K, D] = learned parameters
                           + position_proj(position [B, 4]) added to every
                             global token (budget/step conditioning enters
                             here and only here)
edge type table: embedding of size 2 * edge_type_count + reserved slots;
forward edges use type t, mirrored edges use edge_type_count + t
```

## Trunk

Per layer, pre-norm residual wiring (deviation from whittlezero's
post-norm, chosen deliberately for depth stability; the layer restructure
is already a rewrite, so parity arguments do not apply here):

```text
h = h + node_mask * EdgeAggregation(LN(h), edges)
h = h + node_mask * GlobalExchange(LN(h), LN(g))
g = g + GlobalRead(LN(g), LN(h))
h = h + node_mask * FFN(LN(h))
g = g + FFN_g(LN(g))
```

All masking is multiplicative; padding lanes compute garbage that never
escapes. No branch in forward may depend on tensor values.

### Edge tensors

The wire carries forward edges only. The model materializes, once per
forward:

```text
src = cat(edge_src, edge_dst)   [B, 2E]
dst = cat(edge_dst, edge_src)
typ = cat(edge_type, edge_type + edge_type_count)
mask = cat(edge_mask, edge_mask)
```

Expander edges arrive as one direction of a symmetric family and are
mirrored by the same mechanism.

### EdgeAggregation, aggregation = "attention" (default)

Whittlezero's proven algebra with a correct segment softmax replacing the
exp-clamp approximation:

```text
q, k, v = linear projections of h, viewed [B, N, H, Dh]
e = e_proj(edge_type_table)[typ]          # project the tiny table once,
                                          # then index — whittlezero's trick
score[b, e, h] = (q[b, dst] * k[b, src] * e).sum(Dh) / sqrt(Dh)
score = -inf where !mask
amax  = scatter_reduce(amax, score -> dst)    [B, N, H]
w     = exp(score - amax[dst])                 (0 where masked)
denom = scatter_add(w -> dst), clamp_min(eps)
out   = W_o( scatter_add(w * v[b, src] -> dst) / denom )
```

### EdgeAggregation, aggregation = "gine"

Same gather/scatter plumbing, no per-edge softmax:

```text
msg = act(k_proj(h)[b, src] + e)          (0 where masked)
agg = scatter_add(msg -> dst)
out = MLP((1 + eps) * h + agg)            eps learned scalar
```

### GlobalExchange / GlobalRead

Dense SDPA, K is small (1..32):

```text
GlobalRead:     g attends over nodes, key padding mask = !node_mask
GlobalExchange: nodes attend over g (no mask; all globals are real)
```

One virtual node (K = 1) reproduces whittlezero's connectivity with exact
softmax attention and zero edge traffic. whittlezero's `small_graph_full`
special case is dropped: one code path; the global tokens and expander
cover small graphs.

## Heads

Policy (`[B, A]` raw logits; padded action slots are garbage by contract —
Rust decode truncates by true action count):

```text
subject_pool[b, a] = masked mean over s of h[b, action_subjects[b, a, s]]
                     (0 when subject_count == 0, e.g. STOP)
logit[b, a] = MLP(cat(
    kind_embedding(action_kind[b, a]),
    action_prior[b, a],
    subject_pool[b, a],
    g_readout[b],
))
```

Value:

```text
value[b] = tanh(MLP(g_readout[b]))        # matches the [-1, 1] label scale
g_readout = mean over K of final g
```

Forward-compatibility note (job 2): opponent conditioning lands later as a
second encoded graph block whose readout is concatenated into the value
MLP input. The value head's input layout should make that concat additive;
nothing else is reserved.

## ArchConfig

Declarative, TOML-able, hashed into the checkpoint manifest per
GZ_PYTHON.md:

```text
name = "gz-graph-v1"            # registry key
dim = 128
layers = 4
heads = 4
ffn_dim = 512
dropout = 0.1
activation = "gelu"
aggregation = "attention"       # "attention" | "gine"
global_tokens = 1
```

Rules:

```text
registry.build(schema, arch_config) is the only constructor; the model
reads vocab sizes, attr_dim, edge_type_count, and dims from the schema
and everything else from arch_config. No other inputs, per GZ_PYTHON.md.
Defaults above track whittlezero's trained configuration (dim 128,
heads 4, ffn 512) with layers 4 and gelu as the deliberate deltas.
Post-loop ablations with specific hypotheses: gine at matched wall-clock
(afford ~2x depth/width for the same latency), global_tokens in {8, 16}.
```

## FeatureSchema Amendment (gz-features, Rust work order)

Expander edges become extractor output (amends GZ_FEATURES.md):

```text
FeatureSchemaConfig gains:
  expander_degree u8        (0 disables; default 5 to match whittlezero)
  expander_seed u64
Both hash into FeatureSchemaHash.

Emission: d pseudorandom permutations of the graph's real nodes (seeded
Fisher-Yates keyed by (expander_seed, node_count) — deterministic, one
computation per distinct node count, cacheable), one edge per (node,
permuted node) pair with self-loops skipped, ONE direction only (the
model mirrors), edge type = a dedicated expander type appended after the
engine's edge types. No spectral-gap selection: random permutation
compositions are near-optimal expanders with high probability, and the
selection loop was the reference implementation's largest runtime cost
for negligible quality at this scale.

max_edges accounting becomes part of schema validation:
  max_edges >= engine_edge_budget + expander_degree * max_nodes
Whittle: max_edges = 2 * capacity + expander_degree * capacity.
```

Wire-cost note: at compiler scale (N = 2048, d = 4, B = 64) expander
edges add ~9 MB/batch, a low-single-digit-ms codec cost against 10-30 ms
inference. If measurement disagrees, the zero-wire fallback is a
circulant expander computed in-model (dst = (src + offset_k) mod
node_count — pure index arithmetic, static, per-layer offsets); it is a
schema/arch change gated by a benchmark, not a v1 option.

## Compile And Performance Rules

```text
torch.compile(fullgraph=True) is a hard requirement, asserted by a test
(compile with fullgraph and run; any graph break fails the suite).
Inference: bf16 autocast, mode="reduce-overhead" (CUDA-graph capture)
enabled by evaluator config; warmup at checkpoint load runs every bucket
through compile + capture before the swap, per GZ_PYTHON.md.
One bucket in v1. The bucket table (compiler regime) multiplies compiles,
not code: same model, one compiled entry per (N, E, A) bucket, batcher
groups by bucket (the slot GZ_FEATURES.md reserved).
No CPU<->GPU syncs in forward: no .item(), .any(), data-dependent
control flow, or boolean-mask indexing (the whittlezero lesson, made a
rule).
FlexAttention over block masks derived from edges is the designated
kernel upgrade path when N grows; it must land behind the
EdgeAggregation interface with an A/B benchmark, and pins torch >= 2.5.
Not v1.
```

## Test Contract

Beyond standard shape/roundtrip tests, four properties are binding:

```text
padding invariance: appending padded rows to a batch, or padding a row's
nodes/edges/actions further, must not change any real row's outputs
(tolerance 0 in fp32 eager; documented tolerance under bf16)
batch independence: a row's outputs are identical whether evaluated alone
or alongside arbitrary other rows (same tolerances)
mask correctness: an edge to/from a padding node, an invalid action slot,
and an out-of-range subject index (pointing at padding) must not
influence outputs
compile: fullgraph compile succeeds; compiled and eager outputs agree
within bf16 tolerance; a second forward with a different row_count but
the same shapes triggers no recompile
```

The stub model remains the transport oracle; these tests are the model
oracle. No cross-language bit-exactness is expected of the real model.

## Relationship To Other Specs

```text
GZ_FEATURES.md    input encoding; gains the expander amendment above
GZ_PYTHON.md      registry/build rule, checkpoints, warm-then-swap
GZ_EVAL_PROTOCOL  transport; ModelVersion flows per-result (unchanged)
work orders       (a) Rust: expander emission in gz-features + extractor;
                  (b) Python: this model + gz/checkpoints + evaluator
                  torch backend; (c) Python: trainer
```

## Deferred

```text
FlexAttention kernel; shape buckets > 1
gine/global-token ablations (post-first-learning-curve, hypotheses above)
opponent conditioning block (job 2)
circulant in-model expander (benchmark-gated fallback)
structural node attrs (depth/fanout) as schema evolution
dropout/regularization tuning, EMA weights, training-side details
(GZ_TRAINER.md)
```
