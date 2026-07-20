# GZ Eval Protocol

Status: draft

Purpose: define the language-neutral wire contract between Rust evaluator
clients and Python evaluator servers. `gz-eval-service` owns the Rust side;
`python/gz/proto`, `python/gz/codec`, and `python/gz/evaluator` implement
the Python side.

## Constants

```text
PROTOCOL_VERSION = 2
ENCODING_VERSION = 5
MAX_FRAME = 256 MiB
```

The payload encoding is `GZ_FEATURES.md`: EVAL carries one GZFB batch and
EVAL_RESULT carries one GZFO output.

## Framing

Unix domain socket. Every frame is:

```text
u32 LE body_length, then body: u8 frame_type + fields
```

Rules:

```text
body_length includes the type byte.
body_length must be nonzero and <= MAX_FRAME.
integers are little-endian.
hashes and ids are raw bytes.
unknown frame types are protocol errors.
malformed frames are fatal; no resync.
```

## Frame Types

```text
1 HELLO       client -> server, once, immediately after connect:
              protocol_version u32
              encoding_version u32
              feature_schema_hash 32B
              batch_capacity u32
              engine_id 16B
              engine_version 16B
              action_set_hash 32B

2 HELLO_ACK   server -> client:
              protocol_version u32
              model_version 16B
              model_generation u64

3 EVAL        client -> server:
              batch_id u64
              requested_model_version 16B
              GZFB bytes

4 EVAL_RESULT server -> client:
              batch_id u64
              served_model_version 16B
              active_model_generation u64
              active_model_version 16B
              GZFO bytes

5 PING        client -> server:
              nonce u64

6 PONG        server -> client:
              nonce u64

7 ERROR       server -> client:
              code u32
              msg_len u16
              utf8 message, bounded to 512 bytes

8 MODEL_RELEASE client -> server, no success response:
              model_generation u64
              model_version 16B
```

The server closes after sending ERROR.

## Error Codes

```text
1 protocol
2 encoding
3 schema
4 capacity
5 malformed
```

## Handshake

Stub server behavior:

```text
VALIDATES:
  protocol_version == PROTOCOL_VERSION
  encoding_version == ENCODING_VERSION

ADOPTS from the first client and enforces on every EVAL:
  feature_schema_hash
  batch_capacity

RECORDS only:
  engine_id
  engine_version
  action_set_hash
```

A checkpoint-backed server later validates `feature_schema_hash` against its
checkpoint instead of adopting it. Engine tags are recorded by the stub server
because later checkpoint compatibility checks need the same fields, but the
stub accepts any engine tags.

EVAL validation:

```text
requested_model_version must identify a resident model slot.
GZFB magic and encoding_version must match.
GZFB feature_schema_hash must equal the adopted schema hash.
GZFB batch_capacity must equal the adopted capacity.
GZFB row_count must be <= batch_capacity.
GZFB section dimensions and total byte length must be valid.
```

Responses arrive in request order. `served_model_version` must equal the
request's `requested_model_version`; the active fields advertise the generation
that new episodes should lease. A generation identifier is nonzero and scoped
to one evaluator connection.

The evaluator retains the previous model after activating a replacement. The
client may continue targeting that previous version until it sends
MODEL_RELEASE after its final episode lease and in-flight request are gone.
MODEL_RELEASE for the active or an unknown generation is a protocol error. The
serving implementation bounds residency to the active and one retained
generation; it does not load a third until the retained generation is released.

## Stub Model

Defined over the encoded batch so both implementations consume identical
input. For row `i < row_count`, with `a = node_count[i]`,
`c = action_count[i]`, all arithmetic in u64 with wraparound:

```text
value[i]     = (((a * 2654435761 + c * 40503) % 4096) - 2048) / 2048.0
logits[i][j] = (((a + 31*j + 7*c) % 64) - 32) / 32.0     for j < c
logits[i][j] = 0.0                                        for c <= j < A
rows i >= row_count: all zeros
```

Integer-mod results are < 2^24 and divisions are by powers of two, so both
languages produce bit-identical f32.

Stub model version:

```text
67 7a 2d 73 74 75 62 2d 76 31 00 00 00 00 00 00
```

That is ascii `gz-stub-v1` zero-padded to 16 bytes.

## Sample Protocol

Replay sampling for the trainer uses the same Unix-socket framing:

```text
u32 LE body_length, then body: u8 frame_type + fields
```

Constants:

```text
SAMPLE_PROTOCOL_VERSION = 11
ENCODING_VERSION = 5
MAX_FRAME = 256 MiB
```

Frame types:

```text
1 HELLO       trainer -> server, once:
              protocol_version u32
              encoding_version u32

2 HELLO_ACK   server -> trainer:
              protocol_version u32
              feature_schema_hash 32B
              max_batch u32
              produced_rows u64
              feature_schema_config:
                name_len u16, utf8 name,
                node_vocab_size u16,
                node_attr_dim u16,
                edge_type_count u8,
                action_kind_vocab_size u32,
                max_nodes u32,
                max_edges u32,
                max_actions u32,
                max_subjects u32,
                expander_degree u8,
                expander_seed u64

3 SAMPLE      trainer -> server:
              batch u32, must be 1..=max_batch
              window u64, must be nonzero
              seed u64

4 SAMPLE_RESULT server -> trainer:
              gzfb_len u32
              GZFB bytes
              GZFT bytes

5 ERROR       server -> trainer:
              code u32
              msg_len u16
              utf8 message, bounded to 512 bytes
```

Error codes:

```text
1 protocol
2 encoding
3 empty store
4 bad request
5 missing features
```

The server closes after sending ERROR. One client is served at a time;
requests from that client are processed sequentially and responses are sent
in request order. The server may accept the next client after the current
client disconnects.

`SAMPLE_RESULT` always contains one feature batch (`GZFB`) and one training
target block (`GZFT`) with the same capacity and row_count. Samples are
deterministic for fixed `(store contents, batch, window, seed)`.
