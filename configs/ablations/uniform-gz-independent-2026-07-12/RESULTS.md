# Independent Uniform-GraphZero Ablations

## Control

Every run starts from the neutral-policy GraphZero Exphormer seed-17 control:

- model seed 17, data/self-play seed 42
- fixed root and gated 32-trajectory opponent pool
- 44 lanes x 32 workers, 48 simulations
- GraphZero profile, four-layer Exphormer, mean/shared/kind-prior encodings
- AdamW, constant `3e-4` learning rate, batch 256
- pair/tanh value head with hidden width 512
- 1,000 trainer steps

Each leaf changes only the named factor and run identity. Match and candidate
encoding are one requested joint factor. The SAGE leaf changes only `trunk`, so
it is a trunk isolation rather than the complete WhittleZero profile.

## Results

Lower cost is better. `terminal mean` is row-weighted over the sampled replay
batch. `episode EMA` tracks newly completed self-play episodes. `tail mean`
averages the ten logged replay means from steps 901 through 991.

| Factor | Exact change | Step 501 | Step 751 | Step 991 | Episode EMA 991 | Tail mean | Best |
|---|---|---:|---:|---:|---:|---:|---:|
| Control | none | 168.37 | 170.00 | 168.68 | 135.30 | 169.29 | 101 |
| 1. Muon | `optimizer = "muon_mixed"` | 180.38 | 179.22 | 153.46 | 106.96 | 151.93 | 101 |
| 2. Match/candidate | `subject_encoding = "match"`, `action_encoding = "candidate_only"` | 147.92 | 121.93 | 116.26 | 112.80 | 117.40 | 102 |
| 3. Cosine | cosine to 0.1x over 800 steps, peak LR unchanged | 166.59 | 172.51 | 141.78 | 114.11 | 146.97 | 102 |
| 4. Batch 512 | `batch = 512` | 123.93 | 103.89 | 99.98 | 102.53 | 99.91 | 89 |
| 5. Value width 256 | `value_hidden = 256` | 145.02 | 141.63 | 130.38 | 120.28 | 131.45 | 100 |
| 6. SAGE trunk | `trunk = "sage"` | 162.58 | 161.85 | 163.34 | 128.26 | 163.61 | 102 |
| Whittle-path reference | complete Whittle-compatible recipe | 142.82 | 103.57 | 101.51 | 103.39 | 101.51 | 85 |

## Runtime Context

| Factor | Wall time | Produced rows | Samples per produced row |
|---|---:|---:|---:|
| Control | 6.5 min | 80,814 | 3.14 |
| Muon | 7.8 min | 99,909 | 2.54 |
| Match/candidate | 7.6 min | 87,402 | 2.90 |
| Cosine | 6.3 min | 82,246 | 3.08 |
| Batch 512 | 10.8 min | 200,149 | 2.54 |
| Value width 256 | 7.0 min | 95,871 | 2.65 |
| SAGE trunk | 8.1 min | 96,327 | 2.63 |

Batch 512 is not purely a gradient-noise intervention in this asynchronous
pipeline. With the reuse gate held fixed, doubling the batch also slows trainer
admission, gives actors more wall time, consumes twice as many samples per
optimizer step, and produced 2.48x as many rows by step 991. Its result proves
that this single config change closes the observed learning gap, but does not
separate larger-gradient-batch effects from the resulting actor/learner cadence.

## Verdict

Batch 512 is the only independent factor that reaches Whittle-path population
quality and learning speed by trainer step. Match/candidate encoding is the
second-largest effect. Value width and cosine are useful partial effects; Muon
mainly improves the live episode EMA late; SAGE alone has little effect.
Therefore Exphormer itself is not the observed cause of failure, and Muon is not
required for parity in this seed-17 screen.
