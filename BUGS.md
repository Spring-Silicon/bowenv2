# GraphZero Correctness Audit Findings

Audit target: `/home/ubuntu/graphzero`

Reference implementation for semantics: `/home/ubuntu/whittlezero`

Scope: verified correctness defects only. No style issues and no unverified
candidates.

## Ranked Findings

1. `crates/gz-search/src/gumbel/tree.rs:163` - all-masked nonroot nodes can still select a masked rewrite when `no_backtrack` and `mask_stop` are both enabled. Scenario: graph A rewrites to B, B's only rewrite returns to A, `mask_stop=true`; STOP is set to `-inf` at `crates/gz-search/src/gumbel/task/root.rs:557`, the rewrite is masked at `crates/gz-search/src/gumbel/task/root.rs:460`, `softmax` returns uniform for all `-inf` at `crates/gz-search/src/gumbel/schedule.rs:163`, and `poll_descent` emits the masked Apply again instead of terminating. Severity: High. Confidence: High.

2. `python/gz/evaluator/backends.py:151` - Torch evaluator admits checkpoints by feature schema only, ignoring the engine/action identities sent in `Hello`. Scenario: checkpoint trained under action-set A with the same feature schema is served to Rust selfplay running action-set B; policy logits are interpreted against B's candidate indices with no rejection, and hotswap has the same gap at `python/gz/evaluator/backends.py:322`. `publish_ema` also defaults those manifest tags to zero at `python/gz/trainer/publish.py:86`. Severity: High. Confidence: High.

3. `python/gz/trainer/driver.py:193` - resume loses the bootstrap-row offset used to exclude off-distribution bootstrap rows from training. Scenario: run publishes checkpoint 0 after 64 bootstrap rows, crashes before live rows reach `bootstrap_rows + window_rows`, resumes with `bootstrap_rows = 0` at `python/gz/trainer/driver.py:140`, then starts once `produced_rows >= window_rows`, so the newest sample window still includes bootstrap rows. Severity: Medium. Confidence: High.

4. `crates/gz-orchestrator/src/lanes.rs:1358` - confirmed known defect: replay-time feature extraction leaks re-enumerated candidate handles on early error. Scenario: `engine.candidates` succeeds and candidates are pushed into `created_candidates` at `crates/gz-orchestrator/src/lanes.rs:1376`, then extraction/count/encoding fails at lines 1385, 1391, or 1396; the function returns before `release_episode_handles` receives `feature_rows.candidates` at `crates/gz-orchestrator/src/lanes.rs:1253`. Severity: Medium. Confidence: High.

5. `crates/gz-replay/src/store.rs:317` - `sample_rows` increments the in-memory consumed counter before persisting it. Scenario: rows are MultiGot successfully, `fetch_add` advances `consumed_rows`, then `write_meta_u64` fails; caller receives an error/no batch, but live counters overcount consumed rows until reopen, weakening backlog gating. Severity: Low. Confidence: High.

## Area Verdicts

| Area | Verdict |
| --- | --- |
| Replay store + sample service | dirty for consumed-counter error path; clean for retention floor clamping, window bounds, missing-row errors, append validation, sample framing, and bf16 round-trips. |
| Lane/pool state machines | dirty for known feature-row candidate leak; clean for pool pending/resume tokens, parked eval recovery, lane replay append ordering, backpressure admission, and graph release discipline. |
| Eval service protocol | dirty for checkpoint identity admission; clean for FIFO pending/finish ordering, partial-frame failure, per-result model versions, eval-process striping, hotswap pending results, and CUDA ping-pong host/stager lifetimes. |
| Whittle engine | clean for spot-checked rule ids and semantics including Absorb/DeMorgan/Distribute/Consensus, candidate cache truncation, slot reuse/dedup, compact/serialize round-trips, and v2 fan-in/out/depth features. |
| Driver process management | dirty for resume bootstrap gate; clean for child death checks, process-group kill of torch selfplay/evaluator, stale socket removal, and prefetch error surfacing in normal CLI execution. |
| Search task residue | dirty for all-masked node selection; clean for pending token guards, double-resume rejection, STOP re-eval pending path, and existing `no_backtrack` without `mask_stop`. |

## Known Fragility Checks

- No new `trajectory_id` consumer found; it is still hardcoded at `crates/gz-orchestrator/src/lanes.rs:1409` and projected as `None` at `crates/gz-orchestrator/src/project.rs:30`.
- Shared `step_seed` is still used for sampler seeds and value flip RNG at `python/gz/trainer/driver.py:285`, `python/gz/trainer/driver.py:462`, and `python/gz/trainer/loop.py:205`; not ranked as a correctness defect because no wrong outcome was verified.

## Hardening Tests

| Area | Single most valuable test |
| --- | --- |
| Replay | Fault-inject metadata write failure after successful `sample_rows` MultiGet and assert consumed counters do not advance for undelivered samples. |
| Lane/pool | Make `feature_rows_for_episode` fail after candidate re-enumeration and assert Whittle candidate refs return to baseline. |
| Eval | Publish same-schema checkpoints with mismatched engine/action tags and assert handshake plus hotswap reject them. |
| Engine | Seeded parity corpus against whittlezero native for enumerate/apply across high-risk rules plus compact WAV1 round-trip. |
| Driver | Resume from checkpoint 0 after bootstrap-only startup and assert sampling waits for `bootstrap_rows + window_rows` live rows. |
| Search | Construct a node where all rewrites are backtrack-masked and STOP is mask-stop-masked; assert search terminates or errors, never emits masked Apply repeatedly. |

## Verification Run

These commands passed during the audit:

```bash
cargo test -p gz-replay
cargo test -p gz-engine-whittle
cargo test -p gz-search no_backtrack -- --nocapture
python3 -m pytest python/tests/test_torch_backend.py python/tests/test_trainer_driver.py python/tests/test_trainer_publish.py
python3 -m pytest python/tests/test_sampler.py python/tests/test_server.py
```
