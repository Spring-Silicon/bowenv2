#!/usr/bin/env bash
set -euo pipefail

readonly RUN_DIR="/opt/dlami/nvme/graphzero-runs/sym1m-s42m5-l96w4-r1-vw300k-vtg0p1-c4s48-aux-v8v32score-neuralterminal-clip10-20k-1"
readonly RUNTIME_DIR="$RUN_DIR/queued-runtime-2026-07-21"
readonly CHECKPOINT_DIR="$RUN_DIR/checkpoints"
readonly PYTHON_DIR="$RUNTIME_DIR/python"
readonly CONFIG_DIR="$RUNTIME_DIR/configs"
readonly QUEUE_LOG="$RUN_DIR/lr-queue.log"
readonly STAGE_40_LOG="$RUN_DIR/resume-40k-lr2e5.launch.log"
readonly STAGE_60_LOG="$RUN_DIR/resume-60k-lr2e6.launch.log"
readonly CURRENT_PID="${1:?current trainer PID is required}"

exec 9>"$RUN_DIR/lr-queue.lock"
flock -n 9 || {
    printf 'another LR queue already owns the lock\n' >&2
    exit 1
}
exec >>"$QUEUE_LOG" 2>&1

log() {
    printf '%s %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

verify_step() {
    /usr/bin/python3 - "$CHECKPOINT_DIR" "$1" <<'PY'
import json
import pathlib
import sys

checkpoint_dir = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
pointer = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
manifest = json.loads(
    (checkpoint_dir / pointer["version_dir"] / "manifest.json").read_text(encoding="utf-8")
)
actual = manifest["training_step"]
if actual != expected:
    raise SystemExit(f"expected checkpoint step {expected}, found {actual}")
print(f"verified checkpoint step={actual} model_version={manifest['model_version']}")
PY
}

run_stage() {
    local config="$1"
    local stage_log="$2"
    local expected_step="$3"
    log "starting config=$config expected_step=$expected_step"
    env PYTHONPATH="$PYTHON_DIR" /usr/bin/python3 -m gz.trainer --config "$config" >"$stage_log" 2>&1
    verify_step "$expected_step"
    log "completed expected_step=$expected_step"
}

log "waiting current_pid=$CURRENT_PID"
while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep 30
done

verify_step 20000
log "initial stage complete"
run_stage "$CONFIG_DIR/02-resume-40k-lr2e5.toml" "$STAGE_40_LOG" 40000
run_stage "$CONFIG_DIR/03-resume-60k-lr2e6.toml" "$STAGE_60_LOG" 60000
touch "$RUN_DIR/lr-queue-complete-step-60000"
log "queue complete"
