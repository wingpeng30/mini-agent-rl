#!/usr/bin/env bash
# Fixed wall-clock runner: experiment -> closed result archive -> local ACK -> off.
# Usage: bash run_final_autodl.sh RUN_DIR PYTHON DRIVER [DRIVER_ARGS...]
# FINAL_STARTED_EPOCH may include the already-spent implementation time.
set -Eeuo pipefail

if (( $# < 3 )); then
  printf 'Usage: %s RUN_DIR PYTHON DRIVER [DRIVER_ARGS...]\n' "$0" >&2
  exit 2
fi
run_dir="$1"
python_bin="$2"
driver="$3"
shift 3
mkdir -p -- "$run_dir"
run_dir="$(cd -- "$run_dir" && pwd -P)"
run_parent="$(dirname -- "$run_dir")"
run_name="$(basename -- "$run_dir")"
archive="$run_parent/$run_name-results.tar.gz"
ack_path="$run_dir/local-received.ack"
started="${FINAL_STARTED_EPOCH:-$(date +%s)}"
total="${FINAL_TOTAL_SECONDS:-86400}"
evaluation_reserve="${FINAL_EVAL_RESERVE_SECONDS:-7200}"
finalization_reserve="${FINAL_FINALIZE_RESERVE_SECONDS:-1200}"
ack_seconds="${FINAL_ACK_SECONDS:-900}"
for value in "$started" "$total" "$evaluation_reserve" "$finalization_reserve" "$ack_seconds"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf 'Deadline settings must be nonnegative integer seconds.\n' >&2
    exit 2
  fi
done
if (( total <= evaluation_reserve || evaluation_reserve <= finalization_reserve )); then
  printf 'Require total > evaluation reserve > finalization reserve.\n' >&2
  exit 2
fi
train_deadline=$(( started + total - evaluation_reserve ))
driver_deadline=$(( started + total - finalization_reserve ))
hard_deadline=$(( started + total ))
export FINAL_HARD_DEADLINE_EPOCH="$hard_deadline"
export FINAL_TRAIN_DEADLINE_EPOCH="$train_deadline"
export PYTHONUNBUFFERED=1
driver_pid=""
watchdog_pid=""
finalized=0

write_state() {
  "$python_bin" - "$run_dir" "$1" "$started" "$train_deadline" "$driver_deadline" "$hard_deadline" "${2:-0}" <<'PY'
import json, sys, time
from pathlib import Path
root, phase, started, train, driver, hard, code = sys.argv[1:]
payload = dict(phase=phase, started_epoch=int(started), train_deadline_epoch=int(train),
               driver_deadline_epoch=int(driver), hard_deadline_epoch=int(hard),
               updated_epoch=int(time.time()), driver_exit_code=int(code))
path = Path(root) / 'wrapper-status.json'
temp = path.with_suffix('.tmp')
temp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
temp.replace(path)
PY
}

stop_driver() {
  if [[ -n "$driver_pid" ]] && kill -0 -- "-$driver_pid" 2>/dev/null; then
    # The PID identifies the session created below; never stop unrelated jobs.
    kill -TERM -- "-$driver_pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$driver_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$driver_pid" 2>/dev/null; then
      kill -KILL -- "-$driver_pid" 2>/dev/null || true
    fi
  fi
  if [[ -n "$driver_pid" ]]; then
    wait "$driver_pid" 2>/dev/null || true
  fi
}

stop_additional_producer() {
  local producer="${FINAL_ADDITIONAL_PRODUCER_PID:-}"
  [[ -z "$producer" ]] && return 0
  if [[ ! "$producer" =~ ^[0-9]+$ ]] || (( producer <= 1 )); then
    printf 'Invalid additional experiment producer PID; refusing to signal it.\n' >&2
    return 1
  fi
  local state="$(ps -o stat= -p "$producer" 2>/dev/null)"
  [[ -z "$state" || "$state" == *Z* ]] && return 0
  local group="$(ps -o pgid= -p "$producer" 2>/dev/null | tr -d '[:space:]')"
  if [[ "$group" != "$producer" ]]; then
    printf 'Additional producer is not its expected experiment session leader.\n' >&2
    return 1
  fi
  kill -TERM -- "-$producer" 2>/dev/null || true
  for _ in {1..30}; do
    state="$(ps -o stat= -p "$producer" 2>/dev/null)"
    [[ -z "$state" || "$state" == *Z* ]] && break
    sleep 1
  done
  state="$(ps -o stat= -p "$producer" 2>/dev/null)"
  if [[ -n "$state" && "$state" != *Z* ]]; then
    kill -KILL -- "-$producer" 2>/dev/null || true
  fi
}

finalize() {
  local exit_code="$1"
  if (( finalized )); then return; fi
  finalized=1
  trap - EXIT INT TERM HUP
  set +e
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
  fi
  stop_driver
  # Stop the original model experiment before snapshotting the appended bare run.
  stop_additional_producer
  write_state finalizing "$exit_code"
  # Reporting has no model calls; partial/error runs are still reported.
  local enrich_script="$(dirname -- "$driver")/enrich_final_metrics.py"
  if [[ -f "$enrich_script" ]]; then
    local enrich_args=("$run_dir")
    [[ -n "${FINAL_METRIC_CASES:-}" ]] && enrich_args+=(--cases "$FINAL_METRIC_CASES")
    [[ -n "${FINAL_METRIC_SOURCE_AUDIT:-}" ]] && enrich_args+=(--source-audit "$FINAL_METRIC_SOURCE_AUDIT")
    timeout --signal=TERM --kill-after=10 180 "$python_bin" "$enrich_script" "${enrich_args[@]}" > "$run_dir/enrich.log" 2>&1
  fi
  local summary_script="$(dirname -- "$driver")/summarize_final_online_grpo.py"
  if [[ -f "$summary_script" ]]; then
    timeout --signal=TERM --kill-after=10 120 "$python_bin" "$summary_script" "$run_dir" > "$run_dir/summary.log" 2>&1
  fi
  # All producers of files in run_dir are now stopped. Archive lives outside it.
  local seconds_left=$(( hard_deadline - $(date +%s) - 30 ))
  local archive_limit=300
  (( seconds_left < archive_limit )) && archive_limit="$seconds_left"
  (( archive_limit < 1 )) && archive_limit=1
  local archive_rc=0
  timeout --signal=TERM --kill-after=10 "$archive_limit" tar -czf "$archive.tmp" \
    --exclude='local-received.ack' --exclude='archive-ready.json' --exclude='shutdown-issued.json' \
    --exclude='.env' --exclude='*.tar.gz' \
    -C "$run_parent" "$run_name" || archive_rc=$?
  local digest=""
  if (( archive_rc == 0 )); then
    mv -- "$archive.tmp" "$archive"
    sha256sum -- "$archive" > "$archive.sha256"
    digest="$(cut -d ' ' -f 1 "$archive.sha256")"
  fi
  "$python_bin" - "$run_dir" "$archive" "$digest" "$archive_rc" "$exit_code" <<'PY'
import json, sys, time
from pathlib import Path
root, archive, digest, archive_rc, driver_rc = sys.argv[1:]
payload = dict(archive=archive, sha256=digest or None, archive_exit_code=int(archive_rc),
               driver_exit_code=int(driver_rc), ready_epoch=int(time.time()),
               ack_path=str(Path(root) / 'local-received.ack'))
(Path(root) / 'archive-ready.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps(payload), flush=True)
PY
  local ack_deadline=$(( $(date +%s) + ack_seconds ))
  local latest_ack=$(( hard_deadline - 30 ))
  (( ack_deadline > latest_ack )) && ack_deadline="$latest_ack"
  local acknowledged=false
  if [[ -n "$digest" ]]; then
    while (( $(date +%s) < ack_deadline )); do
      if [[ -f "$ack_path" ]] && [[ "$(tr -d '\r\n[:space:]' < "$ack_path")" == "$digest" ]]; then
        acknowledged=true
        break
      fi
      sleep 5
    done
  fi
  write_state shutting_down "$exit_code"
  # Persist the attempt before a provider shutdown can stop this process.
  "$python_bin" - "$run_dir" "$acknowledged" "${FINAL_DISABLE_SHUTDOWN:-0}" <<'PY'
import json, os, sys, time
from pathlib import Path
root, acknowledged, disabled = sys.argv[1:]
payload = dict(command='/usr/bin/shutdown', attempted=disabled != '1', invoked=None,
               return_code=None, local_archive_acknowledged=acknowledged == 'true',
               attempted_epoch=int(time.time()))
with (Path(root) / 'shutdown-issued.json').open('w', encoding='utf-8') as stream:
    stream.write(json.dumps(payload, indent=2))
    stream.flush(); os.fsync(stream.fileno())
print(json.dumps(payload), flush=True)
PY
  local shutdown_rc=0
  if [[ "${FINAL_DISABLE_SHUTDOWN:-0}" == 1 ]]; then
    shutdown_rc=0
  else
    # The user authorized the provider's documented shutdown command.
    /usr/bin/shutdown > "$run_parent/shutdown-command.log" 2>&1 || shutdown_rc=$?
  fi
  "$python_bin" - "$run_dir" "$acknowledged" "$shutdown_rc" "${FINAL_DISABLE_SHUTDOWN:-0}" <<'PY'
import json, sys, time
from pathlib import Path
root, acknowledged, code, disabled = sys.argv[1:]
path = Path(root) / 'shutdown-issued.json'
payload = json.loads(path.read_text(encoding='utf-8'))
payload.update(invoked=disabled != '1', return_code=int(code), returned_epoch=int(time.time()))
path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps(payload), flush=True)
PY
}

trap 'finalize "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
write_state running 0
remaining=$(( driver_deadline - $(date +%s) ))
if (( remaining <= 0 )); then
  printf 'Experiment deadline already passed; archiving and shutting down.\n' >&2
  exit 124
fi
# setsid makes termination cover only this experiment and its descendants.
setsid "$python_bin" "$driver" "$@" --run-dir "$run_dir" \
  --train-deadline-epoch "$train_deadline" --deadline-epoch "$driver_deadline" \
  >> "$run_dir/driver.log" 2>&1 &
driver_pid=$!
(
  # Short sleeps avoid leaving a day-long orphaned sleep after an early finish.
  while (( $(date +%s) < driver_deadline )); do sleep 5; done
  kill -TERM -- "-$driver_pid" 2>/dev/null || true
  sleep 30
  kill -KILL -- "-$driver_pid" 2>/dev/null || true
) &
watchdog_pid=$!
set +e
wait "$driver_pid"
driver_rc=$?
set -e
exit "$driver_rc"
