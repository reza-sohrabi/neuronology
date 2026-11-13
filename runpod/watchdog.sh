#!/usr/bin/env bash
# Auto-stop wrapper (PLAN.md section 3, cost guardrails).
#
#   runpod/watchdog.sh <status_dir> <max_hours> -- <command...>
#
# - Runs <command> and writes <status_dir>/status.json every HEARTBEAT_S seconds
#   (state, start time, last heartbeat, elapsed, wall-clock limit, exit code).
#   The job itself may write <status_dir>/progress.json (step, loss, ETA); it is
#   merged into status.json on each heartbeat.
# - Kills the command when the wall-clock limit is reached.
# - On ANY exit (success, error, timeout, signal) runs ON_EXIT_HOOK (default: results
#   sync) and then runpod/stop_pod.sh, unless KEEP_POD=1.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_DIR="${1:?status_dir}"; MAX_HOURS="${2:?max_hours}"; shift 2
[[ "${1:-}" == "--" ]] && shift
[[ $# -gt 0 ]] || { echo "usage: watchdog.sh <status_dir> <max_hours> -- <cmd...>" >&2; exit 2; }

HEARTBEAT_S="${HEARTBEAT_S:-30}"
ON_EXIT_HOOK="${ON_EXIT_HOOK:-${HERE}/sync_results.sh}"
mkdir -p "${STATUS_DIR}"
STATUS="${STATUS_DIR}/status.json"
LOG="${STATUS_DIR}/job.log"
START_EPOCH=$(date +%s)
START_ISO=$(date -u +%FT%TZ)
LIMIT_S=$(awk -v h="${MAX_HOURS}" 'BEGIN { printf "%d", h * 3600 }')
STATE="running"; EXIT_CODE="null"; CHILD=""

write_status() {
  local now elapsed progress="null"
  now=$(date +%s); elapsed=$(( now - START_EPOCH ))
  [[ -s "${STATUS_DIR}/progress.json" ]] && progress=$(cat "${STATUS_DIR}/progress.json")
  cat > "${STATUS}.tmp" <<EOF
{"state": "${STATE}", "pod_id": "${RUNPOD_POD_ID:-}", "run_id": "${RUN_ID:-}",
 "git_commit": "${GIT_COMMIT:-}", "started": "${START_ISO}",
 "heartbeat": "$(date -u +%FT%TZ)", "elapsed_s": ${elapsed}, "limit_s": ${LIMIT_S},
 "exit_code": ${EXIT_CODE}, "progress": ${progress}}
EOF
  mv "${STATUS}.tmp" "${STATUS}"
}

finish() {
  local rc=$?
  trap - EXIT INT TERM
  [[ -n "${CHILD}" ]] && kill -TERM "${CHILD}" 2>/dev/null
  [[ "${EXIT_CODE}" == "null" ]] && EXIT_CODE=${rc}
  if [[ "${STATE}" == "running" ]]; then
    if [[ "${EXIT_CODE}" == "0" ]]; then STATE="done"; else STATE="failed"; fi
  fi
  write_status
  echo "[watchdog] state=${STATE} exit=${EXIT_CODE}" | tee -a "${LOG}" >&2
  if [[ -x "${ON_EXIT_HOOK}" ]]; then "${ON_EXIT_HOOK}" || echo "[watchdog] exit hook failed" >&2; fi
  if [[ "${KEEP_POD:-0}" != "1" ]]; then "${HERE}/stop_pod.sh"; fi
  exit "${EXIT_CODE}"
}
trap finish EXIT
trap 'STATE="killed"; EXIT_CODE=143; exit 143' INT TERM

write_status
"$@" >>"${LOG}" 2>&1 &
CHILD=$!
while kill -0 "${CHILD}" 2>/dev/null; do
  if (( $(date +%s) - START_EPOCH >= LIMIT_S )); then
    echo "[watchdog] wall-clock limit ${MAX_HOURS}h reached, killing job" | tee -a "${LOG}" >&2
    STATE="timeout"; EXIT_CODE=124
    kill -TERM "${CHILD}" 2>/dev/null; sleep 30; kill -KILL "${CHILD}" 2>/dev/null
    break
  fi
  write_status
  sleep "${HEARTBEAT_S}" & wait $! 2>/dev/null
done
if [[ "${EXIT_CODE}" == "null" ]]; then
  wait "${CHILD}"; EXIT_CODE=$?
fi
CHILD=""
exit "${EXIT_CODE}"
