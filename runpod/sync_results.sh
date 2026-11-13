#!/usr/bin/env bash
# In-pod results sync, called by watchdog.sh on exit (and safe to call at any time).
#
# Copies the run's committed-size outputs (results/) and status/log to the network
# volume. Progress is read with the RunPod MCP tools (exec-pod, stream-pod-logs); there is
# no external status mirror (owner decision, M1).
#
# Env: VOLUME (default /workspace), RUN_ID, REPO_DIR (checkout), STATUS_DIR.
set -uo pipefail

VOLUME="${VOLUME:-/workspace}"
RUN_ID="${RUN_ID:?RUN_ID}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
STATUS_DIR="${STATUS_DIR:-${VOLUME}/artifacts/runs/${RUN_ID}}"
DEST="${VOLUME}/results_sync/${RUN_ID}"

mkdir -p "${DEST}"
if [[ -d "${REPO_DIR}/results" ]]; then
  rsync -a "${REPO_DIR}/results/" "${DEST}/results/"
fi
for f in status.json progress.json job.log; do
  [[ -f "${STATUS_DIR}/${f}" ]] && cp "${STATUS_DIR}/${f}" "${DEST}/"
done
echo "[sync_results] synced to ${DEST}" >&2

