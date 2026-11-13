#!/usr/bin/env bash
# In-pod results sync, called by watchdog.sh on exit (and safe to call at any time).
#
# Copies the run's committed-size outputs (results/) and status/log to the network
# volume, and optionally mirrors status.json to a private Hugging Face dataset repo so
# progress can be read without a shell.
#
# Env: VOLUME (default /workspace), RUN_ID, REPO_DIR (checkout), STATUS_DIR,
#      HF_STATUS_REPO (optional, e.g. <user>/neuronology-status; needs HF_TOKEN).
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

if [[ -n "${HF_STATUS_REPO:-}" && -n "${HF_TOKEN:-}" && -f "${STATUS_DIR}/status.json" ]]; then
  ( cd "${REPO_DIR}" && uv run --frozen python - "${STATUS_DIR}/status.json" <<'EOF'
import os, sys
from huggingface_hub import HfApi
run_id = os.environ["RUN_ID"]
HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo=f"runs/{run_id}/status.json",
                    repo_id=os.environ["HF_STATUS_REPO"], repo_type="dataset",
                    commit_message=f"status {run_id}")
EOF
  ) || echo "[sync_results] HF status mirror failed (non-fatal)" >&2
fi
