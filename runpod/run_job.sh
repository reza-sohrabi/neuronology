#!/usr/bin/env bash
# Pod start command (PLAN.md section 3):  runpod/run_job.sh <config.yaml>
#
# The pod's start command (set at create-pod; one pod at a time) is:
#   bash -c 'curl -fsSL https://raw.githubusercontent.com/<owner>/neuronology/<GIT_COMMIT>/runpod/run_job.sh \
#            -o /tmp/run_job.sh && bash /tmp/run_job.sh <config.yaml>'
#
# Required env:  REPO_URL, GIT_COMMIT (full SHA; jobs run only at a pinned commit)
# Optional env:  VOLUME (/workspace), RUN_ID (default <config-stem>-<utc>-<sha7>),
#                MAX_HOURS (default from config `max_hours`, else 4), KEEP_POD=1,
#                NEURONOLOGY_MODE=idle (no job: keep the pod up for IDLE_HOURS, default
#                0.5, so results can be read with exec-pod, then stop),
#                MAX_ATTEMPTS (default 2: starts of the same job before it refuses),
#                NEURONOLOGY_DRY_RUN=1 (no pod stop)
#
# Config convention: the YAML has a top-level `script:` naming a module under scripts/
# (e.g. `script: census`). The job runs
#   uv run python -m scripts.<script> --config <config> --run-id <RUN_ID>
# Large outputs go to $VOLUME/artifacts (NEURONOLOGY_ARTIFACTS); results/ is synced to
# $VOLUME/results_sync/<RUN_ID>/. A done marker $VOLUME/jobs/<config-stem>-<sha7>.done
# makes a restarted container stop at once instead of re-running a finished job.
set -euo pipefail

CONFIG="${1:?usage: run_job.sh <config.yaml>}"
VOLUME="${VOLUME:-/workspace}"
REPO_URL="${REPO_URL:?REPO_URL}"
GIT_COMMIT="${GIT_COMMIT:?GIT_COMMIT (full SHA)}"
[[ "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "GIT_COMMIT must be a full SHA" >&2; exit 2; }

# Any failure before the watchdog takes over must still stop the pod: RunPod restarts an
# exited container, so a bare exit would loop (seen in M1 with a CPU reader pod).
early_stop() {
  echo "[run_job] setup failed (line $1); stopping pod" >&2
  if [[ -n "${RUNPOD_POD_ID:-}" && "${NEURONOLOGY_DRY_RUN:-0}" != "1" ]]; then
    runpodctl stop pod "${RUNPOD_POD_ID}" || curl -fsS -X POST \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" || true
  fi
}
trap 'early_stop $LINENO' ERR

# The volume is shared by pods with different uids; let git use checkouts on it.
# Set for this process tree only (no change to any git config file).
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'

export HF_HOME="${VOLUME}/hf"
export UV_CACHE_DIR="${VOLUME}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${VOLUME}/uv-python"
# The venv lives on the (local, fast) container disk; the package cache on the volume.
export UV_PROJECT_ENVIRONMENT="/root/venv"
export UV_LINK_MODE=copy
export NEURONOLOGY_ARTIFACTS="${VOLUME}/artifacts"
REPO_DIR="${VOLUME}/repo/${GIT_COMMIT}"
mkdir -p "${VOLUME}/repo" "${VOLUME}/jobs" "${HF_HOME}" "${UV_CACHE_DIR}"

# 1. Checkout at the pinned commit (fresh clone per commit; reused if present).
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --quiet "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --quiet origin
git -C "${REPO_DIR}" checkout --quiet --detach "${GIT_COMMIT}"
git -C "${REPO_DIR}" reset --quiet --hard "${GIT_COMMIT}"
cd "${REPO_DIR}"

STEM="$(basename "${CONFIG}" .yaml)"
DONE_MARKER="${VOLUME}/jobs/${STEM}-${GIT_COMMIT:0:7}.done"

if [[ "${NEURONOLOGY_MODE:-job}" == "idle" ]]; then
  export RUN_ID="idle-$(date -u +%Y%m%dT%H%M%SZ)"
  export ON_EXIT_HOOK=/bin/true GIT_COMMIT
  echo "[run_job] idle mode for ${IDLE_HOURS:-0.5}h"
  trap - ERR
  exec "${REPO_DIR}/runpod/watchdog.sh" "${VOLUME}/artifacts/runs/${RUN_ID}" \
    "${IDLE_HOURS:-0.5}" -- sleep infinity
fi

if [[ -f "${DONE_MARKER}" ]]; then
  echo "[run_job] ${DONE_MARKER} exists: job already finished, stopping pod"
  "${REPO_DIR}/runpod/stop_pod.sh"
  exit 0
fi

# A container that keeps restarting (e.g. if a stop failed) must not loop on the job.
ATTEMPTS_FILE="${VOLUME}/jobs/${STEM}-${GIT_COMMIT:0:7}.attempts"
ATTEMPTS=$(( $(cat "${ATTEMPTS_FILE}" 2>/dev/null || echo 0) + 1 ))
echo "${ATTEMPTS}" > "${ATTEMPTS_FILE}"
if (( ATTEMPTS > ${MAX_ATTEMPTS:-2} )); then
  echo "[run_job] attempt ${ATTEMPTS} > ${MAX_ATTEMPTS:-2}: refusing to run again, stopping pod"
  "${REPO_DIR}/runpod/stop_pod.sh"
  exit 1
fi

# 2. Environment from the lockfile (CUDA extra: flash-linear-attention on Linux).
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="${HOME}/.local/bin:${PATH}"
timeout "${SETUP_TIMEOUT_MIN:-30}m" uv sync --frozen --extra cuda || { echo "[run_job] uv sync failed or timed out"; "${REPO_DIR}/runpod/stop_pod.sh"; exit 1; }

# 3. Resolve script, run id and time limit from the config.
read -r SCRIPT CFG_HOURS < <(uv run --frozen python - "${CONFIG}" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["script"], cfg.get("max_hours", 4))
EOF
)
[[ "${SCRIPT}" =~ ^[a-z0-9_]+$ ]] || { echo "bad script name '${SCRIPT}'" >&2; exit 2; }
export RUN_ID="${RUN_ID:-${STEM}-$(date -u +%Y%m%dT%H%M%SZ)-${GIT_COMMIT:0:7}}"
export GIT_COMMIT REPO_DIR VOLUME DONE_MARKER
export STATUS_DIR="${VOLUME}/artifacts/runs/${RUN_ID}"
MAX_HOURS="${MAX_HOURS:-${CFG_HOURS}}"

echo "[run_job] run_id=${RUN_ID} script=${SCRIPT} max_hours=${MAX_HOURS} commit=${GIT_COMMIT}"

# 4. Run under the watchdog: heartbeat, wall-clock limit, sync + pod stop on any exit.
trap - ERR
exec "${REPO_DIR}/runpod/watchdog.sh" "${STATUS_DIR}" "${MAX_HOURS}" -- \
  uv run --frozen python -m "scripts.${SCRIPT}" --config "${CONFIG}" --run-id "${RUN_ID}"
