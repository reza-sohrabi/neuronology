#!/usr/bin/env bash
# Pod start command (PLAN.md section 3):  runpod/run_job.sh <config.yaml>
#
# Intended pod start command (set via the create-pod call; one pod at a time):
#   bash -c 'curl -fsSL <raw url of this file at GIT_COMMIT> -o /tmp/run_job.sh \
#            && bash /tmp/run_job.sh configs/<experiment>.yaml'
# or, if the repo is already on the volume:  bash /workspace/repo/runpod/run_job.sh <cfg>
#
# Required env:  REPO_URL, GIT_COMMIT (full SHA; jobs run only at a pinned commit)
# Optional env:  VOLUME (/workspace), RUN_ID (default <config-stem>-<utc>-<sha7>),
#                MAX_HOURS (default from config `max_hours`, else 4), KEEP_POD=1,
#                HF_TOKEN, HF_STATUS_REPO, NEURONOLOGY_DRY_RUN=1 (no pod stop)
#
# Config convention: the YAML has a top-level `script:` naming a module under scripts/
# (e.g. `script: census`). The job runs
#   uv run python -m scripts.<script> --config <config> --run-id <RUN_ID>
# with artifacts under $VOLUME/artifacts/runs/<RUN_ID>/ and HF cache on the volume.
set -euo pipefail

CONFIG="${1:?usage: run_job.sh <config.yaml>}"
VOLUME="${VOLUME:-/workspace}"
REPO_URL="${REPO_URL:?REPO_URL}"
GIT_COMMIT="${GIT_COMMIT:?GIT_COMMIT (full SHA)}"
[[ "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "GIT_COMMIT must be a full SHA" >&2; exit 2; }

export HF_HOME="${VOLUME}/hf"
export UV_CACHE_DIR="${VOLUME}/uv-cache"
REPO_DIR="${VOLUME}/repo/${GIT_COMMIT}"
mkdir -p "${VOLUME}/repo" "${HF_HOME}" "${UV_CACHE_DIR}"

# 1. Checkout at the pinned commit (fresh clone per commit; reused if present).
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --quiet "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --quiet origin
git -C "${REPO_DIR}" checkout --quiet --detach "${GIT_COMMIT}"
git -C "${REPO_DIR}" reset --quiet --hard "${GIT_COMMIT}"
cd "${REPO_DIR}"

# 2. Environment from the lockfile.
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="${HOME}/.local/bin:${PATH}"
uv sync --frozen

# 3. Resolve script, run id and time limit from the config.
read -r SCRIPT CFG_HOURS < <(uv run --frozen python - "${CONFIG}" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["script"], cfg.get("max_hours", 4))
EOF
)
[[ "${SCRIPT}" =~ ^[a-z0-9_]+$ ]] || { echo "bad script name '${SCRIPT}'" >&2; exit 2; }
export RUN_ID="${RUN_ID:-$(basename "${CONFIG}" .yaml)-$(date -u +%Y%m%dT%H%M%SZ)-${GIT_COMMIT:0:7}}"
export GIT_COMMIT REPO_DIR VOLUME
export STATUS_DIR="${VOLUME}/artifacts/runs/${RUN_ID}"
MAX_HOURS="${MAX_HOURS:-${CFG_HOURS}}"

echo "[run_job] run_id=${RUN_ID} script=${SCRIPT} max_hours=${MAX_HOURS} commit=${GIT_COMMIT}"

# 4. Run under the watchdog: heartbeat, wall-clock limit, sync + pod stop on any exit.
exec "${REPO_DIR}/runpod/watchdog.sh" "${STATUS_DIR}" "${MAX_HOURS}" -- \
  uv run --frozen python -m "scripts.${SCRIPT}" --config "${CONFIG}" --run-id "${RUN_ID}"
