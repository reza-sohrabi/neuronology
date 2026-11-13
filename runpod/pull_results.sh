#!/usr/bin/env bash
# Run on the Mac: pull synced results for one run from a pod over SSH into results/.
#
#   runpod/pull_results.sh <ssh-target> <run_id> [<dest_subdir>]
#
# <ssh-target> is what `ssh` accepts, e.g. root@<ip> -p <port> wrapped as an alias in
# ~/.ssh/config (recommended), since rsync -e quoting of ports is error-prone.
# Copies $VOLUME/results_sync/<run_id>/results/ -> ./results/<dest_subdir or run_id>/
# and status.json/job.log -> ./artifacts/runs/<run_id>/ (gitignored).
set -euo pipefail

TARGET="${1:?ssh target}"; RUN_ID="${2:?run_id}"; SUB="${3:-${RUN_ID}}"
VOLUME="${VOLUME:-/workspace}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${VOLUME}/results_sync/${RUN_ID}"

mkdir -p "${ROOT}/results/${SUB}" "${ROOT}/artifacts/runs/${RUN_ID}"
rsync -av "${TARGET}:${SRC}/results/" "${ROOT}/results/${SUB}/"
rsync -av --include='status.json' --include='progress.json' --include='job.log' \
  --exclude='*' "${TARGET}:${SRC}/" "${ROOT}/artifacts/runs/${RUN_ID}/"
echo "pulled ${RUN_ID} -> results/${SUB}/"
