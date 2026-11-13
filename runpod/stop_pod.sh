#!/usr/bin/env bash
# Stop the pod this script runs in. Safe to call anywhere: outside a RunPod pod
# (RUNPOD_POD_ID unset) or with NEURONOLOGY_DRY_RUN=1 it only prints what it would do.
#
# Order of attempts (UNVERIFIED against a live pod, check during the first M1 job):
#   1. `runpodctl stop pod $RUNPOD_POD_ID`  (runpodctl ships in RunPod images)
#   2. RunPod GraphQL `podStop` with $RUNPOD_API_KEY (if the pod was given one)
#   3. Kill PID 1, so the container exits; for a pod with no restart this ends billing
#      of GPU time, but a stopped container may still bill for its disk.
# Never deletes the pod or the network volume.
set -uo pipefail

log() { echo "[stop_pod $(date -u +%FT%TZ)] $*" >&2; }

if [[ -z "${RUNPOD_POD_ID:-}" || "${NEURONOLOGY_DRY_RUN:-0}" == "1" ]]; then
  log "dry run / not in a pod: would stop pod '${RUNPOD_POD_ID:-<unset>}'"
  exit 0
fi

if command -v runpodctl >/dev/null 2>&1; then
  log "runpodctl stop pod ${RUNPOD_POD_ID}"
  if runpodctl stop pod "${RUNPOD_POD_ID}"; then exit 0; fi
  log "runpodctl failed"
fi

if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
  log "GraphQL podStop ${RUNPOD_POD_ID}"
  # Key goes in a header, never in the URL or on a logged command line.
  if curl -fsS -X POST https://api.runpod.io/graphql \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      --data "{\"query\":\"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}" \
      >/dev/null; then
    exit 0
  fi
  log "GraphQL podStop failed"
fi

log "falling back to killing PID 1"
kill -TERM 1 2>/dev/null || true
sleep 20
kill -KILL 1 2>/dev/null || true
