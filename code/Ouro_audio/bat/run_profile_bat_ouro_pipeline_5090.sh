#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

DATASET="${BAT_PROFILE_DATASET:?Set BAT_PROFILE_DATASET}"
OUTPUT_REPORT="${BAT_PROFILE_OUTPUT_REPORT:?Set BAT_PROFILE_OUTPUT_REPORT to a private path}"
WORLD_SIZE="${BAT_PROFILE_WORLD_SIZE:-8}"
STEPS="${BAT_PROFILE_STEPS:-20}"
WARMUP_STEPS="${BAT_PROFILE_WARMUP_STEPS:-3}"
START_INDEX="${BAT_PROFILE_START_INDEX:-0}"
LOCAL_BATCH_SIZE="${BAT_PROFILE_LOCAL_BATCH_SIZE:-2}"
NUM_WORKERS="${BAT_PROFILE_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${BAT_PROFILE_PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${BAT_PROFILE_PERSISTENT_WORKERS:-true}"
OMP_THREADS="${BAT_PROFILE_OMP_NUM_THREADS:-1}"
MKL_THREADS="${BAT_PROFILE_MKL_NUM_THREADS:-1}"
OPENBLAS_THREADS="${BAT_PROFILE_OPENBLAS_NUM_THREADS:-1}"

if [[ "$WORLD_SIZE" != "1" && "$WORLD_SIZE" != "8" ]]; then
  echo "This profiling launcher supports only 1 or 8 pdgpu-5090 ranks; got $WORLD_SIZE" >&2
  exit 2
fi
case "$OUTPUT_REPORT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public profiler output: $OUTPUT_REPORT" >&2
    exit 2
    ;;
esac

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g "$WORLD_SIZE" -n 1 \
  -j "bat-ouro-profile-${WORLD_SIZE}gpu-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_profile_5090.JOB.log" \
  --cmd "BAT_PROFILE_WORLD_SIZE=$(printf '%q' "$WORLD_SIZE") BAT_PROFILE_DATASET=$(printf '%q' "$DATASET") BAT_PROFILE_OUTPUT_REPORT=$(printf '%q' "$OUTPUT_REPORT") BAT_PROFILE_STEPS=$(printf '%q' "$STEPS") BAT_PROFILE_WARMUP_STEPS=$(printf '%q' "$WARMUP_STEPS") BAT_PROFILE_START_INDEX=$(printf '%q' "$START_INDEX") BAT_PROFILE_LOCAL_BATCH_SIZE=$(printf '%q' "$LOCAL_BATCH_SIZE") BAT_PROFILE_NUM_WORKERS=$(printf '%q' "$NUM_WORKERS") BAT_PROFILE_PREFETCH_FACTOR=$(printf '%q' "$PREFETCH_FACTOR") BAT_PROFILE_PERSISTENT_WORKERS=$(printf '%q' "$PERSISTENT_WORKERS") BAT_PROFILE_OMP_NUM_THREADS=$(printf '%q' "$OMP_THREADS") BAT_PROFILE_MKL_NUM_THREADS=$(printf '%q' "$MKL_THREADS") BAT_PROFILE_OPENBLAS_NUM_THREADS=$(printf '%q' "$OPENBLAS_THREADS") bash scripts/profile_bat_ouro_pipeline_remote.sh"
