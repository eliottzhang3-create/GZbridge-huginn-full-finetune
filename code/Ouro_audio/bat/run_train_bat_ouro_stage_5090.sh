#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
STAGE="${BAT_STAGE:?Set BAT_STAGE=I, II or III}"
DATASET="${BAT_STAGE_DATASET:?Set BAT_STAGE_DATASET to a private manifest}"
OUTPUT_DIR="${BAT_STAGE_OUTPUT_DIR:?Set BAT_STAGE_OUTPUT_DIR to a private output directory}"
GPU_COUNT="${BAT_GPU_COUNT:-8}"
WORLD_SIZE="${BAT_WORLD_SIZE:-$GPU_COUNT}"
GRAD_ACCUM="${BAT_GRADIENT_ACCUMULATION_STEPS:-1}"

if [[ "$GPU_COUNT" != "8" || "$WORLD_SIZE" != "8" ]]; then
  echo "This BAT 5090 entry is intentionally fixed to 8 GPUs: GPU_COUNT=$GPU_COUNT WORLD_SIZE=$WORLD_SIZE" >&2
  exit 2
fi
if ! [[ "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]]; then
  echo "BAT_GRADIENT_ACCUMULATION_STEPS must be a positive integer, got: $GRAD_ACCUM" >&2
  exit 2
fi
if [[ "$GRAD_ACCUM" -ne 1 ]]; then
  echo "This BAT run uses the requested global batch 16; accumulation must be 1, got $GRAD_ACCUM" >&2
  exit 2
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g "$GPU_COUNT" -n 1 \
  -j "bat-ouro-stage-${STAGE}-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_stage_${STAGE}_5090.JOB.log" \
  --cmd "BAT_STAGE=$(printf '%q' "$STAGE") BAT_STAGE_DATASET=$(printf '%q' "$DATASET") BAT_STAGE_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_WORLD_SIZE=$(printf '%q' "$WORLD_SIZE") BAT_GRADIENT_ACCUMULATION_STEPS=$(printf '%q' "$GRAD_ACCUM") BAT_LAUNCH_MODE=ddp BAT_RESUME_FROM_CHECKPOINT=$(printf '%q' "${BAT_RESUME_FROM_CHECKPOINT:-}") bash scripts/train_bat_ouro_stage_remote.sh"
