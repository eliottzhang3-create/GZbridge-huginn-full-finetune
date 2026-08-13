#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
STAGE="${BAT_STAGE:?Set BAT_STAGE=I, II or III}"
DATASET="${BAT_STAGE_DATASET:?Set BAT_STAGE_DATASET to a private manifest}"
OUTPUT_DIR="${BAT_STAGE_OUTPUT_DIR:?Set BAT_STAGE_OUTPUT_DIR to a private output directory}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 96G -g "${BAT_GPU_COUNT:-1}" -n 1 \
  -j "bat-ouro-stage-${STAGE}-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_stage_${STAGE}_5090.JOB.log" \
  --cmd "BAT_STAGE=$(printf '%q' "$STAGE") BAT_STAGE_DATASET=$(printf '%q' "$DATASET") BAT_STAGE_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_WORLD_SIZE=$(printf '%q' "${BAT_WORLD_SIZE:-1}") BAT_GRADIENT_ACCUMULATION_STEPS=$(printf '%q' "${BAT_GRADIENT_ACCUMULATION_STEPS:-1}") BAT_RESUME_FROM_CHECKPOINT=$(printf '%q' "${BAT_RESUME_FROM_CHECKPOINT:-}") bash scripts/train_bat_ouro_stage_remote.sh"
