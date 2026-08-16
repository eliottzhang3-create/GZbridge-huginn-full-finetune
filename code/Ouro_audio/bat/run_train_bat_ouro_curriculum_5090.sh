#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
DATASET="${BAT_CURRICULUM_MANIFEST:?Set BAT_CURRICULUM_MANIFEST}"
REPORT="${BAT_CURRICULUM_REPORT:?Set BAT_CURRICULUM_REPORT}"
OUTPUT_DIR="${BAT_CURRICULUM_OUTPUT_DIR:?Set BAT_CURRICULUM_OUTPUT_DIR to a private output path}"
GPU_COUNT="${BAT_GPU_COUNT:-8}"
WORLD_SIZE="${BAT_WORLD_SIZE:-$GPU_COUNT}"
GRAD_ACCUM="${BAT_GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_SEQUENCE_LENGTH="${BAT_MAX_SEQUENCE_LENGTH:-176}"
TORCH_COMPILE="${BAT_TORCH_COMPILE:-true}"
COMPILE_MODE="${BAT_COMPILE_MODE:-reduce-overhead}"
COMPILE_DYNAMIC="${BAT_COMPILE_DYNAMIC:-false}"

if [[ "$GPU_COUNT" != "8" || "$WORLD_SIZE" != "8" || "$GRAD_ACCUM" != "1" ]]; then
  echo "This BAT curriculum entry requires 8 GPUs and accumulation=1" >&2
  exit 2
fi

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-ouro-curriculum-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_curriculum_3090.JOB.log" \
  --cmd "BAT_WORLD_SIZE=8 BAT_GRADIENT_ACCUMULATION_STEPS=1 BAT_MAX_SEQUENCE_LENGTH=$(printf '%q' "$MAX_SEQUENCE_LENGTH") BAT_TORCH_COMPILE=$(printf '%q' "$TORCH_COMPILE") BAT_COMPILE_MODE=$(printf '%q' "$COMPILE_MODE") BAT_COMPILE_DYNAMIC=$(printf '%q' "$COMPILE_DYNAMIC") BAT_CURRICULUM_MANIFEST=$(printf '%q' "$DATASET") BAT_CURRICULUM_REPORT=$(printf '%q' "$REPORT") BAT_CURRICULUM_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_LAUNCH_MODE=ddp bash scripts/train_bat_ouro_curriculum_remote.sh"
