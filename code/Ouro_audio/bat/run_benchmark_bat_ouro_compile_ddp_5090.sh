#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

DATASET="${BAT_COMPILE_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl}"
OUTPUT_REPORT="${BAT_COMPILE_OUTPUT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/compile_ddp_8x5090-$(date +%Y%m%d-%H%M%S).json}"
STEPS="${BAT_COMPILE_STEPS:-10}"
WARMUP_STEPS="${BAT_COMPILE_WARMUP_STEPS:-3}"
LOCAL_BATCH_SIZE="${BAT_COMPILE_LOCAL_BATCH_SIZE:-2}"
NUM_WORKERS="${BAT_COMPILE_NUM_WORKERS:-4}"
COMPILE_MODE="${BAT_COMPILE_MODE:-default}"
COMPILE_DYNAMIC="${BAT_COMPILE_DYNAMIC:-true}"
STATIC_SEQUENCE_LENGTH="${BAT_COMPILE_STATIC_SEQUENCE_LENGTH:-}"

case "$OUTPUT_REPORT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public compile benchmark output: $OUTPUT_REPORT" >&2
    exit 2
    ;;
esac

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 64 -m 256G -g 8 -n 1 \
  -j "bat-ouro-compile-ddp-8x3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_compile_ddp_3090.JOB.log" \
  --cmd "BAT_PROFILE_WORLD_SIZE=8 BAT_PROFILE_DATASET=$(printf '%q' "$DATASET") BAT_PROFILE_OUTPUT_REPORT=$(printf '%q' "$OUTPUT_REPORT") BAT_PROFILE_STEPS=$(printf '%q' "$STEPS") BAT_PROFILE_WARMUP_STEPS=$(printf '%q' "$WARMUP_STEPS") BAT_PROFILE_LOCAL_BATCH_SIZE=$(printf '%q' "$LOCAL_BATCH_SIZE") BAT_PROFILE_NUM_WORKERS=$(printf '%q' "$NUM_WORKERS") BAT_PROFILE_TORCH_COMPILE=true BAT_PROFILE_COMPILE_MODE=$(printf '%q' "$COMPILE_MODE") BAT_PROFILE_COMPILE_DYNAMIC=$(printf '%q' "$COMPILE_DYNAMIC") BAT_PROFILE_STATIC_SEQUENCE_LENGTH=$(printf '%q' "$STATIC_SEQUENCE_LENGTH") BAT_PROFILE_ATTENTION_PROFILE=false bash scripts/profile_bat_ouro_pipeline_remote.sh"
