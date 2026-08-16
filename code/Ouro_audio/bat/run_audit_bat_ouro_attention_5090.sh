#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

DATASET="${BAT_ATTENTION_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl}"
OUTPUT_REPORT="${BAT_ATTENTION_OUTPUT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/attention_kernel_audit-$(date +%Y%m%d-%H%M%S).json}"
STEPS="${BAT_ATTENTION_STEPS:-3}"
WARMUP_STEPS="${BAT_ATTENTION_WARMUP_STEPS:-2}"
LOCAL_BATCH_SIZE="${BAT_ATTENTION_LOCAL_BATCH_SIZE:-8}"
NUM_WORKERS="${BAT_ATTENTION_NUM_WORKERS:-4}"

case "$OUTPUT_REPORT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public attention audit output: $OUTPUT_REPORT" >&2
    exit 2
    ;;
esac

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 1 -n 1 \
  -j "bat-ouro-attention-audit-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_attention_audit_5090.JOB.log" \
  --cmd "BAT_PROFILE_WORLD_SIZE=1 BAT_PROFILE_DATASET=$(printf '%q' "$DATASET") BAT_PROFILE_OUTPUT_REPORT=$(printf '%q' "$OUTPUT_REPORT") BAT_PROFILE_STEPS=$(printf '%q' "$STEPS") BAT_PROFILE_WARMUP_STEPS=$(printf '%q' "$WARMUP_STEPS") BAT_PROFILE_LOCAL_BATCH_SIZE=$(printf '%q' "$LOCAL_BATCH_SIZE") BAT_PROFILE_NUM_WORKERS=$(printf '%q' "$NUM_WORKERS") BAT_PROFILE_TORCH_COMPILE=false BAT_PROFILE_ATTENTION_PROFILE=true bash scripts/profile_bat_ouro_pipeline_remote.sh"
