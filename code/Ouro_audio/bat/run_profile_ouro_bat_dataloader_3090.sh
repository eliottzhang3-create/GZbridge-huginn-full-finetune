#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

DATASET="${OURO_BAT_DATALOADER_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl}"
OUTPUT_REPORT="${OURO_BAT_DATALOADER_OUTPUT_REPORT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/dataloader_profile_stage3_ab_cde.json}"
CMD_PREFIX="OURO_BAT_DATALOADER_DATASET=$(printf '%q' "$DATASET") OURO_BAT_DATALOADER_OUTPUT_REPORT=$(printf '%q' "$OUTPUT_REPORT") "
for name in OURO_MODEL_PATH OURO_BAT_PLUGIN_PATH OURO_BAT_DATALOADER_START_INDEX OURO_BAT_DATALOADER_WARMUP_BATCHES OURO_BAT_DATALOADER_MEASURE_BATCHES OURO_BAT_DATALOADER_PREFETCH_FACTOR OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "ouro-bat-dataloader-profile-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/ouro_bat_dataloader_profile_3090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_profile_ouro_bat_dataloader_remote.sh"
