#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
DATASET="${BAT_SMOKE_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage1_smoke_2.jsonl}"
OUTPUT_DIR="${BAT_SMOKE_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/lora_qformer_smoke-$(date +%Y%m%d-%H%M%S)}"
REPORT="${BAT_SMOKE_REPORT:-$OUTPUT_DIR/audit.json}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 48G -g 1 -n 1 \
  -j "bat-ouro-lora-smoke-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_lora_smoke_5090.JOB.log" \
  --cmd "BAT_SMOKE_DATASET=$(printf '%q' "$DATASET") BAT_SMOKE_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_SMOKE_REPORT=$(printf '%q' "$REPORT") bash scripts/run_bat_ouro_lora_smoke_remote.sh"
