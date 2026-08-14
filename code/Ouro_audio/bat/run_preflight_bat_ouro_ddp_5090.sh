#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

DATASET="${BAT_DDP_PREFLIGHT_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage1_preflight_160.jsonl}"
OUTPUT_DIR="${BAT_DDP_PREFLIGHT_OUTPUT_DIR:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/ddp_preflight_160-$(date +%Y%m%d-%H%M%S)}"
REPORT="${BAT_DDP_PREFLIGHT_REPORT:-$OUTPUT_DIR/audit.json}"

case "$OUTPUT_DIR:$REPORT" in
  /hpc_stor03/public*|*:/hpc_stor03/public*) echo "Refusing public output" >&2; exit 2;;
esac

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-ouro-ddp-preflight-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_ddp_preflight_5090.JOB.log" \
  --cmd "BAT_DDP_PREFLIGHT_DATASET=$(printf '%q' "$DATASET") BAT_DDP_PREFLIGHT_OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") BAT_DDP_PREFLIGHT_REPORT=$(printf '%q' "$REPORT") bash scripts/run_bat_ouro_ddp_preflight_remote.sh"
