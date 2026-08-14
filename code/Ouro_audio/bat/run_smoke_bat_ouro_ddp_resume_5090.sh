#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"

DATASET="${BAT_DDP_RESUME_DATASET:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage1_ddp_smoke_16.jsonl}"
ROOT_DIR="${BAT_DDP_RESUME_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/ddp_resume_smoke_8x5090-$(date +%Y%m%d-%H%M%S)}"

case "$ROOT_DIR" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-ouro-ddp-resume-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_ddp_resume_5090.JOB.log" \
  --cmd "BAT_DDP_RESUME_DATASET=$(printf '%q' "$DATASET") BAT_DDP_RESUME_ROOT=$(printf '%q' "$ROOT_DIR") bash scripts/run_bat_ouro_ddp_resume_remote.sh"
