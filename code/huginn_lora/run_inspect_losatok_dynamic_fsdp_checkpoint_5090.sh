#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 4 -m 16G -g 1 \
  -n 1 \
  -j inspect-losatok-dynamic-fsdp2-checkpoint-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_losatok_dynamic_fsdp_checkpoint_5090.JOB.log" \
  --cmd "bash scripts/inspect_losatok_dynamic_fsdp_checkpoint.sh"
