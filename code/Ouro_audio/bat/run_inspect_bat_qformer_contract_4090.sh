#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

BAT_QFORMER_OUTPUT="${BAT_QFORMER_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/qformer_contract_audit.json}"
BAT_QFORMER_SOURCE="${BAT_QFORMER_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL/src/slam_llm/models/projector.py}"

vc submit \
  -p "${BAT_QUEUE:-pdgpu-5090}" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j "inspect-bat-qformer-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_bat_qformer_contract_5090.JOB.log" \
  --cmd "BAT_QFORMER_SOURCE=$(printf '%q' "$BAT_QFORMER_SOURCE") BAT_QFORMER_OUTPUT=$(printf '%q' "$BAT_QFORMER_OUTPUT") bash scripts/run_bat_qformer_contract_remote.sh"
