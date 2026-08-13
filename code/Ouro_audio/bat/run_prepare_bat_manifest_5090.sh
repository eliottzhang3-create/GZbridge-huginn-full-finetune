#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
STAGE="${BAT_STAGE:?Set BAT_STAGE=I, II or III}"
OUTPUT="${BAT_MANIFEST_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage${STAGE}_train.jsonl}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-manifest-${STAGE}-5090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_manifest_${STAGE}_5090.JOB.log" \
  --cmd "BAT_STAGE=$(printf '%q' "$STAGE") BAT_MANIFEST_OUTPUT=$(printf '%q' "$OUTPUT") bash scripts/prepare_bat_manifest_remote.sh"
