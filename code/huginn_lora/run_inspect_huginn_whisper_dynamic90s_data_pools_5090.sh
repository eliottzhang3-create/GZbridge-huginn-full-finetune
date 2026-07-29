#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

# This is a lightweight metadata-only read. It performs no audio decoding or
# full audio-directory scan. The pdgpu submit path still allocates one unused GPU.
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j inspect-huginn-dyn90-data-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_huginn_whisper_dynamic90s_data_pools_5090.JOB.log" \
  --cmd "bash scripts/inspect_huginn_whisper_dynamic90s_data_pools.sh"
