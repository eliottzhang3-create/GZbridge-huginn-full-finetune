#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

# Metadata-only streaming preparation. One GPU is allocated by the current
# pdgpu submission route but remains unused.
vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j prepare-huginn-dyn90-full-pools-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/prepare_huginn_whisper_dynamic90s_full_atomic_pools_5090.JOB.log" \
  --cmd "bash scripts/prepare_huginn_whisper_dynamic90s_full_atomic_pools.sh"
