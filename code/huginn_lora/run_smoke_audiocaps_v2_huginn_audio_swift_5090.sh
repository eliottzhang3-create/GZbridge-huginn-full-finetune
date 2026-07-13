#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j smoke-audiocaps-v2-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_audiocaps_v2_huginn_audio_swift_5090.JOB.log" \
  --cmd "AUDIOCAPS_OUTPUT_DIR=outputs/huginn_audio_audiocaps_v2_smoke20_b8ga4_5090 AUDIOCAPS_LOGGING_DIR=outputs/huginn_audio_audiocaps_v2_smoke20_b8ga4_5090/tensorboard AUDIOCAPS_MAX_STEPS=20 AUDIOCAPS_SAVE_STRATEGY=steps AUDIOCAPS_SAVE_STEPS=20 AUDIOCAPS_SAVE_TOTAL_LIMIT=1 AUDIOCAPS_LOGGING_STEPS=1 AUDIOCAPS_REPORT_TO=none bash scripts/train_audiocaps_v2_huginn_audio_swift_5090.sh"
