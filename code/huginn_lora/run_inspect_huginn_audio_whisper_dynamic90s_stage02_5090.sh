#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
if [ -n "${HUGINN_AUDIO_DYNAMIC90S_STAGE02_WORK_DIR:-}" ]; then
  printf -v quoted_work_dir '%q' "$HUGINN_AUDIO_DYNAMIC90S_STAGE02_WORK_DIR"
  CMD_PREFIX="HUGINN_AUDIO_DYNAMIC90S_STAGE02_WORK_DIR=${quoted_work_dir} "
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j inspect-whisper-dynamic90s-stage02-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_huginn_audio_whisper_dynamic90s_stage02_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_huginn_audio_whisper_dynamic90s_stage02.sh"
