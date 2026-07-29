#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
if [ -n "${HUGINN_AUDIO_DYNAMIC90S_STAGE5_RUN_ROOT:-}" ]; then
  printf -v quoted_run_root '%q' "$HUGINN_AUDIO_DYNAMIC90S_STAGE5_RUN_ROOT"
  CMD_PREFIX="HUGINN_AUDIO_DYNAMIC90S_STAGE5_RUN_ROOT=${quoted_run_root} "
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 128G -g 4 \
  -n 1 \
  -j smoke-whisper-dynamic90s-stage5-fsdp4-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_huginn_audio_whisper_dynamic90s_stage5_stability_fsdp4_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_huginn_audio_whisper_dynamic90s_stage5_stability_fsdp4.sh"
