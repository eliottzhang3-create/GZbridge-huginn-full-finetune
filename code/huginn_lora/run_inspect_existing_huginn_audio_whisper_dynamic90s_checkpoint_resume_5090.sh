#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

RUN_ROOT="${1:-${HUGINN_AUDIO_DYNAMIC90S_EXISTING_CHECKPOINT_RUN_ROOT:-}}"
if [ -z "$RUN_ROOT" ]; then
  echo "Set HUGINN_AUDIO_DYNAMIC90S_EXISTING_CHECKPOINT_RUN_ROOT or pass the existing run root as argument 1." >&2
  exit 2
fi
printf -v quoted_run_root '%q' "$RUN_ROOT"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j inspect-whisper-dyn90-existing-ckpt-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_existing_huginn_audio_whisper_dynamic90s_checkpoint_resume_5090.JOB.log" \
  --cmd "HUGINN_AUDIO_DYNAMIC90S_EXISTING_CHECKPOINT_RUN_ROOT=${quoted_run_root} bash scripts/inspect_existing_huginn_audio_whisper_dynamic90s_checkpoint_resume.sh"
