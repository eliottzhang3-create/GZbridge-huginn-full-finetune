#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-20000"
PLUGIN_PATH="$SCRIPT_DIR/plugins/huginn_audio_whisper_dynamic30s_eval_swift.py"
OUTPUT_DIR="outputs/huginn_whisper_dynamic30s_multiplier_checkpoint20000_clotho_retrieval"

vc submit \
  -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j eval-whisper-dynamic30s-multiplier-20000-4090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/eval_huginn_audio_whisper_dynamic30s_multiplier_checkpoint20000_4090.JOB.log" \
  --cmd "SWIFT_RETRIEVAL_CHECKPOINTS='$CHECKPOINT' SWIFT_RETRIEVAL_OUTPUT_DIR='$OUTPUT_DIR' SWIFT_RETRIEVAL_PLUGIN_PATH='$PLUGIN_PATH' SWIFT_RETRIEVAL_SAMPLE_COUNT='all' bash scripts/eval_huginn_audio_text_retrieval_swift.sh"

