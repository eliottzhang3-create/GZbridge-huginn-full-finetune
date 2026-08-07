#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-46050"
PLUGIN_PATH="$SCRIPT_DIR/plugins/huginn_audio_whisper_dynamic30s_eval_swift.py"
OUTPUT_DIR="outputs/huginn_whisper_dynamic30s_multiplier_checkpoint46050_clotho_caption_samples"

vc submit \
  -p pdgpu-4090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j generate-whisper-dynamic30s-multiplier-46050-4090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/generate_clotho_caption_samples_huginn_audio_whisper_dynamic30s_multiplier_checkpoint46050_4090.JOB.log" \
  --cmd "CLOTHO_CAPTION_CHECKPOINT='$CHECKPOINT' CLOTHO_CAPTION_OUTPUT_DIR='$OUTPUT_DIR' CLOTHO_CAPTION_PLUGIN_PATH='$PLUGIN_PATH' CLOTHO_CAPTION_SAMPLE_COUNT='3' CLOTHO_CAPTION_MAX_NEW_TOKENS='64' bash scripts/generate_clotho_caption_samples_swift.sh"
