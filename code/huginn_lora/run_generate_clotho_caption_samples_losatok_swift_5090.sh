#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CHECKPOINT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_dynamic90s_audiocaps_v2_e3_b4ga4_fsdp2/v0-20260723-054928
PLUGIN_PATH="$SCRIPT_DIR/plugins/huginn_losatok_swift.py"
CHECKPOINTS="$CHECKPOINT_ROOT/checkpoint-2802;$CHECKPOINT_ROOT/checkpoint-5604"
OUTPUT_DIR="outputs/huginn_losatok_dynamic90s_clotho_caption_samples_2802_5604"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j generate-losatok-dynamic90s-clotho-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/generate_clotho_caption_losatok_dynamic90s_5090.JOB.log" \
  --cmd "HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1 HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1 CLOTHO_CAPTION_CHECKPOINTS='$CHECKPOINTS' CLOTHO_CAPTION_OUTPUT_DIR='$OUTPUT_DIR' CLOTHO_CAPTION_PLUGIN_PATH='$PLUGIN_PATH' bash scripts/generate_clotho_caption_samples_swift.sh"
