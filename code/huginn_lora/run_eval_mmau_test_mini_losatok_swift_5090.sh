#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

PLUGIN_PATH="$SCRIPT_DIR/plugins/huginn_losatok_swift.py"
CHECKPOINT_2802=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_dynamic90s_audiocaps_v2_e3_b4ga4_fsdp2/v0-20260723-054928/checkpoint-2802
CHECKPOINT_5604=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_dynamic90s_audiocaps_v2_e3_b4ga4_fsdp2/v0-20260723-054928/checkpoint-5604
OUTPUT_DIR="outputs/mmau_test_mini_losatok_dynamic90s_audiocaps_v2_fsdp2"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j eval-mmau-losatok-dynamic90s-fsdp2-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/eval_mmau_test_mini_losatok_5090.JOB.log" \
  --cmd "HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1 HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1 MMAU_CHECKPOINTS='$CHECKPOINT_2802;$CHECKPOINT_5604' MMAU_OUTPUT_DIR='$OUTPUT_DIR' MMAU_PLUGIN_PATH='$PLUGIN_PATH' bash scripts/eval_mmau_test_mini_swift.sh"
