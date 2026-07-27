#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_acavcaps_wds_legacy_quarter_fixed32_warmstart2802_e1_b8ga4_5090/run-20260724_073239/v0-20260724-073259/checkpoint-36741
PLUGIN_PATH="$SCRIPT_DIR/plugins/huginn_losatok_swift.py"
OUTPUT_DIR="outputs/mmau_test_mini_losatok_legacy_fixed32_acavcaps_quarter_e1_checkpoint36741"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j eval-mmau-losatok-legacy32-acav-quarter-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/eval_mmau_test_mini_losatok_legacy_acavcaps_quarter_5090.JOB.log" \
  --cmd "HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=0 HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=0 MMAU_CHECKPOINT='$CHECKPOINT' MMAU_OUTPUT_DIR='$OUTPUT_DIR' MMAU_PLUGIN_PATH='$PLUGIN_PATH' bash scripts/eval_mmau_test_mini_swift.sh"
