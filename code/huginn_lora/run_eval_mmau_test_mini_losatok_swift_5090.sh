#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

PLUGIN_PATH="$SCRIPT_DIR/plugins/huginn_losatok_swift.py"
CHECKPOINT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_clothoaqa_e1_warmstart2802_b8ga4_5090/v0-20260722-024418/checkpoint-659
OUTPUT_DIR="outputs/mmau_test_mini_losatok_clothoaqa_e1_checkpoint659"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j eval-mmau-losatok-clothoaqa-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/eval_mmau_test_mini_losatok_5090.JOB.log" \
  --cmd "MMAU_CHECKPOINT='$CHECKPOINT' MMAU_OUTPUT_DIR='$OUTPUT_DIR' MMAU_PLUGIN_PATH='$PLUGIN_PATH' bash scripts/eval_mmau_test_mini_swift.sh"
