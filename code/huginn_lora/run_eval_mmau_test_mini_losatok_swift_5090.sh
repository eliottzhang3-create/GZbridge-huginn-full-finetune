#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CHECKPOINT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090/v1-20260720-162632
PLUGIN_PATH="$SCRIPT_DIR/plugins/huginn_losatok_swift.py"
CHECKPOINTS="$CHECKPOINT_ROOT/checkpoint-5604;$CHECKPOINT_ROOT/checkpoint-8406"
OUTPUT_DIR="outputs/mmau_test_mini_losatok_e3_5604_8406"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j eval-mmau-losatok-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/eval_mmau_test_mini_losatok_5090.JOB.log" \
  --cmd "MMAU_CHECKPOINTS='$CHECKPOINTS' MMAU_OUTPUT_DIR='$OUTPUT_DIR' MMAU_PLUGIN_PATH='$PLUGIN_PATH' bash scripts/eval_mmau_test_mini_swift.sh"
