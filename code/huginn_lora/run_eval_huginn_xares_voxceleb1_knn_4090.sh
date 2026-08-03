#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CHECKPOINT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-20000"
PLUGIN_PATH="$SCRIPT_DIR/plugins/huginn_audio_whisper_dynamic90s_swift.py"
DATA_ROOT="/hpc_stor03/public/shared/data/mml/VoxCeleb1_origin"
WORK_ROOT="outputs/xares_voxceleb1_knn_full"

vc submit   -p pdgpu-4090   -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1   -c 8 -m 32G -g 1   -n 1   -j eval-huginn-xares-voxceleb1-knn-4090-$(date +%m%d%H%M)   -d "$SCRIPT_DIR"   JOB=1:1 "$SCRIPT_DIR/log/eval_huginn_xares_voxceleb1_knn_4090.JOB.log"   --cmd "HUGINN_XARES_CHECKPOINT='$CHECKPOINT' HUGINN_XARES_PLUGIN_PATH='$PLUGIN_PATH' HUGINN_XARES_VOXCELEB1_ROOT='$DATA_ROOT' HUGINN_XARES_VOXCELEB1_WORK_ROOT='$WORK_ROOT' HUGINN_XARES_VOXCELEB1_USE_MINI_DATASET=0 HUGINN_XARES_VOXCELEB1_FORCE_ENCODE=1 HUGINN_XARES_VOXCELEB1_DO_KNN=1 bash scripts/run_huginn_xares_voxceleb1_knn.sh"
