#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${HUGINN_XARES_CONDA_ENV:-env_xares}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${HUGINN_XARES_CUDA_VISIBLE_DEVICES:-0}"
export HUGINN_AUDIO_DYNAMIC90S_FSDP2_NONPERSISTENT_ROPE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

XARES_ROOT="${HUGINN_XARES_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares}"
CHECKPOINT="${HUGINN_XARES_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-20000}"
PLUGIN_PATH="${HUGINN_XARES_PLUGIN_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py}"
DATA_ROOT="${HUGINN_XARES_VOXCELEB1_ROOT:-/hpc_stor03/public/shared/data/mml/VoxCeleb1_origin}"
REAL_COUNT="${HUGINN_XARES_REAL_SAMPLE_COUNT:-4}"
OUTPUT_DIR="${HUGINN_XARES_ENCODER_SMOKE_OUTPUT_DIR:-$REPO_ROOT/outputs/xares_huginn_encoder_smoke}"
REPORT="$OUTPUT_DIR/huginn_xares_encoder_smoke_report.json"

export PYTHONPATH="$XARES_ROOT/src:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========== HUGINN X-ARES ENCODER SYNTHETIC/REAL SMOKE START =========="
echo "active_env=$CONDA_DEFAULT_ENV"
echo "checkpoint=$CHECKPOINT"
echo "plugin_path=$PLUGIN_PATH"
echo "data_root=$DATA_ROOT"
echo "real_count=$REAL_COUNT"
echo "device=cuda:0"
echo "full_xares_knn=false"
python -u "$SCRIPT_DIR/smoke_huginn_whisper_xares_encoder.py" \
  --checkpoint "$CHECKPOINT" \
  --plugin-path "$PLUGIN_PATH" \
  --xares-root "$XARES_ROOT" \
  --data-root "$DATA_ROOT" \
  --real-count "$REAL_COUNT" \
  --output-report "$REPORT" \
  --device cuda:0
echo "========== HUGINN X-ARES ENCODER SYNTHETIC/REAL SMOKE EXIT =========="
