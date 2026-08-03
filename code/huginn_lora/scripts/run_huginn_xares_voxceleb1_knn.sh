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
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

XARES_ROOT="${HUGINN_XARES_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares}"
CHECKPOINT="${HUGINN_XARES_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-20000}"
PLUGIN_PATH="${HUGINN_XARES_PLUGIN_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py}"
DATA_ROOT="${HUGINN_XARES_VOXCELEB1_ROOT:-/hpc_stor03/public/shared/data/mml/VoxCeleb1_origin}"
WORK_ROOT="${HUGINN_XARES_VOXCELEB1_WORK_ROOT:-$REPO_ROOT/outputs/xares_voxceleb1_knn}"
ENCODER_PATH="$SCRIPT_DIR/huginn_whisper_xares_encoder_entry.py"
TASK_PATH="$SCRIPT_DIR/huginn_xares_voxceleb1_task.py"

export HUGINN_XARES_ROOT="$XARES_ROOT"
export HUGINN_XARES_CHECKPOINT="$CHECKPOINT"
export HUGINN_XARES_PLUGIN_PATH="$PLUGIN_PATH"
export HUGINN_XARES_VOXCELEB1_ROOT="$DATA_ROOT"
export HUGINN_XARES_VOXCELEB1_CONFIG_REPORT="$WORK_ROOT/task_config.json"
export PYTHONPATH="$XARES_ROOT/src:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$WORK_ROOT"

echo "========== HUGINN X-ARES VOXCELEB1 KNN START =========="
echo "active_env=$CONDA_DEFAULT_ENV"
echo "xares_root=$XARES_ROOT"
echo "checkpoint=$CHECKPOINT"
echo "data_root=$DATA_ROOT"
echo "encoder=$ENCODER_PATH"
echo "task=$TASK_PATH"
echo "work_root=$WORK_ROOT"
echo "batch_size_encode=${HUGINN_XARES_VOXCELEB1_BATCH_SIZE_ENCODE:-1}"
echo "num_encoder_workers=${HUGINN_XARES_VOXCELEB1_NUM_ENCODER_WORKERS:-0}"
echo "mini=${HUGINN_XARES_VOXCELEB1_USE_MINI_DATASET:-0}"
echo "force_encode=${HUGINN_XARES_VOXCELEB1_FORCE_ENCODE:-1}"
echo "do_knn=${HUGINN_XARES_VOXCELEB1_DO_KNN:-1}"

python -u -m xares.run   --max-jobs 1   "$ENCODER_PATH"   "$TASK_PATH"

echo "========== HUGINN X-ARES VOXCELEB1 KNN EXIT =========="

