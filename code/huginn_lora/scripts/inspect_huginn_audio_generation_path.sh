#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CHECKPOINT="${HUGINN_AUDIO_GENERATION_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406}"
OUTPUT_DIR="${HUGINN_AUDIO_GENERATION_INSPECT_OUTPUT_DIR:-$REPO_ROOT/outputs/huginn_audio_generation_path_inspect}"

echo "========== INSPECT HUGINN AUDIO GENERATION PATH =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "checkpoint=$CHECKPOINT"
echo "output_dir=$OUTPUT_DIR"

python -u code/huginn_lora/scripts/inspect_huginn_audio_generation_path.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"
