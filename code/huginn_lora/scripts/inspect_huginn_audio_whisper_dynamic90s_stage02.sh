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
export HUGINN_AUDIO_DYNAMIC90S_TRAIN_CHAIN_AUDIT=1

WORK_DIR="${HUGINN_AUDIO_DYNAMIC90S_STAGE02_WORK_DIR:-$REPO_ROOT/outputs/huginn_audio_whisper_dynamic90s_stage02}"

echo "========== HUGINN WHISPER DYNAMIC90S STAGE 0-2 =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WORK_DIR=$WORK_DIR"
echo "data_source=job_generated_synthetic_wav"
echo "formal_dataset_dependency=none"
echo "whisper_encoder=frozen"
echo "lora_rank=8 lora_alpha=16 lora_dropout=0.05"
echo "audio_tokens=dynamic complete_120ms_per_token"

python -u code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic90s_stage02.py \
  --work-dir "$WORK_DIR"
