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

WORK_DIR="${HUGINN_AUDIO_DYNAMIC90S_STAGE02_WORK_DIR:-$REPO_ROOT/outputs/huginn_audio_whisper_dynamic90s_stage02/run-$(date +%Y%m%d_%H%M%S)}"
if [ -e "$WORK_DIR" ]; then
  echo "Stage 0-2 work directory already exists; choose a fresh HUGINN_AUDIO_DYNAMIC90S_STAGE02_WORK_DIR: $WORK_DIR" >&2
  exit 1
fi

echo "========== HUGINN WHISPER DYNAMIC90S STAGE 0-2 =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WORK_DIR=$WORK_DIR"
echo "data_source=job_generated_synthetic_wav"
echo "formal_dataset_dependency=none"
echo "whisper_encoder=fully_trainable learning_rate=1e-4"
echo "lora_rank=8 lora_alpha=16 lora_dropout=0.05"
echo "lora_scope=huginn_transformer_only"
echo "fsdp_units_to_validate=whisper_whole,aligner_whole,prelude_2blocks,core_adapter_plus_4blocks,coda_2blocks"
echo "fsdp_reshard_after_forward=true for_all_units=true"
echo "audio_tokens=dynamic complete_120ms_per_token"
echo "audio_over_90s=retain_first_90s no_duration_discard=true"

python -u code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic90s_stage02.py \
  --work-dir "$WORK_DIR"
