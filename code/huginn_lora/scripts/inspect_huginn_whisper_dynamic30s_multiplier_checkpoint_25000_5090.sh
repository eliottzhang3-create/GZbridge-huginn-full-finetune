#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${HUGINN_AUDIO_DYNAMIC30S_MULTIPLIER_CHECKPOINT_25000:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-25000}"
REPORT="${HUGINN_AUDIO_DYNAMIC30S_MULTIPLIER_CHECKPOINT_25000_REPORT:-$REPO_ROOT/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/checkpoint-25000.single_audit.json}"

python -u code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_single_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --step 25000 \
  --world-size 4 \
  --phase multiplier_formal_checkpoint \
  --output-report "$REPORT" \
  --require-formal-training
