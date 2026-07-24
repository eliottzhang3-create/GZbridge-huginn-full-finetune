#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

CHECKPOINT_ROOT="${LOSATOK_DYNAMIC_FSDP2_CHECKPOINT_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_dynamic90s_audiocaps_v2_e3_b4ga4_fsdp2/v0-20260723-054928}"

python -u code/huginn_lora/scripts/inspect_losatok_dynamic_fsdp_checkpoint.py \
  --checkpoint "$CHECKPOINT_ROOT/checkpoint-2802" \
  --checkpoint "$CHECKPOINT_ROOT/checkpoint-5604"
