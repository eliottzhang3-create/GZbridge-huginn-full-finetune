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

echo "========== INSPECT HUGINN LOSATOK SWIFT TRAINABLES =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "repo_root=$REPO_ROOT"
echo "losatok_root=/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok"
echo "losatok_code_dir=$REPO_ROOT/code/huginn_lora/LosatokCode"
python -u code/huginn_lora/scripts/inspect_huginn_losatok_swift_trainables.py
