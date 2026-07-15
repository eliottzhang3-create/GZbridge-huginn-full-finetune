#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
echo "========== RUN SWIFT FSDP2 LAUNCH PATH INSPECT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
python -u code/huginn_lora/scripts/inspect_swift_fsdp2_launch_path.py
