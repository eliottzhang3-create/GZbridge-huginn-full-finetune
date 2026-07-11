#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
which python || true
python -V || true

python code/huginn_lora/scripts/inspect_acavcaps_dataset.py \
  --samples_per_tar 32 \
  --preview_count 4
