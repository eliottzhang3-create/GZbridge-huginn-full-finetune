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

export CUDA_VISIBLE_DEVICES=0

python code/huginn_lora/scripts/huginnI.py \
  --steps 200000000 \
  --log-interval 2000000 \
  --num-samples 8 \
  --batch-size 4 \
  --input-dim 4096 \
  --hidden-dim 8192 \
  --output-dim 4096 \
  --depth 3 \
  --lr 1e-3 \
  --dtype bf16
