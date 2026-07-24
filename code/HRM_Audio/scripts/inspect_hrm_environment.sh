#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_HRM"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

OUTPUT_REPORT="${HRM_ENV_OUTPUT_REPORT:-$REPO_ROOT/outputs/hrm_text/environment_inspect.json}"

echo "========== INSPECT HRM-TEXT ENVIRONMENT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "OUTPUT_REPORT=$OUTPUT_REPORT"

python -u code/HRM_Audio/scripts/inspect_hrm_environment.py \
  --device cuda:0 \
  --output-report "$OUTPUT_REPORT"
