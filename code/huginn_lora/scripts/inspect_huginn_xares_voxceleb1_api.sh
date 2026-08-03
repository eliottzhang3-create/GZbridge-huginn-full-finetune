#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${HUGINN_XARES_CONDA_ENV:-env_xares}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
XARES_ROOT="${HUGINN_XARES_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares}"
OUTPUT_DIR="${HUGINN_XARES_VOXCELEB1_API_OUTPUT_DIR:-$REPO_ROOT/outputs/xares_voxceleb1_api_contract}"
REPORT="$OUTPUT_DIR/voxceleb1_xares_api_contract.json"

export PYTHONPATH="$XARES_ROOT/src:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========== HUGINN X-ARES VOXCELEB1 API CONTRACT START =========="
echo "active_env=$CONDA_DEFAULT_ENV"
echo "xares_root=$XARES_ROOT"
echo "output_report=$REPORT"
python -u "$SCRIPT_DIR/inspect_huginn_xares_voxceleb1_api.py" \
  --xares-root "$XARES_ROOT" \
  --output-report "$REPORT"
echo "========== HUGINN X-ARES VOXCELEB1 API CONTRACT EXIT =========="

