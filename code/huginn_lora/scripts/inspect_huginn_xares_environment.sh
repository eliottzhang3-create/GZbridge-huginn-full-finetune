#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${HUGINN_XARES_CUDA_VISIBLE_DEVICES:-0}"
XARES_ROOT="${HUGINN_XARES_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares}"
OUTPUT_DIR="${HUGINN_XARES_ENV_OUTPUT_DIR:-$REPO_ROOT/outputs/xares_environment_preflight}"
REPORT="$OUTPUT_DIR/xares_import_environment_report.json"

export PYTHONPATH="$XARES_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "========== HUGINN X-ARES IMPORT/ENVIRONMENT PREFLIGHT START =========="
echo "active_env=$CONDA_DEFAULT_ENV"
echo "xares_root=$XARES_ROOT"
echo "output_report=$REPORT"
echo "network_or_data_access=false"
echo "audio_decode=false"
echo "checkpoint_load=false"
python -u "$SCRIPT_DIR/inspect_huginn_xares_environment.py" \
  --xares-root "$XARES_ROOT" \
  --output-report "$REPORT"
echo "========== HUGINN X-ARES IMPORT/ENVIRONMENT PREFLIGHT EXIT =========="
