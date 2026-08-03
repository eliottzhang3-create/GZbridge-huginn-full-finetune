#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${HUGINN_XARES_CONDA_ENV:-env_xares}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${HUGINN_XARES_CUDA_VISIBLE_DEVICES:-0}"
XARES_ROOT="${HUGINN_XARES_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares}"
DATA_ROOT="${HUGINN_XARES_VOXCELEB1_ROOT:-/hpc_stor03/public/shared/data/mml/VoxCeleb1_origin}"
OUTPUT_DIR="${HUGINN_XARES_VOXCELEB1_OUTPUT_DIR:-$REPO_ROOT/outputs/xares_voxceleb1_data_audit}"
REPORT="$OUTPUT_DIR/voxceleb1_data_path_report.json"

export PYTHONPATH="$XARES_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "========== HUGINN X-ARES VOXCELEB1 DATA PATH AUDIT START =========="
echo "active_env=$CONDA_DEFAULT_ENV"
echo "xares_root=$XARES_ROOT"
echo "data_root=$DATA_ROOT"
echo "output_report=$REPORT"
echo "audio_decode=false"
echo "audio_copy=false"
echo "full_audio_scan=false"
python -u "$SCRIPT_DIR/inspect_huginn_xares_voxceleb1_data.py" \
  --xares-root "$XARES_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-report "$REPORT"
echo "========== HUGINN X-ARES VOXCELEB1 DATA PATH AUDIT EXIT =========="
