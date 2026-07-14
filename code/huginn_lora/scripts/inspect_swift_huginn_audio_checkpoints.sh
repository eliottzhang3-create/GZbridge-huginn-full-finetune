#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
OUTPUT_REPORT="${SWIFT_AUDIO_CHECKPOINT_INSPECT_REPORT:-$REPO_ROOT/outputs/huginn_audio_retrieval_clotho_v2_swift/checkpoint_inspect.json}"

echo "========== INSPECT SWIFT HUGINN AUDIO CHECKPOINTS =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "output_report=$OUTPUT_REPORT"
python -u code/huginn_lora/scripts/inspect_swift_huginn_audio_checkpoints.py \
  --output_report "$OUTPUT_REPORT"
