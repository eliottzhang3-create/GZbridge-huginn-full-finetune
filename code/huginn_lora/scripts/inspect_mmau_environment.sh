#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

DATASET_ROOT_ARGS=()
if [ -n "${MMAU_DATASET_ROOT:-}" ]; then
  DATASET_ROOT_ARGS=(--dataset-root "$MMAU_DATASET_ROOT")
fi

OUTPUT_REPORT="${MMAU_INSPECT_OUTPUT:-$REPO_ROOT/data/audio_swift/mmau/mmau_environment_inspect.json}"

echo "========== INSPECT MMAU ENVIRONMENT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "dataset_root=${MMAU_DATASET_ROOT:-<auto-discover>}"
echo "output_report=$OUTPUT_REPORT"

python -u code/huginn_lora/scripts/inspect_mmau_environment.py \
  "${DATASET_ROOT_ARGS[@]}" \
  --output-report "$OUTPUT_REPORT"
