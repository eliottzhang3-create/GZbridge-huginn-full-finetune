#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
MANIFEST_DIR="${FORMAL_MANIFEST_DIR:-$REPO_ROOT/data/audio_swift/acavcaps/subset_56_full_1tar_chunks}"

echo "========== INSPECT ACAVCAPS SUBSET DURATIONS =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "manifest_dir=$MANIFEST_DIR"
python -u code/huginn_lora/scripts/inspect_acavcaps_manifest_durations.py \
  --manifest_dir "$MANIFEST_DIR"
