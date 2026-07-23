#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_sampled.json}"
export ACAVCAPS_WDS_MAX_TARS_PER_STAGE="${ACAVCAPS_WDS_MAX_TARS_PER_STAGE:-2}"

echo "========== ACAVCAPS SWIFT ITERABLE DATASET INSPECT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "manifest=$ACAVCAPS_WDS_MANIFEST"
echo "max_tars_per_stage=$ACAVCAPS_WDS_MAX_TARS_PER_STAGE"
python -u code/huginn_lora/scripts/inspect_acavcaps_wds_swift_dataset.py
