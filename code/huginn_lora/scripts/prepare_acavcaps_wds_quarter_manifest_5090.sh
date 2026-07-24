#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
SOURCE_MANIFEST="${ACAVCAPS_WDS_FULL_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json}"
QUARTER_MANIFEST="${ACAVCAPS_WDS_QUARTER_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_quarter_ceil_seed20260723.json}"

echo "========== PREPARE ACAVCAPS WDS QUARTER MANIFEST =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "source_manifest=$SOURCE_MANIFEST"
echo "quarter_manifest=$QUARTER_MANIFEST"
echo "selection=ceil(N/4) within each category's source randomized order"
echo "stage_tar_order=preserve source global randomized order"
echo "public_dataset_policy=read_only"

python -u code/huginn_lora/scripts/prepare_acavcaps_wds_quarter_manifest.py \
  --source_manifest "$SOURCE_MANIFEST" \
  --output_manifest "$QUARTER_MANIFEST"

python -u code/huginn_lora/scripts/inspect_acavcaps_wds_quarter_manifest.py \
  --manifest "$QUARTER_MANIFEST" \
  --world_size 1 \
  --per_device_batch_size 8 \
  --gradient_accumulation_steps 4

echo "========== ACAVCAPS WDS QUARTER MANIFEST PREPARATION PASSED =========="
