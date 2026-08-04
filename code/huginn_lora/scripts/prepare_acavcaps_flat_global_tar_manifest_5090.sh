#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
SOURCE_MANIFEST="${ACAVCAPS_FULL_SOURCE_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json}"
OUTPUT_MANIFEST="${ACAVCAPS_FLAT_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps/acavcaps_flat_global_tar_shuffle_seed20260723.json}"
SEED="${ACAVCAPS_FLAT_TAR_SHUFFLE_SEED:-20260723}"
BUFFER="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"

echo "========== PREPARE ACAVCAPS FLAT GLOBAL-TAR MANIFEST =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "source_manifest=$SOURCE_MANIFEST"
echo "output_manifest=$OUTPUT_MANIFEST"
echo "seed=$SEED"
echo "sample_shuffle_buffer=$BUFFER"
echo "schedule=all_1071_tars_one_global_permutation_no_stage_boundaries"

python -u code/huginn_lora/scripts/prepare_acavcaps_flat_global_tar_manifest.py \
  --source_manifest "$SOURCE_MANIFEST" \
  --output_manifest "$OUTPUT_MANIFEST" \
  --seed "$SEED" \
  --sample_shuffle_buffer "$BUFFER" \
  --expected_tar_count 1071 \
  --expected_sample_count 4664169 \
  --overwrite

python -u code/huginn_lora/scripts/inspect_acavcaps_flat_global_tar_manifest.py \
  --manifest "$OUTPUT_MANIFEST" \
  --world_size 8 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --expected_tar_count 1071 \
  --expected_sample_count 4664169 \
  --expected_buffer "$BUFFER" \
  --check_tar_files
