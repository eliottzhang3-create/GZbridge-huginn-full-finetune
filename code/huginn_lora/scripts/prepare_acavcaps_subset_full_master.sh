#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
SOURCE_CHUNK_DIR="${SOURCE_CHUNK_DIR:-$REPO_ROOT/data/audio_swift/acavcaps/subset_56_full_1tar_chunks}"
MASTER_MANIFEST="${MASTER_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps/acavcaps_subset_56_full_master_shuffled.jsonl}"

echo "========== PREPARE ACAVCAPS SUBSET MASTER =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "source_chunk_dir=$SOURCE_CHUNK_DIR"
echo "master_manifest=$MASTER_MANIFEST"
python -u code/huginn_lora/scripts/prepare_acavcaps_subset_full_master.py \
  --manifest_dir "$SOURCE_CHUNK_DIR" \
  --output_manifest "$MASTER_MANIFEST" \
  --seed 42 \
  --expected_chunk_count 56 \
  --expected_record_count 235333
