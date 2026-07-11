#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
which python || true
python -V || true

export PYTHONUNBUFFERED=1
echo "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"

mkdir -p data/audio_swift/acavcaps

PILOT_MANIFEST="$REPO_ROOT/data/audio_swift/acavcaps/acavcaps_caption_long_pilot_swift.jsonl"

echo "========== PREPARE ACAVCAPS PILOT ONLY =========="
python -u code/huginn_lora/scripts/prepare_acavcaps_swift_dataset.py \
  --output_manifest "$PILOT_MANIFEST" \
  --text_field long \
  --text_index 0 \
  --samples_per_tar 64
