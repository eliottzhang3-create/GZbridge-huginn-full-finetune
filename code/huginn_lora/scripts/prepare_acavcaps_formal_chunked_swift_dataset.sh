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

mkdir -p data/audio_swift/acavcaps/formal_chunks

FORMAL_CHUNK_DIR="$REPO_ROOT/data/audio_swift/acavcaps/formal_chunks"

echo "========== PREPARE ACAVCAPS FORMAL CHUNKS =========="
python -u code/huginn_lora/scripts/prepare_acavcaps_formal_chunked_swift_dataset.py \
  --output_dir "$FORMAL_CHUNK_DIR" \
  --text_field long \
  --text_index 0 \
  --chunk_size_tars 8
