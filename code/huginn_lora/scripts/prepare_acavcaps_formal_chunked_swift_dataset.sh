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
FORMAL_CHUNK_SIZE_TARS="${FORMAL_CHUNK_SIZE_TARS:-8}"
FORMAL_SAMPLES_PER_TAR="${FORMAL_SAMPLES_PER_TAR:-}"
FORMAL_START_CHUNK="${FORMAL_START_CHUNK:-}"
FORMAL_END_CHUNK="${FORMAL_END_CHUNK:-}"
FORMAL_SKIP_EXISTING="${FORMAL_SKIP_EXISTING:-0}"

ARGS=(
  --output_dir "$FORMAL_CHUNK_DIR"
  --text_field long
  --text_index 0
  --chunk_size_tars "$FORMAL_CHUNK_SIZE_TARS"
)

if [ -n "$FORMAL_SAMPLES_PER_TAR" ]; then
  ARGS+=(--samples_per_tar "$FORMAL_SAMPLES_PER_TAR")
fi

if [ -n "$FORMAL_START_CHUNK" ]; then
  ARGS+=(--start_chunk "$FORMAL_START_CHUNK")
fi

if [ -n "$FORMAL_END_CHUNK" ]; then
  ARGS+=(--end_chunk "$FORMAL_END_CHUNK")
fi

if [ "$FORMAL_SKIP_EXISTING" = "1" ]; then
  ARGS+=(--skip_existing)
fi

echo "========== PREPARE ACAVCAPS FORMAL CHUNKS =========="
echo "FORMAL_CHUNK_SIZE_TARS=$FORMAL_CHUNK_SIZE_TARS"
echo "FORMAL_SAMPLES_PER_TAR=${FORMAL_SAMPLES_PER_TAR:-<full>}"
echo "FORMAL_START_CHUNK=${FORMAL_START_CHUNK:-<auto>}"
echo "FORMAL_END_CHUNK=${FORMAL_END_CHUNK:-<auto>}"
echo "FORMAL_SKIP_EXISTING=$FORMAL_SKIP_EXISTING"
python -u code/huginn_lora/scripts/prepare_acavcaps_formal_chunked_swift_dataset.py "${ARGS[@]}"
