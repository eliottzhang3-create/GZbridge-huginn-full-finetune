#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

CONTRACT="${HUGINN_DYNAMIC90S_DATA_CONTRACT:-$REPO_ROOT/code/huginn_lora/configs/huginn_whisper_dynamic90s_data_contract_v1.json}"
INVENTORY="${HUGINN_DYNAMIC90S_DATA_INSPECT_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/audits/data_pool_inventory.json}"
OUTPUT_DIR="${HUGINN_DYNAMIC90S_ATOMIC_PILOT_DIR:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/pilot}"
RECORDS_PER_POOL="${HUGINN_DYNAMIC90S_ATOMIC_PILOT_RECORDS:-16}"
AUDIOCAPS_MANIFEST="${AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
CLOTHO_ROOT="${CLOTHO_CAPTION_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn}"
CLOTHO_TRAIN_MANIFEST="${CLOTHO_TRAIN_MANIFEST:-train_expand.json}"
WAVCAPS_ROOT="${WAVCAPS_DATASET_ROOT:-/hpc_stor03/public/shared/data/raa/WavCaps}"
GIGASPEECH_ROOT="${GIGASPEECH_DATASET_ROOT:-/hpc_stor03/public/shared/data/asr/am/GigaSpeech}"
GIGASPEECH_METADATA="${GIGASPEECH_METADATA_NAME:-GigaSpeech.json}"

echo "========== PREPARE HUGINN WHISPER DYNAMIC90S ATOMIC PILOT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "contract=$CONTRACT"
echo "inventory=$INVENTORY"
echo "output_dir=$OUTPUT_DIR"
echo "records_per_pool=$RECORDS_PER_POOL"
echo "full_manifests=false audio_decode=false audio_copy=false token_accounting=false"

python -u code/huginn_lora/scripts/prepare_huginn_whisper_dynamic90s_atomic_pilot.py \
  --contract "$CONTRACT" \
  --inventory_report "$INVENTORY" \
  --output_dir "$OUTPUT_DIR" \
  --records_per_pool "$RECORDS_PER_POOL" \
  --audiocaps_manifest "$AUDIOCAPS_MANIFEST" \
  --clotho_root "$CLOTHO_ROOT" \
  --clotho_train_manifest "$CLOTHO_TRAIN_MANIFEST" \
  --wavcaps_root "$WAVCAPS_ROOT" \
  --gigaspeech_root "$GIGASPEECH_ROOT" \
  --gigaspeech_metadata "$GIGASPEECH_METADATA"

echo "========== PREPARE HUGINN WHISPER DYNAMIC90S ATOMIC PILOT EXIT =========="
