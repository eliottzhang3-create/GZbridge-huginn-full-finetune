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
PILOT_REPORT="${HUGINN_DYNAMIC90S_ATOMIC_PILOT_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/pilot/atomic_pilot_report.json}"
OUTPUT_ROOT="${HUGINN_DYNAMIC90S_FULL_POOL_ROOT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s}"
AUDIOCAPS_MANIFEST="${AUDIOCAPS_TRAIN_MANIFEST:-$REPO_ROOT/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl}"
CLOTHO_ROOT="${CLOTHO_CAPTION_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn}"
CLOTHO_TRAIN_MANIFEST="${CLOTHO_TRAIN_MANIFEST:-train_expand.json}"
GIGASPEECH_ROOT="${GIGASPEECH_DATASET_ROOT:-/hpc_stor03/public/shared/data/asr/am/GigaSpeech}"
GIGASPEECH_METADATA="${GIGASPEECH_METADATA_NAME:-GigaSpeech.json}"
PROGRESS_EVERY="${HUGINN_DYNAMIC90S_FULL_POOL_PROGRESS_EVERY:-100000}"
MIN_FREE_GIB="${HUGINN_DYNAMIC90S_FULL_POOL_MIN_FREE_GIB:-10}"

echo "========== PREPARE HUGINN WHISPER DYNAMIC30S FULL ATOMIC POOLS =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "inventory=$INVENTORY"
echo "pilot_report=$PILOT_REPORT"
echo "output_root=$OUTPUT_ROOT"
echo "min_free_gib=$MIN_FREE_GIB"
echo "duration_policy=retain_all_retain_first30s metadata_only=true"
echo "audio_decode=false audio_copy=false full_audio_scan=false token_accounting=false"

python -u code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_contract.py

python -u code/huginn_lora/scripts/prepare_huginn_whisper_dynamic90s_full_atomic_pools.py \
  --contract "$CONTRACT" \
  --inventory_report "$INVENTORY" \
  --pilot_report "$PILOT_REPORT" \
  --output_root "$OUTPUT_ROOT" \
  --audiocaps_manifest "$AUDIOCAPS_MANIFEST" \
  --clotho_root "$CLOTHO_ROOT" \
  --clotho_train_manifest "$CLOTHO_TRAIN_MANIFEST" \
  --gigaspeech_root "$GIGASPEECH_ROOT" \
  --gigaspeech_metadata "$GIGASPEECH_METADATA" \
  --progress_every "$PROGRESS_EVERY" \
  --min_free_gib "$MIN_FREE_GIB"

echo "========== PREPARE HUGINN WHISPER DYNAMIC30S FULL ATOMIC POOLS EXIT =========="
