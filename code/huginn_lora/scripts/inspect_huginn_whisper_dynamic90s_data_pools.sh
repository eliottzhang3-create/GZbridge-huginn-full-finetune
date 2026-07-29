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
AUDIOCAPS_ROOT="${AUDIOCAPS_DATASET_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2}"
WAVCAPS_ROOT="${WAVCAPS_DATASET_ROOT:-/hpc_stor03/public/shared/data/raa/WavCaps}"
CLOTHO_ROOT="${CLOTHO_CAPTION_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn}"
CLOTHO_TRAIN_MANIFEST="${CLOTHO_TRAIN_MANIFEST:-train_expand.json}"
GIGASPEECH_ROOT="${GIGASPEECH_DATASET_ROOT:-/hpc_stor03/public/shared/data/asr/am/GigaSpeech}"
GIGASPEECH_METADATA="${GIGASPEECH_METADATA_NAME:-GigaSpeech.json}"
OUTPUT_REPORT="${HUGINN_DYNAMIC90S_DATA_INSPECT_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/audits/data_pool_inventory.json}"
PROBE_COUNT="${HUGINN_DYNAMIC90S_DATA_PROBE_COUNT:-4}"
METADATA_SCHEMA_RECORDS="${HUGINN_DYNAMIC90S_METADATA_SCHEMA_RECORDS:-20}"

echo "========== INSPECT HUGINN WHISPER DYNAMIC90S DATA POOLS =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "repo_root=$REPO_ROOT"
echo "contract=$CONTRACT"
echo "audiocaps_root=$AUDIOCAPS_ROOT"
echo "wavcaps_root=$WAVCAPS_ROOT read_only=true"
echo "clotho_root=$CLOTHO_ROOT split=train_only manifest=$CLOTHO_TRAIN_MANIFEST"
echo "gigaspeech_root=$GIGASPEECH_ROOT read_only=true metadata=$GIGASPEECH_METADATA"
echo "output_report=$OUTPUT_REPORT"
echo "probe_count=$PROBE_COUNT"
echo "metadata_schema_records=$METADATA_SCHEMA_RECORDS"
echo "metadata_only=true downloads=0 copies=0 conversions=0 audio_decodes=0 full_audio_scans=0"
echo "gpu_is_not_used_by_this_read_only_inventory=true"

python -u code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_data_pools.py \
  --contract "$CONTRACT" \
  --audiocaps_root "$AUDIOCAPS_ROOT" \
  --audiocaps_split train \
  --wavcaps_root "$WAVCAPS_ROOT" \
  --clotho_root "$CLOTHO_ROOT" \
  --clotho_train_manifest "$CLOTHO_TRAIN_MANIFEST" \
  --gigaspeech_root "$GIGASPEECH_ROOT" \
  --gigaspeech_metadata "$GIGASPEECH_METADATA" \
  --probe_count "$PROBE_COUNT" \
  --metadata_schema_records "$METADATA_SCHEMA_RECORDS" \
  --output_report "$OUTPUT_REPORT"

echo "========== INSPECT HUGINN WHISPER DYNAMIC90S DATA POOLS EXIT =========="
