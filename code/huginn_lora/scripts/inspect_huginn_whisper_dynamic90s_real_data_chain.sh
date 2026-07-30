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

REGISTRY="${HUGINN_DYNAMIC90S_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/pool_registry.json}"
PLUGIN="${HUGINN_DYNAMIC90S_MIXTURE_PLUGIN:-$REPO_ROOT/code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_swift.py}"
REPORT="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/real_data_chain_report.json}"
SEED="${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}"
ROWS="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_ROWS:-256}"
RESUME_POSITION="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_RESUME_POSITION:-37}"
RESUME_ROWS="${HUGINN_DYNAMIC90S_REAL_DATA_CHAIN_RESUME_ROWS:-8}"

echo "========== INSPECT HUGINN WHISPER DYNAMIC30S REAL DATA CHAIN =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "registry=$REGISTRY"
echo "plugin=$PLUGIN"
echo "report=$REPORT"
echo "seed=$SEED rows=$ROWS resume_position=$RESUME_POSITION resume_rows=$RESUME_ROWS"
echo "scope=model_load:false whisper_load:false real_audio_decodes:4 audio_copy:false opus_conversion:false"

python -u code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_real_data_chain.py \
  --registry "$REGISTRY" \
  --plugin "$PLUGIN" \
  --output_report "$REPORT" \
  --seed "$SEED" \
  --rows "$ROWS" \
  --resume_position "$RESUME_POSITION" \
  --resume_probe_rows "$RESUME_ROWS" \
  --overwrite

echo "========== INSPECT HUGINN WHISPER DYNAMIC30S REAL DATA CHAIN EXIT =========="
