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
REGISTRY="${HUGINN_DYNAMIC90S_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/pool_registry.json}"
FULL_REPORT="${HUGINN_DYNAMIC90S_FULL_POOL_REPORT:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/full_pool_report.json}"
OUTPUT_DIR="${HUGINN_DYNAMIC90S_SAMPLER_DIR:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/sampler}"
SEED="${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}"
SIMULATION_DRAWS="${HUGINN_DYNAMIC90S_MIXTURE_SIMULATION_DRAWS:-1000000}"
SCHEDULE_RECORDS="${HUGINN_DYNAMIC90S_MIXTURE_PILOT_RECORDS:-4096}"

echo "========== INSPECT HUGINN WHISPER DYNAMIC30S INDEXED MIXTURE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "registry=$REGISTRY"
echo "full_report=$FULL_REPORT"
echo "output_dir=$OUTPUT_DIR"
echo "seed=$SEED world_size=4 simulation_draws=$SIMULATION_DRAWS"
echo "audio_read=false audio_decode=false token_accounting=false"

python -u code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_indexed_mixture.py \
  --contract "$CONTRACT" \
  --registry "$REGISTRY" \
  --full_report "$FULL_REPORT" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --world_size 4 \
  --simulation_draws "$SIMULATION_DRAWS" \
  --schedule_records "$SCHEDULE_RECORDS" \
  --overwrite

echo "========== INSPECT HUGINN WHISPER DYNAMIC30S INDEXED MIXTURE EXIT =========="
