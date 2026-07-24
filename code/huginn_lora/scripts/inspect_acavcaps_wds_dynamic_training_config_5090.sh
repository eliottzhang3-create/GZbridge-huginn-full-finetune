#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1
unset ACAVCAPS_WDS_MAX_TARS_PER_STAGE

MANIFEST="${ACAVCAPS_WDS_FULL_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json}"
STATS="${ACAVCAPS_WDS_FULL_STATS:-${MANIFEST%.json}.stats.json}"

echo "========== ACAVCAPS DYNAMIC LOSATOK TRAINING CONFIG =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "manifest=$MANIFEST"
echo "stats=$STATS"
echo "dynamic_audio_tokens=$HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS"
echo "max_tars_per_stage=<unset/all>"

python -u code/huginn_lora/scripts/inspect_acavcaps_wds_dynamic_training_config.py \
  --manifest "$MANIFEST" \
  --stats "$STATS" \
  --world_size 2 \
  --per_device_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs "${ACAVCAPS_NUM_TRAIN_EPOCHS:-1}"
