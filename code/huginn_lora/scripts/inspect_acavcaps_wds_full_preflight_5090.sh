#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ACAVCAPS_WDS_SEED="${ACAVCAPS_WDS_SEED:-20260723}"
export ACAVCAPS_WDS_SAMPLE_SHUFFLE_BUFFER="${ACAVCAPS_WDS_SAMPLE_SHUFFLE_BUFFER:-512}"

PRIVATE_ROOT="${ACAVCAPS_WDS_PRIVATE_ROOT:-$REPO_ROOT/data/audio_swift/acavcaps_wds}"
MANIFEST_OUT="${ACAVCAPS_WDS_FULL_MANIFEST_OUT:-$PRIVATE_ROOT/acavcaps_wds_stage_schedule_full_seed${ACAVCAPS_WDS_SEED}.json}"

echo "========== ACAVCAPS WEBDATASET FULL PREFLIGHT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "repo_root=$REPO_ROOT"
echo "public_dataset_root=/hpc_stor03/public/shared/data/raa/ACAVCAPS"
echo "private_manifest=$MANIFEST_OUT"
echo "private_stats=${MANIFEST_OUT%.json}.stats.json"
echo "seed=$ACAVCAPS_WDS_SEED"
echo "sample_shuffle_buffer=$ACAVCAPS_WDS_SAMPLE_SHUFFLE_BUFFER"
echo "scan_mode=full"
echo "scan_tars_per_stage=ignored_in_full_mode"
echo "public_root_policy=read_only"

python -u code/huginn_lora/scripts/inspect_acavcaps_wds_preflight.py \
  --dataset_root /hpc_stor03/public/shared/data/raa/ACAVCAPS \
  --manifest_out "$MANIFEST_OUT" \
  --seed "$ACAVCAPS_WDS_SEED" \
  --sample_shuffle_buffer "$ACAVCAPS_WDS_SAMPLE_SHUFFLE_BUFFER" \
  --scan_mode full \
  --scan_tars_per_stage 1

