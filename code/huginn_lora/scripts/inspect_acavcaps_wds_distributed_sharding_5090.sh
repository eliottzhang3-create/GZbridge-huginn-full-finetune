#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"
export ACAVCAPS_WDS_MAX_TARS_PER_STAGE="${ACAVCAPS_WDS_MAX_TARS_PER_STAGE:-2}"

PROBE_SAMPLES_PER_RANK="${ACAVCAPS_WDS_PROBE_SAMPLES_PER_RANK:-256}"

echo "========== ACAVCAPS WEBDATASET DISTRIBUTED SHARD INSPECT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "manifest=$ACAVCAPS_WDS_MANIFEST"
echo "buffer_size=$ACAVCAPS_WDS_BUFFER_SIZE"
echo "max_tars_per_stage=$ACAVCAPS_WDS_MAX_TARS_PER_STAGE"
echo "probe_samples_per_rank=$PROBE_SAMPLES_PER_RANK"
echo "consume_all=true"
echo "audio_decode=disabled"

torchrun --standalone --nproc_per_node=2 \
  code/huginn_lora/scripts/inspect_acavcaps_wds_distributed_sharding.py \
  --manifest "$ACAVCAPS_WDS_MANIFEST" \
  --probe_samples_per_rank "$PROBE_SAMPLES_PER_RANK" \
  --consume_all \
  --max_tars_per_stage "$ACAVCAPS_WDS_MAX_TARS_PER_STAGE" \
  --expected_world_size 2
