#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export ACAVCAPS_WDS_STREAM_MANIFEST="${ACAVCAPS_WDS_STREAM_MANIFEST:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_sampled.json}"
export ACAVCAPS_WDS_STREAM_BUFFER_SIZE="${ACAVCAPS_WDS_STREAM_BUFFER_SIZE:-512}"
export ACAVCAPS_WDS_STREAM_MAX_TARS_PER_STAGE="${ACAVCAPS_WDS_STREAM_MAX_TARS_PER_STAGE:-2}"
export ACAVCAPS_WDS_STREAM_DECODE_EVERY="${ACAVCAPS_WDS_STREAM_DECODE_EVERY:-512}"
export ACAVCAPS_WDS_STREAM_LOG_EVERY="${ACAVCAPS_WDS_STREAM_LOG_EVERY:-1000}"

echo "========== ACAVCAPS WEBDATASET STREAM INSPECT =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "repo_root=$REPO_ROOT"
echo "manifest=$ACAVCAPS_WDS_STREAM_MANIFEST"
echo "buffer_size=$ACAVCAPS_WDS_STREAM_BUFFER_SIZE"
echo "max_tars_per_stage=$ACAVCAPS_WDS_STREAM_MAX_TARS_PER_STAGE"
echo "decode_every=$ACAVCAPS_WDS_STREAM_DECODE_EVERY"

python -u code/huginn_lora/scripts/inspect_acavcaps_wds_stream.py \
  --manifest "$ACAVCAPS_WDS_STREAM_MANIFEST" \
  --buffer-size "$ACAVCAPS_WDS_STREAM_BUFFER_SIZE" \
  --max-tars-per-stage "$ACAVCAPS_WDS_STREAM_MAX_TARS_PER_STAGE" \
  --decode-every "$ACAVCAPS_WDS_STREAM_DECODE_EVERY" \
  --log-every "$ACAVCAPS_WDS_STREAM_LOG_EVERY"
