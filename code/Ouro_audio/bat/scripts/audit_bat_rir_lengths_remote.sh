#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

REVERB_ROOT="${BAT_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
OUTPUT="${BAT_RIR_AUDIT_OUTPUT:?Set BAT_RIR_AUDIT_OUTPUT to a private output path}"
SAMPLE_RATE="${BAT_RIR_SAMPLE_RATE:-32000}"
TARGET_SECONDS="${BAT_RIR_TARGET_SECONDS:-2.0}"
LIMIT="${BAT_RIR_AUDIT_LIMIT:-0}"
PREVIEW_COUNT="${BAT_RIR_AUDIT_PREVIEW_COUNT:-20}"

case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/bat/scripts/audit_bat_rir_lengths.py \
  --reverb-root "$REVERB_ROOT" \
  --output "$OUTPUT" \
  --sample-rate "$SAMPLE_RATE" \
  --target-seconds "$TARGET_SECONDS" \
  --limit "$LIMIT" \
  --preview-count "$PREVIEW_COUNT"
