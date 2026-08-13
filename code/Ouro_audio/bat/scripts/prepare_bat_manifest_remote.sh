#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"

QA_ROOT="${BAT_QA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end}"
STAGE="${BAT_STAGE:?Set BAT_STAGE to I, II or III}"
case "$STAGE" in
  I) SOURCE="$QA_ROOT/stage1-clsdoa/train.json";;
  II) SOURCE="$QA_ROOT/stage2-single/train.json";;
  III) SOURCE="$QA_ROOT/stage3-mixup/train.json";;
  *) echo "Invalid BAT_STAGE=$STAGE" >&2; exit 2;;
esac
OUTPUT="${BAT_MANIFEST_OUTPUT:?Set BAT_MANIFEST_OUTPUT to a private output path}"
case "$OUTPUT" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac

python -u code/Ouro_audio/bat/scripts/prepare_bat_swift_manifest.py \
  --qa-json "$SOURCE" --stage "$STAGE" --output "$OUTPUT" \
  --seed "${BAT_MANIFEST_SEED:-42}" --limit "${BAT_MANIFEST_LIMIT:-0}"
