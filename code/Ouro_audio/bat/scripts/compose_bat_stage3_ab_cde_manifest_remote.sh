#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SOURCE="${BAT_STAGE3_SOURCE_MANIFEST:?Set BAT_STAGE3_SOURCE_MANIFEST}"
OUTPUT="${BAT_STAGE3_AB_CDE_MANIFEST:?Set BAT_STAGE3_AB_CDE_MANIFEST}"
REPORT="${BAT_STAGE3_AB_CDE_REPORT:?Set BAT_STAGE3_AB_CDE_REPORT}"

python -u code/Ouro_audio/bat/scripts/compose_bat_stage3_ab_cde_manifest.py \
  --stage3-manifest "$SOURCE" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --global-batch-size 64 \
  --epochs 2 \
  --warmup-ratio 0.13 \
  --learning-rate 0.002 \
  --shuffle-seed "${BAT_STAGE3_SHUFFLE_SEED:-42}"
