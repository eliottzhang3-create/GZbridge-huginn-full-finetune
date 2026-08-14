#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STAGE1="${BAT_STAGE1_MANIFEST:?Set BAT_STAGE1_MANIFEST}"
STAGE2="${BAT_STAGE2_MANIFEST:?Set BAT_STAGE2_MANIFEST}"
STAGE3="${BAT_STAGE3_MANIFEST:?Set BAT_STAGE3_MANIFEST}"
OUTPUT="${BAT_CURRICULUM_MANIFEST:?Set BAT_CURRICULUM_MANIFEST to a private output path}"
REPORT="${BAT_CURRICULUM_REPORT:?Set BAT_CURRICULUM_REPORT to a private output path}"
case "$OUTPUT:$REPORT" in
  */hpc_stor03/public*:*|*:/hpc_stor03/public*) echo "Refusing public output" >&2; exit 2;;
esac

python -u code/Ouro_audio/bat/scripts/compose_bat_curriculum_manifest.py \
  --stage1-manifest "$STAGE1" --stage2-manifest "$STAGE2" --stage3-manifest "$STAGE3" \
  --output "$OUTPUT" --report "$REPORT" --global-batch-size 16
