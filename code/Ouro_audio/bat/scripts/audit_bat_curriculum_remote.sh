#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MANIFEST="${BAT_CURRICULUM_MANIFEST:?Set BAT_CURRICULUM_MANIFEST}"
REPORT="${BAT_CURRICULUM_REPORT:?Set BAT_CURRICULUM_REPORT}"
AUDIT_REPORT="${BAT_CURRICULUM_AUDIT_REPORT:?Set BAT_CURRICULUM_AUDIT_REPORT}"
case "$AUDIT_REPORT" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac

python -u code/Ouro_audio/bat/scripts/audit_bat_curriculum_manifest.py \
  --manifest "$MANIFEST" --report "$REPORT" --output-report "$AUDIT_REPORT" --global-batch-size 16
echo "========== BAT GLOBAL CURRICULUM MANIFEST AUDIT PASSED =========="
