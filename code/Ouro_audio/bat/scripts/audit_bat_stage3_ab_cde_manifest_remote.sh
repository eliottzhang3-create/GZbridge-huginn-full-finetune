#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MANIFEST="${BAT_STAGE3_AB_CDE_MANIFEST:?Set BAT_STAGE3_AB_CDE_MANIFEST}"
REPORT="${BAT_STAGE3_AB_CDE_REPORT:?Set BAT_STAGE3_AB_CDE_REPORT}"
OUTPUT_REPORT="${BAT_STAGE3_AB_CDE_MANIFEST_AUDIT_REPORT:?Set BAT_STAGE3_AB_CDE_MANIFEST_AUDIT_REPORT}"

echo "========== BAT STAGE-III MANIFEST/REPORT ORDER + DIGEST AUDIT =========="
echo "[manifest] $MANIFEST"
echo "[source-report] $REPORT"
echo "[output-report] $OUTPUT_REPORT"

python -u code/Ouro_audio/bat/scripts/audit_bat_stage3_ab_cde_manifest.py \
  --manifest "$MANIFEST" \
  --report "$REPORT" \
  --output-report "$OUTPUT_REPORT" \
  --expected-global-batch-size 64
