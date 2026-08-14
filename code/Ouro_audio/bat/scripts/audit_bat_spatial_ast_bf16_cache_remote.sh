#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"

REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SOURCE_MANIFEST="${BAT_FEATURE_SOURCE_MANIFEST:?Set BAT_FEATURE_SOURCE_MANIFEST}"
CACHE_DIR="${BAT_FEATURE_CACHE_DIR:?Set BAT_FEATURE_CACHE_DIR}"
OUTPUT="${BAT_FEATURE_AUDIT_OUTPUT:?Set BAT_FEATURE_AUDIT_OUTPUT to a private output path}"
FINITE_MODE="${BAT_FEATURE_AUDIT_FINITE_MODE:-all}"

case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/bat/scripts/audit_bat_spatial_ast_bf16_cache.py \
  --source-manifest "$SOURCE_MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --output "$OUTPUT" \
  --finite-mode "$FINITE_MODE"
