#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
cd "$ROOT"
export PYTHONPATH="$ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false

MANIFEST="${BAT_ARROW_AUDIT_MANIFEST:?Set BAT_ARROW_AUDIT_MANIFEST}"
OUTPUT="${BAT_ARROW_AUDIT_OUTPUT:?Set BAT_ARROW_AUDIT_OUTPUT}"
CACHE_DIR="${BAT_ARROW_AUDIT_CACHE_DIR:-}"
LIMIT="${BAT_ARROW_AUDIT_LIMIT:-8500}"
RANK="${BAT_ARROW_AUDIT_RANK:-0}"
WORLD="${BAT_ARROW_AUDIT_WORLD_SIZE:-1}"
BATCH="${BAT_ARROW_AUDIT_LOCAL_BATCH_SIZE:-8}"
ARGS=(--manifest "$MANIFEST" --output-report "$OUTPUT" --limit "$LIMIT" --rank "$RANK" --world-size "$WORLD" --local-batch-size "$BATCH")
if [[ -n "$CACHE_DIR" ]]; then ARGS+=(--cache-dir "$CACHE_DIR"); fi
python -u code/Ouro_audio/bat/scripts/audit_bat_arrow_cache_paths.py "${ARGS[@]}"
