#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MANIFEST="${BAT_CURRICULUM_MANIFEST:?Set BAT_CURRICULUM_MANIFEST}"
REPORT="${BAT_CURRICULUM_REPORT:?Set BAT_CURRICULUM_REPORT}"
OUTPUT="${BAT_ARROW_SCHEMA_REPORT:?Set BAT_ARROW_SCHEMA_REPORT to a private output path}"
case "$OUTPUT" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac

python -u code/Ouro_audio/bat/scripts/audit_bat_jsonl_arrow_schema.py \
  --manifest "$MANIFEST" \
  --curriculum-report "$REPORT" \
  --expected-count 4204096 \
  --output-report "$OUTPUT"
