#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
if [[ -f "$USER_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx 'swift_ouro'; then
    conda activate swift_ouro
  fi
fi

REPO_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

BIDEPTH_ROOT="${OWL_BIDEPTH_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth}"
OUTPUT="${OWL_PHASE1_PAPER_TYPE_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/phase1_paper_type_audit.json}"

case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/owl/scripts/audit_bidepth_paper_types.py \
  --bidepth-root "$BIDEPTH_ROOT" \
  --output "$OUTPUT"
