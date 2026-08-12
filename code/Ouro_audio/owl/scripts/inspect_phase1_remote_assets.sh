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

SAGE_PATH="${OWL_SAGE_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/OWL/SAGE/finetuned.pth}"
BIDEPTH_ROOT="${OWL_BIDEPTH_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth}"
OUTPUT="${OWL_PHASE1_AUDIT_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/phase1_asset_audit.json}"

case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

ARGS=(
  --sage-path "$SAGE_PATH"
  --bidepth-root "$BIDEPTH_ROOT"
  --output "$OUTPUT"
)
if [[ "${OWL_PHASE1_AUDIT_SHA256:-0}" == "1" ]]; then
  ARGS+=(--sha256)
fi
if [[ "${OWL_PHASE1_AUDIT_SKIP_SAGE:-0}" == "1" ]]; then
  ARGS+=(--skip-sage)
fi

python -u code/Ouro_audio/owl/scripts/inspect_phase1_remote_assets.py "${ARGS[@]}"
