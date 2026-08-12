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
SOURCE_ROOT="${OWL_SOURCE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL}"
OUTPUT="${OWL_SAGE_NATIVE_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/phase1_sage_native_audit.json}"

case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

ARGS=(
  --sage-path "$SAGE_PATH"
  --owl-source-root "$SOURCE_ROOT"
  --bidepth-root "$BIDEPTH_ROOT"
  --output "$OUTPUT"
  --device "${OWL_SAGE_DEVICE:-cuda}"
  --real-sample-count "${OWL_SAGE_REAL_SAMPLE_COUNT:-6}"
)
if [[ -n "${OWL_AUDIO_ROOT:-}" ]]; then
  IFS=':' read -r -a AUDIO_ROOTS <<< "$OWL_AUDIO_ROOT"
  for root in "${AUDIO_ROOTS[@]}"; do
    ARGS+=(--audio-root "$root")
  done
fi

python -u code/Ouro_audio/owl/scripts/inspect_sage_native.py "${ARGS[@]}"
