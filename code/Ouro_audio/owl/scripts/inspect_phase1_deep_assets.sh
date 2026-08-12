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
ARCHIVE="${OWL_REVERB_ARCHIVE:-$BIDEPTH_ROOT/reverb.tar.gz}"
OUTPUT="${OWL_PHASE1_DEEP_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/phase1_deep_asset_audit.json}"

ARGS=(
  --bidepth-root "$BIDEPTH_ROOT"
  --reverb-archive "$ARCHIVE"
  --output "$OUTPUT"
  --sample-npy-count "${OWL_SAMPLE_NPY_COUNT:-12}"
)

if [[ -n "${OWL_AUDIO_ROOT:-}" ]]; then
  IFS=':' read -r -a AUDIO_ROOTS <<< "$OWL_AUDIO_ROOT"
  for root in "${AUDIO_ROOTS[@]}"; do
    ARGS+=(--audio-root "$root")
  done
fi
if [[ -n "${OWL_SOURCE_ROOT:-}" ]]; then
  ARGS+=(--owl-source-root "$OWL_SOURCE_ROOT")
fi
if [[ "${OWL_PHASE1_DEEP_SHA256:-0}" == "1" ]]; then
  ARGS+=(--sha256)
fi

python -u code/Ouro_audio/owl/scripts/audit_bidepth_deep.py "${ARGS[@]}"
