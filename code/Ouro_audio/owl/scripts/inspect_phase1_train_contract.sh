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
REVERB_ROOT="${OWL_REVERB_ROOT:-$BIDEPTH_ROOT/reverb_extracted/mp3d_reverb}"
OWL_SOURCE_ROOT="${OWL_SOURCE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL}"
OUTPUT="${OWL_PHASE1_TRAIN_CONTRACT_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/phase1_train_contract_audit.json}"
AUDIO_ROOT="${OWL_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"

case "$OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $OUTPUT" >&2
    exit 2
    ;;
esac

ARGS=(
  --bidepth-root "$BIDEPTH_ROOT"
  --reverb-root "$REVERB_ROOT"
  --owl-source-root "$OWL_SOURCE_ROOT"
  --output "$OUTPUT"
  --sample-rir-count "${OWL_SAMPLE_RIR_COUNT:-24}"
  --audio-root "$AUDIO_ROOT"
)

if [[ -n "${OWL_AUDIO_ROOT:-}" ]]; then
  IFS=':' read -r -a AUDIO_ROOTS <<< "$OWL_AUDIO_ROOT"
  for root in "${AUDIO_ROOTS[@]}"; do
    ARGS+=(--audio-root "$root")
  done
fi

python -u code/Ouro_audio/owl/scripts/audit_bidepth_train_contract.py "${ARGS[@]}"
