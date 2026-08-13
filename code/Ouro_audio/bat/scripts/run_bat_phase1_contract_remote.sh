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

BAT_DATA_ROOT="${BAT_DATA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA}"
BAT_QA_ROOT="${BAT_QA_ROOT:-$BAT_DATA_ROOT/closed-end}"
BAT_AUDIO_ROOT="${BAT_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
BAT_REVERB_ROOT="${BAT_REVERB_ROOT:-$BAT_DATA_ROOT/mp3d_reverb}"
BAT_SPATIAL_AST_CODE_ROOT="${BAT_SPATIAL_AST_CODE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST}"
BAT_QFORMER_PATH="${BAT_QFORMER_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL/src/slam_llm/models/projector.py}"
BAT_OUTPUT="${BAT_PHASE1_CONTRACT_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/phase1_data_contract_audit.json}"

case "$BAT_OUTPUT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public output path: $BAT_OUTPUT" >&2
    exit 2
    ;;
esac

python -u code/Ouro_audio/bat/scripts/audit_bat_phase1_data_contract.py \
  --qa-root "$BAT_QA_ROOT" \
  --audio-root "$BAT_AUDIO_ROOT" \
  --reverb-root "$BAT_REVERB_ROOT" \
  --spatial-ast-root "$BAT_SPATIAL_AST_CODE_ROOT" \
  --qformer-path "$BAT_QFORMER_PATH" \
  --output "$BAT_OUTPUT"
