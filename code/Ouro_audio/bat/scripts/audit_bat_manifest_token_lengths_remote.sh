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
export OMP_NUM_THREADS="${BAT_TOKEN_AUDIT_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${BAT_TOKEN_AUDIT_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BAT_TOKEN_AUDIT_OPENBLAS_NUM_THREADS:-1}"

MANIFEST="${BAT_TOKEN_AUDIT_MANIFEST:?Set BAT_TOKEN_AUDIT_MANIFEST}"
OUTPUT_REPORT="${BAT_TOKEN_AUDIT_OUTPUT_REPORT:?Set BAT_TOKEN_AUDIT_OUTPUT_REPORT}"
MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
PROGRESS_EVERY="${BAT_TOKEN_AUDIT_PROGRESS_EVERY:-10000}"
TAIL_RECORDS="${BAT_TOKEN_AUDIT_TAIL_RECORDS:-650000}"

case "$OUTPUT_REPORT" in
  /hpc_stor03/public|/hpc_stor03/public/*)
    echo "Refusing public audit output: $OUTPUT_REPORT" >&2
    exit 2
    ;;
esac

echo "========== BAT OURO PRODUCTION TOKEN-LENGTH AUDIT =========="
echo "manifest=$MANIFEST"
echo "output=$OUTPUT_REPORT"
echo "model=$MODEL_PATH"
echo "audio_io=disabled dummy_waveform_only"
echo "tail_records=$TAIL_RECORDS"

python -u code/Ouro_audio/bat/scripts/audit_bat_manifest_token_lengths.py \
  --manifest "$MANIFEST" \
  --model-path "$MODEL_PATH" \
  --plugin-path "$PLUGIN_PATH" \
  --output-report "$OUTPUT_REPORT" \
  --progress-every "$PROGRESS_EVERY" \
  --tail-records "$TAIL_RECORDS"
