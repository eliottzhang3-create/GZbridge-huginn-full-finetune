#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
cd "$ROOT"
export PYTHONPATH="$ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${BAT_DIAG_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${BAT_DIAG_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BAT_DIAG_OPENBLAS_NUM_THREADS:-1}"

MANIFEST="${BAT_AUDIO_AUDIT_MANIFEST:?Set BAT_AUDIO_AUDIT_MANIFEST}"
OUTPUT="${BAT_AUDIO_AUDIT_OUTPUT:?Set BAT_AUDIO_AUDIT_OUTPUT}"
PROGRESS="${BAT_AUDIO_AUDIT_PROGRESS:-${OUTPUT%.json}.progress.json}"
AUDIO_ROOT="${BAT_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
REVERB_ROOT="${BAT_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
LIMIT="${BAT_AUDIO_AUDIT_LIMIT:-8500}"
START="${BAT_AUDIO_AUDIT_START_INDEX:-0}"

case "$OUTPUT$PROGRESS" in
  /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;;
esac
python -u code/Ouro_audio/bat/scripts/audit_bat_audio_rows.py \
  --manifest "$MANIFEST" --audio-root "$AUDIO_ROOT" --reverb-root "$REVERB_ROOT" \
  --output-report "$OUTPUT" --progress-file "$PROGRESS" --limit "$LIMIT" --start-index "$START"
