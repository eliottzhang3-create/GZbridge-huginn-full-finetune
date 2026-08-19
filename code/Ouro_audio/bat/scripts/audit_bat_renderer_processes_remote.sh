#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
cd "$ROOT"
export PYTHONPATH="$ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${BAT_RENDER_DDP_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${BAT_RENDER_DDP_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BAT_RENDER_DDP_OPENBLAS_NUM_THREADS:-1}"

MANIFEST="${BAT_RENDER_DDP_MANIFEST:?Set BAT_RENDER_DDP_MANIFEST}"
OUTPUT="${BAT_RENDER_DDP_OUTPUT:?Set BAT_RENDER_DDP_OUTPUT}"
PROGRESS="${BAT_RENDER_DDP_PROGRESS:-${OUTPUT%.json}.progress}"
AUDIO_ROOT="${BAT_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
REVERB_ROOT="${BAT_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
WORLD="${BAT_RENDER_DDP_WORLD_SIZE:-8}"
BATCH="${BAT_RENDER_DDP_LOCAL_BATCH_SIZE:-8}"
LIMIT="${BAT_RENDER_DDP_GLOBAL_RECORD_LIMIT:-8500}"
ARGS=(--manifest "$MANIFEST" --audio-root "$AUDIO_ROOT" --reverb-root "$REVERB_ROOT" --output-prefix "$OUTPUT" --progress-prefix "$PROGRESS" --global-record-limit "$LIMIT" --local-batch-size "$BATCH")
torchrun --standalone --nproc_per_node="$WORLD" code/Ouro_audio/bat/scripts/audit_bat_renderer_processes.py "${ARGS[@]}"
python -u code/Ouro_audio/bat/scripts/audit_bat_renderer_processes.py "${ARGS[@]}" --combine --world-size-for-combine "$WORLD"
