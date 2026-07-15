#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
DATASET_ROOT="${WAVCAPS_DATASET_ROOT:-/hpc_stor03/public/shared/data/raa/WavCaps}"
AUDIO_SUBDIR="${WAVCAPS_AUDIO_SUBDIR:-AudioSet_SL_flac}"
OUTPUT_REPORT="${WAVCAPS_INSPECT_REPORT:-$REPO_ROOT/data/audio_swift/wavcaps_audioset/wavcaps_audioset_inspect.json}"

echo "========== INSPECT WAVCAPS AUDIOSET =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "dataset_root=$DATASET_ROOT"
echo "audio_subdir=$AUDIO_SUBDIR"
echo "output_report=$OUTPUT_REPORT"

python -u code/huginn_lora/scripts/inspect_wavcaps_audioset_dataset.py \
  --dataset_root "$DATASET_ROOT" \
  --audio_subdir "$AUDIO_SUBDIR" \
  --output_report "$OUTPUT_REPORT"
