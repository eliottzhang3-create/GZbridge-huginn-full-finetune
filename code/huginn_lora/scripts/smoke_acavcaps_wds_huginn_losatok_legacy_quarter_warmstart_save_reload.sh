#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_QUARTER_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_quarter_ceil_seed20260723.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"
# This is deliberately a short train/save/reload smoke.  It checks the first
# two selected tar files from each stage; formal training must unset this cap.
export ACAVCAPS_WDS_MAX_TARS_PER_STAGE="${ACAVCAPS_WDS_MAX_TARS_PER_STAGE:-2}"
export LOSATOK_LEGACY_ACAV_WDS_SMOKE_ROOT="${LOSATOK_LEGACY_ACAV_WDS_QUARTER_SMOKE_ROOT:-outputs/huginn_losatok_acavcaps_wds_legacy_quarter_warmstart_save_reload/run-$(date +%Y%m%d_%H%M%S)}"

echo "========== ACAVCAPS LEGACY LOSATOK QUARTER WARM-START SMOKE =========="
echo "quarter_manifest=$ACAVCAPS_WDS_MANIFEST"
echo "buffer_size=$ACAVCAPS_WDS_BUFFER_SIZE max_tars_per_stage=$ACAVCAPS_WDS_MAX_TARS_PER_STAGE"
echo "scope=metadata_preflight_all_271_tars+training_smoke_first_2_selected_tars_per_stage"

python -u code/huginn_lora/scripts/inspect_acavcaps_wds_quarter_manifest.py \
  --manifest "$ACAVCAPS_WDS_MANIFEST" \
  --world_size 1 \
  --per_device_batch_size 8 \
  --gradient_accumulation_steps 4

exec bash "$SCRIPT_DIR/smoke_acavcaps_wds_huginn_losatok_legacy_warmstart_save_reload.sh"
