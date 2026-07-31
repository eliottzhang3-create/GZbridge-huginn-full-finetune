#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
REGISTRY="${HUGINN_MULTIPLIER_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m/multiplier_pool_registry.json}"
FORMAL_ROOT="${HUGINN_MULTIPLIER_FORMAL_OUTPUT_ROOT:-$REPO_ROOT/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4}"
RUN_ROOT="${HUGINN_MULTIPLIER_LIVE_AUDIT_RUN_ROOT:-${HUGINN_MULTIPLIER_FORMAL_RUN_ROOT:-}}"
RECENT_WINDOW="${HUGINN_MULTIPLIER_LIVE_AUDIT_RECENT_WINDOW:-8192}"
REPORT="${HUGINN_MULTIPLIER_LIVE_AUDIT_REPORT:-$(dirname "$REGISTRY")/audits/multiplier_live_usage_$(date +%Y%m%d_%H%M%S).json}"

ARGS=(
  --registry "$REGISTRY"
  --formal-root "$FORMAL_ROOT"
  --recent-duration-window "$RECENT_WINDOW"
  --output-report "$REPORT"
)
if [ -n "$RUN_ROOT" ]; then
  ARGS+=(--run-root "$RUN_ROOT")
fi

echo "========== HUGINN WHISPER DYNAMIC30S MULTIPLIER LIVE USAGE AUDIT START =========="
echo "scope=read_only model_load=false audio_decode=false audio_copy=false"
echo "registry=$REGISTRY"
echo "formal_root=$FORMAL_ROOT run_root=${RUN_ROOT:-<latest-with-statistics>}"
echo "recent_duration_window=$RECENT_WINDOW report=$REPORT"

python -u "$SCRIPT_DIR/inspect_huginn_whisper_dynamic30s_multiplier_live_usage.py" "${ARGS[@]}"

echo "========== HUGINN WHISPER DYNAMIC30S MULTIPLIER LIVE USAGE AUDIT EXIT =========="
