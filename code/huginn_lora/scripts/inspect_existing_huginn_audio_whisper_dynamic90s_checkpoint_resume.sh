#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

RUN_ROOT="${1:-${HUGINN_AUDIO_DYNAMIC90S_EXISTING_CHECKPOINT_RUN_ROOT:-}}"
if [ -z "$RUN_ROOT" ]; then
  echo "Set HUGINN_AUDIO_DYNAMIC90S_EXISTING_CHECKPOINT_RUN_ROOT or pass the existing run root as argument 1." >&2
  exit 2
fi
if [ ! -d "$RUN_ROOT" ]; then
  echo "Existing checkpoint smoke root is missing: $RUN_ROOT" >&2
  exit 1
fi
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

SAVE_STEP=4
RESUME_STEP=6
WORLD_SIZE=4
SEED="${HUGINN_DYNAMIC90S_MIXTURE_SEED:-20260730}"
SAVE_AUDIT_DIR="$RUN_ROOT/save_rank_audits"
RESUME_AUDIT_DIR="$RUN_ROOT/resume_rank_audits"
DATA_AUDIT_DIR="$RUN_ROOT/data_position_audits"
FORWARD_AUDIT_DIR="$RUN_ROOT/forward_consumption_audits"
SAVE_OUTPUT_DIR="$RUN_ROOT/save_phase"
RESUME_OUTPUT_DIR="$RUN_ROOT/resume_phase"
CONTENT_REPORT="$RUN_ROOT/checkpoint_content_report.json"
REGISTRY="${HUGINN_DYNAMIC90S_POOL_REGISTRY:-$REPO_ROOT/data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/pool_registry.json}"
MARKER_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_checkpoint_resume_markers.py"
CHECKPOINT_INSPECTOR="$REPO_ROOT/code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_fsdp_checkpoints.py"

find_checkpoint() {
  local output_dir=$1
  local checkpoint_name=$2
  mapfile -t matches < <(find "$output_dir" -type d -name "$checkpoint_name" -print | sort)
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one $checkpoint_name below $output_dir; found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]:-<none>}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

for required_path in \
  "$SAVE_AUDIT_DIR" "$RESUME_AUDIT_DIR" "$DATA_AUDIT_DIR" \
  "$FORWARD_AUDIT_DIR" \
  "$SAVE_OUTPUT_DIR" "$RESUME_OUTPUT_DIR" "$REGISTRY" \
  "$MARKER_INSPECTOR" "$CHECKPOINT_INSPECTOR"; do
  if [ ! -e "$required_path" ]; then
    echo "Required existing-run audit path is missing: $required_path" >&2
    exit 1
  fi
done

SAVE_CHECKPOINT="$(find_checkpoint "$SAVE_OUTPUT_DIR" "checkpoint-$SAVE_STEP")"
RESUME_CHECKPOINT="$(find_checkpoint "$RESUME_OUTPUT_DIR" "checkpoint-$RESUME_STEP")"
SAVE_STATS_STATE="$SAVE_CHECKPOINT/audio_training_statistics.json"
RESUME_STATS_STATE="$RESUME_CHECKPOINT/audio_training_statistics.json"
for stats_state in "$SAVE_STATS_STATE" "$RESUME_STATS_STATE"; do
  if [ ! -s "$stats_state" ]; then
    echo "Required cumulative training statistics state is missing: $stats_state" >&2
    exit 1
  fi
done

echo "========== HUGINN WHISPER DYNAMIC90S EXISTING CHECKPOINT AUDIT START =========="
echo "scope=posthoc_read_existing_markers_and_fsdp_checkpoints no_training=true no_model_load=true no_audio_decode=true"
echo "run_root=$RUN_ROOT"
echo "save_checkpoint=$SAVE_CHECKPOINT"
echo "resume_checkpoint=$RESUME_CHECKPOINT"

set +e
python -u "$MARKER_INSPECTOR" \
  --save-audit-dir "$SAVE_AUDIT_DIR" \
  --resume-audit-dir "$RESUME_AUDIT_DIR" \
  --data-audit-dir "$DATA_AUDIT_DIR" \
  --forward-audit-dir "$FORWARD_AUDIT_DIR" \
  --save-stats-state "$SAVE_STATS_STATE" \
  --resume-stats-state "$RESUME_STATS_STATE" \
  --registry "$REGISTRY" \
  --seed "$SEED" \
  --save-step "$SAVE_STEP" \
  --resume-step "$RESUME_STEP" \
  --world-size "$WORLD_SIZE"
MARKER_STATUS=$?

python -u "$CHECKPOINT_INSPECTOR" \
  --save-checkpoint "$SAVE_CHECKPOINT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --save-step "$SAVE_STEP" \
  --resume-step "$RESUME_STEP" \
  --world-size "$WORLD_SIZE" \
  --output-report "$CONTENT_REPORT"
CHECKPOINT_STATUS=$?
set -e

echo "[audit-status] markers=$MARKER_STATUS checkpoint_content=$CHECKPOINT_STATUS"
if [ "$MARKER_STATUS" -ne 0 ] || [ "$CHECKPOINT_STATUS" -ne 0 ]; then
  echo "========== HUGINN WHISPER DYNAMIC90S EXISTING CHECKPOINT AUDIT FAILED ==========" >&2
  exit 1
fi

echo "========== HUGINN WHISPER DYNAMIC90S EXISTING CHECKPOINT AUDIT PASSED =========="
echo "content_report=$CONTENT_REPORT"
