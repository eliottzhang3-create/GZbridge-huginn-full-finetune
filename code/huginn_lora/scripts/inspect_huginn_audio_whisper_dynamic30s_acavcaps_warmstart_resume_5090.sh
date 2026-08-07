#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${HUGINN_AUDIO_DYNAMIC30S_ACAV_SMOKE_RUN_ROOT:-$REPO_ROOT/outputs/huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart_save_resume/run-20260807_021831}"
SAVE_CHECKPOINT="${HUGINN_AUDIO_DYNAMIC30S_ACAV_SMOKE_SAVE_CHECKPOINT:-$RUN_ROOT/save_phase/v0-20260807-021901/checkpoint-2}"
RESUME_CHECKPOINT="${HUGINN_AUDIO_DYNAMIC30S_ACAV_SMOKE_RESUME_CHECKPOINT:-$RUN_ROOT/resume_phase/v0-20260807-024005/checkpoint-3}"
REPORT="${HUGINN_AUDIO_DYNAMIC30S_ACAV_SMOKE_REPORT:-$RUN_ROOT/acavcaps_warmstart_resume_report.json}"

for required_path in \
  "$SAVE_CHECKPOINT/pytorch_model_fsdp_0" \
  "$RESUME_CHECKPOINT/pytorch_model_fsdp_0" \
  "$SAVE_CHECKPOINT/model_only_warmstart.json"; do
  if [ ! -e "$required_path" ]; then
    echo "Required ACAVCAPS smoke audit path is missing: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$REPORT")"
echo "========== ACAVCAPS WARMSTART/RESUME OFFLINE INSPECTOR =========="
echo "save_checkpoint=$SAVE_CHECKPOINT"
echo "resume_checkpoint=$RESUME_CHECKPOINT"
echo "report=$REPORT"

python -u code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic30s_acavcaps_warmstart_resume.py \
  --save-checkpoint "$SAVE_CHECKPOINT" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --save-step 2 \
  --resume-step 3 \
  --world-size 8 \
  --output-report "$REPORT"
