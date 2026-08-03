#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${HUGINN_XARES_CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="${HUGINN_XARES_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-15000}"
EXPECTED_STEP="${HUGINN_XARES_EXPECTED_STEP:-15000}"
EXPECTED_PHASE="${HUGINN_XARES_EXPECTED_PHASE:-multiplier_formal_checkpoint}"
WORLD_SIZE="${HUGINN_XARES_WORLD_SIZE:-4}"
MODEL_CONFIG="${HUGINN_XARES_MODEL_CONFIG:-$REPO_ROOT/models/huginn-audio-whisper-dynamic90s-v1/config.json}"
OUTPUT_DIR="${HUGINN_XARES_INSPECT_OUTPUT_DIR:-$REPO_ROOT/outputs/xares_checkpoint_inspect}"
REPORT="$OUTPUT_DIR/checkpoint-${EXPECTED_STEP}_readonly_report.json"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --expected-step "$EXPECTED_STEP"
  --expected-phase "$EXPECTED_PHASE"
  --world-size "$WORLD_SIZE"
  --model-config "$MODEL_CONFIG"
  --output-report "$REPORT"
)
if [ "${HUGINN_XARES_SKIP_TENSOR_PROBES:-0}" = "1" ]; then
  ARGS+=(--skip-tensor-probes)
fi

echo "========== HUGINN X-ARES CHECKPOINT INSPECT START =========="
echo "checkpoint=$CHECKPOINT"
echo "expected_step=$EXPECTED_STEP"
echo "world_size=$WORLD_SIZE"
echo "model_config=$MODEL_CONFIG"
echo "output_report=$REPORT"
python -u "$SCRIPT_DIR/inspect_huginn_xares_checkpoint.py" "${ARGS[@]}"
echo "========== HUGINN X-ARES CHECKPOINT INSPECT EXIT =========="
