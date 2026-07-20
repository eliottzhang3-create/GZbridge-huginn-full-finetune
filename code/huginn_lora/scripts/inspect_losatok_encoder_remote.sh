#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOSATOK_ROOT="${LOSATOK_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok}"
LOSATOK_CODE_DIR="${LOSATOK_CODE_DIR:-$REPO_ROOT/code/huginn_lora/LosatokCode}"
LOSATOK_CHECKPOINT_NAME="${LOSATOK_CHECKPOINT_NAME:-losatok_kl1e-3.pth}"
LOSATOK_INPUT_WAV="${LOSATOK_INPUT_WAV:-}"
LOSATOK_OUTPUT_REPORT="${LOSATOK_OUTPUT_REPORT:-$REPO_ROOT/outputs/losatok/losatok_encoder_inspect.json}"

INPUT_ARGS=()
if [ -n "$LOSATOK_INPUT_WAV" ]; then
  INPUT_ARGS=(--input-wav "$LOSATOK_INPUT_WAV")
fi

echo "========== INSPECT LOSATOK REMOTE ENCODER =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "losatok_root=$LOSATOK_ROOT"
echo "losatok_code_dir=$LOSATOK_CODE_DIR"
echo "losatok_checkpoint_name=$LOSATOK_CHECKPOINT_NAME"
echo "losatok_input_wav=${LOSATOK_INPUT_WAV:-<code-example/en.wav>}"
echo "output_report=$LOSATOK_OUTPUT_REPORT"
echo "offline_hf=true"

python -u code/huginn_lora/scripts/inspect_losatok_encoder_remote.py \
  --losatok-root "$LOSATOK_ROOT" \
  --code-dir "$LOSATOK_CODE_DIR" \
  --checkpoint-name "$LOSATOK_CHECKPOINT_NAME" \
  --output-report "$LOSATOK_OUTPUT_REPORT" \
  "${INPUT_ARGS[@]}"
