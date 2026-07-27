#!/bin/bash
set -euo pipefail

# Independent HRM-Text + Whisper Clotho-v2 samples generator.
# No Huginn generator is imported or modified by this route.
CONDA_BASE="$(printenv USER_CONDA_BASE 2>/dev/null || echo /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/swift_HRM"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_DIR="$(printenv HRM_CLOTHO_CAPTION_DATASET_DIR 2>/dev/null || true)"
if [[ -z "$DATASET_DIR" ]]; then
  DATASET_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn"
fi
EVAL_MANIFEST="$(printenv HRM_CLOTHO_CAPTION_MANIFEST 2>/dev/null || true)"
if [[ -z "$EVAL_MANIFEST" ]]; then
  EVAL_MANIFEST="test_expand.jsonl"
fi
WRAPPER_MODEL_PATH="$(printenv HRM_AUDIO_WRAPPER_MODEL_PATH 2>/dev/null || true)"
if [[ -z "$WRAPPER_MODEL_PATH" ]]; then
  WRAPPER_MODEL_PATH="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/hrm-text-audio-v1"
fi
PLUGIN_PATH="$(printenv HRM_CLOTHO_CAPTION_PLUGIN_PATH 2>/dev/null || true)"
if [[ -z "$PLUGIN_PATH" ]]; then
  PLUGIN_PATH="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/HRM_Audio/plugins/hrm_text_audio_swift.py"
fi
OUTPUT_ROOT="$(printenv HRM_CLOTHO_CAPTION_OUTPUT_ROOT 2>/dev/null || true)"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/clotho_caption_samples"
fi

CHECKPOINTS=(
  "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-2802"
  "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-5604"
)
CHECKPOINTS_VALUE="$(printenv HRM_CLOTHO_CAPTION_CHECKPOINTS 2>/dev/null || true)"
if [[ -n "$CHECKPOINTS_VALUE" ]]; then
  IFS=':' read -r -a CHECKPOINTS <<< "$CHECKPOINTS_VALUE"
fi

SAMPLE_COUNT="$(printenv HRM_CLOTHO_CAPTION_SAMPLE_COUNT 2>/dev/null || echo 3)"
SEED="$(printenv HRM_CLOTHO_CAPTION_SEED 2>/dev/null || echo 74)"
MAX_NEW_TOKENS="$(printenv HRM_CLOTHO_CAPTION_MAX_NEW_TOKENS 2>/dev/null || echo 64)"
DEVICE="$(printenv HRM_CLOTHO_CAPTION_DEVICE 2>/dev/null || echo cuda:0)"

echo "========== RUN HRM AUDIO CLOTHO CAPTION SAMPLES =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$(which python)"
echo "dataset_dir=$DATASET_DIR"
echo "eval_manifest=$EVAL_MANIFEST"
echo "wrapper_model_path=$WRAPPER_MODEL_PATH"
echo "plugin_path=$PLUGIN_PATH"
echo "output_root=$OUTPUT_ROOT"
echo "sample_count=$SAMPLE_COUNT seed=$SEED max_new_tokens=$MAX_NEW_TOKENS"
echo "generation_path=hrm_audio_manual_prefill_cache"
printf 'checkpoint=%s\n' "${CHECKPOINTS[@]}"

ARGS=(
  --dataset-dir "$DATASET_DIR"
  --eval-manifest "$EVAL_MANIFEST"
  --wrapper-model-path "$WRAPPER_MODEL_PATH"
  --plugin-path "$PLUGIN_PATH"
  --output-root "$OUTPUT_ROOT"
  --sample-count "$SAMPLE_COUNT"
  --seed "$SEED"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
)
for checkpoint in "${CHECKPOINTS[@]}"; do
  ARGS+=(--checkpoint "$checkpoint")
done

exec python -u code/HRM_Audio/scripts/generate_clotho_caption_samples_hrm_audio_swift.py "${ARGS[@]}"
