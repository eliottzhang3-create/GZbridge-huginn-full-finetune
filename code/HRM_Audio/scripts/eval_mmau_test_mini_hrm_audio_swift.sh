#!/bin/bash
set -euo pipefail

# Independent HRM-Text + Whisper MMAU test-mini evaluator.
# No Huginn evaluator is imported or modified by this route.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(printenv HRM_MMAU_PYTHON 2>/dev/null || true)"
if [[ -z "$PYTHON" ]]; then
  PYTHON="$(printenv PYTHON 2>/dev/null || true)"
fi
if [[ -z "$PYTHON" ]]; then
  PYTHON="python"
fi

DATASET_PATH="$(printenv MMAU_TEST_MINI_PATH 2>/dev/null || true)"
if [[ -z "$DATASET_PATH" ]]; then
  DATASET_PATH="/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU test_mini/test_mini.parquet"
fi
WRAPPER_MODEL_PATH="$(printenv HRM_AUDIO_WRAPPER_MODEL_PATH 2>/dev/null || true)"
if [[ -z "$WRAPPER_MODEL_PATH" ]]; then
  WRAPPER_MODEL_PATH="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/hrm-text-audio-v1"
fi
PLUGIN_PATH="$(printenv MMAU_PLUGIN_PATH 2>/dev/null || true)"
if [[ -z "$PLUGIN_PATH" ]]; then
  PLUGIN_PATH="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/HRM_Audio/plugins/hrm_text_audio_swift.py"
fi
OUTPUT_ROOT="$(printenv MMAU_OUTPUT_ROOT 2>/dev/null || true)"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/mmau_test_mini"
fi

CHECKPOINTS=(
  "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-2802"
  "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-5604"
)

CHECKPOINTS_VALUE="$(printenv MMAU_CHECKPOINTS 2>/dev/null || true)"
if [[ -n "$CHECKPOINTS_VALUE" ]]; then
  IFS=':' read -r -a CHECKPOINTS <<< "$CHECKPOINTS_VALUE"
fi

ARGS=(
  --dataset-path "$DATASET_PATH"
  --wrapper-model-path "$WRAPPER_MODEL_PATH"
  --plugin-path "$PLUGIN_PATH"
  --output-root "$OUTPUT_ROOT"
  --start-offset "$(printenv MMAU_START_OFFSET 2>/dev/null || echo 0)"
  --log-every "$(printenv MMAU_LOG_EVERY 2>/dev/null || echo 10)"
  --device "$(printenv MMAU_DEVICE 2>/dev/null || echo cuda:0)"
)
MAX_SAMPLES="$(printenv MMAU_MAX_SAMPLES 2>/dev/null || true)"
if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi
PRINT_SAMPLES="$(printenv MMAU_PRINT_SAMPLES 2>/dev/null || true)"
if [[ "$PRINT_SAMPLES" == "true" ]]; then
  ARGS+=(--print-samples)
fi
for checkpoint in "${CHECKPOINTS[@]}"; do
  ARGS+=(--checkpoint "$checkpoint")
done

echo "========== RUN HRM AUDIO MMAU TEST-MINI EVAL =========="
echo "python=$PYTHON"
echo "dataset=$DATASET_PATH"
echo "wrapper=$WRAPPER_MODEL_PATH"
echo "plugin=$PLUGIN_PATH"
echo "output_root=$OUTPUT_ROOT"
printf 'checkpoint=%s\n' "${CHECKPOINTS[@]}"

exec "$PYTHON" -u "$SCRIPT_DIR/eval_mmau_test_mini_hrm_audio_swift.py" "${ARGS[@]}"
