#!/bin/bash
set -euo pipefail

# Independent Whisper-large LoRA MMAU evaluation route.
# This is intentionally separate from the active LoSATok evaluation wrapper.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CHECKPOINT="${MMAU_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-2802}"
OUTPUT_DIR="${MMAU_OUTPUT_DIR:-outputs/mmau_test_mini_whisper_audiocaps_v2_checkpoint2802}"
PLUGIN_PATH="${MMAU_PLUGIN_PATH:-$SCRIPT_DIR/plugins/huginn_audio_swift.py}"

quote_env() {
  local name="$1"
  local value="$2"
  local quoted
  printf -v quoted '%q' "$value"
  CMD_PREFIX="${CMD_PREFIX}${name}=${quoted} "
}

CMD_PREFIX=""
quote_env MMAU_CHECKPOINT "$CHECKPOINT"
quote_env MMAU_OUTPUT_DIR "$OUTPUT_DIR"
quote_env MMAU_PLUGIN_PATH "$PLUGIN_PATH"

# Preserve optional evaluator controls supplied by the caller.
for name in MMAU_CHECKPOINTS MMAU_TEST_MINI_PATH MMAU_START_OFFSET MMAU_MAX_SAMPLES MMAU_LOG_EVERY MMAU_NUM_STEPS HUGINN_AUDIO_FSDP_EVAL_EXPORT_DIR; do
  if [ -n "${!name:-}" ]; then
    quote_env "$name" "${!name}"
  fi
done

echo "========== SUBMIT WHISPER-LARGE MMAU EVAL =========="
echo "checkpoint=$CHECKPOINT"
echo "output_dir=$OUTPUT_DIR"
echo "plugin_path=$PLUGIN_PATH"
echo "dataset_path=${MMAU_TEST_MINI_PATH:-<default evaluator path>}"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j eval-mmau-whisper-cp2802-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/eval_mmau_whisper_cp2802_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/eval_mmau_test_mini_swift.sh"
