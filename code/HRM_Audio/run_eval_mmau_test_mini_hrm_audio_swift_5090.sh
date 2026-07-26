#!/bin/bash
set -euo pipefail

# Remote 5090 launcher for the independent HRM-Text MMAU test-mini route.
# Huginn code is intentionally outside this execution path.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
quote_env() {
  local name="$1"
  local value="$2"
  local quoted
  printf -v quoted '%q' "$value"
  CMD_PREFIX="${CMD_PREFIX}${name}=${quoted} "
}

for name in \
  USER_CONDA_BASE \
  HRM_TEXT_MODEL_PATH \
  HRM_AUDIO_WHISPER_MODEL_PATH \
  HRM_AUDIO_WRAPPER_MODEL_PATH \
  MMAU_PLUGIN_PATH \
  MMAU_TEST_MINI_PATH \
  MMAU_OUTPUT_ROOT \
  MMAU_CHECKPOINTS \
  MMAU_START_OFFSET \
  MMAU_MAX_SAMPLES \
  MMAU_LOG_EVERY \
  MMAU_DEVICE \
  MMAU_PRINT_SAMPLES \
  MMAU_MAX_AUDIO_SECONDS; do
  value="$(printenv "$name" 2>/dev/null || true)"
  if [[ -n "$value" ]]; then
    quote_env "$name" "$value"
  fi
done

echo "========== SUBMIT HRM AUDIO MMAU TEST-MINI EVAL =========="
echo "output_root=$(printenv MMAU_OUTPUT_ROOT 2>/dev/null || echo '<default>')"
echo "checkpoints=$(printenv MMAU_CHECKPOINTS 2>/dev/null || echo '<formal checkpoint-2802 and checkpoint-5604 defaults>')"
echo "dataset=$(printenv MMAU_TEST_MINI_PATH 2>/dev/null || echo '<default MMAU test-mini path>')"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j eval-mmau-hrm-audio-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/eval_mmau_hrm_audio_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/eval_mmau_test_mini_hrm_audio_swift.sh"
