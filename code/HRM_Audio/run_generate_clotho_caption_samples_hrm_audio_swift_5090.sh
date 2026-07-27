#!/bin/bash
set -euo pipefail

# Remote 5090 launcher for the independent HRM-Text Clotho-v2 samples route.
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
  HRM_CLOTHO_CAPTION_DATASET_DIR \
  HRM_CLOTHO_CAPTION_MANIFEST \
  HRM_CLOTHO_CAPTION_PLUGIN_PATH \
  HRM_CLOTHO_CAPTION_OUTPUT_ROOT \
  HRM_CLOTHO_CAPTION_CHECKPOINTS \
  HRM_CLOTHO_CAPTION_SAMPLE_COUNT \
  HRM_CLOTHO_CAPTION_SEED \
  HRM_CLOTHO_CAPTION_MAX_NEW_TOKENS \
  HRM_CLOTHO_CAPTION_DEVICE; do
  value="$(printenv "$name" 2>/dev/null || true)"
  if [[ -n "$value" ]]; then
    quote_env "$name" "$value"
  fi
done

echo "========== SUBMIT HRM AUDIO CLOTHO CAPTION SAMPLES =========="
echo "output_root=$(printenv HRM_CLOTHO_CAPTION_OUTPUT_ROOT 2>/dev/null || echo '<default>')"
echo "checkpoints=$(printenv HRM_CLOTHO_CAPTION_CHECKPOINTS 2>/dev/null || echo '<formal checkpoint-2802 and checkpoint-5604 defaults>')"
echo "dataset_dir=$(printenv HRM_CLOTHO_CAPTION_DATASET_DIR 2>/dev/null || echo '<default Clotho root>')"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j generate-clotho-hrm-audio-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/generate_clotho_hrm_audio_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/generate_clotho_caption_samples_hrm_audio_swift.sh"
