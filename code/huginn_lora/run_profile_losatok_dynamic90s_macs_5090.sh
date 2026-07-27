#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

MODEL_DIR="${TORCHPROFILE_MODEL_DIR:-$SCRIPT_DIR/../../models/huginn-audio-losatok-v1}"
PLUGIN_PATH="${TORCHPROFILE_PLUGIN_PATH:-$SCRIPT_DIR/plugins/huginn_losatok_swift.py}"
OUTPUT_DIR="${TORCHPROFILE_OUTPUT_DIR:-$SCRIPT_DIR/../../outputs/torchprofile_losatok_dynamic90s_macs}"
OUTPUT_JSON="$OUTPUT_DIR/profile_$(date +%Y%m%d_%H%M%S).json"

for path in "$MODEL_DIR" "$PLUGIN_PATH"; do
  if [ ! -e "$path" ]; then
    echo "Required MACs profiling path is missing: $path" >&2
    exit 1
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j profile-losatok-dynamic90s-macs-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/profile_losatok_dynamic90s_macs_5090.JOB.log" \
  --cmd "HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1 HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=0 python -u scripts/profile_losatok_dynamic90s_macs.py --model_dir '$MODEL_DIR' --plugin '$PLUGIN_PATH' --with_lora --audio_seconds 5 30 90 --text_tokens 64 --no_grad_steps 24 --grad_steps 8 --output_json '$OUTPUT_JSON'"
