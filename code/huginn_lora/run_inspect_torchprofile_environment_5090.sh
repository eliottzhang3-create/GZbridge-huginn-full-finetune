#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

TRAINING_SCRIPT="${TORCHPROFILE_TRAINING_SCRIPT:-$SCRIPT_DIR/scripts/train_acavcaps_wds_huginn_losatok_dynamic90s_quarter_fsdp2_5090.sh}"
MODEL_DIR="${TORCHPROFILE_MODEL_DIR:-$SCRIPT_DIR/../../models/huginn-audio-losatok-v1}"
PLUGIN_PATH="${TORCHPROFILE_PLUGIN_PATH:-$SCRIPT_DIR/plugins/huginn_losatok_swift.py}"

for path in "$TRAINING_SCRIPT" "$MODEL_DIR" "$PLUGIN_PATH"; do
  if [ ! -e "$path" ]; then
    echo "Required profiling preflight path is missing: $path" >&2
    exit 1
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j inspect-torchprofile-losatok-fsdp2-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_torchprofile_environment_5090.JOB.log" \
  --cmd "python -u scripts/inspect_torchprofile_environment.py --training_script '$TRAINING_SCRIPT' --model_dir '$MODEL_DIR' --plugin '$PLUGIN_PATH' --strict"
