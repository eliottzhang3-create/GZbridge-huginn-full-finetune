#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p log

QUEUE_NAME="${VC_QUEUE:-pdgpu-5090}"
CONTAINER_IMAGE="${VC_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1}"
JOB_NAME="inspect-huginn-audio-swift-trainables-$(date +%m%d%H%M)"

echo "QUEUE_NAME=$QUEUE_NAME"
echo "CONTAINER_IMAGE=$CONTAINER_IMAGE"

vc submit \
  -p "$QUEUE_NAME" \
  -i "$CONTAINER_IMAGE" \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j "$JOB_NAME" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_huginn_audio_swift_trainables.JOB.log" \
  --cmd "bash scripts/inspect_huginn_audio_swift_trainables.sh"
