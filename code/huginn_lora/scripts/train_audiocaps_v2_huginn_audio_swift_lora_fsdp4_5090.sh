#!/bin/bash
set -euo pipefail

# Explicit entry point for the current Whisper-large dynamic-90s LoRA/FSDP4
# route. Keep the implementation in the canonical training script so the
# generic and descriptive submit wrappers cannot drift apart.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/train_audiocaps_v2_huginn_audio_swift_5090.sh" "$@"
