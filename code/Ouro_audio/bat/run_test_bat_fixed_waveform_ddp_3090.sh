#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
MODEL_TYPE="$(printenv BAT_FIXED_DDP_MODEL_TYPE 2>/dev/null || true)"
if [ -z "$MODEL_TYPE" ]; then MODEL_TYPE=ouro; fi
MODEL_PATH="$(printenv BAT_FIXED_DDP_MODEL_PATH 2>/dev/null || true)"
PLUGIN_PATH="$(printenv BAT_FIXED_DDP_PLUGIN_PATH 2>/dev/null || true)"
OUTPUT="$(printenv BAT_FIXED_DDP_OUTPUT 2>/dev/null || true)"
if [ -z "$MODEL_PATH" ] || [ -z "$PLUGIN_PATH" ] || [ -z "$OUTPUT" ]; then echo "Set BAT_FIXED_DDP_MODEL_PATH, BAT_FIXED_DDP_PLUGIN_PATH and BAT_FIXED_DDP_OUTPUT" >&2; exit 2; fi
CMD_PREFIX="BAT_FIXED_DDP_MODEL_TYPE=$(printf '%q' "$MODEL_TYPE") BAT_FIXED_DDP_MODEL_PATH=$(printf '%q' "$MODEL_PATH") BAT_FIXED_DDP_PLUGIN_PATH=$(printf '%q' "$PLUGIN_PATH") BAT_FIXED_DDP_OUTPUT=$(printf '%q' "$OUTPUT") "
append_env() { local name="$1"; local value; value="$(printenv "$name" 2>/dev/null || true)"; if [ -n "$value" ]; then CMD_PREFIX="$CMD_PREFIX$name=$(printf '%q' "$value") "; fi; }
for name in BAT_FIXED_DDP_WORLD_SIZE BAT_FIXED_DDP_LOCAL_BATCH_SIZE BAT_FIXED_DDP_STEPS OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS; do append_env "$name"; done
vc submit -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-fixed-waveform-$MODEL_TYPE-3090-$(date +%m%d%H%M)" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat-fixed-waveform-$MODEL_TYPE-3090.JOB.log" \
  --cmd "$CMD_PREFIX bash scripts/test_bat_fixed_waveform_ddp_remote.sh"

