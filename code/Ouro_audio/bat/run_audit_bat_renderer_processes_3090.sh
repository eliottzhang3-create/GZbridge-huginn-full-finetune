#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
MANIFEST="$(printenv BAT_RENDER_DDP_MANIFEST 2>/dev/null || true)"
OUTPUT="$(printenv BAT_RENDER_DDP_OUTPUT 2>/dev/null || true)"
if [ -z "$MANIFEST" ] || [ -z "$OUTPUT" ]; then echo "Set BAT_RENDER_DDP_MANIFEST and BAT_RENDER_DDP_OUTPUT" >&2; exit 2; fi
CMD_PREFIX="BAT_RENDER_DDP_MANIFEST=$(printf '%q' "$MANIFEST") BAT_RENDER_DDP_OUTPUT=$(printf '%q' "$OUTPUT") "
append_env() { local name="$1"; local value; value="$(printenv "$name" 2>/dev/null || true)"; if [ -n "$value" ]; then CMD_PREFIX="$CMD_PREFIX$name=$(printf '%q' "$value") "; fi; }
for name in BAT_RENDER_DDP_PROGRESS BAT_RENDER_DDP_WORLD_SIZE BAT_RENDER_DDP_LOCAL_BATCH_SIZE BAT_RENDER_DDP_GLOBAL_RECORD_LIMIT BAT_AUDIO_ROOT BAT_REVERB_ROOT; do append_env "$name"; done
vc submit -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-renderer-procs-3090-$(date +%m%d%H%M)" -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat-renderer-procs-3090.JOB.log" \
  --cmd "$CMD_PREFIX bash scripts/audit_bat_renderer_processes_remote.sh"
