#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in LOSATOK_DYNAMIC_MODULES_SAVE_SMOKE_MANIFEST LOSATOK_DYNAMIC_MODULES_SAVE_SMOKE_ROOT; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 64G -g 2 \
  -n 1 \
  -j smoke-losatok-fsdp2-modsave-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_audiocaps_v2_huginn_losatok_dynamic90s_modules_save_fsdp2_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_audiocaps_v2_huginn_losatok_dynamic90s_modules_save_fsdp2_5090.sh"
