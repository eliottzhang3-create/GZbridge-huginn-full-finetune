#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in \
  LOSATOK_DYNAMIC_ACAV_WDS_INIT_CHECKPOINT \
  ACAVCAPS_WDS_QUARTER_MANIFEST \
  ACAVCAPS_WDS_BUFFER_SIZE \
  ACAVCAPS_WDS_MAX_TARS_PER_STAGE \
  LOSATOK_DYNAMIC_ACAV_WDS_QUARTER_SMOKE_ROOT; do
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
  -j smoke-acav-dyn90-quarter-warmstart-f2-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_acavcaps_wds_huginn_losatok_dynamic90s_quarter_warmstart_save_reload_fsdp2_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_acavcaps_wds_huginn_losatok_dynamic90s_quarter_warmstart_save_reload_fsdp2.sh"
