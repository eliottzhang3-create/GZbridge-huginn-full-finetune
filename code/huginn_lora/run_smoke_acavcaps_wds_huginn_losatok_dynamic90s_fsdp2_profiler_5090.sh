#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
for name in ACAVCAPS_WDS_MANIFEST ACAVCAPS_WDS_BUFFER_SIZE ACAVCAPS_WDS_MAX_TARS_PER_STAGE ACAVCAPS_WDS_PROFILE_OUTPUT_DIR ACAVCAPS_WDS_PROFILE_MAX_STEPS HUGINN_TORCH_PROFILER_WAIT HUGINN_TORCH_PROFILER_WARMUP HUGINN_TORCH_PROFILER_ACTIVE; do
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
  -j profile-acav-dynamic-fsdp2-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/smoke_acavcaps_wds_huginn_losatok_dynamic90s_fsdp2_profiler_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/smoke_acavcaps_wds_huginn_losatok_dynamic90s_fsdp2_profiler.sh"
