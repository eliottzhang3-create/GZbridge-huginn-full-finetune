#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
if [ -n "${HRM_ENV_OUTPUT_REPORT:-}" ]; then
  printf -v quoted_report '%q' "$HRM_ENV_OUTPUT_REPORT"
  CMD_PREFIX="HRM_ENV_OUTPUT_REPORT=${quoted_report} "
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j inspect-hrm-environment-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_hrm_environment_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_hrm_environment.sh"
