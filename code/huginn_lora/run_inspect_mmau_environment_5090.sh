#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

CMD_PREFIX=""
if [ -n "${MMAU_DATASET_ROOT:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}MMAU_DATASET_ROOT=${MMAU_DATASET_ROOT} "
fi
if [ -n "${MMAU_INSPECT_OUTPUT:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX}MMAU_INSPECT_OUTPUT=${MMAU_INSPECT_OUTPUT} "
fi

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j inspect-mmau-environment-5090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_mmau_environment_5090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/inspect_mmau_environment.sh"
