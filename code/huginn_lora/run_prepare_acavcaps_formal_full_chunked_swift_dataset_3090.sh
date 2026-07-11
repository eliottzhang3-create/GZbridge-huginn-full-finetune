#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p log

CMD_PREFIX=""
if [ -n "${FORMAL_START_CHUNK:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX} FORMAL_START_CHUNK=${FORMAL_START_CHUNK}"
fi
if [ -n "${FORMAL_END_CHUNK:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX} FORMAL_END_CHUNK=${FORMAL_END_CHUNK}"
fi
if [ -n "${FORMAL_SKIP_EXISTING:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX} FORMAL_SKIP_EXISTING=${FORMAL_SKIP_EXISTING}"
fi
if [ -n "${FORMAL_CHUNK_SIZE_TARS:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX} FORMAL_CHUNK_SIZE_TARS=${FORMAL_CHUNK_SIZE_TARS}"
fi
if [ -n "${FORMAL_CHUNK_DIR:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX} FORMAL_CHUNK_DIR=${FORMAL_CHUNK_DIR}"
fi
if [ -n "${FORMAL_SAMPLES_PER_TAR:-}" ]; then
  CMD_PREFIX="${CMD_PREFIX} FORMAL_SAMPLES_PER_TAR=${FORMAL_SAMPLES_PER_TAR}"
fi

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j prepare-acavcaps-formal-full-chunks-3090-$(date +%m%d%H%M) \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/prepare_acavcaps_formal_full_chunked_swift_dataset_3090.JOB.log" \
  --cmd "${CMD_PREFIX} bash scripts/prepare_acavcaps_formal_full_chunked_swift_dataset.sh"
