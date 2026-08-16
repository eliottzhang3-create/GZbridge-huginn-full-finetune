#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
STAGE1="${BAT_STAGE1_MANIFEST:?Set BAT_STAGE1_MANIFEST}"
STAGE2="${BAT_STAGE2_MANIFEST:?Set BAT_STAGE2_MANIFEST}"
STAGE3="${BAT_STAGE3_MANIFEST:?Set BAT_STAGE3_MANIFEST}"
ROOT="${BAT_COMPILE_SMOKE_ROOT:?Set BAT_COMPILE_SMOKE_ROOT to a private output root}"

case "$ROOT" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 -n 1 \
  -j "bat-ouro-curriculum-compile-smoke-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat_ouro_curriculum_compile_smoke_3090.JOB.log" \
  --cmd "BAT_STAGE1_MANIFEST=$(printf '%q' "$STAGE1") BAT_STAGE2_MANIFEST=$(printf '%q' "$STAGE2") BAT_STAGE3_MANIFEST=$(printf '%q' "$STAGE3") BAT_COMPILE_SMOKE_ROOT=$(printf '%q' "$ROOT") bash scripts/smoke_bat_ouro_curriculum_compile_resume_remote.sh"
