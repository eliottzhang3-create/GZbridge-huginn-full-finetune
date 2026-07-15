#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export WAVCAPS_INIT_CHECKPOINT="${WAVCAPS_INIT_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-5604}"
export WAVCAPS_REQUIRE_BOUNDARY_CHECKPOINT=0
export WAVCAPS_MAX_STEPS="${WAVCAPS_MAX_STEPS:-20}"
export WAVCAPS_SAVE_STRATEGY="${WAVCAPS_SAVE_STRATEGY:-steps}"
export WAVCAPS_SAVE_STEPS="${WAVCAPS_SAVE_STEPS:-$WAVCAPS_MAX_STEPS}"
export WAVCAPS_SAVE_TOTAL_LIMIT="${WAVCAPS_SAVE_TOTAL_LIMIT:-1}"
export WAVCAPS_LOGGING_STEPS="${WAVCAPS_LOGGING_STEPS:-1}"
export WAVCAPS_OUTPUT_DIR="${WAVCAPS_OUTPUT_DIR:-outputs/huginn_audio_wavcaps_audioset_sl_warmstart5604_smoke20_b8ga4_5090_v2}"

echo "========== WAVCAPS WARM-START SMOKE =========="
echo "max_steps=$WAVCAPS_MAX_STEPS"
echo "output_dir=$WAVCAPS_OUTPUT_DIR"
bash "$REPO_ROOT/code/huginn_lora/scripts/train_wavcaps_audioset_huginn_audio_swift_5090.sh"
