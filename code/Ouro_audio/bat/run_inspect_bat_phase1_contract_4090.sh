#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

BAT_DATA_ROOT="${BAT_DATA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA}"
BAT_QA_ROOT="${BAT_QA_ROOT:-$BAT_DATA_ROOT/closed-end}"
BAT_AUDIO_ROOT="${BAT_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
BAT_REVERB_ROOT="${BAT_REVERB_ROOT:-$BAT_DATA_ROOT/mp3d_reverb}"
BAT_SPATIAL_AST_CODE_ROOT="${BAT_SPATIAL_AST_CODE_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST}"
BAT_QFORMER_PATH="${BAT_QFORMER_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL/src/slam_llm/models/projector.py}"
BAT_OUTPUT="${BAT_PHASE1_CONTRACT_OUTPUT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/phase1_data_contract_audit.json}"

CMD_PREFIX=""
for name in BAT_DATA_ROOT BAT_QA_ROOT BAT_AUDIO_ROOT BAT_REVERB_ROOT \
  BAT_SPATIAL_AST_CODE_ROOT BAT_QFORMER_PATH BAT_PHASE1_CONTRACT_OUTPUT; do
  value="${!name:-}"
  if [ -n "$value" ]; then
    printf -v quoted_value '%q' "$value"
    CMD_PREFIX="${CMD_PREFIX}${name}=${quoted_value} "
  fi
done

vc submit \
  -p "${BAT_QUEUE:-pdgpu-4090}" \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j "inspect-bat-phase1-contract-4090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/inspect_bat_phase1_contract_4090.JOB.log" \
  --cmd "${CMD_PREFIX}bash scripts/run_bat_phase1_contract_remote.sh"
