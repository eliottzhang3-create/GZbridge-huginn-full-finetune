#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
OUTPUT="${BAT_EVAL_CONTRACT_OUTPUT:?Set BAT_EVAL_CONTRACT_OUTPUT to a private report path}"

CMD_PREFIX="BAT_EVAL_CONTRACT_OUTPUT=$(printf '%q' "$OUTPUT") "
append_env() {
  local name="$1" value
  value="$(printenv "$name" 2>/dev/null || true)"
  if [ -n "$value" ]; then CMD_PREFIX="$CMD_PREFIX$name=$(printf '%q' "$value") "; fi
}
for name in BAT_EVAL_QA_ROOT BAT_EVAL_AUDIO_ROOT BAT_EVAL_REVERB_ROOT BAT_EVAL_SPATIAL_AST_ROOT BAT_EVAL_SPATIAL_AST_CHECKPOINT BAT_EVAL_QFORMER_SOURCE BAT_EVAL_OURO_MODEL_PATH BAT_EVAL_OURO_PLUGIN_PATH BAT_EVAL_OURO_CHECKPOINT BAT_EVAL_QWEN3_MODEL_PATH BAT_EVAL_QWEN3_PLUGIN_PATH BAT_EVAL_QWEN3_CHECKPOINT; do
  append_env "$name"
done

vc submit -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-eval-contract-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat-eval-contract-3090.JOB.log" \
  --cmd "$CMD_PREFIX bash scripts/audit_bat_eval_contract_remote.sh"
