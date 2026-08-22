#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/log"
MODEL_KIND="${BAT_EVAL_MODEL_KIND:?Set BAT_EVAL_MODEL_KIND=ouro or qwen3}"
OUTPUT_JSONL="${BAT_EVAL_OUTPUT_JSONL:?Set BAT_EVAL_OUTPUT_JSONL to a private JSONL path}"
OUTPUT_REPORT="${BAT_EVAL_OUTPUT_REPORT:?Set BAT_EVAL_OUTPUT_REPORT to a private report path}"

CMD_PREFIX="BAT_EVAL_MODEL_KIND=$(printf '%q' "$MODEL_KIND") BAT_EVAL_OUTPUT_JSONL=$(printf '%q' "$OUTPUT_JSONL") BAT_EVAL_OUTPUT_REPORT=$(printf '%q' "$OUTPUT_REPORT") "
append_env() {
  local name="$1" value
  value="$(printenv "$name" 2>/dev/null || true)"
  if [ -n "$value" ]; then CMD_PREFIX="$CMD_PREFIX$name=$(printf '%q' "$value") "; fi
}
for name in BAT_EVAL_MODEL_PATH BAT_EVAL_PLUGIN_PATH BAT_EVAL_CHECKPOINT BAT_EVAL_QA_ROOT BAT_EVAL_AUDIO_ROOT BAT_EVAL_REVERB_ROOT BAT_EVAL_SPATIAL_AST_ROOT BAT_EVAL_SPATIAL_AST_CHECKPOINT BAT_EVAL_QFORMER_SOURCE BAT_EVAL_MAX_RECORDS_PER_SPLIT BAT_EVAL_REPEAT BAT_EVAL_MAX_NEW_TOKENS BAT_EVAL_NUM_BEAMS BAT_EVAL_RIR_POLICY BAT_EVAL_INCLUDE_NONBINARY BAT_EVAL_BINARY_ANSWER_PROMPT; do
  append_env "$name"
done

vc submit -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "bat-eval-smoke-${MODEL_KIND}-3090-$(date +%m%d%H%M)" \
  -d "$SCRIPT_DIR" \
  JOB=1:1 "$SCRIPT_DIR/log/bat-eval-smoke-${MODEL_KIND}-3090.JOB.log" \
  --cmd "$CMD_PREFIX bash scripts/smoke_bat_eval_generation_remote.sh"
