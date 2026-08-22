#!/usr/bin/env bash
set -euo pipefail

: "${BAT_EVAL_TYPE:?Set BAT_EVAL_TYPE to A, B, C, or D}"
: "${BAT_EVAL_OUTPUT_JSONL:?Set BAT_EVAL_OUTPUT_JSONL}"
: "${BAT_EVAL_OUTPUT_REPORT:?Set BAT_EVAL_OUTPUT_REPORT}"

case "${BAT_EVAL_TYPE}" in
  A|B|C|D) ;;
  *) echo "Qwen3 evaluator only supports A/B/C/D; refusing ${BAT_EVAL_TYPE}" >&2; exit 2 ;;
esac

if [[ "$(dirname -- "$BAT_EVAL_OUTPUT_JSONL")" == "/" || "$(dirname -- "$BAT_EVAL_OUTPUT_REPORT")" == "/" ]]; then
  echo "Qwen3 evaluation outputs must be under a private directory; got JSONL=${BAT_EVAL_OUTPUT_JSONL} REPORT=${BAT_EVAL_OUTPUT_REPORT}" >&2
  exit 2
fi

REPO=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
STAMP=$(date +%m%d%H%M)
JOB_NAME="bat-qwen3-eval-${BAT_EVAL_TYPE}-${STAMP}"
LOG_PATH=${BAT_EVAL_LOG_PATH:-${REPO}/code/Ouro_audio/bat/log/${JOB_NAME}.JOB.log}

CMD_PREFIX="BAT_EVAL_TYPE=$(printf '%q' "$BAT_EVAL_TYPE") BAT_EVAL_OUTPUT_JSONL=$(printf '%q' "$BAT_EVAL_OUTPUT_JSONL") BAT_EVAL_OUTPUT_REPORT=$(printf '%q' "$BAT_EVAL_OUTPUT_REPORT") "
append_env() {
  local name="$1" value
  value="$(printenv "$name" 2>/dev/null || true)"
  if [ -n "$value" ]; then CMD_PREFIX="$CMD_PREFIX$name=$(printf '%q' "$value") "; fi
}
for name in BAT_EVAL_MODEL_PATH BAT_EVAL_CHECKPOINT BAT_EVAL_QA_ROOT BAT_EVAL_AUDIO_ROOT BAT_EVAL_REVERB_ROOT BAT_EVAL_SPATIAL_AST_ROOT BAT_EVAL_SPATIAL_AST_CHECKPOINT BAT_EVAL_QFORMER_SOURCE BAT_EVAL_START_INDEX BAT_EVAL_MAX_RECORDS BAT_EVAL_MAX_NEW_TOKENS BAT_EVAL_NUM_BEAMS BAT_EVAL_RIR_POLICY BAT_EVAL_DETECTION_MODE BAT_EVAL_LABEL_CSV BAT_EVAL_OVERWRITE BAT_EVAL_LAUNCHER_STATUS; do
  append_env "$name"
done

vc submit \
  -p pdgpu-3090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 -n 1 \
  -j "${JOB_NAME}" \
  -d "${REPO}/code/Ouro_audio/bat" \
  JOB=1:1 "${LOG_PATH}" \
  --cmd "cd ${REPO} && ${CMD_PREFIX}bash code/Ouro_audio/bat/scripts/eval_bat_qwen3_online_remote.sh"
