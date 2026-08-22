#!/usr/bin/env bash
set -euo pipefail

: "${BAT_EVAL_TYPE:?Set BAT_EVAL_TYPE to A, B, C, D, E-direction, or E-distance}"
: "${BAT_EVAL_OUTPUT_JSONL:?Set BAT_EVAL_OUTPUT_JSONL}"
: "${BAT_EVAL_OUTPUT_REPORT:?Set BAT_EVAL_OUTPUT_REPORT}"

ENV_PREFIX=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
REPO=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
PYTHON=${ENV_PREFIX}/bin/python

source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
export PYTHONPATH="${REPO}/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

MODEL_PATH=${BAT_EVAL_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}
CHECKPOINT=${BAT_EVAL_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_localcache_0819_v2/v0-20260819-120617/checkpoint-10500}
QA_ROOT=${BAT_EVAL_QA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end}
AUDIO_ROOT=${BAT_EVAL_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}
REVERB_ROOT=${BAT_EVAL_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}
SPATIAL_AST_ROOT=${BAT_EVAL_SPATIAL_AST_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST}
SPATIAL_AST_CHECKPOINT=${BAT_EVAL_SPATIAL_AST_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth}
QFORMER_SOURCE=${BAT_EVAL_QFORMER_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL/src/slam_llm/models/projector.py}
LABEL_CSV=${BAT_EVAL_LABEL_CSV:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/class_labels_indices_subset.csv}
LAUNCHER_STATUS="${BAT_EVAL_LAUNCHER_STATUS:-${BAT_EVAL_OUTPUT_REPORT}.launcher_status.txt}"

write_launcher_status() {
  local code="$?"
  {
    echo "exit_code=${code}"
    echo "hostname=$(hostname)"
    echo "pid=$$"
    echo "updated_unix=$(date +%s)"
    echo "heartbeat=${BAT_EVAL_OUTPUT_JSONL}.heartbeat.json"
    echo "faulthandler=${BAT_EVAL_OUTPUT_JSONL}.faulthandler.log"
  } > "${LAUNCHER_STATUS}"
  return "${code}"
}
trap write_launcher_status EXIT

ARGS=(
  --eval-type "${BAT_EVAL_TYPE}"
  --model-path "${MODEL_PATH}"
  --plugin-path "${REPO}/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py"
  --checkpoint "${CHECKPOINT}"
  --qa-root "${QA_ROOT}"
  --audio-root "${AUDIO_ROOT}"
  --reverb-root "${REVERB_ROOT}"
  --spatial-ast-root "${SPATIAL_AST_ROOT}"
  --spatial-ast-checkpoint "${SPATIAL_AST_CHECKPOINT}"
  --qformer-source "${QFORMER_SOURCE}"
  --output-jsonl "${BAT_EVAL_OUTPUT_JSONL}"
  --output-report "${BAT_EVAL_OUTPUT_REPORT}"
  --device "${BAT_EVAL_DEVICE:-cuda:0}"
  --start-index "${BAT_EVAL_START_INDEX:-0}"
  --max-records "${BAT_EVAL_MAX_RECORDS:-0}"
  --rir-policy "${BAT_EVAL_RIR_POLICY:-official_bat}"
  --binary-answer-prompt "${BAT_EVAL_BINARY_ANSWER_PROMPT:-off}"
  --detection-mode "${BAT_EVAL_DETECTION_MODE:-model_output_embedding}"
  --label-csv "${LABEL_CSV}"
)

# Leave generation limits unset by default so the Python evaluator selects the
# common BAT contract: every evaluation type uses 10 new tokens and greedy
# single-beam generation. Explicit larger values are rejected by the evaluator.
if [[ -n "${BAT_EVAL_MAX_NEW_TOKENS:-}" ]]; then
  ARGS+=(--max-new-tokens "${BAT_EVAL_MAX_NEW_TOKENS}")
fi
if [[ -n "${BAT_EVAL_NUM_BEAMS:-}" ]]; then
  ARGS+=(--num-beams "${BAT_EVAL_NUM_BEAMS}")
fi

if [[ "${BAT_EVAL_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

cd "${REPO}"
"${PYTHON}" -u code/Ouro_audio/bat/scripts/eval_bat_ouro_online.py "${ARGS[@]}"
