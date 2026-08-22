#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
cd "$ROOT"
export PYTHONPATH="$ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

MODEL_KIND="${BAT_EVAL_MODEL_KIND:?Set BAT_EVAL_MODEL_KIND=ouro or qwen3}"
QA_ROOT="${BAT_EVAL_QA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end}"
AUDIO_ROOT="${BAT_EVAL_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
REVERB_ROOT="${BAT_EVAL_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
SPATIAL_AST_ROOT="${BAT_EVAL_SPATIAL_AST_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST}"
SPATIAL_AST_CHECKPOINT="${BAT_EVAL_SPATIAL_AST_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth}"
QFORMER_SOURCE="${BAT_EVAL_QFORMER_SOURCE:-/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL/src/slam_llm/models/projector.py}"

if [ "$MODEL_KIND" = "ouro" ]; then
  MODEL_PATH="${BAT_EVAL_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
  PLUGIN_PATH="${BAT_EVAL_PLUGIN_PATH:-$ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
  CHECKPOINT="${BAT_EVAL_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_localcache_0819_v2/v0-20260819-120617/checkpoint-10500}"
else
  MODEL_PATH="${BAT_EVAL_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
  PLUGIN_PATH="${BAT_EVAL_PLUGIN_PATH:-$ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
  CHECKPOINT="${BAT_EVAL_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/stage3_ab_cde_localcache_0819_v2/v0-20260819-150938/checkpoint-10500}"
fi

OUTPUT_JSONL="${BAT_EVAL_OUTPUT_JSONL:?Set BAT_EVAL_OUTPUT_JSONL to a private JSONL path}"
OUTPUT_REPORT="${BAT_EVAL_OUTPUT_REPORT:?Set BAT_EVAL_OUTPUT_REPORT to a private report path}"
MAX_RECORDS="${BAT_EVAL_MAX_RECORDS_PER_SPLIT:-1}"
REPEAT="${BAT_EVAL_REPEAT:-2}"
MAX_NEW="${BAT_EVAL_MAX_NEW_TOKENS:-24}"
BEAMS="${BAT_EVAL_NUM_BEAMS:-1}"
RIR_POLICY="${BAT_EVAL_RIR_POLICY:-official_bat}"
INCLUDE_NONBINARY="${BAT_EVAL_INCLUDE_NONBINARY:-0}"
BINARY_ANSWER_PROMPT="${BAT_EVAL_BINARY_ANSWER_PROMPT:-auto}"

ARGS=(
  --model-kind "$MODEL_KIND"
  --model-path "$MODEL_PATH"
  --plugin-path "$PLUGIN_PATH"
  --checkpoint "$CHECKPOINT"
  --qa-root "$QA_ROOT"
  --audio-root "$AUDIO_ROOT"
  --reverb-root "$REVERB_ROOT"
  --spatial-ast-root "$SPATIAL_AST_ROOT"
  --spatial-ast-checkpoint "$SPATIAL_AST_CHECKPOINT"
  --qformer-source "$QFORMER_SOURCE"
  --output-jsonl "$OUTPUT_JSONL"
  --output-report "$OUTPUT_REPORT"
  --max-records-per-split "$MAX_RECORDS"
  --repeat "$REPEAT"
  --max-new-tokens "$MAX_NEW"
  --num-beams "$BEAMS"
  --rir-policy "$RIR_POLICY"
  --binary-answer-prompt "$BINARY_ANSWER_PROMPT"
)
if [ "$INCLUDE_NONBINARY" = "1" ]; then ARGS+=(--include-nonbinary); fi

echo "========== BAT PHASE 2 EVALUATION GENERATION SMOKE LAUNCH =========="
echo "[model] $MODEL_KIND checkpoint=$CHECKPOINT"
echo "[scope] records_per_split=$MAX_RECORDS repeat=$REPEAT rir_policy=$RIR_POLICY binary_answer_prompt=$BINARY_ANSWER_PROMPT"
python -u code/Ouro_audio/bat/scripts/smoke_bat_eval_generation.py "${ARGS[@]}"
