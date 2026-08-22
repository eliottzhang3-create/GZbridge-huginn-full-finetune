#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
source /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/etc/profile.d/conda.sh
conda activate /hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
cd "$ROOT"
export PYTHONPATH="$ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

QA_ROOT="${BAT_EVAL_QA_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end}"
AUDIO_ROOT="${BAT_EVAL_AUDIO_ROOT:-/hpc_stor03/public/shared/data/raa/AudioSet}"
REVERB_ROOT="${BAT_EVAL_REVERB_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb}"
SPATIAL_AST_ROOT="${BAT_EVAL_SPATIAL_AST_ROOT:-/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST}"
SPATIAL_AST_CHECKPOINT="${BAT_EVAL_SPATIAL_AST_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth}"
QFORMER_SOURCE="${BAT_EVAL_QFORMER_SOURCE:-$ROOT/code/OWL/src/slam_llm/models/projector.py}"
OURO_MODEL_PATH="${BAT_EVAL_OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
OURO_PLUGIN_PATH="${BAT_EVAL_OURO_PLUGIN_PATH:-$ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
OURO_CHECKPOINT="${BAT_EVAL_OURO_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_localcache_0819_v2/v0-20260819-120617/checkpoint-10500}"
QWEN3_MODEL_PATH="${BAT_EVAL_QWEN3_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base}"
QWEN3_PLUGIN_PATH="${BAT_EVAL_QWEN3_PLUGIN_PATH:-$ROOT/code/Ouro_audio/plugins/qwen3_bat_spatial_ast_swift.py}"
QWEN3_CHECKPOINT="${BAT_EVAL_QWEN3_CHECKPOINT:-/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/stage3_ab_cde_localcache_0819_v2/v0-20260819-150938/checkpoint-10500}"
OUTPUT="${BAT_EVAL_CONTRACT_OUTPUT:?Set BAT_EVAL_CONTRACT_OUTPUT to a private report path}"

echo "========== BAT PHASE 1 EVALUATION CONTRACT AUDIT LAUNCH =========="
echo "[scope] metadata-only; no audio decode, RIR load, convolution, or model load"
echo "[output] $OUTPUT"
python -u code/Ouro_audio/bat/scripts/audit_bat_eval_contract.py \
  --qa-root "$QA_ROOT" \
  --audio-root "$AUDIO_ROOT" \
  --reverb-root "$REVERB_ROOT" \
  --spatial-ast-root "$SPATIAL_AST_ROOT" \
  --spatial-ast-checkpoint "$SPATIAL_AST_CHECKPOINT" \
  --qformer-source "$QFORMER_SOURCE" \
  --ouro-model-path "$OURO_MODEL_PATH" \
  --ouro-plugin-path "$OURO_PLUGIN_PATH" \
  --ouro-checkpoint "$OURO_CHECKPOINT" \
  --qwen3-model-path "$QWEN3_MODEL_PATH" \
  --qwen3-plugin-path "$QWEN3_PLUGIN_PATH" \
  --qwen3-checkpoint "$QWEN3_CHECKPOINT" \
  --output "$OUTPUT"
