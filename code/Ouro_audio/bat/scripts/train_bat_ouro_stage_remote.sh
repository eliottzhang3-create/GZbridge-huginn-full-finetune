#!/usr/bin/env bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_ouro"
REPO_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/code/Ouro_audio:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
WORLD_SIZE="${BAT_WORLD_SIZE:-1}"
if ! [[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BAT_WORLD_SIZE must be a positive integer, got: $WORLD_SIZE" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "$WORLD_SIZE" -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    last_gpu=$((WORLD_SIZE - 1))
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$last_gpu")"
  fi
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
MODEL_PATH="${OURO_MODEL_PATH:-/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B}"
PLUGIN_PATH="${OURO_BAT_PLUGIN_PATH:-$REPO_ROOT/code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py}"
DATASET="${BAT_STAGE_DATASET:?Set BAT_STAGE_DATASET to a private JSONL manifest}"
STAGE="${BAT_STAGE:?Set BAT_STAGE=I, II or III}"
OUTPUT_DIR="${BAT_STAGE_OUTPUT_DIR:?Set BAT_STAGE_OUTPUT_DIR to a private checkpoint directory}"
GRAD_ACCUM="${BAT_GRADIENT_ACCUMULATION_STEPS:-1}"
LAUNCH_MODE="${BAT_LAUNCH_MODE:-ddp}"

case "$OUTPUT_DIR" in /hpc_stor03/public|/hpc_stor03/public/*) echo "Refusing public output" >&2; exit 2;; esac
ARGS=(--model-path "$MODEL_PATH" --plugin-path "$PLUGIN_PATH" --dataset "$DATASET" --stage "$STAGE" --output-dir "$OUTPUT_DIR" --world-size "$WORLD_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM")
if [[ -n "${BAT_RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$BAT_RESUME_FROM_CHECKPOINT")
fi

echo "========== BAT OURO STAGE LAUNCH =========="
echo "launch_mode=$LAUNCH_MODE"
echo "world_size=$WORLD_SIZE"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "per_device_batch_size=2 gradient_accumulation_steps=$GRAD_ACCUM global_batch_size=$((2 * WORLD_SIZE * GRAD_ACCUM))"

case "$LAUNCH_MODE" in
  single)
    if [[ "$WORLD_SIZE" -ne 1 ]]; then
      echo "single launch mode requires BAT_WORLD_SIZE=1, got $WORLD_SIZE" >&2
      exit 2
    fi
    python -u code/Ouro_audio/bat/scripts/train_bat_ouro_stage.py "${ARGS[@]}"
    ;;
  ddp)
    if [[ "$WORLD_SIZE" -le 1 ]]; then
      python -u code/Ouro_audio/bat/scripts/train_bat_ouro_stage.py "${ARGS[@]}"
    else
      export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
      export MASTER_PORT="${MASTER_PORT:-29517}"
      export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
      torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
        code/Ouro_audio/bat/scripts/train_bat_ouro_stage.py "${ARGS[@]}"
    fi
    ;;
  *)
    echo "Unsupported BAT_LAUNCH_MODE=$LAUNCH_MODE; expected single or ddp" >&2
    exit 2
    ;;
esac
