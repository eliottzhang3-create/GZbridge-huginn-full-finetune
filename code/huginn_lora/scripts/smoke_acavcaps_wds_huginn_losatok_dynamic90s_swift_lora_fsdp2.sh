#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export OMP_NUM_THREADS=4
export HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1
export HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1
export HUGINN_LOSATOK_TRAIN_CHAIN_AUDIT=1
export ACAVCAPS_WDS_MANIFEST="${ACAVCAPS_WDS_MANIFEST:-$REPO_ROOT/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json}"
export ACAVCAPS_WDS_BUFFER_SIZE="${ACAVCAPS_WDS_BUFFER_SIZE:-512}"
export ACAVCAPS_WDS_MAX_TARS_PER_STAGE="${ACAVCAPS_WDS_MAX_TARS_PER_STAGE:-2}"

PLUGIN_PATH="$REPO_ROOT/code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py"
MODEL_PATH="$REPO_ROOT/models/huginn-audio-losatok-v1"
LOSATOK_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok
LOSATOK_CODE_DIR="$REPO_ROOT/code/huginn_lora/LosatokCode"
OUTPUT_DIR="${ACAVCAPS_WDS_DYNAMIC_SMOKE_OUTPUT_DIR:-outputs/huginn_losatok_acavcaps_dynamic90s_fsdp2_smoke2_b4ga4}"
MAX_STEPS="${ACAVCAPS_WDS_DYNAMIC_SMOKE_MAX_STEPS:-2}"

for required_path in "$ACAVCAPS_WDS_MANIFEST" "$PLUGIN_PATH" "$MODEL_PATH" \
  "$LOSATOK_ROOT/ckpts/losatok_kl1e-3.pth" "$LOSATOK_ROOT/ckpts/semantic_encoder.pth" \
  "$LOSATOK_ROOT/midashenglm" "$LOSATOK_CODE_DIR/config/16k_16k_25Hz_losatok.yml"; do
  if [ ! -e "$required_path" ]; then
    echo "Required dynamic LoSATok ACAVCAPS smoke path is missing: $required_path" >&2
    exit 1
  fi
done
if ! [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACAVCAPS_WDS_DYNAMIC_SMOKE_MAX_STEPS must be a positive integer" >&2
  exit 1
fi
if find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
  echo "Smoke output already contains checkpoints; choose a fresh ACAVCAPS_WDS_DYNAMIC_SMOKE_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
FSDP_CONFIG_PATH="$OUTPUT_DIR/fsdp2_lora_no_activation.json"
printf '%s\n' '{"fsdp":"full_shard auto_wrap","fsdp_config":{"activation_checkpointing":false,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":true,"fsdp_version":2,"reshard_after_forward":true,"state_dict_type":"SHARDED_STATE_DICT"}}' > "$FSDP_CONFIG_PATH"

echo "========== ACAVCAPS DYNAMIC LOSATOK FSDP2-2GPU SMOKE =========="
echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
echo "manifest=$ACAVCAPS_WDS_MANIFEST"
echo "buffer_size=$ACAVCAPS_WDS_BUFFER_SIZE"
echo "max_tars_per_stage=$ACAVCAPS_WDS_MAX_TARS_PER_STAGE"
echo "dynamic_audio_tokens=true max_audio_seconds=90 max_audio_tokens=375"
echo "compressor_kernel=11 stride=6 adaptive_pool=false"
echo "world_size=2 per_device_batch_size=4 gradient_accumulation_steps=4 global_effective_batch_size=32"
echo "max_steps=$MAX_STEPS checkpoint_saving=disabled"
echo "streaming=true dataset_shuffle=false train_dataloader_shuffle=false"
echo "policy=frozen_losatok aligner_trainable huginn_lora_trainable"
echo "decode_policy=training_time_only"

python - <<'PY'
import torch
import torchaudio
print(f"[env] torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda}")
if torch.__version__ != torchaudio.__version__:
    raise SystemExit("Torch and torchaudio versions must match for the dynamic LoSATok smoke")
PY

swift sft \
  --model "$MODEL_PATH" \
  --model_type huginn_losatok_raven \
  --template huginn_losatok_text \
  --external_plugins "$PLUGIN_PATH" \
  --dataset "$ACAVCAPS_WDS_MANIFEST" \
  --streaming true \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --sortish_sampler false \
  --group_by_length false \
  --max_length 192 \
  --output_dir "$OUTPUT_DIR" \
  --tuner_type lora_llm \
  --freeze_aligner false \
  --learning_rate 1e-4 \
  --aligner_lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --fsdp "$FSDP_CONFIG_PATH" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing false \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --save_only_model false \
  --report_to none \
  --bf16 true

if find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
  echo "Smoke must not save checkpoints: found one under $OUTPUT_DIR" >&2
  exit 1
fi
echo "========== ACAVCAPS DYNAMIC LOSATOK FSDP2-2GPU SMOKE PASSED =========="
