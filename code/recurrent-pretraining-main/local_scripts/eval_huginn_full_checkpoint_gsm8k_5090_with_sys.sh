#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

WORKDIR="$(pwd)"
CHECKPOINT_DIR="/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}"
  exit 1
fi

TASK="${TASK:-gsm8k}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
MEAN_RECURRENCE="${MEAN_RECURRENCE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant that can assist users with mathematical reasoning.}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/outputs/lm_eval_${TASK}_with_sys_$(basename "${CHECKPOINT_DIR}")}"

mkdir -p scripts/fake_bin
cat > scripts/fake_bin/git <<'EOF'
#!/bin/bash
if [ "$1" = "describe" ] && [ "$2" = "--always" ]; then
  echo "nogit"
  exit 0
fi
echo "unsupported fake git command: $@" >&2
exit 0
EOF
chmod +x scripts/fake_bin/git

export PATH="${WORKDIR}/scripts/fake_bin:/usr/bin:/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SYSTEM_PROMPT

echo "[step] env ready"
echo "ACTIVE_ENV=${CONDA_DEFAULT_ENV}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TASK=${TASK}"
echo "NUM_FEWSHOT=${NUM_FEWSHOT}"
echo "DEVICE=${DEVICE}"
echo "DTYPE=${DTYPE}"
echo "MEAN_RECURRENCE=${MEAN_RECURRENCE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "APPLY_CHAT_TEMPLATE=true"
echo "SYSTEM_PROMPT=${SYSTEM_PROMPT}"

echo "[step] python"
which python || true
python -V || true

echo "[step] lm_eval"
which lm_eval || true

echo "[step] git"
which git || true
git describe --always || true

mkdir -p "${OUTPUT_DIR}"

echo "[step] before lm_eval run"
lm_eval run \
  --model hf \
  --model_args "pretrained=${CHECKPOINT_DIR},trust_remote_code=True,dtype=${DTYPE},mean_recurrence=${MEAN_RECURRENCE}" \
  --tasks "${TASK}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --system_instruction "${SYSTEM_PROMPT}" \
  --apply_chat_template \
  --output_path "${OUTPUT_DIR}" \
  --log_samples

echo "[step] after lm_eval run"
