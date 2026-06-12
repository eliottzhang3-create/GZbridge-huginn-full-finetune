#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
which python
python -V
python -c "import torch; print('torch =', torch.__version__); print('cuda available =', torch.cuda.is_available()); print('gpu_count =', torch.cuda.device_count())"

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval_audio_whisper_clotho_caption_loss.py
