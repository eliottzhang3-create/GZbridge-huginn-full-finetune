#!/bin/bash
set -euo pipefail

USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_BASE/envs/swift_huginn"

echo "ACTIVE_ENV=$CONDA_DEFAULT_ENV"
which python
python -V

python eval_audio_text_retrieval.py "$@"
