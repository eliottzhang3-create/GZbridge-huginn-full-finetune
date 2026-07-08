#!/bin/bash
set -euo pipefail
USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate swift_huginn_v100_2
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/huginn_pipeline
python scripts/debug_huginn_sft.py
