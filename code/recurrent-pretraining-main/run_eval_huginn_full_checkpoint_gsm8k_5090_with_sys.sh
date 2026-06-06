#!/bin/bash
set -euo pipefail

mkdir -p log

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j huginn-lmeval-5090-with-sys-$(date +%m%d%H%M) \
  -d "$(pwd)" \
  JOB=1:1 "$(pwd)/log/huginn_eval_5090_with_sys.JOB.log" \
  --cmd "bash local_scripts/eval_huginn_full_checkpoint_gsm8k_5090_with_sys.sh"
