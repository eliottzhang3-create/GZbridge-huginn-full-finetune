#!/bin/bash
set -euo pipefail

mkdir -p log

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 16 -m 128G -g 1 \
  -n 1 \
  -j huginn-full-gsm8k-lmeval-5090-$(date +%m%d%H%M) \
  -d "$(pwd)" \
  JOB=1:1 "$(pwd)/log/huginn_eval_gsm8k_5090.JOB.log" \
  --cmd "bash local_scripts/eval_huginn_full_checkpoint_gsm8k_5090.sh"
