#!/bin/bash
mkdir -p log

vc submit \
  -p pdgpu-v100 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 32 -m 256G -g 8 \
  -n 1 \
  -j huginn-full-gsm8k-fsdp-$(date +%m%d%H%M) \
  -d "$(pwd)" \
  JOB=1:1 "$(pwd)/log/huginn_full_gsm8k_fsdp.JOB.log" \
  --cmd "bash local_scripts/train_huginn_full_gsm8k_fsdp.sh"
