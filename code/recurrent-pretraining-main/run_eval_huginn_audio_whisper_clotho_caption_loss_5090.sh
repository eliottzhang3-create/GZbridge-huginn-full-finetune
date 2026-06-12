#!/bin/bash
set -euo pipefail

mkdir -p log

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j huginn-audio-clotho-eval-5090-$(date +%m%d%H%M) \
  -d "$(pwd)" \
  JOB=1:1 "$(pwd)/log/huginn_audio_whisper_clotho_caption_loss_5090.JOB.log" \
  --cmd "bash local_scripts/eval_huginn_audio_whisper_clotho_caption_loss_5090.sh"
