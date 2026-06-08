#!/bin/bash
set -euo pipefail

mkdir -p log

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j huginn-audio-whisper-tiny-5090-$(date +%m%d%H%M) \
  -d "$(pwd)" \
  JOB=1:1 "$(pwd)/log/huginn_audio_whisper_tiny_overfit_5090.JOB.log" \
  --cmd "bash local_scripts/train_huginn_audio_whisper_tiny_overfit_5090.sh"
