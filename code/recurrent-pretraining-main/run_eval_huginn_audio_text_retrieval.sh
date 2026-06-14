#!/bin/bash
set -euo pipefail

mkdir -p log
EXTRA_ARGS="$*"

vc submit \
  -p pdgpu-5090 \
  -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 \
  -c 8 -m 32G -g 1 \
  -n 1 \
  -j huginn-audio-text-retrieval-$(date +%m%d%H%M) \
  -d "$(pwd)" \
  JOB=1:1 "$(pwd)/log/huginn_audio_text_retrieval.JOB.log" \
  --cmd "bash local_scripts/eval_huginn_audio_text_retrieval.sh $EXTRA_ARGS"
