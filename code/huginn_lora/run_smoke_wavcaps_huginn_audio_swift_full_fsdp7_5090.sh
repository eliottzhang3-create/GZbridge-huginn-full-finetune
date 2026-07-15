#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

vc submit -p pdgpu-5090 -i docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1 -c 28 -m 224G -g 7 -n 1 -j smoke-huginn-audio-full-fsdp7-5090-$(date +%m%d%H%M) -d "$SCRIPT_DIR" JOB=1:1 "$SCRIPT_DIR/log/smoke_wavcaps_huginn_audio_swift_full_fsdp7_5090.JOB.log" --cmd "bash scripts/smoke_wavcaps_huginn_audio_swift_full_fsdp7.sh"
