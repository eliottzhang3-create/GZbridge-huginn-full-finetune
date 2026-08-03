#!/usr/bin/env python3
"""X-ARES encoder entrypoint for the frozen Huginn Whisper audio front-end."""

from __future__ import annotations

import os
from pathlib import Path

from huginn_whisper_xares_encoder import HuginnWhisperXaresEncoder


CHECKPOINT = os.environ.get(
    "HUGINN_XARES_CHECKPOINT",
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/"
    "swift_output/v0-20260731-085036/checkpoint-20000",
)
PLUGIN_PATH = os.environ.get(
    "HUGINN_XARES_PLUGIN_PATH",
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py",
)
DEVICE = os.environ.get("HUGINN_XARES_ENCODER_DEVICE", "cuda:0")

encoder = HuginnWhisperXaresEncoder(
    checkpoint=Path(CHECKPOINT),
    plugin_path=Path(PLUGIN_PATH),
    device=DEVICE,
)

print(
    f"[xares-encoder-entry] checkpoint={CHECKPOINT} plugin={PLUGIN_PATH} "
    f"device={DEVICE} output_dim={encoder.output_dim} hop_ms={encoder.hop_size_in_ms}",
    flush=True,
)

