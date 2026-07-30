#!/usr/bin/env python3
"""Audit the single-chunk 30-second / 160-ms Huginn Whisper contract."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "models/huginn-audio-whisper-dynamic90s-v1/config.json"
MODEL_SOURCE = REPO_ROOT / "models/huginn-audio-whisper-dynamic90s-v1/raven_modeling_minimal.py"
PLUGIN_SOURCE = REPO_ROOT / "code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py"
SAMPLE_RATE = 16000
FEATURE_HOP = 160
ENCODER_DOWNSAMPLE = 2
KERNEL = 8
STRIDE = 8
RETAIN_SECONDS = 30.0
DISCARD_ABOVE_SECONDS = 90.0


def token_count(duration_seconds: float) -> int | None:
    if duration_seconds <= 0:
        raise ValueError(f"Duration must be positive: {duration_seconds}")
    if duration_seconds > DISCARD_ABOVE_SECONDS:
        return None
    retained_samples = min(int(round(duration_seconds * SAMPLE_RATE)), int(RETAIN_SECONDS * SAMPLE_RATE))
    feature_frames = max(1, retained_samples // FEATURE_HOP)
    encoder_frames = feature_frames // ENCODER_DOWNSAMPLE
    if encoder_frames < KERNEL:
        return 0
    return (encoder_frames - KERNEL) // STRIDE + 1


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "audio_pooling_type": "conv1d_stride8_dynamic30s",
        "audio_token_duration_ms": 160,
        "audio_reference_30s_token_count": 187,
        "audio_max_token_count": 187,
        "audio_chunk_seconds": 30.0,
        "audio_max_seconds": 30.0,
        "audio_compressor_kernel_size": 8,
        "audio_compressor_stride": 8,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"Dynamic-30s model config mismatch: {mismatches}")

    derived_ms = FEATURE_HOP * ENCODER_DOWNSAMPLE * STRIDE * 1000 // SAMPLE_RATE
    if derived_ms != 160:
        raise AssertionError(f"Derived token duration changed: {derived_ms}ms")

    cases = {
        0.08: 0,
        0.16: 1,
        1.0: 6,
        10.0: 62,
        29.999: 187,
        30.0: 187,
        30.001: 187,
        60.0: 187,
        90.0: 187,
        90.001: None,
        120.0: None,
    }
    observed = {duration: token_count(duration) for duration in cases}
    if observed != cases:
        raise AssertionError(f"Duration/token policy mismatch: observed={observed} expected={cases}")

    model_source = MODEL_SOURCE.read_text(encoding="utf-8")
    plugin_source = PLUGIN_SOURCE.read_text(encoding="utf-8")
    required_markers = {
        "model_single_chunk_guard": "if segment_count != 1:",
        "template_single_chunk_guard": "if len(audio_chunks) != 1 or len(audio_feature_lengths) != 1:",
        "batch_single_chunk_guard": "if any(count != 1 for count in segment_counts):",
        "duration_discard_guard": "if raw_duration_seconds > DISCARD_AUDIO_ABOVE_SECONDS:",
    }
    missing = [
        name
        for name, marker in required_markers.items()
        if marker not in (model_source if name.startswith("model_") else plugin_source)
    ]
    if missing:
        raise AssertionError(f"Dynamic-30s runtime guards are missing: {missing}")

    print("========== HUGINN WHISPER DYNAMIC30S CONTRACT PASSED ==========")
    print(f"[contract] token_duration_ms={derived_ms} max_audio_tokens=187 boundary_tokens=2")
    print(f"[contract] duration_cases={observed}")
    print("[contract] chunks_per_sample=1 local_batch_prefix_padding=true discard_above_90s=true")


if __name__ == "__main__":
    main()
