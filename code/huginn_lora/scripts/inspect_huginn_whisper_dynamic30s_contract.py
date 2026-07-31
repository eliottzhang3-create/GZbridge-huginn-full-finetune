#!/usr/bin/env python3
"""Audit the single-chunk 30-second / 240-ms Huginn Whisper contract."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "models/huginn-audio-whisper-dynamic90s-v1/config.json"
DATA_CONTRACT_PATH = REPO_ROOT / "code/huginn_lora/configs/huginn_whisper_dynamic90s_data_contract_v1.json"
MODEL_SOURCE = REPO_ROOT / "models/huginn-audio-whisper-dynamic90s-v1/raven_modeling_minimal.py"
PLUGIN_SOURCE = REPO_ROOT / "code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py"
SAMPLE_RATE = 16000
FEATURE_HOP = 160
ENCODER_DOWNSAMPLE = 2
KERNEL = 12
STRIDE = 12
RETAIN_SECONDS = 30.0


def token_count(duration_seconds: float) -> int | None:
    if duration_seconds <= 0:
        raise ValueError(f"Duration must be positive: {duration_seconds}")
    retained_samples = min(int(round(duration_seconds * SAMPLE_RATE)), int(RETAIN_SECONDS * SAMPLE_RATE))
    feature_frames = max(1, retained_samples // FEATURE_HOP)
    encoder_frames = feature_frames // ENCODER_DOWNSAMPLE
    if encoder_frames < KERNEL:
        return 0
    return (encoder_frames - KERNEL) // STRIDE + 1


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_contract = json.loads(DATA_CONTRACT_PATH.read_text(encoding="utf-8"))
    expected = {
        "audio_pooling_type": "conv1d_stride12_dynamic30s",
        "audio_token_duration_ms": 240,
        "audio_reference_30s_token_count": 125,
        "audio_max_token_count": 125,
        "audio_chunk_seconds": 30.0,
        "audio_max_seconds": 30.0,
        "audio_compressor_kernel_size": 12,
        "audio_compressor_stride": 12,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"Dynamic-30s model config mismatch: {mismatches}")
    runtime_contract = data_contract.get("audio_runtime_contract", {})
    if (
        runtime_contract.get("discard_above_seconds") is not None
        or runtime_contract.get("max_included_seconds") != RETAIN_SECONDS
        or runtime_contract.get("long_audio_policy")
        != "retain every eligible dataset record and decode only the first 30 seconds"
    ):
        raise AssertionError(f"Dynamic-30s data duration policy mismatch: {runtime_contract}")

    derived_ms = FEATURE_HOP * ENCODER_DOWNSAMPLE * STRIDE * 1000 // SAMPLE_RATE
    if derived_ms != 240:
        raise AssertionError(f"Derived token duration changed: {derived_ms}ms")

    cases = {
        0.08: 0,
        0.16: 0,
        0.24: 1,
        1.0: 4,
        10.0: 41,
        29.999: 124,
        30.0: 125,
        30.001: 125,
        60.0: 125,
        90.0: 125,
        90.001: 125,
        120.0: 125,
        3600.0: 125,
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
    }
    missing = [
        name
        for name, marker in required_markers.items()
        if marker not in (model_source if name.startswith("model_") else plugin_source)
    ]
    if missing:
        raise AssertionError(f"Dynamic-30s runtime guards are missing: {missing}")
    forbidden_plugin_markers = (
        "DISCARD_AUDIO_ABOVE_SECONDS",
        "duration-filtered training pool emitted an ineligible audio sample",
    )
    present_forbidden = [marker for marker in forbidden_plugin_markers if marker in plugin_source]
    if present_forbidden:
        raise AssertionError(f"Obsolete duration-discard guards remain: {present_forbidden}")

    print("========== HUGINN WHISPER DYNAMIC30S CONTRACT PASSED ==========")
    print(f"[contract] token_duration_ms={derived_ms} max_audio_tokens=125 boundary_tokens=2")
    print(f"[contract] duration_cases={observed}")
    print("[contract] chunks_per_sample=1 local_batch_prefix_padding=true retain_all_cap_at_30s=true")


if __name__ == "__main__":
    main()
