#!/usr/bin/env python3
"""Prepare repeated exact-30-second synthetic audio for acceleration Stage 2."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from array import array
from pathlib import Path


SAMPLE_RATE = 16_000
DURATION_SECONDS = 30.0
EXPECTED_SEGMENTS = 1
EXPECTED_AUDIO_TOKENS = 125
EXPECTED_PREFIX_TOKENS = 127
RECORD_COUNT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def write_wav(path: Path) -> int:
    sample_count = int(SAMPLE_RATE * DURATION_SECONDS)
    one_second = array(
        "h",
        (
            int(round(0.06 * 32767.0 * math.sin(2.0 * math.pi * 220.0 * index / SAMPLE_RATE)))
            for index in range(SAMPLE_RATE)
        ),
    )
    if sys.byteorder != "little":
        one_second.byteswap()
    pcm_bytes = one_second.tobytes() * int(DURATION_SECONDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)
    return sample_count


def main() -> None:
    args = parse_args()
    fixture_dir = args.work_dir.expanduser().resolve() / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    wav_path = fixture_dir / "synthetic_exact30s.wav"
    sample_count = write_wav(wav_path)
    expected_sample_count = int(DURATION_SECONDS * SAMPLE_RATE)
    if sample_count != expected_sample_count:
        raise AssertionError(
            f"30-second WAV sample count mismatch: actual={sample_count} expected={expected_sample_count}"
        )

    manifest_path = fixture_dir / "dynamic30s_acceleration_stage2_fsdp4.jsonl"
    with manifest_path.open("w", encoding="utf-8") as output_file:
        for index in range(RECORD_COUNT):
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can understand audio and respond accurately.",
                    },
                    {
                        "role": "user",
                        "content": "Listen to the synthetic audio and describe the validation tone.",
                    },
                    {
                        "role": "assistant",
                        "content": "The audio contains a steady synthetic validation tone.",
                    },
                ],
                "audios": [str(wav_path)],
                "metadata": {
                    "dataset": "synthetic_dynamic30s_acceleration_stage2_fsdp4",
                    "record_index": index,
                    "duration_seconds": DURATION_SECONDS,
                    "expected_segments": EXPECTED_SEGMENTS,
                    "expected_audio_tokens": EXPECTED_AUDIO_TOKENS,
                    "expected_prefix_tokens": EXPECTED_PREFIX_TOKENS,
                },
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "gate": "huginn_whisper_dynamic30s_240ms_acceleration_stage2_fixture_v2",
        "dataset": "synthetic_dynamic30s_acceleration_stage2_fsdp4",
        "record_count": RECORD_COUNT,
        "duration_seconds": DURATION_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "sample_count": sample_count,
        "segments_per_sample": EXPECTED_SEGMENTS,
        "audio_tokens_per_sample": EXPECTED_AUDIO_TOKENS,
        "prefix_tokens_per_sample": EXPECTED_PREFIX_TOKENS,
        "per_device_train_batch_size": 2,
        "world_size": 4,
        "gradient_accumulation_steps": 4,
        "global_batch_size": 32,
        "manifest": str(manifest_path),
        "wav": str(wav_path),
    }
    summary_path = fixture_dir / "dynamic30s_acceleration_stage2_fsdp4.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("========== HUGINN WHISPER DYNAMIC30S ACCELERATION STAGE 2 FIXTURE PREPARED ==========")
    print(f"[fixture] manifest={manifest_path}")
    print(f"[fixture] summary={summary_path}")
    print(f"[fixture] contract={summary}")


if __name__ == "__main__":
    main()
