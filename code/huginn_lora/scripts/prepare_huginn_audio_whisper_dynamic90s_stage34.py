#!/usr/bin/env python3
"""Prepare deterministic synthetic data for the merged Stage 3-4 FSDP4 gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


STAGE34_CASES = (
    # The first four records are deliberately mapped one-per-rank on the first
    # non-shuffled distributed step: short, 30s, 60s, and >120s->90s.
    (1.00, 8, 1),
    (30.00, 250, 1),
    (60.00, 500, 2),
    (120.01, 750, 3),
    (15.00, 125, 1),
    (45.00, 375, 2),
    (75.00, 625, 3),
    (90.00, 750, 3),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def build_record(plugin: Any, wav_path: Path, duration: float, plan: Any, index: int) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": plugin.DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Listen to the synthetic audio and state that it is a validation tone.",
            },
            {"role": "assistant", "content": "This is a synthetic validation tone."},
        ],
        "audios": [str(wav_path.resolve())],
        "metadata": {
            "dataset": "synthetic_dynamic90s_stage34_fsdp4",
            "record_index": index,
            "duration_seconds": duration,
            "included_seconds": plan.included_samples / 16_000,
            "expected_audio_tokens": plan.total_audio_tokens,
            "expected_prefix_tokens": plan.total_audio_tokens + 2,
            "expected_segments": plan.segment_count,
            "expected_first_step_rank": index if index < 4 else None,
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    stage02 = load_module(
        "huginn_audio_whisper_dynamic90s_stage34_stage02_helpers",
        repo_root / "code" / "huginn_lora" / "scripts" / "inspect_huginn_audio_whisper_dynamic90s_stage02.py",
    )
    plugin = stage02.import_plugin(
        repo_root / "code" / "huginn_lora" / "plugins" / "huginn_audio_whisper_dynamic90s_swift.py"
    )
    production_plans = stage02.assert_contract(plugin)

    fixture_dir = args.work_dir / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fixture_dir / "dynamic90s_stage34_fsdp4.jsonl"
    records: list[dict[str, Any]] = []
    for index, (duration, expected_tokens, expected_segments) in enumerate(STAGE34_CASES):
        plan = production_plans[duration]
        if plan.total_audio_tokens != expected_tokens or plan.segment_count != expected_segments:
            raise AssertionError(
                f"Stage 3-4 contract mismatch for {duration}s: "
                f"tokens={plan.total_audio_tokens} segments={plan.segment_count}"
            )
        wav_path = fixture_dir / f"stage34_{index:02d}_{duration:06.2f}s.wav"
        sample_count = stage02.write_synthetic_wav(wav_path, duration)
        if sample_count != plan.total_samples:
            raise AssertionError(f"Synthetic WAV sample count mismatch for {duration}s")
        records.append(build_record(plugin, wav_path, duration, plan, index))

    with manifest_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "synthetic_dynamic90s_stage34_fsdp4",
        "record_count": len(records),
        "world_size": 4,
        "first_step_rank_prefix_tokens": {str(rank): records[rank]["metadata"]["expected_prefix_tokens"] for rank in range(4)},
        "manifest": str(manifest_path.resolve()),
    }
    summary_path = fixture_dir / "dynamic90s_stage34_fsdp4.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("========== HUGINN WHISPER DYNAMIC90S STAGE 3-4 FIXTURE PREPARED ==========")
    print(f"[fixture] manifest={manifest_path}")
    print(f"[fixture] summary={summary_path}")
    print(f"[fixture] first_step_rank_prefix_tokens={summary['first_step_rank_prefix_tokens']}")


if __name__ == "__main__":
    main()
