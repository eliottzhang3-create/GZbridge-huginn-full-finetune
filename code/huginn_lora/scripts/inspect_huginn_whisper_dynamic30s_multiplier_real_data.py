#!/usr/bin/env python3
"""Decode one scheduled real sample from every multiplier component."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import TASK_PROMPTS  # noqa: E402
from data_pipeline.finite_multiplier_pool import (  # noqa: E402
    COMPONENT_ORDER,
    FiniteMultiplierPool,
    render_multiplier_row,
)


BASE_PLUGIN_PATH = HUGINN_LORA_ROOT / "plugins/huginn_audio_whisper_dynamic90s_swift.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def load_base_plugin() -> Any:
    module_name = "huginn_audio_whisper_dynamic30s_multiplier_realdata_base"
    spec = importlib.util.spec_from_file_location(module_name, BASE_PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Huginn audio plugin: {BASE_PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    plugin = load_base_plugin()
    registry_path = args.registry.expanduser().resolve()
    first_positions: dict[str, int] = {}
    results: dict[str, Any] = {}
    with FiniteMultiplierPool(registry_path) as pool:
        for position in range(pool.total_records):
            selection = pool.selection(position)
            first_positions.setdefault(selection.component_name, position)
            if len(first_positions) == len(COMPONENT_ORDER):
                break
        if tuple(sorted(first_positions, key=COMPONENT_ORDER.index)) != COMPONENT_ORDER:
            raise AssertionError(f"Multiplier schedule does not cover every component: {first_positions}")
        for name in COMPONENT_ORDER:
            position = first_positions[name]
            selection = pool.selection(position)
            record = pool.record(selection)
            row = render_multiplier_row(record, selection, pool.seed)
            expected_prompt = TASK_PROMPTS[selection.task]
            if row["messages"][1] != {"role": "user", "content": expected_prompt}:
                raise AssertionError(f"Multiplier task prompt mismatch for {name}")
            waveform = plugin.load_audio_item(
                row["audios"][0],
                target_sr=16000,
                max_audio_seconds=30.0,
            )
            duration = len(waveform) / 16000.0
            if not 0.0 < duration <= 30.000001:
                raise AssertionError(f"Multiplier real audio duration is invalid: {name}={duration}")
            results[name] = {
                "global_position": position,
                "uid": row["metadata"]["uid"],
                "record_index": selection.record_index,
                "replica_id": selection.replica_id,
                "task": selection.task,
                "audio_path": row["audios"][0]["path"],
                "audio_format": row["audios"][0]["format"],
                "start_sec": row["audios"][0]["start_sec"],
                "end_sec": row["audios"][0]["end_sec"],
                "decoded_duration_seconds": duration,
                "finite": bool(plugin.np.isfinite(waveform).all()),
            }
            if not results[name]["finite"]:
                raise AssertionError(f"Multiplier real waveform contains non-finite values: {name}")
            print(
                f"[multiplier-realdata] component={name} position={position} "
                f"format={results[name]['audio_format']} duration={duration:.6f}",
                flush=True,
            )
    report = {
        "gate": "huginn_whisper_dynamic30s_multiplier_real_data_v1",
        "validation_passed": True,
        "registry": str(registry_path),
        "audio_decode_count": len(results),
        "model_load": False,
        "whisper_load": False,
        "audio_copy": False,
        "components": results,
    }
    write_json_atomic(args.output_report.expanduser().resolve(), report)
    print(f"[multiplier-realdata] report={args.output_report.expanduser().resolve()}", flush=True)
    print("========== HUGINN WHISPER DYNAMIC30S MULTIPLIER REAL DATA PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
