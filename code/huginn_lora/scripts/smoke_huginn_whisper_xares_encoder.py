#!/usr/bin/env python3
"""Synthetic and small-real-data smoke for the Huginn X-ARES encoder wrapper."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

from inspect_huginn_xares_voxceleb1_data import (
    AUDIO_SUFFIXES,
    bounded_files,
    import_voxceleb_task,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/"
    "huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/"
    "swift_output/v0-20260731-085036/checkpoint-20000"
)
DEFAULT_PLUGIN = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py"
)
DEFAULT_XARES_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares"
DEFAULT_DATA_ROOT = "/hpc_stor03/public/shared/data/mml/VoxCeleb1_origin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--plugin-path", default=DEFAULT_PLUGIN)
    parser.add_argument("--xares-root", default=DEFAULT_XARES_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--real-count", type=int, default=4)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_wrapper_module() -> Any:
    path = SCRIPT_DIR / "huginn_whisper_xares_encoder.py"
    spec = importlib.util.spec_from_file_location("huginn_whisper_xares_encoder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expected_tokens(sample_count: int, sample_rate: int = 16000) -> int:
    feature_length = min(3000, max(1, sample_count // 160))
    encoder_length = feature_length // 2
    if encoder_length < 12:
        return 0
    return (encoder_length - 12) // 12 + 1


def main() -> None:
    args = parse_args()
    xares_root = Path(args.xares_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    task_report = import_voxceleb_task(xares_root)
    if not task_report.get("ok"):
        raise RuntimeError(f"VoxCeleb1 task import failed during wrapper smoke: {task_report}")
    print(
        f"[xares-smoke][task] config_instances={task_report.get('task_config_instances', {})}",
        flush=True,
    )
    audio_paths = bounded_files(data_root / "wav_total", AUDIO_SUFFIXES, args.real_count)
    if len(audio_paths) < args.real_count:
        part_dirs = sorted(
            path for path in data_root.iterdir() if path.name.startswith(("vox1_dev_wav_part", "vox1_test_wav"))
        )
        for part_dir in part_dirs:
            audio_paths.extend(
                bounded_files(part_dir, AUDIO_SUFFIXES, args.real_count - len(audio_paths))
            )
            if len(audio_paths) >= args.real_count:
                break
    if not audio_paths:
        raise FileNotFoundError(f"No real VoxCeleb1 audio samples found below {data_root}")

    wrapper_module = load_wrapper_module()
    encoder = wrapper_module.HuginnWhisperXaresEncoder(
        checkpoint=args.checkpoint,
        plugin_path=args.plugin_path,
        device=args.device,
    )

    synthetic_cases = []
    for seconds in (1.0, 3.125, 10.0, 29.999, 30.0):
        sample_count = int(round(seconds * encoder.sampling_rate))
        waveform = torch.linspace(-0.25, 0.25, sample_count, dtype=torch.float32).unsqueeze(0)
        output = encoder(waveform)
        expected = expected_tokens(sample_count)
        actual = tuple(int(value) for value in output.shape)
        if actual != (1, expected, encoder.output_dim):
            raise RuntimeError(
                f"Synthetic output contract mismatch: seconds={seconds} expected={(1, expected, encoder.output_dim)} actual={actual}"
            )
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError(f"Synthetic output is non-finite: seconds={seconds}")
        synthetic_cases.append({
            "seconds": seconds,
            "samples": sample_count,
            "expected_tokens": expected,
            "output_shape": list(actual),
            "finite": True,
        })
        print(f"[xares-smoke][synthetic] seconds={seconds} output_shape={actual}", flush=True)

    real_cases = []
    for audio_path in audio_paths[: args.real_count]:
        waveform = encoder.plugin.load_audio_file(
            Path(audio_path),
            target_sr=encoder.sampling_rate,
            max_audio_seconds=30.0,
        )
        output = encoder(torch.from_numpy(waveform).unsqueeze(0))
        actual = tuple(int(value) for value in output.shape)
        if actual[0] != 1 or actual[2] != encoder.output_dim or actual[1] <= 0:
            raise RuntimeError(f"Real output contract mismatch: path={audio_path} shape={actual}")
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError(f"Real output is non-finite: {audio_path}")
        real_cases.append({
            "audio_path": str(audio_path),
            "decoded_samples": int(waveform.shape[0]),
            "retained_seconds": float(waveform.shape[0] / encoder.sampling_rate),
            "output_shape": list(actual),
            "finite": True,
        })
        print(f"[xares-smoke][real] path={audio_path} output_shape={actual}", flush=True)

    report = {
        "gate": "huginn_xares_encoder_synthetic_and_voxceleb1_smoke_v1",
        "validation_passed": True,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "plugin_path": str(Path(args.plugin_path).expanduser().resolve()),
        "xares_root": str(xares_root),
        "data_root": str(data_root),
        "task_import": task_report,
        "wrapper_contract": {
            "sampling_rate": encoder.sampling_rate,
            "output_dim": encoder.output_dim,
            "hop_size_in_ms": encoder.hop_size_in_ms,
            "maximum_seconds": 30.0,
            "boundary_tokens_included": False,
            "source_path": [
                "audio_encoder.encoder",
                "audio_aligner.temporal_compressor",
                "audio_aligner.audio_projector",
            ],
            "restored_audio_and_aligner": encoder.restore_report,
        },
        "synthetic_cases": synthetic_cases,
        "real_cases": real_cases,
        "full_xares_knn": False,
    }
    output = args.output_report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[xares-smoke] report={output}", flush=True)
    print("========== HUGINN X-ARES ENCODER SYNTHETIC/REAL SMOKE PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
