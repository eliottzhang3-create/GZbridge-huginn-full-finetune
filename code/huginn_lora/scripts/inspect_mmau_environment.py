#!/usr/bin/env python3
"""Read-only MMAU readiness inspection for the remote Swift/Huginn environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import wave
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune")
DEFAULT_SEARCH_ROOTS = [
    Path("/hpc_stor03/sjtu_home/jinwei.zhang/data"),
    Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data"),
]
DEFAULT_CHECKPOINTS = [
    Path(
        "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/"
        "huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-5604"
    ),
    Path(
        "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/"
        "huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406"
    ),
]
PACKAGE_NAMES = [
    "torch",
    "transformers",
    "swift",
    "peft",
    "safetensors",
    "numpy",
    "tqdm",
    "datasets",
    "pyarrow",
    "soundfile",
    "librosa",
    "torchaudio",
    "torchcodec",
    "huggingface_hub",
    "lm_eval",
]
EXPECTED_DATA_NAMES = {
    "test_mini.parquet",
    "test.parquet",
    "mmau-test-mini.json",
    "mmau-test.json",
    "test-audios.tar.gz",
}


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def path_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return summary
    summary["is_dir"] = path.is_dir()
    if path.is_file():
        summary["size_bytes"] = path.stat().st_size
        summary["size_human"] = format_bytes(path.stat().st_size)
    return summary


def package_summary(name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"package": name, "installed": False}
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # Optional packages may fail because of system libraries.
        result["import_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["installed"] = True
    result["module_path"] = getattr(module, "__file__", None)
    try:
        result["version"] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["version"] = getattr(module, "__version__", "unknown")
    return result


def find_mmau_roots(search_roots: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for child in search_root.iterdir():
            if child.is_dir() and "mmau" in child.name.lower():
                discovered.add(child.resolve())
            if child.is_file() and child.name.lower() in EXPECTED_DATA_NAMES:
                discovered.add(search_root.resolve())
    return sorted(discovered)


def list_relevant_files(dataset_root: Path) -> list[dict[str, Any]]:
    relevant: list[dict[str, Any]] = []
    if not dataset_root.is_dir():
        return relevant
    for item in sorted(dataset_root.iterdir()):
        name = item.name.lower()
        if (
            name in EXPECTED_DATA_NAMES
            or "test-mini" in name
            or "test_mini" in name
            or "test-audios" in name
            or "test_audios" in name
            or "mmau" in name
        ):
            relevant.append(path_summary(item))
    return relevant


def inspect_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        result["record_count"] = len(records) if isinstance(records, list) else None
        if isinstance(records, list) and records:
            first = records[0]
            result["first_record_keys"] = sorted(first) if isinstance(first, dict) else None
            result["first_record"] = first if isinstance(first, dict) else repr(first)
            audio_values = [
                record.get("audio_id")
                for record in records
                if isinstance(record, dict) and isinstance(record.get("audio_id"), str)
            ]
            result["audio_id_count"] = len(audio_values)
            result["first_audio_id"] = audio_values[0] if audio_values else None
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_parquet(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        result["error"] = f"pyarrow unavailable: {type(exc).__name__}: {exc}"
        return result
    try:
        parquet_file = pq.ParquetFile(path)
        result["row_count"] = parquet_file.metadata.num_rows
        result["row_group_count"] = parquet_file.metadata.num_row_groups
        result["schema"] = str(parquet_file.schema_arrow)
        result["column_names"] = parquet_file.schema_arrow.names
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_audio_directory(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    if not path.is_dir():
        result["error"] = "not a directory"
        return result
    wav_files = sorted(path.glob("*.wav"))
    result["wav_count"] = len(wav_files)
    if not wav_files:
        return result
    sample = wav_files[0]
    result["first_wav"] = str(sample)
    result["first_wav_size"] = format_bytes(sample.stat().st_size)
    try:
        with wave.open(str(sample), "rb") as handle:
            result["first_wav_format"] = {
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "frame_count": handle.getnframes(),
                "duration_seconds": handle.getnframes() / float(handle.getframerate()),
            }
    except Exception as exc:
        result["first_wav_read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_dataset_root(dataset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(dataset_root),
        "summary": path_summary(dataset_root),
        "relevant_files": list_relevant_files(dataset_root),
    }
    if not dataset_root.is_dir():
        return result
    json_files = sorted(dataset_root.glob("*.json"))
    parquet_files = sorted(dataset_root.glob("*.parquet"))
    audio_dirs = [
        child
        for child in sorted(dataset_root.iterdir())
        if child.is_dir() and ("audio" in child.name.lower() or "test" in child.name.lower())
    ]
    result["json_inspection"] = [inspect_json(path) for path in json_files if "mmau" in path.name.lower()]
    result["parquet_inspection"] = [inspect_parquet(path) for path in parquet_files]
    result["audio_directory_inspection"] = [inspect_audio_directory(path) for path in audio_dirs]
    disk = shutil.disk_usage(dataset_root)
    result["disk_free"] = format_bytes(disk.free)
    result["disk_total"] = format_bytes(disk.total)
    return result


def print_section(title: str) -> None:
    print(f"========== {title} ==========")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--output-report", type=Path, default=None)
    args = parser.parse_args()

    search_roots = args.search_root or DEFAULT_SEARCH_ROOTS
    report_path = args.output_report or DEFAULT_REPO_ROOT / "data/audio_swift/mmau/mmau_environment_inspect.json"

    print_section("MMAU ENVIRONMENT INSPECT")
    print(f"[env] python={sys.version.split()[0]}")
    print(f"[env] platform={platform.platform()}")
    print(f"[env] executable={sys.executable}")
    print(f"[env] cwd={Path.cwd()}")
    print(f"[dataset] explicit_root={args.dataset_root}")
    print(f"[dataset] search_roots={[str(path) for path in search_roots]}")
    print(f"[report] output={report_path}")

    print_section("PACKAGES")
    packages = [package_summary(name) for name in PACKAGE_NAMES]
    for package in packages:
        if package["installed"]:
            print(f"[package] {package['package']} installed version={package.get('version')} path={package.get('module_path')}")
        else:
            print(f"[package] {package['package']} unavailable error={package.get('import_error')}")

    print_section("SYSTEM TOOLS")
    tool_paths = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "sox", "flac")}
    for name, location in tool_paths.items():
        print(f"[tool] {name}={location}")

    print_section("CHECKPOINTS")
    checkpoints = [path_summary(path) for path in DEFAULT_CHECKPOINTS]
    for checkpoint in checkpoints:
        print(f"[checkpoint] {checkpoint}")

    print_section("DATASET DISCOVERY")
    discovered_roots = find_mmau_roots(search_roots)
    if args.dataset_root is not None:
        dataset_roots = [args.dataset_root]
    else:
        dataset_roots = discovered_roots
    print(f"[dataset] discovered_roots={[str(path) for path in discovered_roots]}")
    if not dataset_roots:
        print("[dataset] no MMAU root found at top level of the configured search roots")

    dataset_reports = [inspect_dataset_root(path) for path in dataset_roots]
    for dataset_report in dataset_reports:
        print(f"[dataset] root={dataset_report['root']}")
        print(f"[dataset] summary={json.dumps(dataset_report['summary'], ensure_ascii=True)}")
        for item in dataset_report["relevant_files"]:
            print(f"[dataset-file] {json.dumps(item, ensure_ascii=True)}")
        for item in dataset_report.get("json_inspection", []):
            print(f"[json] {json.dumps(item, ensure_ascii=True)}")
        for item in dataset_report.get("parquet_inspection", []):
            print(f"[parquet] {json.dumps(item, ensure_ascii=True)}")
        for item in dataset_report.get("audio_directory_inspection", []):
            print(f"[audio-dir] {json.dumps(item, ensure_ascii=True)}")
        print(f"[dataset] disk_free={dataset_report.get('disk_free')}")

    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "packages": packages,
        "system_tools": tool_paths,
        "checkpoints": checkpoints,
        "search_roots": [str(path) for path in search_roots],
        "discovered_roots": [str(path) for path in discovered_roots],
        "datasets": dataset_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_section("MMAU ENVIRONMENT INSPECT DONE")
    print(f"[report] output={report_path}")
    print(f"[dataset] inspected_root_count={len(dataset_reports)}")


if __name__ == "__main__":
    main()
