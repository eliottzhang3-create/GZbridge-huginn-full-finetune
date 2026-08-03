#!/usr/bin/env python3
"""Read-only audit of the X-ARES VoxCeleb1 task and shared data layout."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any


DEFAULT_XARES_ROOT = Path("/hpc_stor03/public/shared/data/mml/VoxCeleb1_origin")
DEFAULT_XARES_CODE_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares")
EXPECTED_TOP_LEVEL = ("dev_test_split", "txt", "wav_total")
PART_PREFIXES = ("vox1_dev_wav_part", "vox1_test_wav")
TEXT_SUFFIXES = (".txt", ".lst", ".csv", ".json", ".jsonl")
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".opus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xares-root", type=Path, default=DEFAULT_XARES_CODE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_XARES_ROOT)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-audio-samples", type=int, default=8)
    return parser.parse_args()


def import_voxceleb_task(xares_root: Path) -> dict[str, Any]:
    src_root = xares_root / "src"
    task_path = src_root / "tasks" / "voxceleb1_task.py"
    if not task_path.is_file():
        return {"ok": False, "error": f"Missing task file: {task_path}"}
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    try:
        spec = importlib.util.spec_from_file_location("xares_voxceleb1_task_audit", task_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec: {task_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        configs = {}
        for name, value in vars(module).items():
            if value.__class__.__name__ == "TaskConfig":
                try:
                    configs[name] = dataclasses.asdict(value)
                except TypeError:
                    configs[name] = repr(value)
        return {
            "ok": True,
            "path": str(task_path),
            "module_symbols": sorted(name for name in vars(module) if not name.startswith("__"))[:80],
            "task_config_instances": configs,
        }
    except Exception as exc:  # noqa: BLE001 - preserve remote audit diagnostics
        return {
            "ok": False,
            "path": str(task_path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def bounded_children(path: Path, limit: int = 20) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    if not path.is_dir():
        return {"exists": True, "is_dir": False, "path": str(path)}
    names = []
    try:
        with os.scandir(path) as iterator:
            for index, entry in enumerate(iterator):
                if index >= limit:
                    break
                names.append(entry.name)
    except OSError as exc:
        return {"exists": True, "is_dir": True, "path": str(path), "error": str(exc)}
    return {
        "exists": True,
        "is_dir": True,
        "path": str(path),
        "child_preview": sorted(names),
        "preview_limit": limit,
    }


def bounded_files(path: Path, suffixes: tuple[str, ...], limit: int) -> list[str]:
    if not path.is_dir():
        return []
    found: list[str] = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.lower().endswith(suffixes):
                found.append(str(Path(root) / filename))
                if len(found) >= limit:
                    return found
    return found


def source_path_mentions(task_path: Path) -> list[str]:
    text = task_path.read_text(encoding="utf-8")
    matches = re.findall(r"[^\"' ]*(?:vox|wav|txt|split|Vox)[^\"' ]*", text)
    unique: list[str] = []
    for match in matches:
        cleaned = match.strip("()[]{}:,=")
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique[:100]


def main() -> None:
    args = parse_args()
    xares_root = args.xares_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    report_path = args.output_report.expanduser().resolve()
    if not xares_root.is_dir():
        raise FileNotFoundError(f"X-ARES root does not exist: {xares_root}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"VoxCeleb1 data root does not exist: {data_root}")

    task_path = xares_root / "src" / "tasks" / "voxceleb1_task.py"
    task_import = import_voxceleb_task(xares_root)
    top_level = {
        name: bounded_children(data_root / name)
        for name in EXPECTED_TOP_LEVEL
    }
    parts = sorted(
        entry.name
        for entry in data_root.iterdir()
        if entry.name.startswith(PART_PREFIXES)
    )
    text_search_roots = [data_root / "txt", data_root / "dev_test_split"]
    text_files: list[str] = []
    for search_root in text_search_roots:
        text_files.extend(bounded_files(search_root, TEXT_SUFFIXES, 40 - len(text_files)))
        if len(text_files) >= 40:
            break
    audio_source_candidates = [
        data_root / "wav_total",
        *(data_root / name for name in parts),
    ]
    existing_audio_sources = [str(path) for path in audio_source_candidates if path.exists()]
    audio_samples: list[str] = []
    for source in audio_source_candidates:
        audio_samples.extend(bounded_files(source, AUDIO_SUFFIXES, max(1, args.max_audio_samples) - len(audio_samples)))
        if len(audio_samples) >= max(1, args.max_audio_samples):
            break

    blocking_issues = []
    if not task_import.get("ok"):
        blocking_issues.append("voxceleb1_task_import_failed")
    if not existing_audio_sources:
        blocking_issues.append("no_audio_source_directory_found")
    if not text_files:
        blocking_issues.append("no_split_or_label_text_files_found")
    missing_top_level = [name for name, payload in top_level.items() if not payload.get("exists")]
    if missing_top_level:
        blocking_issues.append(f"missing_expected_top_level={missing_top_level}")

    task_mentions = source_path_mentions(task_path) if task_path.is_file() else []
    report = {
        "gate": "huginn_xares_voxceleb1_data_path_audit_v1",
        "validation_passed": not blocking_issues,
        "audio_decode": False,
        "audio_copy": False,
        "full_audio_scan": False,
        "xares_root": str(xares_root),
        "voxceleb1_data_root": str(data_root),
        "task_import": task_import,
        "task_source_path_mentions": task_mentions,
        "expected_top_level": top_level,
        "detected_archive_or_audio_part_entries": parts,
        "label_or_split_file_samples": text_files,
        "audio_file_samples": audio_samples,
        "existing_audio_source_candidates": existing_audio_sources,
        "blocking_issues": blocking_issues,
        "next_step_contract": {
            "read_only_public_root": True,
            "requires_local_manifest_or_index_only_if_task_layout_mismatches": True,
            "audio_decode_deferred_to_real_data_smoke": True,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[voxceleb1] task_import={task_import.get('ok')}", flush=True)
    if not task_import.get("ok"):
        print(f"[voxceleb1][task-import-error] {task_import.get('error')}", flush=True)
        traceback_text = task_import.get("traceback")
        if traceback_text:
            print(f"[voxceleb1][task-import-traceback]\n{traceback_text}", flush=True)
    print(f"[voxceleb1] data_root={data_root}", flush=True)
    print(f"[voxceleb1] audio_sources={existing_audio_sources}", flush=True)
    print(f"[voxceleb1] text_file_samples={len(text_files)} audio_file_samples={len(audio_samples)}", flush=True)
    print(f"[voxceleb1] full_audio_scan=false audio_decode=false audio_copy=false", flush=True)
    print(f"[voxceleb1] report={report_path}", flush=True)
    if blocking_issues:
        print(f"[voxceleb1][blocking_issues]={json.dumps(blocking_issues, ensure_ascii=False)}", flush=True)
        raise SystemExit(1)
    print("========== HUGINN X-ARES VOXCELEB1 DATA PATH AUDIT PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
