#!/usr/bin/env python3
"""Read-only Stage 1 preflight for the remote X-ARES source and environment.

This audit does not download data, decode audio, load a checkpoint, or modify
the X-ARES tree.  It only checks source layout, imports, CLI help, installed
package versions, CUDA visibility, and declared dependency risks.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "pyproject.toml",
    "src/xares/run.py",
    "src/xares/task.py",
    "src/xares/trainer.py",
    "src/xares/audiowebdataset.py",
)
REQUIRED_DIRS = ("src/xares", "src/tasks")
CORE_IMPORTS = (
    "xares",
    "xares.run",
    "xares.task",
    "xares.trainer",
    "xares.audiowebdataset",
)
TASK_FILES = (
    "voxceleb1_task.py",
    "gtzan_task.py",
    "nsynth_instument_task.py",
)
PACKAGE_IMPORTS = (
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "peft",
    "jiwer",
    "soundfile",
    "webdataset",
    "numpy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xares-root",
        type=Path,
        default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares"),
    )
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def version_of(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_report(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "module": module_name,
            "ok": True,
            "file": str(getattr(module, "__file__", "")),
            "version": str(getattr(module, "__version__", "")),
        }
    except Exception as exc:  # noqa: BLE001 - report every environment failure
        return {
            "module": module_name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }


def import_task_file(path: Path) -> dict[str, Any]:
    module_name = f"xares_task_preflight_{path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            "file": str(path),
            "ok": True,
            "module_name": module_name,
            "symbols": sorted(name for name in dir(module) if not name.startswith("_"))[:40],
        }
    except Exception as exc:  # noqa: BLE001 - report every task failure
        return {
            "file": str(path),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }


def run_cli_help(xares_root: Path, env: dict[str, str]) -> dict[str, Any]:
    child_env = dict(env)
    source_path = str(xares_root / "src")
    old_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = source_path + (os.pathsep + old_pythonpath if old_pythonpath else "")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "xares.run", "--help"],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout_preview": completed.stdout[:4000],
            "stderr_preview": completed.stderr[:4000],
        }
    except Exception as exc:  # noqa: BLE001 - report subprocess failure
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }


def declared_torch_constraint(pyproject_text: str) -> str | None:
    for line in pyproject_text.splitlines():
        if "torch" in line.lower() and any(symbol in line for symbol in (">", "<", "=", "~")):
            return line.strip()
    return None


def main() -> None:
    args = parse_args()
    xares_root = args.xares_root.expanduser().resolve()
    report_path = args.output_report.expanduser().resolve()
    if not xares_root.is_dir():
        raise FileNotFoundError(f"X-ARES root does not exist: {xares_root}")

    missing_files = [str(xares_root / path) for path in REQUIRED_FILES if not (xares_root / path).is_file()]
    missing_dirs = [str(xares_root / path) for path in REQUIRED_DIRS if not (xares_root / path).is_dir()]
    if missing_files or missing_dirs:
        raise FileNotFoundError(f"Incomplete X-ARES source: missing_files={missing_files} missing_dirs={missing_dirs}")

    source_path = str(xares_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    package_versions = {
        distribution: version_of(distribution)
        for distribution in PACKAGE_IMPORTS
    }
    package_import_reports = [import_report(name) for name in PACKAGE_IMPORTS]
    core_import_reports = [import_report(name) for name in CORE_IMPORTS]
    task_import_reports = [
        import_task_file(xares_root / "src" / "tasks" / filename)
        for filename in TASK_FILES
    ]

    pyproject_path = xares_root / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    pyproject_summary = {
        "path": str(pyproject_path),
        "bytes": pyproject_path.stat().st_size,
        "declared_torch_constraint": declared_torch_constraint(pyproject_text),
        "has_project_table": "[project]" in pyproject_text,
        "has_build_system_table": "[build-system]" in pyproject_text,
    }

    cuda = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    torch_report = next((item for item in package_import_reports if item["module"] == "torch"), None)
    if torch_report and torch_report["ok"]:
        try:
            torch = importlib.import_module("torch")
            cuda["cuda_available"] = bool(torch.cuda.is_available())
            cuda["device_count"] = int(torch.cuda.device_count())
            cuda["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
        except Exception as exc:  # noqa: BLE001 - include CUDA diagnostic
            cuda["error"] = f"{type(exc).__name__}: {exc}"

    cli_help = run_cli_help(xares_root, os.environ.copy())
    import_blockers = [item for item in core_import_reports if not item["ok"]]
    task_blockers = [item for item in task_import_reports if not item["ok"]]
    package_blockers = [item for item in package_import_reports if not item["ok"]]
    warnings = []
    torch_version = package_versions.get("torch")
    constraint = pyproject_summary["declared_torch_constraint"]
    if torch_version and constraint and "<2.9" in constraint:
        try:
            major, minor = (int(value) for value in torch_version.split(".")[:2])
            if (major, minor) >= (2, 9):
                warnings.append(
                    f"Installed torch={torch_version} is newer than the declared X-ARES constraint {constraint}; "
                    "do not modify swift_huginn until a compatibility smoke passes."
                )
        except ValueError:
            warnings.append(f"Could not parse installed torch version: {torch_version}")
    if not cuda["cuda_available"]:
        warnings.append("CUDA is unavailable; this is acceptable for import-only preflight but not for encoder smoke.")
    if not shutil.which("uv"):
        warnings.append("uv executable is not present; source-mode execution is still possible if imports pass.")

    report = {
        "gate": "huginn_xares_import_environment_preflight_v1",
        "validation_passed": not import_blockers and not task_blockers and not package_blockers and cli_help["ok"],
        "network_or_data_access": "not_attempted",
        "audio_decode": False,
        "checkpoint_load": False,
        "xares_root": str(xares_root),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "package_versions": package_versions,
        "package_imports": package_import_reports,
        "core_imports": core_import_reports,
        "task_imports": task_import_reports,
        "pyproject": pyproject_summary,
        "cli_help": cli_help,
        "cuda": cuda,
        "warnings": warnings,
        "blocking_issues": {
            "package_imports": package_blockers,
            "core_imports": import_blockers,
            "task_imports": task_blockers,
            "cli_help": [] if cli_help["ok"] else [cli_help],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[xares-env] root={xares_root}", flush=True)
    print(f"[xares-env] torch={package_versions.get('torch')} torchaudio={package_versions.get('torchaudio')}", flush=True)
    print(f"[xares-env] core_imports={sum(item['ok'] for item in core_import_reports)}/{len(core_import_reports)}", flush=True)
    print(f"[xares-env] task_imports={sum(item['ok'] for item in task_import_reports)}/{len(task_import_reports)}", flush=True)
    print(f"[xares-env] cli_help={cli_help['ok']} cuda={cuda['cuda_available']} devices={cuda['device_count']}", flush=True)
    for warning in warnings:
        print(f"[xares-env][warning] {warning}", flush=True)
    print(f"[xares-env] report={report_path}", flush=True)
    if any(report["blocking_issues"].values()):
        print(f"[xares-env][blocking_issues]={json.dumps(report['blocking_issues'], ensure_ascii=False)}", flush=True)
        raise SystemExit(1)
    print("========== HUGINN X-ARES IMPORT/ENVIRONMENT PREFLIGHT PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
