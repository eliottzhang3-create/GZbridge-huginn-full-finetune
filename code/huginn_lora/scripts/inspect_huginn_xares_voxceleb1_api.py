#!/usr/bin/env python3
"""Read-only runtime contract audit for the remote X-ARES VoxCeleb1 API.

The X-ARES source tree is intentionally kept on the remote server. This audit
inspects the actual remote package at execution time instead of guessing its
task, encoder-checker, or K-NN signatures locally.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_XARES_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares")
DEFAULT_TASK = "voxceleb1_task.py"
MODULES = (
    "xares",
    "xares.run",
    "xares.task",
    "xares.trainer",
    "xares.audio_encoder_checker",
    "xares.models.retrival",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xares-root", type=Path, default=DEFAULT_XARES_ROOT)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def describe_module(module: ModuleType) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    functions: dict[str, Any] = {}
    task_configs: dict[str, Any] = {}
    for name, value in sorted(vars(module).items()):
        if name.startswith("_"):
            continue
        if inspect.isclass(value) and getattr(value, "__module__", None) == module.__name__:
            classes[name] = {
                "qualname": getattr(value, "__qualname__", name),
                "signature": safe_signature(value),
                "methods": {
                    method_name: safe_signature(method)
                    for method_name, method in sorted(vars(value).items())
                    if not method_name.startswith("_")
                    and callable(method)
                    and method_name in {
                        "__call__",
                        "forward",
                        "run",
                        "evaluate",
                        "build",
                        "load",
                        "train",
                        "predict",
                        "encode",
                    }
                },
            }
        elif inspect.isfunction(value) and getattr(value, "__module__", None) == module.__name__:
            functions[name] = safe_signature(value)
        elif value.__class__.__name__ == "TaskConfig":
            try:
                task_configs[name] = dataclasses.asdict(value)
            except Exception:
                task_configs[name] = repr(value)
    return {
        "file": str(getattr(module, "__file__", "")),
        "classes": classes,
        "functions": functions,
        "task_config_instances": task_configs,
        "public_symbols": [name for name in sorted(vars(module)) if not name.startswith("_")],
    }


def import_file(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def run_cli_help(xares_root: Path) -> dict[str, Any]:
    child_env = dict(os.environ)
    source = str(xares_root / "src")
    child_env["PYTHONPATH"] = source + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, "-m", "xares.run", "--help"],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": completed.stdout[:12000],
        "stderr": completed.stderr[:12000],
    }


def main() -> None:
    args = parse_args()
    xares_root = args.xares_root.expanduser().resolve()
    source_root = xares_root / "src"
    task_path = source_root / "tasks" / DEFAULT_TASK
    if not source_root.is_dir():
        raise FileNotFoundError(f"X-ARES source directory does not exist: {source_root}")
    if not task_path.is_file():
        raise FileNotFoundError(f"VoxCeleb1 task does not exist: {task_path}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    report: dict[str, Any] = {
        "gate": "huginn_xares_voxceleb1_api_contract_v1",
        "validation_passed": False,
        "xares_root": str(xares_root),
        "task_path": str(task_path),
        "modules": {},
        "task_module": {},
        "cli_help": {},
        "blocking_issues": [],
    }
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            report["modules"][module_name] = {
                "ok": True,
                **describe_module(module),
            }
        except Exception as exc:
            report["modules"][module_name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            report["blocking_issues"].append(f"module_import_failed:{module_name}")

    try:
        task_module = import_file(task_path, "xares_voxceleb1_task_contract")
        report["task_module"] = {"ok": True, **describe_module(task_module)}
    except Exception as exc:
        report["task_module"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
        report["blocking_issues"].append("voxceleb1_task_import_failed")

    try:
        report["cli_help"] = run_cli_help(xares_root)
        if not report["cli_help"]["ok"]:
            report["blocking_issues"].append("xares_run_help_failed")
    except Exception as exc:
        report["cli_help"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
        report["blocking_issues"].append("xares_run_help_failed")

    report["validation_passed"] = not report["blocking_issues"]
    output = args.output_report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("========== HUGINN X-ARES VOXCELEB1 API CONTRACT START ==========")
    print(f"[xares-api] root={xares_root}")
    print(f"[xares-api] task={task_path}")
    for module_name, payload in report["modules"].items():
        print(f"[xares-api] module={module_name} ok={payload.get('ok')}")
    print(f"[xares-api] task_import={report['task_module'].get('ok')}")
    print(f"[xares-api] cli_help={report['cli_help'].get('ok')}")
    print(f"[xares-api] report={output}")
    if report["blocking_issues"]:
        print(f"[xares-api][blocking_issues]={json.dumps(report['blocking_issues'])}")
        raise SystemExit(1)
    print("========== HUGINN X-ARES VOXCELEB1 API CONTRACT PASSED ==========")


if __name__ == "__main__":
    main()

