#!/usr/bin/env python3
"""X-ARES VoxCeleb1 task adapter for the shared read-only dataset."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


XARES_ROOT = Path(
    os.environ.get(
        "HUGINN_XARES_ROOT",
        "/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares",
    )
).expanduser().resolve()
OFFICIAL_TASK_PATH = XARES_ROOT / "src" / "tasks" / "voxceleb1_task.py"
DATA_ROOT = Path(
    os.environ.get(
        "HUGINN_XARES_VOXCELEB1_ROOT",
        "/hpc_stor03/public/shared/data/mml/VoxCeleb1_origin",
    )
).expanduser().resolve()
WORK_ROOT = Path(
    os.environ.get(
        "HUGINN_XARES_VOXCELEB1_WORK_ROOT",
        "outputs/xares_voxceleb1_knn",
    )
).expanduser().resolve()
CONFIG_REPORT = Path(
    os.environ.get(
        "HUGINN_XARES_VOXCELEB1_CONFIG_REPORT",
        "outputs/xares_voxceleb1_knn/task_config.json",
    )
).expanduser().resolve()


def _load_official_task() -> ModuleType:
    if not OFFICIAL_TASK_PATH.is_file():
        raise FileNotFoundError(f"Official VoxCeleb1 task not found: {OFFICIAL_TASK_PATH}")
    module_name = "xares_official_voxceleb1_task"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, OFFICIAL_TASK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official task: {OFFICIAL_TASK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    return module


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if callable(value):
        return f"<callable:{getattr(value, '__qualname__', type(value).__name__)}>"
    if hasattr(value, "__dict__"):
        return f"<{type(value).__module__}.{type(value).__name__}>"
    return repr(value)


def _set_if_present(config: Any, name: str, value: Any) -> None:
    if not hasattr(config, name):
        raise AttributeError(f"VoxCeleb1 TaskConfig has no expected field: {name}")
    setattr(config, name, value)


def _redirect_encoded_outputs(config: Any) -> None:
    """Keep generated X-ARES embeddings outside the public read-only tree."""
    names = getattr(config, "encoded_tar_name_of_split", None)
    if not isinstance(names, dict):
        raise TypeError(
            "VoxCeleb1 TaskConfig.encoded_tar_name_of_split must be a dict for "
            f"safe cache redirection, got {type(names).__name__}"
        )
    redirected = {}
    for split, raw_name in names.items():
        source_name = Path(str(raw_name)).name
        redirected[split] = str(WORK_ROOT / f"encoded_{split}_{source_name}")
    config.encoded_tar_name_of_split = redirected


def _write_config_report(config: Any) -> None:
    payload = {
        "data_root": str(DATA_ROOT),
        "config": _json_safe(config),
        "config_type": f"{type(config).__module__}.{type(config).__name__}",
    }
    output = CONFIG_REPORT.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def voxceleb1_config(encoder):
    official = _load_official_task()
    config = official.voxceleb1_config(encoder)

    # The public VoxCeleb1 tree is read-only. The task owns its standard split
    # and label protocol; we only override execution/cache controls and the
    # shared data root supplied by the user.
    _set_if_present(config, "env_root", DATA_ROOT)
    _set_if_present(
        config,
        "batch_size_encode",
        int(os.environ.get("HUGINN_XARES_VOXCELEB1_BATCH_SIZE_ENCODE", "1")),
    )
    _set_if_present(
        config,
        "num_encoder_workers",
        int(os.environ.get("HUGINN_XARES_VOXCELEB1_NUM_ENCODER_WORKERS", "0")),
    )
    _set_if_present(
        config,
        "use_mini_dataset",
        os.environ.get("HUGINN_XARES_VOXCELEB1_USE_MINI_DATASET", "0") == "1",
    )
    _set_if_present(
        config,
        "force_download",
        os.environ.get("HUGINN_XARES_VOXCELEB1_FORCE_DOWNLOAD", "0") == "1",
    )
    _set_if_present(
        config,
        "force_encode",
        os.environ.get("HUGINN_XARES_VOXCELEB1_FORCE_ENCODE", "1") == "1",
    )
    _set_if_present(
        config,
        "do_knn",
        os.environ.get("HUGINN_XARES_VOXCELEB1_DO_KNN", "1") == "1",
    )
    _set_if_present(config, "encoder", encoder)
    _redirect_encoded_outputs(config)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _write_config_report(config)

    print(
        f"[xares-voxceleb1-task] data_root={DATA_ROOT} "
        f"mini={config.use_mini_dataset} force_encode={config.force_encode} "
        f"do_knn={config.do_knn} batch_size_encode={config.batch_size_encode} "
        f"num_encoder_workers={config.num_encoder_workers}",
        flush=True,
    )
    return config
