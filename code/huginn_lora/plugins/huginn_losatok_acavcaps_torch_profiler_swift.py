"""Combined external plugin for the ACAVCAPS Dynamic-90s profiler smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent


def _load(path: Path, name: str) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load profiling external plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


_load(PLUGIN_DIR / "huginn_losatok_acavcaps_wds_swift.py", "huginn_losatok_acavcaps_wds_profile_dataset")
_load(PLUGIN_DIR / "huginn_losatok_torch_profiler.py", "huginn_losatok_profile_hooks")
