"""Combined external plugin for real-data Whisper dynamic-90s profiling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent


def _load(path: Path, name: str) -> None:
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load profiling external plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise


_load(
    PLUGIN_DIR / "huginn_audio_whisper_dynamic90s_mixture_swift.py",
    "huginn_audio_whisper_dynamic90s_profile_dataset",
)
_load(
    PLUGIN_DIR / "huginn_whisper_dynamic90s_torch_profiler.py",
    "huginn_audio_whisper_dynamic90s_profile_hooks",
)
