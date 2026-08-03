"""Evaluation-only adapter for the current Huginn Whisper dynamic-30s route.

This module delegates model, processor, audio loading, and prompt behavior to
the current dynamic Whisper plugin.  It freezes the reconstructed Whisper
encoder only for evaluation, because the training route intentionally
unfreezes it.  No training registration or training behavior is changed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


_BASE_PATH = Path(__file__).with_name("huginn_audio_whisper_dynamic90s_swift.py")
_BASE_MODULE_NAME = "huginn_audio_whisper_dynamic30s_eval_base"


def _load_base() -> ModuleType:
    existing = sys.modules.get(_BASE_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dynamic Whisper base plugin: {_BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BASE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(_BASE_MODULE_NAME) is module:
            sys.modules.pop(_BASE_MODULE_NAME, None)
        raise
    return module


_BASE = _load_base()


def build_huginn_audio_model(model_dir: str) -> Any:
    model = _BASE.build_huginn_audio_model(model_dir)
    if not hasattr(model, "audio_encoder"):
        raise AttributeError("Evaluation model has no audio_encoder")
    model.audio_encoder.requires_grad_(False)
    print(
        "[HuginnWhisperDynamic30sEval] evaluation-only Whisper freeze applied",
        flush=True,
    )
    return model


def __getattr__(name: str) -> Any:
    return getattr(_BASE, name)

