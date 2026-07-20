"""Helpers for importing the original Huginn implementation as a sibling package."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

_BASE_PACKAGE = "huginn_base_original"
_BASE_DIR = pathlib.Path(__file__).resolve().parent.parent / "huginn-0125"

if _BASE_PACKAGE not in sys.modules:
    package = types.ModuleType(_BASE_PACKAGE)
    package.__path__ = [str(_BASE_DIR)]
    sys.modules[_BASE_PACKAGE] = package


def _load_module(module_name: str):
    qualified_name = f"{_BASE_PACKAGE}.{module_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    module_path = _BASE_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load base Huginn module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


RavenConfig = _load_module("raven_config_minimal").RavenConfig
_model_module = _load_module("raven_modeling_minimal")
RavenForCausalLM = _model_module.RavenForCausalLM
CausalLMOutputRecurrentLatents = _model_module.CausalLMOutputRecurrentLatents
