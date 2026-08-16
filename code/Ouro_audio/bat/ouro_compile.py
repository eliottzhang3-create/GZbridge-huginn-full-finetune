"""Shared, narrowly scoped ``torch.compile`` setup for BAT/Ouro training.

Only ``OuroForCausalLM.model`` is compiled.  The recurrent Transformer core
contains Ouro's four universal-transformer steps; the multimodal wrapper,
Spatial-AST, Q-Former, audio renderer, and LM head stay eager.  This module is
kept separate from the profiler so the production trainer and its smoke use
exactly the same compiler configuration.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def find_ouro_causal_model(model: nn.Module) -> nn.Module:
    """Locate the native ``OuroForCausalLM`` under PEFT/DDP wrappers."""

    current: nn.Module = model
    if hasattr(current, "module") and isinstance(getattr(current, "module"), nn.Module):
        current = current.module
    if hasattr(current, "get_base_model"):
        current = current.get_base_model()
    if current.__class__.__name__ == "OuroForCausalLM":
        return current
    for module in current.modules():
        if module.__class__.__name__ == "OuroForCausalLM":
            return module
    raise RuntimeError("Unable to locate OuroForCausalLM after Swift/PEFT wrapping")


def prepare_compile_runtime() -> dict[str, Any]:
    """Apply the cluster-safe compiler settings used by validated DDP smoke."""

    report: dict[str, Any] = {
        "repro_after_aot_disabled": False,
        "inductor_graph_repro_disabled": False,
        "dynamo_optimize_ddp_disabled": False,
    }

    # PyTorch 2.11 can enter a compiler-repro path after a real Inductor
    # error and execute ``nvcc --version``.  The cluster image's nvcc is not
    # executable for this user, so disable only that diagnostic side effect;
    # real compilation errors still propagate.
    os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
    os.environ.pop("TORCHDYNAMO_REPRO_AFTER_AOT", None)
    os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "0"

    from torch import _dynamo
    from torch._dynamo.repro import after_aot

    _dynamo.reset()
    dynamo_config = getattr(_dynamo, "config", None)
    if dynamo_config is not None:
        # Keep ordinary DDP reduction.  Dynamo's separate DDP optimizer is
        # incompatible with Ouro's auxiliary recurrent return structure.
        if hasattr(dynamo_config, "optimize_ddp"):
            dynamo_config.optimize_ddp = False
            report["dynamo_optimize_ddp_disabled"] = True
        if hasattr(dynamo_config, "repro_after"):
            dynamo_config.repro_after = None
        if hasattr(dynamo_config, "repro_after_aot"):
            dynamo_config.repro_after_aot = False
        if hasattr(dynamo_config, "repro_level"):
            dynamo_config.repro_level = 0

    # See the profiler's validated workaround: this prevents a failed
    # compile from being masked by the inaccessible nvcc probe.
    after_aot.save_graph_repro = lambda *args, **kwargs: None
    report["repro_after_aot_disabled"] = True
    report["inductor_graph_repro_disabled"] = True
    return report


def compile_ouro_transformer_core(
    causal: nn.Module,
    *,
    mode: str = "reduce-overhead",
    dynamic: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Compile only the native Ouro recurrent Transformer core."""

    transformer_core = getattr(causal, "model", None)
    if not isinstance(transformer_core, nn.Module):
        raise RuntimeError("Unable to locate OuroForCausalLM.model Transformer core")
    original_class = type(transformer_core).__name__
    compiled_core = torch.compile(
        transformer_core,
        backend="inductor",
        mode=mode,
        dynamic=dynamic,
        fullgraph=False,
    )
    causal.model = compiled_core
    return compiled_core, {
        "target": "OuroForCausalLM.model",
        "target_class_before_compile": original_class,
        "target_class_after_compile": type(compiled_core).__name__,
        "outer_multimodal_model_compiled": False,
        "spatial_ast_compiled": False,
        "qformer_compiled": False,
        "audio_renderer_compiled": False,
        "lm_head_compiled": False,
        "mode": mode,
        "dynamic": bool(dynamic),
    }


def dynamo_counter_summary() -> dict[str, Any]:
    """Return stable JSON-safe compile counters for a training audit."""

    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {"unique_graphs": 0, "graph_break_count": 0, "calls_captured": 0}
    stats = counters.get("stats", {})
    graph_breaks = counters.get("graph_break", {})
    return {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "graph_break_count": int(sum(graph_breaks.values())) if hasattr(graph_breaks, "values") else 0,
        "calls_captured": int(stats.get("calls_captured", 0)),
    }

