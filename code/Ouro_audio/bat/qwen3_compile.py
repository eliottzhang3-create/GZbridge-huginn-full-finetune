"""Narrow torch.compile helpers for the Qwen3 BAT compile smoke.

Only ``Qwen3ForCausalLM.model`` is compiled.  This is Qwen3's native
Transformer stack; the multimodal wrapper, Spatial-AST, Q-Former, audio
renderer and ``lm_head`` remain eager.  The fixed BAT sequence length is
therefore the only sequence shape seen by the compiled core.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def find_qwen3_causal_model(model: nn.Module) -> nn.Module:
    """Locate the native Qwen3 causal model under PEFT/DDP wrappers."""

    current: nn.Module = model
    if hasattr(current, "module") and isinstance(getattr(current, "module"), nn.Module):
        current = current.module
    if hasattr(current, "get_base_model"):
        current = current.get_base_model()
    if current.__class__.__name__ == "Qwen3ForCausalLM":
        return current
    for module in current.modules():
        if module.__class__.__name__ == "Qwen3ForCausalLM":
            return module
    raise RuntimeError("Unable to locate Qwen3ForCausalLM after Swift/PEFT wrapping")


def prepare_compile_runtime() -> dict[str, Any]:
    """Disable only PyTorch's inaccessible compiler-repro side path."""

    report: dict[str, Any] = {
        "repro_after_aot_disabled": False,
        "inductor_graph_repro_disabled": False,
        "dynamo_optimize_ddp_disabled": False,
        "compile_threads_env": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
    }
    os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
    os.environ.pop("TORCHDYNAMO_REPRO_AFTER_AOT", None)
    os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "0"

    from torch import _dynamo
    from torch._dynamo.repro import after_aot

    _dynamo.reset()
    dynamo_config = getattr(_dynamo, "config", None)
    if dynamo_config is not None:
        if hasattr(dynamo_config, "optimize_ddp"):
            dynamo_config.optimize_ddp = False
            report["dynamo_optimize_ddp_disabled"] = True
        if hasattr(dynamo_config, "repro_after"):
            dynamo_config.repro_after = None
        if hasattr(dynamo_config, "repro_after_aot"):
            dynamo_config.repro_after_aot = False
        if hasattr(dynamo_config, "repro_level"):
            dynamo_config.repro_level = 0

    after_aot.save_graph_repro = lambda *args, **kwargs: None
    report["repro_after_aot_disabled"] = True
    report["inductor_graph_repro_disabled"] = True
    return report


def compile_qwen3_transformer_core(
    causal: nn.Module,
    *,
    mode: str = "default",
    dynamic: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Compile only ``Qwen3ForCausalLM.model`` with a static shape contract."""

    transformer_core = getattr(causal, "model", None)
    if not isinstance(transformer_core, nn.Module):
        raise RuntimeError("Unable to locate Qwen3ForCausalLM.model Transformer core")
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
        "target": "Qwen3ForCausalLM.model",
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


def dynamo_counter_summary() -> dict[str, int]:
    """Return JSON-safe compile counters."""

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
