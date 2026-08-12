#!/usr/bin/env python3
"""Run a submitted native Ouro-1.4B load, forward, cache, and generation smoke test."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compat.ouro_cache import patch_ouro_cache


EXPECTED_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "configuration_ouro.py",
    "modeling_ouro.py",
    "tokenizer.json",
    "tokenizer_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("outputs/ouro/native_smoke.json"),
    )
    parser.add_argument(
        "--skip-forward-profile",
        action="store_true",
        help="Skip the additional no-cache four-step forward profile.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def require_files(model_path: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Ouro model snapshot is incomplete: missing={missing} path={model_path}")


def profile_no_cache_forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    device_index: int,
) -> dict[str, Any]:
    """Profile one full four-step forward and account for recurrent reuse.

    ``active_unique_parameter_count`` counts distinct physical parameters
    used by the forward. ``active_parameter_use_count`` counts parameter
    elements on every actual module invocation, so shared Ouro layers are
    counted once per recurrent step. The latter is not extra storage: it is a
    measure of parameter reuse across the four recurrent steps.

    PyTorch profiler FLOPs cover operators for which the installed profiler
    has a FLOP formula. The independent linear count covers every executed
    ``nn.Linear`` projection and is reported separately because custom
    attention operators may not expose FLOPs to the profiler.
    """

    parameter_by_id = {
        id(parameter): (name, int(parameter.numel()))
        for name, parameter in model.named_parameters()
    }
    active_parameter_ids: set[int] = set()
    active_parameter_use_count = 0
    linear_flops = 0
    module_call_counts: dict[str, int] = {}
    recurrent_layer_call_counts: dict[str, int] = {}
    module_name_by_id = {
        id(module): (name or module.__class__.__name__)
        for name, module in model.named_modules()
    }
    layer_name_by_id: dict[int, str] = {}
    hook_handles: list[Any] = []

    def on_parameter_module(module: torch.nn.Module, _args: Any, output: Any) -> None:
        nonlocal active_parameter_use_count, linear_flops
        module_name = module_name_by_id.get(id(module), module.__class__.__name__)
        module_call_counts[module_name] = module_call_counts.get(module_name, 0) + 1
        for parameter in module.parameters(recurse=False):
            parameter_id = id(parameter)
            active_parameter_ids.add(parameter_id)
            active_parameter_use_count += int(parameter.numel())
        if isinstance(module, torch.nn.Linear) and isinstance(output, torch.Tensor):
            # 2 FLOPs per multiply-add; bias addition is intentionally omitted.
            linear_flops += 2 * int(output.numel()) * int(module.in_features)

    def on_recurrent_layer(module: torch.nn.Module, _args: Any, _output: Any) -> None:
        layer_name = layer_name_by_id[id(module)]
        recurrent_layer_call_counts[layer_name] = (
            recurrent_layer_call_counts.get(layer_name, 0) + 1
        )

    for module in model.modules():
        if any(True for _ in module.parameters(recurse=False)):
            hook_handles.append(module.register_forward_hook(on_parameter_module))

    ouro_core = getattr(model, "model", None)
    layers = getattr(ouro_core, "layers", None)
    if layers is not None:
        for index, layer in enumerate(layers):
            layer_name_by_id[id(layer)] = f"layers.{index}"
            hook_handles.append(layer.register_forward_hook(on_recurrent_layer))
    if layers is None or len(layers) == 0:
        raise RuntimeError("Could not locate Ouro physical decoder layers for forward profiling")

    torch.cuda.synchronize(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    baseline_allocated = int(torch.cuda.memory_allocated(device_index))
    profile_started = time.perf_counter()
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            with torch.inference_mode():
                profiled_forward = model(**inputs, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device_index)
    finally:
        for handle in hook_handles:
            handle.remove()
    profile_seconds = time.perf_counter() - profile_started

    expected_steps = int(getattr(getattr(model, "config", None), "total_ut_steps", -1))
    expected_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", -1))
    observed_counts = list(recurrent_layer_call_counts.values())
    if expected_steps > 0 and set(observed_counts) != {expected_steps}:
        raise RuntimeError(
            "Ouro recurrent profile did not execute every physical layer the expected number of times: "
            f"expected_steps={expected_steps} observed_counts={recurrent_layer_call_counts}"
        )
    if expected_layers > 0 and len(recurrent_layer_call_counts) != expected_layers:
        raise RuntimeError(
            "Ouro recurrent profile did not observe all physical layers: "
            f"expected_layers={expected_layers} observed_layers={len(recurrent_layer_call_counts)}"
        )

    profiler_flops = 0
    profiler_events: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        event_flops = int(getattr(event, "flops", 0) or 0)
        profiler_flops += event_flops
        if event_flops:
            profiler_events.append(
                {
                    "name": event.key,
                    "flops": event_flops,
                    "count": int(getattr(event, "count", 0)),
                }
            )
    profiler_events.sort(key=lambda item: item["flops"], reverse=True)

    active_unique_parameter_names = sorted(
        parameter_by_id[parameter_id][0]
        for parameter_id in active_parameter_ids
        if parameter_id in parameter_by_id
    )
    active_unique_parameter_count = sum(
        parameter_by_id[parameter_id][1]
        for parameter_id in active_parameter_ids
        if parameter_id in parameter_by_id
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))

    return {
        "forward_mode": "inference_mode_use_cache_false",
        "logits_shape": list(profiled_forward.logits.shape),
        "profile_seconds": profile_seconds,
        "active_unique_parameter_count": active_unique_parameter_count,
        "active_parameter_use_count": active_parameter_use_count,
        "active_parameter_reuse_multiplier": (
            active_parameter_use_count / active_unique_parameter_count
            if active_unique_parameter_count
            else None
        ),
        "active_unique_parameter_name_count": len(active_unique_parameter_names),
        "active_unique_parameter_name_preview": active_unique_parameter_names[:20],
        "recurrent_layer_call_counts": recurrent_layer_call_counts,
        "recurrent_steps": expected_steps,
        "recurrent_physical_layers": len(recurrent_layer_call_counts),
        "recurrent_layer_call_count_values": sorted(
            set(recurrent_layer_call_counts.values())
        ),
        "linear_projection_flops": linear_flops,
        "profiler_supported_flops": profiler_flops,
        "profiler_top_flop_operators": profiler_events[:20],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        "peak_increment_over_baseline_bytes": peak_allocated - baseline_allocated,
        "interpretation": {
            "active_unique_parameter_count": "distinct physical parameters used; shared recurrent weights counted once",
            "active_parameter_use_count": "parameter elements counted for every actual module invocation; shared recurrent weights counted once per loop",
            "linear_projection_flops": "2 x output_elements x in_features for executed Linear modules, excluding bias additions",
            "profiler_supported_flops": "PyTorch profiler operator FLOPs; unsupported custom operators may be absent",
            "peak_increment_over_baseline_bytes": "additional peak CUDA allocation above the loaded-model baseline; includes activations and workspaces",
        },
    }


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    require_files(model_path)

    if not torch.cuda.is_available():
        raise RuntimeError("Ouro native smoke requires CUDA and must run inside a submitted GPU job")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This smoke test is GPU-only; got device={device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("========== OURO-1.4B NATIVE SMOKE ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[platform] {platform.platform()}", flush=True)
    print(f"[torch] version={torch.__version__} device={torch.cuda.get_device_name(device_index)}", flush=True)
    print(f"[model] path={model_path}", flush=True)
    print(f"[model] expected_revision={EXPECTED_REVISION}", flush=True)
    print(f"[model] local_files_only=True transformers={package_version('transformers')}", flush=True)

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    config_summary = {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "intermediate_size": int(getattr(config, "intermediate_size", -1)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "total_ut_steps": int(getattr(config, "total_ut_steps", -1)),
        "vocab_size": int(getattr(config, "vocab_size", -1)),
    }
    expected_config = {
        "model_type": "ouro",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "total_ut_steps": 4,
        "vocab_size": 49152,
    }
    for key, expected in expected_config.items():
        if config_summary.get(key) != expected:
            raise RuntimeError(f"Unexpected Ouro config field {key}: expected={expected} actual={config_summary.get(key)}")
    print(f"[config] {json.dumps(config_summary, ensure_ascii=False)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[1])
    print(f"[tokenizer] prompt_tokens={prompt_length}", flush=True)

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    print(f"[load] class={model.__class__.__module__}.{model.__class__.__name__} seconds={load_seconds:.3f}", flush=True)

    cache_patch = patch_ouro_cache(model)
    print(f"[cache] patch={json.dumps(cache_patch, ensure_ascii=False)}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count < 1_000_000_000 or parameter_count > 2_000_000_000:
        raise RuntimeError(f"Unexpected Ouro-1.4B parameter count: {parameter_count}")
    print(f"[load] parameters={parameter_count}", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    with torch.inference_mode():
        forward = model(**inputs, use_cache=True, logits_to_keep=1)
    cache = getattr(forward, "past_key_values", None)
    if cache is None or not hasattr(cache, "get_seq_length"):
        raise RuntimeError(f"Native Ouro forward did not return a compatible cache: {type(cache)}")
    cache_length = int(cache.get_seq_length())
    if cache_length != prompt_length:
        raise RuntimeError(f"Unexpected native cache length: expected={prompt_length} actual={cache_length}")
    print(f"[forward] logits_shape={tuple(forward.logits.shape)} cache_length={cache_length}", flush=True)

    if args.skip_forward_profile:
        forward_profile = {"skipped": True}
        print("[profile] skipped", flush=True)
    else:
        forward_profile = profile_no_cache_forward(model, inputs, device_index)
        print(
            "[profile] "
            f"active_unique_params={forward_profile['active_unique_parameter_count']} "
            f"parameter_use_count={forward_profile['active_parameter_use_count']} "
            f"layer_calls={forward_profile['recurrent_layer_call_count_values']} "
            f"profiler_flops={forward_profile['profiler_supported_flops']} "
            f"linear_flops={forward_profile['linear_projection_flops']} "
            f"peak_increment_bytes={forward_profile['peak_increment_over_baseline_bytes']}",
            flush=True,
        )

    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generation_started
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    generated_length = int(generated.shape[1])
    if generated_length <= prompt_length:
        raise RuntimeError(f"Generation produced no new tokens: prompt={prompt_length} output={generated_length}")
    print(f"[generate] output_tokens={generated_length} seconds={generation_seconds:.3f}", flush=True)
    print(f"[generate] text={generated_text!r}", flush=True)

    report = {
        "status": "ok",
        "model_path": str(model_path),
        "expected_revision": EXPECTED_REVISION,
        "packages": {
            "transformers": package_version("transformers"),
            "torch": package_version("torch"),
        },
        "device": {
            "name": torch.cuda.get_device_name(device_index),
            "index": device_index,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
        },
        "config": config_summary,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "cache_patch": cache_patch,
        "parameter_count": parameter_count,
        "prompt": prompt,
        "prompt_length": prompt_length,
        "forward_logits_shape": list(forward.logits.shape),
        "cache_length": cache_length,
        "generated_length": generated_length,
        "generated_text": generated_text,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "forward_profile": forward_profile,
    }
    atomic_write_json(args.output_report, report)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
