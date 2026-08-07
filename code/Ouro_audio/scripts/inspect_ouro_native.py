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
        "parameter_count": parameter_count,
        "prompt": prompt,
        "prompt_length": prompt_length,
        "forward_logits_shape": list(forward.logits.shape),
        "cache_length": cache_length,
        "generated_length": generated_length,
        "generated_text": generated_text,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
    }
    atomic_write_json(args.output_report, report)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
