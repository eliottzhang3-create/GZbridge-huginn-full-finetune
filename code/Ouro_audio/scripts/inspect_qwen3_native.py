#!/usr/bin/env python3
"""Audit native Qwen3-4B-Base loading, forward, cache, and generation."""

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


EXPECTED_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
}

REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="The future of artificial intelligence is")
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("outputs/ouro/qwen3/native_smoke.json"),
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def require_files(model_path: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Qwen3-4B-Base snapshot is incomplete: missing={missing} path={model_path}"
        )


def config_summary(config: Any) -> dict[str, Any]:
    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", None) or []),
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "intermediate_size": int(getattr(config, "intermediate_size", -1)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", -1)),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
        "vocab_size": int(getattr(config, "vocab_size", -1)),
        "max_position_embeddings": int(
            getattr(config, "max_position_embeddings", -1)
        ),
        "use_cache_default": bool(getattr(config, "use_cache", True)),
        "torch_dtype": str(getattr(config, "torch_dtype", None)),
    }


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    require_files(model_path)

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen3 native audit requires CUDA and must run inside a submitted GPU job"
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This audit is GPU-only; got device={device}")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    torch.cuda.set_device(device_index)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("========== QWEN3-4B-BASE NATIVE AUDIT ==========", flush=True)
    print(
        f"[python] version={sys.version.split()[0]} executable={sys.executable}",
        flush=True,
    )
    print(f"[platform] {platform.platform()}", flush=True)
    print(
        f"[torch] version={torch.__version__} device="
        f"{torch.cuda.get_device_name(device_index)}",
        flush=True,
    )
    print(f"[model] path={model_path}", flush=True)
    print("[model] local_files_only=True trust_remote_code=False", flush=True)
    print(
        f"[packages] transformers={package_version('transformers')} "
        f"torch={package_version('torch')}",
        flush=True,
    )

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    summary = config_summary(config)
    for key, expected in EXPECTED_CONFIG.items():
        actual = summary.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Unexpected Qwen3 config field {key}: "
                f"expected={expected!r} actual={actual!r}"
            )
    print(f"[config] {json.dumps(summary, ensure_ascii=False)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(args.prompt.strip(), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(inputs["input_ids"].shape[1])
    print(
        f"[tokenizer] class={tokenizer.__class__.__name__} "
        f"prompt_tokens={prompt_length} vocab={len(tokenizer)} "
        f"bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id} "
        f"pad={tokenizer.pad_token_id}",
        flush=True,
    )

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_started
    model_class = f"{model.__class__.__module__}.{model.__class__.__name__}"
    print(f"[load] class={model_class} seconds={load_seconds:.3f}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 3_500_000_000 <= parameter_count <= 4_500_000_000:
        raise RuntimeError(f"Unexpected Qwen3-4B parameter count: {parameter_count}")
    dtype_set = sorted({str(parameter.dtype) for parameter in model.parameters()})
    print(f"[load] parameters={parameter_count} dtypes={dtype_set}", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    with torch.inference_mode():
        no_cache = model(**inputs, use_cache=False)
    torch.cuda.synchronize(device_index)
    no_cache_shape = list(no_cache.logits.shape)
    expected_logits_shape = [1, prompt_length, int(config.vocab_size)]
    if no_cache_shape != expected_logits_shape:
        raise RuntimeError(
            f"Unexpected no-cache logits shape: "
            f"expected={expected_logits_shape} actual={no_cache_shape}"
        )
    if not bool(torch.isfinite(no_cache.logits).all().item()):
        raise RuntimeError("No-cache forward logits contain NaN or Inf")
    print(f"[forward] use_cache=False logits_shape={no_cache_shape}", flush=True)

    with torch.inference_mode():
        cached = model(**inputs, use_cache=True)
    cache = getattr(cached, "past_key_values", None)
    cache_length = None
    if cache is not None and hasattr(cache, "get_seq_length"):
        cache_length = int(cache.get_seq_length())
        if cache_length != prompt_length:
            raise RuntimeError(
                f"Unexpected Qwen cache length: expected={prompt_length} "
                f"actual={cache_length}"
            )
    print(
        f"[cache] class={type(cache).__name__ if cache is not None else None} "
        f"sequence_length={cache_length}",
        flush=True,
    )

    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generation_started
    generated_ids = generated[0, prompt_length:]
    if generated_ids.numel() == 0:
        raise RuntimeError("Qwen generation produced no new tokens")
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(
        f"[generate] new_tokens={generated_ids.numel()} "
        f"seconds={generation_seconds:.3f}",
        flush=True,
    )
    print(f"[generate] continuation={continuation!r}", flush=True)

    report = {
        "status": "ok",
        "model_path": str(model_path),
        "model_class": model_class,
        "packages": {
            "transformers": package_version("transformers"),
            "torch": package_version("torch"),
        },
        "device": {
            "name": torch.cuda.get_device_name(device_index),
            "index": device_index,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        },
        "config": summary,
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "parameter_count": parameter_count,
        "parameter_dtypes": dtype_set,
        "prompt": args.prompt.strip(),
        "prompt_length": prompt_length,
        "no_cache_logits_shape": no_cache_shape,
        "cache": {
            "class": type(cache).__name__ if cache is not None else None,
            "sequence_length": cache_length,
        },
        "generated_token_count": int(generated_ids.numel()),
        "generated_text": generated_text,
        "continuation": continuation,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
    }
    atomic_write_json(args.output_report, report)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
