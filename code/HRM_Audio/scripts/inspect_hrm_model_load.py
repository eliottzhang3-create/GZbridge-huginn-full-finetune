#!/usr/bin/env python3
"""Load the official local HRM-Text checkpoint without running generation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text"
EXPECTED_WEIGHT_BYTES = 2_365_606_568
EXPECTED_WEIGHT_SHA256 = "f8fe2b2bf6948414e8e8d6538659198726d98f967c55b533b7aabe8a1fa9a584"
EXPECTED_PARAMETER_COUNT = 1_182_795_264
REQUIRED_FILES = (
    "config.json",
    "LICENSE",
    "model.safetensors",
    "README.md",
    "tokenizer_config.json",
    "tokenizer.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_parameters(model: torch.nn.Module) -> tuple[int, Counter[str], Counter[str]]:
    total = 0
    dtype_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        dtype_counts[str(parameter.dtype)] += count
        device_counts[str(parameter.device)] += count
    return total, dtype_counts, device_counts


def loading_info_report(loading_info: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        value = loading_info.get(key, [])
        result[key] = [str(item) for item in value]
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("HRM-Text model loading inspect requires CUDA")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path.resolve()
    missing_files = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Incomplete HRM-Text snapshot at {model_path}: missing={missing_files}")

    weight_path = model_path / "model.safetensors"
    weight_bytes = weight_path.stat().st_size
    if weight_bytes != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Unexpected model.safetensors size: expected={EXPECTED_WEIGHT_BYTES}, actual={weight_bytes}"
        )

    print("========== HRM-TEXT MODEL LOAD INSPECT ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[path] model={model_path}", flush=True)
    print(f"[weight] bytes={weight_bytes}", flush=True)

    weight_sha256 = None
    if not args.skip_sha256:
        print("[stage] sha256=model.safetensors", flush=True)
        weight_sha256 = sha256sum(weight_path)
        print(f"[weight] sha256={weight_sha256}", flush=True)
        if weight_sha256 != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError(
                f"Unexpected model.safetensors SHA256: expected={EXPECTED_WEIGHT_SHA256}, actual={weight_sha256}"
            )

    raw_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "model_type": "hrm_text",
        "architectures": ["HrmTextForCausalLM"],
        "hidden_size": 1536,
        "num_hidden_layers": 32,
        "H_cycles": 2,
        "L_cycles": 3,
        "prefix_lm": True,
        "vocab_size": 65536,
    }
    config_mismatches = {
        key: {"expected": expected, "actual": raw_config.get(key)}
        for key, expected in expected_config.items()
        if raw_config.get(key) != expected
    }
    if config_mismatches:
        raise RuntimeError(f"Unexpected HRM-Text config values: {config_mismatches}")

    print("[stage] load=AutoConfig", flush=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.__class__.__name__ != "HrmTextConfig":
        raise RuntimeError(f"Expected HrmTextConfig, got {config.__class__.__module__}.{config.__class__.__name__}")

    print("[stage] load=AutoTokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if len(tokenizer) != config.vocab_size:
        raise RuntimeError(f"Tokenizer/config vocabulary mismatch: tokenizer={len(tokenizer)}, config={config.vocab_size}")

    device = torch.device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print("[stage] load=AutoModelForCausalLM dtype=bfloat16 attention=sdpa", flush=True)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
        output_loading_info=True,
    )
    model.eval()
    torch.cuda.synchronize(device)

    normalized_loading_info = loading_info_report(loading_info)
    nonempty_loading_info = {key: value for key, value in normalized_loading_info.items() if value}
    if nonempty_loading_info:
        raise RuntimeError(f"Checkpoint loading mismatch: {nonempty_loading_info}")
    if model.__class__.__name__ != "HrmTextForCausalLM":
        raise RuntimeError(
            f"Expected HrmTextForCausalLM, got {model.__class__.__module__}.{model.__class__.__name__}"
        )
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    if attention_implementation != "sdpa":
        raise RuntimeError(f"Expected SDPA attention, got {attention_implementation!r}")

    parameter_count, dtype_counts, device_counts = count_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Unexpected parameter count: expected={EXPECTED_PARAMETER_COUNT}, actual={parameter_count}"
        )
    if set(dtype_counts) != {"torch.bfloat16"}:
        raise RuntimeError(f"Expected all parameters in BF16, got dtype_counts={dict(dtype_counts)}")
    if set(device_counts) != {str(device)}:
        raise RuntimeError(f"Expected all parameters on {device}, got device_counts={dict(device_counts)}")

    input_embeddings = model.get_input_embeddings()
    expected_embedding_shape = (config.vocab_size, config.hidden_size)
    if tuple(input_embeddings.weight.shape) != expected_embedding_shape:
        raise RuntimeError(
            f"Unexpected input embedding shape: expected={expected_embedding_shape}, "
            f"actual={tuple(input_embeddings.weight.shape)}"
        )

    forward_signature = inspect.signature(model.forward)
    forward_parameters = set(forward_signature.parameters)
    required_forward_parameters = {"input_ids", "inputs_embeds", "token_type_ids"}
    missing_forward_parameters = sorted(required_forward_parameters - forward_parameters)
    if missing_forward_parameters:
        raise RuntimeError(f"HRM forward lacks required inputs: {missing_forward_parameters}; {forward_signature}")
    if not model.can_generate():
        raise RuntimeError("Loaded HRM-Text model reports can_generate() == False")

    report = {
        "status": "ok",
        "model_path": str(model_path),
        "snapshot": {
            "required_files": list(REQUIRED_FILES),
            "weight_bytes": weight_bytes,
            "weight_sha256": weight_sha256,
        },
        "config": {
            "class": f"{config.__class__.__module__}.{config.__class__.__name__}",
            **{key: getattr(config, key) for key in expected_config},
            "attn_implementation": attention_implementation,
        },
        "tokenizer": {
            "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
            "length": len(tokenizer),
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "special_tokens_map": tokenizer.special_tokens_map,
        },
        "model": {
            "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "parameter_count": parameter_count,
            "parameter_dtype_counts": dict(dtype_counts),
            "parameter_device_counts": dict(device_counts),
            "input_embedding_shape": list(input_embeddings.weight.shape),
            "forward_signature": str(forward_signature),
            "supports_inputs_embeds": "inputs_embeds" in forward_parameters,
            "supports_token_type_ids": "token_type_ids" in forward_parameters,
            "can_generate": model.can_generate(),
            "hf_device_map": getattr(model, "hf_device_map", None),
            "loading_info": normalized_loading_info,
        },
        "cuda_memory": {
            "allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        },
    }
    atomic_write_json(args.output_report, report)
    print(
        f"[model] class={report['model']['class']} parameters={parameter_count} "
        f"dtype_counts={dict(dtype_counts)} device_counts={dict(device_counts)}",
        flush=True,
    )
    print(f"[forward] {forward_signature}", flush=True)
    print(f"[memory] {json.dumps(report['cuda_memory'], ensure_ascii=False)}", flush=True)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
