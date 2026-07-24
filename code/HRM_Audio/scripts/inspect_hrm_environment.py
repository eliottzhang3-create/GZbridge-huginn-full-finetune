#!/usr/bin/env python3
"""Validate the pinned HRM-Text software and CUDA runtime before loading weights."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


EXPECTED_PACKAGE_VERSIONS = {
    "ms-swift": "4.4.2",
    "transformers": "5.9.0",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "torchaudio": "2.11.0+cu128",
    "accelerate": "1.13.0",
    "peft": "0.18.1",
    "trl": "0.29.1",
    "datasets": "3.6.0",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "huggingface-hub": "1.24.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    missing: list[str] = []
    mismatched: dict[str, dict[str, str]] = {}
    for package_name, expected in EXPECTED_PACKAGE_VERSIONS.items():
        try:
            actual = version(package_name)
        except PackageNotFoundError:
            missing.append(package_name)
            continue
        installed[package_name] = actual
        if actual != expected:
            mismatched[package_name] = {"expected": expected, "actual": actual}
    if missing or mismatched:
        raise RuntimeError(f"Pinned package check failed: missing={missing}, mismatched={mismatched}")
    return installed


def pip_check() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"pip check failed with code {completed.returncode}: {output}")
    return output


def main() -> None:
    args = parse_args()

    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import HrmTextConfig, HrmTextForCausalLM

    try:
        import swift
    except Exception as exc:
        raise RuntimeError(f"Unable to import ms-swift: {type(exc).__name__}: {exc}") from exc

    print("========== HRM-TEXT ENVIRONMENT INSPECT ==========", flush=True)
    installed = package_versions()
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[platform] {platform.platform()}", flush=True)
    print(f"[packages] {json.dumps(installed, ensure_ascii=False)}", flush=True)

    dependency_result = pip_check()
    print(f"[pip-check] {dependency_result}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    if not args.device.startswith("cuda"):
        raise ValueError(f"This inspect expects a CUDA device, got {args.device!r}")

    device = torch.device(args.device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    bf16_supported = torch.cuda.is_bf16_supported()
    if not bf16_supported:
        raise RuntimeError("The selected CUDA runtime does not report BF16 support")
    if not hasattr(functional, "scaled_dot_product_attention"):
        raise RuntimeError("PyTorch scaled_dot_product_attention is unavailable")

    print(
        f"[cuda] torch_cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} "
        f"device_count={torch.cuda.device_count()} device={device} name={properties.name!r} "
        f"capability={properties.major}.{properties.minor} total_gib={properties.total_memory / (1024**3):.3f} "
        f"bf16_supported={bf16_supported}",
        flush=True,
    )
    print(
        f"[hrm-import] config={HrmTextConfig.__module__}.{HrmTextConfig.__name__} "
        f"model={HrmTextForCausalLM.__module__}.{HrmTextForCausalLM.__name__}",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats(device_index)
    left = torch.arange(256, device=device, dtype=torch.float32).reshape(16, 16).to(torch.bfloat16)
    right = torch.eye(16, device=device, dtype=torch.bfloat16)
    product = left @ right
    if not torch.isfinite(product).all() or not torch.allclose(product, left, rtol=0.0, atol=0.0):
        raise RuntimeError("BF16 CUDA matrix multiplication smoke failed")

    query = torch.randn((1, 2, 8, 64), device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention_output = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
    )
    if attention_output.shape != query.shape or not torch.isfinite(attention_output).all():
        raise RuntimeError("BF16 CUDA SDPA smoke failed")
    torch.cuda.synchronize(device_index)

    report = {
        "status": "ok",
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": installed,
        "pip_check": dependency_result,
        "imports": {
            "swift_version": str(getattr(swift, "__version__", "unknown")),
            "transformers_version": transformers.__version__,
            "hrm_config_class": f"{HrmTextConfig.__module__}.{HrmTextConfig.__name__}",
            "hrm_model_class": f"{HrmTextForCausalLM.__module__}.{HrmTextForCausalLM.__name__}",
            "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
        },
        "cuda": {
            "available": True,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "selected_device": str(device),
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_gib": properties.total_memory / (1024**3),
            "bf16_supported": bf16_supported,
            "bf16_matmul_finite": bool(torch.isfinite(product).all().item()),
            "sdpa_finite": bool(torch.isfinite(attention_output).all().item()),
            "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
        },
    }
    atomic_write_json(args.output_report, report)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
