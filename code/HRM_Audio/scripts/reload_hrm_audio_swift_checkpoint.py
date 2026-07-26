#!/usr/bin/env python3
"""Fresh-process reload audit for an HRM audio Swift lora_llm checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

import inspect_hrm_audio_swift_trainability as trainability_audit
from smoke_hrm_swift_trainer import logits_difference_report, validate_bfloat16_cross_instance


MODEL_TYPE = "hrm_text_audio_whisper"
EXPECTED_ADAPTER_TENSORS = 512
EXPECTED_ALIGNER_TENSORS = 20
ALIGNER_MARKERS = ("temporal_compressor.", "audio_projector.", "audio_boundary_embeddings.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_plugin(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("hrm_audio_checkpoint_reload_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import HRM audio Swift plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def canonical_aligner_key(key: str) -> str | None:
    marker = next((candidate for candidate in ALIGNER_MARKERS if candidate in key), None)
    if marker is None:
        return None
    suffix = key[key.index(marker) :]
    return suffix.replace("original_module.", "").replace("modules_to_save.default.", "")


def canonical_adapter_key(key: str) -> str:
    for marker in ("H_module.", "L_module."):
        if marker in key:
            return key[key.index(marker) :].replace(".default.", ".")
    raise RuntimeError(f"Unexpected HRM adapter key: {key}")


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint tensor file is missing: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def load_aligner_sidecar(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    source = read_safetensors(path)
    source_by_canonical: dict[str, torch.Tensor] = {}
    invalid_source = []
    for key, tensor in source.items():
        canonical = canonical_aligner_key(key)
        if canonical is None:
            invalid_source.append(key)
        elif canonical in source_by_canonical:
            raise RuntimeError(f"Duplicate canonical aligner key in sidecar: {canonical}")
        else:
            source_by_canonical[canonical] = tensor
    if invalid_source or len(source_by_canonical) != EXPECTED_ALIGNER_TENSORS:
        raise RuntimeError(
            f"Invalid vit.safetensors contents: count={len(source_by_canonical)} invalid={invalid_source[:20]}"
        )

    target_state = model.state_dict()
    target_by_canonical: dict[str, str] = {}
    for key in target_state:
        canonical = canonical_aligner_key(key)
        if canonical is not None:
            if canonical in target_by_canonical:
                raise RuntimeError(f"Fresh base has duplicate canonical aligner key: {canonical}")
            target_by_canonical[canonical] = key
    if set(source_by_canonical) != set(target_by_canonical):
        raise RuntimeError(
            "Aligner sidecar/fresh-base key mismatch: "
            f"missing={sorted(set(target_by_canonical) - set(source_by_canonical))[:20]} "
            f"unexpected={sorted(set(source_by_canonical) - set(target_by_canonical))[:20]}"
        )
    selected: dict[str, torch.Tensor] = {}
    for canonical, target_key in target_by_canonical.items():
        source_tensor = source_by_canonical[canonical]
        target_tensor = target_state[target_key]
        if source_tensor.shape != target_tensor.shape:
            raise RuntimeError(
                f"Aligner shape mismatch for {canonical}: sidecar={tuple(source_tensor.shape)} "
                f"target={tuple(target_tensor.shape)}"
            )
        selected[target_key] = source_tensor.to(dtype=target_tensor.dtype)
    result = model.load_state_dict(selected, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected aligner restore keys: {result.unexpected_keys}")
    return {
        "tensor_count": len(selected),
        "source_dtype_counts": dict(Counter(str(tensor.dtype) for tensor in source_by_canonical.values())),
        "canonical_keys": sorted(source_by_canonical),
    }


def force_expected_policy(model: torch.nn.Module) -> torch.nn.Module:
    wrapper = trainability_audit.find_unique_module(model, "HrmTextAudioForConditionalGeneration")
    wrapper.audio_encoder.requires_grad_(False)
    for name, parameter in wrapper.model.named_parameters():
        parameter.requires_grad = "lora_" in name
    wrapper.lm_head.requires_grad_(False)
    for module_name in ("temporal_compressor", "audio_projector", "audio_boundary_embeddings"):
        module = getattr(wrapper, module_name)
        module.requires_grad_(True)
    wrapper.train()
    if wrapper.audio_encoder.training:
        raise RuntimeError("Fresh reload put the frozen Whisper encoder in training mode")
    return wrapper


def audit_adapter_state(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    checkpoint_state = read_safetensors(checkpoint_path)
    runtime_state = get_peft_model_state_dict(model)
    checkpoint_canonical = {canonical_adapter_key(key): tensor for key, tensor in checkpoint_state.items()}
    runtime_canonical = {canonical_adapter_key(key): tensor.detach().cpu() for key, tensor in runtime_state.items()}
    if len(checkpoint_canonical) != EXPECTED_ADAPTER_TENSORS or set(checkpoint_canonical) != set(runtime_canonical):
        raise RuntimeError(
            "Fresh adapter key mismatch: "
            f"checkpoint={len(checkpoint_canonical)} runtime={len(runtime_canonical)} "
            f"missing={sorted(set(checkpoint_canonical) - set(runtime_canonical))[:20]} "
            f"unexpected={sorted(set(runtime_canonical) - set(checkpoint_canonical))[:20]}"
        )
    max_abs_diff = 0.0
    dtype_pairs = Counter()
    for key in checkpoint_canonical:
        left = checkpoint_canonical[key]
        right = runtime_canonical[key]
        dtype_pairs[(str(left.dtype), str(right.dtype))] += 1
        max_abs_diff = max(max_abs_diff, float((left.float() - right.float()).abs().max().item()))
    if max_abs_diff != 0.0:
        raise RuntimeError(f"Fresh adapter reload is not exact: max_abs_diff={max_abs_diff}")
    return {
        "tensor_count": len(checkpoint_canonical),
        "max_abs_diff": max_abs_diff,
        "dtype_pairs": {f"{left}->{right}": count for (left, right), count in dtype_pairs.items()},
    }


def audit_aligner_state(model: torch.nn.Module, sidecar_path: Path) -> dict[str, Any]:
    source = {
        canonical_aligner_key(key): tensor
        for key, tensor in read_safetensors(sidecar_path).items()
        if canonical_aligner_key(key) is not None
    }
    wrapper = trainability_audit.find_unique_module(model, "HrmTextAudioForConditionalGeneration")
    runtime = {
        canonical_aligner_key(key): tensor.detach().cpu()
        for key, tensor in wrapper.state_dict().items()
        if canonical_aligner_key(key) is not None
    }
    if len(source) != EXPECTED_ALIGNER_TENSORS or set(source) != set(runtime):
        raise RuntimeError(
            f"Fresh aligner key mismatch: sidecar={len(source)} runtime={len(runtime)}"
        )
    max_abs_diff = 0.0
    dtype_pairs = Counter()
    for key in source:
        left = source[key]
        right = runtime[key]
        dtype_pairs[(str(left.dtype), str(right.dtype))] += 1
        max_abs_diff = max(max_abs_diff, float((left.float() - right.float()).abs().max().item()))
    if max_abs_diff != 0.0:
        raise RuntimeError(f"Fresh aligner reload is not exact: max_abs_diff={max_abs_diff}")
    return {
        "tensor_count": len(source),
        "max_abs_diff": max_abs_diff,
        "dtype_pairs": {f"{left}->{right}": count for (left, right), count in dtype_pairs.items()},
    }


def main() -> None:
    args = parse_args()
    wrapper_model_path = args.wrapper_model_path.resolve()
    plugin_path = args.plugin_path.resolve()
    checkpoint = args.checkpoint.resolve()
    reference_payload_path = args.reference_payload.resolve()
    output_report = args.output_report.resolve()
    for path, description in (
        (wrapper_model_path, "wrapper model"),
        (plugin_path, "plugin"),
        (checkpoint, "checkpoint"),
        (reference_payload_path, "reference payload"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh HRM audio checkpoint reload requires CUDA")

    import_plugin(plugin_path)
    try:
        from swift import get_model_processor
    except ImportError:
        from swift.model import get_model_processor
    from swift.tuners import Swift

    device = torch.device("cuda:0")
    torch.cuda.set_device(device.index or 0)
    base_model, _ = get_model_processor(
        str(wrapper_model_path),
        model_type=MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    if base_model is None:
        raise RuntimeError("Fresh Swift HRM audio base load returned None")
    base_model.config.use_cache = False
    aligner_load_report = load_aligner_sidecar(base_model, checkpoint / "vit.safetensors")
    reloaded_model = Swift.from_pretrained(base_model, str(checkpoint), is_trainable=True)
    wrapper = force_expected_policy(reloaded_model)
    parameter_report = trainability_audit.audit_parameters(reloaded_model, wrapper)
    adapter_report = audit_adapter_state(reloaded_model, checkpoint / "adapter_model.safetensors")
    aligner_report = audit_aligner_state(reloaded_model, checkpoint / "vit.safetensors")

    payload = torch.load(reference_payload_path, map_location="cpu", weights_only=False)
    batch = {
        key: value.to(device=device) if torch.is_tensor(value) else value
        for key, value in payload["batch"].items()
    }
    if torch.is_floating_point(batch["audio_input_features"]):
        batch["audio_input_features"] = batch["audio_input_features"].to(dtype=torch.bfloat16)
    reference_logits = payload["reference_logits"]
    reloaded_model.eval()
    with torch.inference_mode():
        logits = reloaded_model(**batch, use_cache=False).logits.detach().cpu()
        repeat_logits = reloaded_model(**batch, use_cache=False).logits.detach().cpu()
    self_repeat = logits_difference_report(logits, repeat_logits)
    if not self_repeat["exact"]:
        raise RuntimeError(f"Fresh reloaded model is not self-deterministic: {self_repeat}")
    cross_instance = logits_difference_report(reference_logits, logits)
    numerical_validation = validate_bfloat16_cross_instance(
        cross_instance,
        name="HRM audio fresh-process checkpoint reload",
    )

    memory = {
        "device": torch.cuda.get_device_name(0),
        "allocated_gib": torch.cuda.memory_allocated(0) / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved(0) / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
    }
    report = {
        "status": "OK",
        "checkpoint": str(checkpoint),
        "aligner_load": aligner_load_report,
        "parameters": parameter_report,
        "adapter": adapter_report,
        "aligner": aligner_report,
        "logits": {
            "self_repeat": self_repeat,
            "cross_instance": cross_instance,
            "validation": numerical_validation,
        },
        "memory": memory,
    }
    atomic_write_json(output_report, report)
    print("========== HRM AUDIO FRESH-PROCESS CHECKPOINT RELOAD ==========" , flush=True)
    print(f"[checkpoint] {checkpoint}", flush=True)
    print(f"[adapter] {adapter_report}", flush=True)
    print(f"[aligner] {aligner_report}", flush=True)
    print(f"[trainables] total={parameter_report['trainable']} groups={parameter_report['groups']}", flush=True)
    print(f"[logits] self_repeat={self_repeat}", flush=True)
    print(f"[logits] cross_instance={cross_instance}", flush=True)
    print(f"[memory] {memory}", flush=True)
    print(f"[result] status=OK output_report={output_report}", flush=True)


if __name__ == "__main__":
    main()
