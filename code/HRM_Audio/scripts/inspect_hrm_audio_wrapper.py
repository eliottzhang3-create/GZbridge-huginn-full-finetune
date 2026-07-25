#!/usr/bin/env python3
"""Audit the native HRM-Text Whisper wrapper before Swift multimodal registration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, HrmTextForCausalLM


EXPECTED_BASE_PARAMETERS = 1_182_795_264
EXPECTED_ALIGNER_PARAMETERS = 39_538_176
EXPECTED_AUDIO_PREFIX_TOKENS = 34
EXPECTED_RECURRENT_SEQUENCE = ["L", "L", "L", "H", "L", "L", "L", "H"]
EXPECTED_GRAD_ENABLED = [False, False, False, True, True, True, True, True]
PROMPT = "<|im_start|><|object_ref_start|>What is 1 + 1?<|im_end|>"
RESPONSE = "2."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrm-model-path", type=Path, required=True)
    parser.add_argument("--whisper-model-path", type=Path, required=True)
    parser.add_argument("--wrapper-model-path", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_wrapper_package(wrapper_path: Path):
    package_name = "hrm_text_audio_v1_wrapper_audit"
    init_path = wrapper_path / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"Wrapper package entry point not found: {init_path}")
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(wrapper_path)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import wrapper package from {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def count_parameters(model: torch.nn.Module) -> dict[str, Any]:
    groups = defaultdict(lambda: {"total": 0, "trainable": 0, "tensor_count": 0})
    dtype_counts: dict[str, int] = defaultdict(int)
    device_counts: dict[str, int] = defaultdict(int)
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith("audio_encoder."):
            group = "audio_encoder"
        elif name.startswith(("temporal_compressor.", "audio_projector.", "audio_boundary_embeddings.")):
            group = "aligner"
        elif name.startswith(("model.", "lm_head.")):
            group = "hrm_base"
        else:
            group = "other"
        count = parameter.numel()
        groups[group]["total"] += count
        groups[group]["tensor_count"] += 1
        if parameter.requires_grad:
            groups[group]["trainable"] += count
            trainable_names.append(name)
        dtype_counts[str(parameter.dtype)] += count
        device_counts[str(parameter.device)] += count
    return {
        "groups": dict(groups),
        "total": sum(item["total"] for item in groups.values()),
        "trainable": sum(item["trainable"] for item in groups.values()),
        "trainable_names": trainable_names,
        "dtype_counts": dict(dtype_counts),
        "device_counts": dict(device_counts),
    }


def build_text_batch(tokenizer: Any, device: torch.device) -> dict[str, torch.Tensor | int | list[int]]:
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    response_ids = tokenizer.encode(RESPONSE, add_special_tokens=False)
    eos_id = int(tokenizer.eos_token_id)
    if not response_ids or response_ids[-1] != eos_id:
        response_ids.append(eos_id)
    input_ids_list = prompt_ids + response_ids
    labels_list = [-100] * len(prompt_ids) + response_ids
    token_type_list = [1] * len(prompt_ids) + [0] * len(response_ids)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    labels = torch.tensor([labels_list], dtype=torch.long, device=device)
    token_type_ids = torch.tensor([token_type_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    compact_length = len(response_ids) + 1
    compact_labels = labels[:, -compact_length:].clone()
    if compact_labels[0, 0].item() != -100:
        raise RuntimeError("Compact labels must retain one ignored leading prediction position")
    loss_mask = (labels != -100)[0]
    tensor_compact_labels = F.pad(labels[:, loss_mask], (1, 0), value=-100)
    tensor_logits_to_keep = F.pad(loss_mask[1:], (0, 1), value=True)
    if tensor_compact_labels.shape[1] != int(tensor_logits_to_keep.sum().item()):
        raise RuntimeError("Swift single-sample compact labels/mask cardinality mismatch")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "token_type_ids": token_type_ids,
        "compact_labels": compact_labels,
        "compact_length": compact_length,
        "tensor_compact_labels": tensor_compact_labels,
        "tensor_logits_to_keep": tensor_logits_to_keep,
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
    }


def manual_shifted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def install_forward_audit_hooks(model: torch.nn.Module):
    inner_capture: dict[str, Any] = {}
    recurrence: list[dict[str, Any]] = []

    def inner_pre_hook(_module, _args, kwargs):
        if inner_capture:
            return
        for name in ("inputs_embeds", "attention_mask", "token_type_ids"):
            value = kwargs.get(name)
            if not torch.is_tensor(value):
                raise RuntimeError(f"HRM inner forward did not receive tensor {name}")
            inner_capture[name] = value.detach().cpu().clone()
        inner_capture["input_ids_is_none"] = kwargs.get("input_ids") is None

    def make_stack_hook(stack_name: str):
        def hook(_module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            recurrence.append(
                {
                    "index": len(recurrence),
                    "stack": stack_name,
                    "grad_enabled": torch.is_grad_enabled(),
                    "input_requires_grad": bool(getattr(hidden_states, "requires_grad", False)),
                }
            )

        return hook

    handles = [model.model.register_forward_pre_hook(inner_pre_hook, with_kwargs=True)]
    handles.append(model.model.L_module.register_forward_pre_hook(make_stack_hook("L"), with_kwargs=True))
    handles.append(model.model.H_module.register_forward_pre_hook(make_stack_hook("H"), with_kwargs=True))
    return inner_capture, recurrence, handles


def validate_parameter_policy(parameter_report: dict[str, Any]) -> None:
    groups = parameter_report["groups"]
    required_groups = {"audio_encoder", "aligner", "hrm_base"}
    if not required_groups.issubset(groups):
        raise RuntimeError(f"Wrapper parameter groups are incomplete: {groups}")
    if groups["hrm_base"]["total"] != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(f"Unexpected HRM base parameter count: {groups['hrm_base']}")
    if groups["aligner"]["total"] != EXPECTED_ALIGNER_PARAMETERS:
        raise RuntimeError(f"Unexpected fixed-32 aligner parameter count: {groups['aligner']}")
    if groups["audio_encoder"]["trainable"] != 0:
        raise RuntimeError(f"Whisper encoder is not fully frozen: {groups['audio_encoder']}")
    if groups["hrm_base"]["trainable"] != 0:
        raise RuntimeError(f"HRM base is not fully frozen: {groups['hrm_base']}")
    if groups["aligner"]["trainable"] != EXPECTED_ALIGNER_PARAMETERS:
        raise RuntimeError(f"Aligner is not fully trainable: {groups['aligner']}")
    if groups.get("other", {}).get("total", 0) != 0:
        raise RuntimeError(f"Unclassified wrapper parameters found: {groups['other']}")


def gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    groups = {
        "temporal_compressor": [],
        "audio_projector": [],
        "audio_boundary_embeddings": [],
    }
    frozen_gradient_names: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith(("audio_encoder.", "model.", "lm_head.")):
            if parameter.grad is not None:
                frozen_gradient_names.append(name)
            continue
        for prefix in groups:
            if name.startswith(f"{prefix}."):
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all().item()):
                        raise RuntimeError(f"Non-finite aligner gradient: {name}")
                    groups[prefix].append(
                        {
                            "name": name,
                            "norm": float(parameter.grad.float().norm().item()),
                        }
                    )
                break
    if frozen_gradient_names:
        raise RuntimeError(f"Frozen model parameters received gradients: {frozen_gradient_names[:20]}")
    report: dict[str, Any] = {"frozen_gradient_names": frozen_gradient_names}
    for prefix, entries in groups.items():
        nonzero = [item for item in entries if item["norm"] > 0]
        if not nonzero:
            raise RuntimeError(f"No nonzero gradients reached {prefix}: {entries[:20]}")
        report[prefix] = {
            "grad_tensor_count": len(entries),
            "nonzero_grad_tensor_count": len(nonzero),
            "norm_sum": sum(item["norm"] for item in entries),
            "preview": entries[:8],
        }
    return report


def main() -> None:
    args = parse_args()
    expected_versions = {"transformers": "5.9.0", "torch": "2.11.0+cu128"}
    mismatches = {
        name: {"expected": expected, "actual": version(name)}
        for name, expected in expected_versions.items()
        if version(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Unexpected wrapper-audit environment: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM audio wrapper audit requires CUDA")

    hrm_model_path = args.hrm_model_path.resolve()
    whisper_model_path = args.whisper_model_path.resolve()
    wrapper_model_path = args.wrapper_model_path.resolve()
    for path, description in (
        (hrm_model_path, "HRM model"),
        (whisper_model_path, "Whisper model"),
        (wrapper_model_path, "wrapper model"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description} path: {path}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"Wrapper audit requires a CUDA device, got {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    package = import_wrapper_package(wrapper_model_path)
    config = package.HrmTextAudioConfig.from_pretrained(wrapper_model_path, local_files_only=True)
    expected_config = {
        "hidden_size": 1536,
        "num_layers_per_stack": 16,
        "H_cycles": 2,
        "L_cycles": 3,
        "L_bp_cycles": [0, 3],
        "prefix_lm": True,
        "audio_encoder_hidden_size": 1280,
        "audio_target_token_count": 32,
        "audio_compressor_intermediate_size": 1536,
        "audio_projector_hidden_size": 2048,
    }
    config_mismatches = {
        name: {"expected": expected, "actual": getattr(config, name, None)}
        for name, expected in expected_config.items()
        if getattr(config, name, None) != expected
    }
    if config_mismatches:
        raise RuntimeError(f"Unexpected HRM audio config: {config_mismatches}")

    print("========== HRM AUDIO WRAPPER AUDIT ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(
        f"[packages] transformers={version('transformers')} torch={version('torch')} ",
        f"torchaudio={version('torchaudio')}",
        flush=True,
    )
    print(f"[hrm-model] {hrm_model_path}", flush=True)
    print(f"[whisper-model] {whisper_model_path}", flush=True)
    print(f"[wrapper-model] {wrapper_model_path}", flush=True)
    print(
        f"[cuda] device={device} name={properties.name!r} total_gib={properties.total_memory / (1024**3):.3f}",
        flush=True,
    )

    load_started = time.perf_counter()
    model = package.HrmTextAudioForConditionalGeneration.from_hrm_text_pretrained(
        hrm_model_path,
        audio_encoder_path=whisper_model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_started
    if model.audio_encoder is None:
        raise RuntimeError("Wrapper load did not attach Whisper encoder")
    if model.audio_encoder.training:
        raise RuntimeError("Frozen Whisper encoder must be in eval mode")
    if model.audio_prefix_length != EXPECTED_AUDIO_PREFIX_TOKENS:
        raise RuntimeError(f"Unexpected audio prefix length: {model.audio_prefix_length}")
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError(f"Wrapper HRM attention is not SDPA: {model.config._attn_implementation}")
    parameter_report = count_parameters(model)
    validate_parameter_policy(parameter_report)
    if set(parameter_report["dtype_counts"]) != {"torch.bfloat16"}:
        raise RuntimeError(f"Wrapper parameters are not uniformly BF16: {parameter_report['dtype_counts']}")
    if set(parameter_report["device_counts"]) != {str(device)}:
        raise RuntimeError(f"Wrapper parameters are not uniformly on {device}: {parameter_report['device_counts']}")
    print(
        f"[load] seconds={load_seconds:.6f} total={parameter_report['total']} "
        f"trainable={parameter_report['trainable']} groups={json.dumps(parameter_report['groups'])}",
        flush=True,
    )
    print(f"[hrm-load] {json.dumps(model._hrm_base_loading_info, ensure_ascii=False)}", flush=True)
    print(f"[whisper-load] {json.dumps(model._whisper_loading_info, ensure_ascii=False)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hrm_model_path, local_files_only=True, use_fast=True)
    batch = build_text_batch(tokenizer, device)
    text_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "token_type_ids": batch["token_type_ids"],
        "use_cache": False,
    }
    with torch.inference_mode():
        wrapper_text_logits = model(**text_kwargs).logits
        native_text_logits = HrmTextForCausalLM.forward(model, **text_kwargs).logits
    text_max_abs_diff = float((wrapper_text_logits.float() - native_text_logits.float()).abs().max().item())
    if not torch.equal(wrapper_text_logits, native_text_logits):
        raise RuntimeError(f"Text-only wrapper is not an exact native passthrough: max_abs_diff={text_max_abs_diff}")
    print(
        f"[text-passthrough] shape={tuple(wrapper_text_logits.shape)} max_abs_diff={text_max_abs_diff}",
        flush=True,
    )

    feature_values = torch.linspace(
        -1.0,
        1.0,
        steps=int(config.audio_feature_size) * 3000,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_input_features = feature_values.view(1, int(config.audio_feature_size), 3000)
    with torch.inference_mode():
        audio_prefix = model.build_audio_prefix(audio_input_features)
    expected_prefix_shape = (1, EXPECTED_AUDIO_PREFIX_TOKENS, int(config.hidden_size))
    if tuple(audio_prefix.shape) != expected_prefix_shape or not bool(torch.isfinite(audio_prefix).all().item()):
        raise RuntimeError(
            f"Invalid audio prefix: shape={tuple(audio_prefix.shape)} finite={torch.isfinite(audio_prefix).all().item()}"
        )
    print(f"[audio-prefix] shape={tuple(audio_prefix.shape)} dtype={audio_prefix.dtype}", flush=True)
    del audio_prefix

    full_labels = batch["labels"]
    combined_labels = torch.cat(
        [
            torch.full(
                (1, EXPECTED_AUDIO_PREFIX_TOKENS),
                -100,
                dtype=full_labels.dtype,
                device=device,
            ),
            full_labels,
        ],
        dim=1,
    )
    with torch.inference_mode():
        full_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=full_labels,
            audio_input_features=audio_input_features,
            use_cache=False,
        )
    expected_full_length = EXPECTED_AUDIO_PREFIX_TOKENS + batch["input_ids"].shape[1]
    if full_outputs.logits.shape[1] != expected_full_length:
        raise RuntimeError(
            f"Full-label audio logits length mismatch: expected={expected_full_length} "
            f"actual={full_outputs.logits.shape[1]}"
        )
    full_manual_loss = manual_shifted_loss(full_outputs.logits, combined_labels)
    full_loss_diff = float((full_outputs.loss.float() - full_manual_loss).abs().item())
    if full_loss_diff > 1e-5:
        raise RuntimeError(f"Full-label audio NTP loss mismatch: diff={full_loss_diff}")
    print(
        f"[full-label-loss] model={full_outputs.loss.item():.9f} manual={full_manual_loss.item():.9f} "
        f"diff={full_loss_diff}",
        flush=True,
    )
    del full_outputs, full_manual_loss

    with torch.inference_mode():
        integer_compact_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["compact_labels"],
            logits_to_keep=int(batch["compact_length"]),
            audio_input_features=audio_input_features,
            use_cache=False,
        )
    integer_compact_manual_loss = manual_shifted_loss(
        integer_compact_outputs.logits,
        batch["compact_labels"],
    )
    integer_compact_loss_diff = float(
        (integer_compact_outputs.loss.float() - integer_compact_manual_loss).abs().item()
    )
    if integer_compact_loss_diff > 1e-5:
        raise RuntimeError(f"Integer compact-label audio NTP loss mismatch: diff={integer_compact_loss_diff}")
    print(
        f"[integer-compact-loss] model={integer_compact_outputs.loss.item():.9f} "
        f"manual={integer_compact_manual_loss.item():.9f} diff={integer_compact_loss_diff}",
        flush=True,
    )
    del integer_compact_outputs, integer_compact_manual_loss

    model.train()
    if model.audio_encoder.training:
        raise RuntimeError("model.train() incorrectly switched the frozen Whisper encoder to train mode")
    model.zero_grad(set_to_none=True)
    inner_capture, recurrence, hook_handles = install_forward_audit_hooks(model)
    compact_length = int(batch["tensor_compact_labels"].shape[1])
    forward_started = time.perf_counter()
    try:
        compact_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["tensor_compact_labels"],
            logits_to_keep=batch["tensor_logits_to_keep"],
            audio_input_features=audio_input_features,
            use_cache=False,
        )
    finally:
        for handle in hook_handles:
            handle.remove()
    torch.cuda.synchronize(device_index)
    forward_seconds = time.perf_counter() - forward_started
    if compact_outputs.loss is None or not bool(torch.isfinite(compact_outputs.loss).item()):
        raise RuntimeError(f"Compact-label wrapper loss is invalid: {compact_outputs.loss}")
    if tuple(compact_outputs.logits.shape[:2]) != (1, compact_length):
        raise RuntimeError(f"Compact logits shape mismatch: {tuple(compact_outputs.logits.shape)}")
    compact_manual_loss = manual_shifted_loss(
        compact_outputs.logits.detach(),
        batch["tensor_compact_labels"],
    )
    compact_loss_diff = float((compact_outputs.loss.float() - compact_manual_loss).abs().item())
    if compact_loss_diff > 1e-5:
        raise RuntimeError(f"Compact-label audio NTP loss mismatch: diff={compact_loss_diff}")

    expected_combined_types = torch.cat(
        [
            torch.ones((1, EXPECTED_AUDIO_PREFIX_TOKENS), dtype=torch.long),
            batch["token_type_ids"].detach().cpu(),
        ],
        dim=1,
    )
    if not inner_capture.get("input_ids_is_none", False):
        raise RuntimeError("Audio wrapper did not call native HRM through inputs_embeds")
    if tuple(inner_capture["inputs_embeds"].shape) != (1, expected_full_length, int(config.hidden_size)):
        raise RuntimeError(f"Combined inputs_embeds shape mismatch: {inner_capture['inputs_embeds'].shape}")
    if not torch.equal(inner_capture["token_type_ids"], expected_combined_types):
        raise RuntimeError("Combined audio/text PrefixLM token_type_ids are wrong")
    if not bool((inner_capture["attention_mask"] == 1).all().item()):
        raise RuntimeError("Fixed-32 wrapper combined attention mask must be all ones")
    recurrent_sequence = [item["stack"] for item in recurrence]
    recurrent_grad_enabled = [item["grad_enabled"] for item in recurrence]
    if recurrent_sequence != EXPECTED_RECURRENT_SEQUENCE:
        raise RuntimeError(f"HRM recurrence order changed: {recurrent_sequence}")
    if recurrent_grad_enabled != EXPECTED_GRAD_ENABLED:
        raise RuntimeError(f"HRM static-K=5 gradient suffix changed: {recurrent_grad_enabled}")
    print(
        f"[compact-label-loss] model={compact_outputs.loss.item():.9f} "
        f"manual={compact_manual_loss.item():.9f} diff={compact_loss_diff} "
        f"logits_shape={tuple(compact_outputs.logits.shape)}",
        flush=True,
    )
    print(f"[recurrence] {json.dumps(recurrence, ensure_ascii=False)}", flush=True)

    backward_started = time.perf_counter()
    compact_outputs.loss.backward()
    torch.cuda.synchronize(device_index)
    backward_seconds = time.perf_counter() - backward_started
    gradients = gradient_report(model)
    print(f"[gradients] {json.dumps(gradients, ensure_ascii=False)}", flush=True)

    if not math.isfinite(float(compact_outputs.loss.item())):
        raise RuntimeError("Compact loss became non-finite after backward")
    memory_report = {
        "device_index": device_index,
        "device_name": properties.name,
        "allocated_gib": torch.cuda.memory_allocated(device_index) / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved(device_index) / (1024**3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / (1024**3),
    }
    report = {
        "status": "ok",
        "paths": {
            "hrm_model": str(hrm_model_path),
            "whisper_model": str(whisper_model_path),
            "wrapper_model": str(wrapper_model_path),
        },
        "packages": {
            "transformers": version("transformers"),
            "torch": version("torch"),
            "torchaudio": version("torchaudio"),
        },
        "config": {name: getattr(config, name) for name in expected_config},
        "loading": {
            "seconds": load_seconds,
            "hrm": model._hrm_base_loading_info,
            "whisper": model._whisper_loading_info,
        },
        "parameters": parameter_report,
        "text_passthrough": {
            "shape": list(wrapper_text_logits.shape),
            "max_abs_diff": text_max_abs_diff,
            "exact": True,
        },
        "audio_prefix": {
            "shape": list(expected_prefix_shape),
            "combined_sequence_length": expected_full_length,
            "combined_token_type_ids": expected_combined_types.tolist(),
        },
        "loss": {
            "full_labels_abs_diff": full_loss_diff,
            "integer_compact_labels_abs_diff": integer_compact_loss_diff,
            "compact_labels_model": float(compact_outputs.loss.item()),
            "compact_labels_manual": float(compact_manual_loss.item()),
            "compact_labels_abs_diff": compact_loss_diff,
            "compact_length": compact_length,
            "tensor_logits_to_keep_length_before_audio": int(batch["tensor_logits_to_keep"].numel()),
            "tensor_logits_to_keep_selected": int(batch["tensor_logits_to_keep"].sum().item()),
        },
        "recurrence": recurrence,
        "gradients": gradients,
        "timing": {
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
        },
        "cuda_memory": memory_report,
    }
    atomic_write_json(args.output_report.resolve(), report)
    print(f"[memory] {json.dumps(memory_report, ensure_ascii=False)}", flush=True)
    print(f"[result] status=OK output_report={args.output_report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
