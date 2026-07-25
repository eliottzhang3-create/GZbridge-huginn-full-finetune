#!/usr/bin/env python3
"""Audit independent HRM audio registration, processor, template, collator, and Swift load."""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import math
import struct
import sys
import time
import wave
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


TEXT_MODEL_TYPE = "hrm_text_native"
TEXT_TEMPLATES = ("hrm_text_direct", "hrm_text_synth_cot")
AUDIO_MODEL_TYPE = "hrm_text_audio_whisper"
AUDIO_TEMPLATE_TYPE = "hrm_text_audio"
AUDIO_MODEL_ARCH = "hrm_text_audio_whisper"
QUESTION = "What is 1 + 1?"
RESPONSE = "2."
EXPECTED_PROMPT = f"<|im_start|><|object_ref_start|>{QUESTION}<|im_end|>"
EXPECTED_ALIGNER_PARAMETERS = 39_538_176
EXPECTED_HRM_PARAMETERS = 1_182_795_264


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-model-path", type=Path, required=True)
    parser.add_argument("--text-plugin-path", type=Path, required=True)
    parser.add_argument("--audio-plugin-path", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_plugin(path: Path, module_name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Plugin is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_mapping(modules: list[Any], required_key: str, preferred_name: str) -> tuple[str, Mapping[Any, Any]]:
    candidates: list[tuple[str, Mapping[Any, Any]]] = []
    for module in modules:
        for attribute, value in vars(module).items():
            if "MAPPING" in attribute.upper() and isinstance(value, Mapping):
                candidates.append((f"{module.__name__}.{attribute}", value))
    exact_preferred = [
        item
        for item in candidates
        if item[0].rsplit(".", 1)[-1].upper() == preferred_name.upper() and required_key in item[1]
    ]
    if exact_preferred:
        return exact_preferred[0]
    exact = [item for item in candidates if required_key in item[1]]
    if not exact:
        raise RuntimeError(f"No Swift mapping contains key={required_key!r}; candidates={[item[0] for item in candidates]}")
    exact.sort(key=lambda item: len(item[1]), reverse=True)
    return exact[0]


def as_int_list(value: Any, *, name: str) -> list[int]:
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be list/tuple/tensor, got {type(value)}")
    return [int(item) for item in value]


def create_test_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    frame_count = sample_rate // 4
    frames = bytearray()
    for index in range(frame_count):
        value = int(0.2 * 32767 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def build_template(get_template, processor: Any):
    return get_template(
        template_type=AUDIO_TEMPLATE_TYPE,
        processor=processor,
        max_length=512,
        use_chat_template=True,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )


def audit_encoded(template: Any, tokenizer: Any, wav_path: Path, *, response: str | None):
    messages = [{"role": "user", "content": QUESTION}]
    mode = "transformers"
    if response is not None:
        messages.append({"role": "assistant", "content": response})
        mode = "train"
    sample = {"messages": messages, "audios": [str(wav_path)]}
    template.set_mode(mode)
    encoded = template.encode(sample)
    input_ids = as_int_list(encoded["input_ids"], name="input_ids")
    token_types = as_int_list(encoded.get("token_type_ids"), name="token_type_ids")
    labels_value = encoded.get("labels")
    labels = as_int_list(labels_value, name="labels") if labels_value is not None else None
    expected_prompt_ids = as_int_list(
        tokenizer(EXPECTED_PROMPT, add_special_tokens=False)["input_ids"],
        name="expected_prompt_ids",
    )
    if input_ids[: len(expected_prompt_ids)] != expected_prompt_ids:
        raise RuntimeError(
            f"HRM audio prompt token mismatch: expected={expected_prompt_ids} actual={input_ids[:len(expected_prompt_ids)]}"
        )
    if len(token_types) != len(input_ids):
        raise RuntimeError(f"HRM audio token type length mismatch: input={len(input_ids)} types={len(token_types)}")
    features = encoded.get("audio_input_features")
    if not torch.is_tensor(features) or tuple(features.shape) != (80, 3000):
        raise RuntimeError(f"HRM audio encoded feature shape mismatch: {getattr(features, 'shape', None)}")
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("HRM audio encoded features contain NaN or Inf")

    if response is None:
        if input_ids != expected_prompt_ids or set(token_types) != {1}:
            raise RuntimeError(f"Inference PrefixLM encoding is invalid: ids={input_ids} types={token_types}")
        if labels is not None and any(label != -100 for label in labels):
            raise RuntimeError(f"Inference sample unexpectedly has supervised labels: {labels}")
        prefix_length = len(input_ids)
    else:
        if labels is None or len(labels) != len(input_ids):
            raise RuntimeError("Training encoding lacks sequence-aligned labels")
        supervised = [index for index, label in enumerate(labels) if label != -100]
        if not supervised:
            raise RuntimeError("Training encoding has no supervised response tokens")
        prefix_length = supervised[0]
        if token_types[:prefix_length] != [1] * prefix_length:
            raise RuntimeError("Training prompt token types are not all PrefixLM ones")
        if token_types[prefix_length:] != [0] * (len(token_types) - prefix_length):
            raise RuntimeError("Training response token types are not all causal zeros")
        if any(label != -100 for label in labels[:prefix_length]):
            raise RuntimeError("Training prompt labels are not ignored")

    report = {
        "mode": mode,
        "sample": sample,
        "sequence_length": len(input_ids),
        "prefix_length": prefix_length,
        "input_ids": input_ids,
        "labels": labels,
        "token_type_ids": token_types,
        "audio_shape": list(features.shape),
        "audio_dtype": str(features.dtype),
    }
    return encoded, report


def parameter_policy(model: torch.nn.Module) -> dict[str, Any]:
    groups = {
        "audio_encoder": {"total": 0, "trainable": 0},
        "aligner": {"total": 0, "trainable": 0},
        "hrm_base": {"total": 0, "trainable": 0},
        "other": {"total": 0, "trainable": 0},
    }
    for name, parameter in model.named_parameters():
        if name.startswith("audio_encoder."):
            group = "audio_encoder"
        elif name.startswith(("temporal_compressor.", "audio_projector.", "audio_boundary_embeddings.")):
            group = "aligner"
        elif name.startswith(("model.", "lm_head.")):
            group = "hrm_base"
        else:
            group = "other"
        groups[group]["total"] += parameter.numel()
        if parameter.requires_grad:
            groups[group]["trainable"] += parameter.numel()
    if groups["hrm_base"] != {"total": EXPECTED_HRM_PARAMETERS, "trainable": 0}:
        raise RuntimeError(f"Swift-loaded HRM base policy mismatch: {groups['hrm_base']}")
    if groups["aligner"] != {"total": EXPECTED_ALIGNER_PARAMETERS, "trainable": EXPECTED_ALIGNER_PARAMETERS}:
        raise RuntimeError(f"Swift-loaded aligner policy mismatch: {groups['aligner']}")
    if groups["audio_encoder"]["total"] <= 0 or groups["audio_encoder"]["trainable"] != 0:
        raise RuntimeError(f"Swift-loaded Whisper policy mismatch: {groups['audio_encoder']}")
    if groups["other"]["total"] != 0:
        raise RuntimeError(f"Unclassified Swift-loaded parameters: {groups['other']}")
    return groups


def main() -> None:
    args = parse_args()
    required_versions = {"ms-swift": "4.4.2", "transformers": "5.9.0", "torch": "2.11.0+cu128"}
    mismatches = {
        name: {"expected": expected, "actual": version(name)}
        for name, expected in required_versions.items()
        if version(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Unexpected HRM audio Swift environment: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM audio Swift registration audit requires CUDA")

    try:
        from swift import get_model_processor, get_template
    except ImportError:
        from swift.model import get_model_processor
        from swift.template import get_template
    import swift.model as swift_model
    import swift.template as swift_template

    model_register = importlib.import_module("swift.model.register")
    template_register = importlib.import_module("swift.template.register")
    registry_modules = [swift_model, model_register]
    template_modules = [swift_template, template_register]
    arch_modules = [swift_model, model_register]
    for module_name in ("swift.model.model_arch", "swift.model.model_archs"):
        try:
            arch_modules.append(importlib.import_module(module_name))
        except ImportError:
            pass

    text_plugin = import_plugin(args.text_plugin_path.resolve(), "hrm_text_swift_audio_isolation_audit")
    model_registry_name, model_registry = discover_mapping(registry_modules, TEXT_MODEL_TYPE, "MODEL_MAPPING")
    template_registry_name, template_registry = discover_mapping(template_modules, TEXT_TEMPLATES[0], "TEMPLATE_MAPPING")
    text_model_meta_before = model_registry[TEXT_MODEL_TYPE]
    text_template_meta_before = {name: template_registry[name] for name in TEXT_TEMPLATES}

    audio_plugin = import_plugin(args.audio_plugin_path.resolve(), "hrm_text_audio_swift_registration_audit")
    if audio_plugin.MODEL_TYPE != AUDIO_MODEL_TYPE or audio_plugin.TEMPLATE_TYPE != AUDIO_TEMPLATE_TYPE:
        raise RuntimeError("HRM audio plugin identifiers do not match the audit contract")
    model_registry_name, model_registry = discover_mapping(registry_modules, AUDIO_MODEL_TYPE, "MODEL_MAPPING")
    template_registry_name, template_registry = discover_mapping(template_modules, AUDIO_TEMPLATE_TYPE, "TEMPLATE_MAPPING")
    arch_registry_name, arch_registry = discover_mapping(arch_modules, AUDIO_MODEL_ARCH, "MODEL_ARCH_MAPPING")
    if model_registry.get(TEXT_MODEL_TYPE) is not text_model_meta_before:
        raise RuntimeError("Audio plugin replaced the existing HRM text model registration")
    for name, meta in text_template_meta_before.items():
        if template_registry.get(name) is not meta:
            raise RuntimeError(f"Audio plugin replaced the existing HRM text template: {name}")
    if AUDIO_MODEL_TYPE not in model_registry or AUDIO_TEMPLATE_TYPE not in template_registry:
        raise RuntimeError("HRM audio model/template registration is missing")
    if AUDIO_MODEL_ARCH not in arch_registry:
        raise RuntimeError("HRM audio MultiModelKeys registration is missing")

    model_meta = model_registry[AUDIO_MODEL_TYPE]
    if not model_meta.is_multimodal or model_meta.template != AUDIO_TEMPLATE_TYPE:
        raise RuntimeError(f"HRM audio ModelMeta is invalid: {model_meta}")
    resolved_model_arch = model_meta.model_arch
    if isinstance(resolved_model_arch, str):
        resolved_arch_name = resolved_model_arch
    else:
        resolved_arch_name = getattr(resolved_model_arch, "arch_name", None)
    if resolved_arch_name != AUDIO_MODEL_ARCH:
        raise RuntimeError(
            f"HRM audio model_arch mismatch: expected={AUDIO_MODEL_ARCH!r} "
            f"resolved_name={resolved_arch_name!r} value={resolved_model_arch!r}"
        )
    if model_meta.loader.__name__ != "HrmTextAudioLoader":
        raise RuntimeError(f"HRM audio loader mismatch: {model_meta.loader}")
    if model_meta.architectures != ["HrmTextAudioForConditionalGeneration"]:
        raise RuntimeError(f"HRM audio architecture metadata mismatch: {model_meta.architectures}")
    template_meta = template_registry[AUDIO_TEMPLATE_TYPE]
    if template_meta.prompt != ["<|im_start|><|object_ref_start|>{{QUERY}}<|im_end|>"]:
        raise RuntimeError(f"HRM audio template prompt mismatch: {template_meta.prompt}")
    if template_meta.template_cls.__name__ != "HrmTextAudioTemplate":
        raise RuntimeError(f"HRM audio template class mismatch: {template_meta.template_cls}")

    arch_meta = arch_registry[AUDIO_MODEL_ARCH]

    def arch_value(name: str):
        return arch_meta[name] if isinstance(arch_meta, Mapping) else getattr(arch_meta, name)

    arch_groups = {
        name: list(arch_value(name))
        for name in ("language_model", "aligner", "generator")
    }
    expected_arch_groups = {
        "language_model": ["model", "lm_head"],
        "aligner": ["temporal_compressor", "audio_projector", "audio_boundary_embeddings"],
        "generator": ["audio_encoder"],
    }
    if arch_groups != expected_arch_groups:
        raise RuntimeError(f"HRM audio MultiModelKeys groups mismatch: {arch_groups}")
    if not isinstance(resolved_model_arch, str):
        resolved_arch_groups = {
            name: list(getattr(resolved_model_arch, name))
            for name in ("language_model", "aligner", "generator")
        }
        if resolved_arch_groups != expected_arch_groups or resolved_arch_groups != arch_groups:
            raise RuntimeError(
                "Resolved ModelMeta/MODEL_ARCH_MAPPING groups differ: "
                f"resolved={resolved_arch_groups} registry={arch_groups}"
            )

    wrapper_model_path = args.wrapper_model_path.resolve()
    model, processor = get_model_processor(
        str(wrapper_model_path),
        model_type=AUDIO_MODEL_TYPE,
        load_model=False,
        download_model=False,
    )
    if model is not None:
        raise RuntimeError("load_model=False unexpectedly returned an HRM audio model")
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id != 11 or len(tokenizer) != 65536:
        raise RuntimeError(f"Unexpected HRM audio tokenizer: vocab={len(tokenizer)} eos={tokenizer.eos_token_id}")

    output_dir = args.output_report.resolve().parent
    wav_path = output_dir / "synthetic_16k_mono.wav"
    create_test_wav(wav_path)
    infer_template = build_template(get_template, processor)
    infer_encoded, infer_report = audit_encoded(infer_template, tokenizer, wav_path, response=None)
    train_template = build_template(get_template, processor)
    train_encoded, train_report = audit_encoded(train_template, tokenizer, wav_path, response=RESPONSE)
    collated = train_template.data_collator([copy.deepcopy(train_encoded), copy.deepcopy(train_encoded)])
    required_batch_keys = {"input_ids", "attention_mask", "labels", "token_type_ids", "audio_input_features"}
    if not required_batch_keys.issubset(collated):
        raise RuntimeError(f"HRM audio collator dropped fields: keys={sorted(collated)}")
    if tuple(collated["audio_input_features"].shape) != (2, 80, 3000):
        raise RuntimeError(f"HRM audio collator feature shape mismatch: {tuple(collated['audio_input_features'].shape)}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"HRM audio Swift load requires CUDA, got {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    load_started = time.perf_counter()
    model, loaded_processor = get_model_processor(
        str(wrapper_model_path),
        model_type=AUDIO_MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    model.eval()
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_started
    if model.__class__.__name__ != "HrmTextAudioForConditionalGeneration":
        raise RuntimeError(f"Swift loaded unexpected model class: {type(model)}")
    if type(loaded_processor).__name__ != "HrmTextAudioProcessor":
        raise RuntimeError(f"Swift loaded unexpected processor: {type(loaded_processor)}")
    parameters = parameter_policy(model)

    input_ids = torch.tensor([as_int_list(infer_encoded["input_ids"], name="input_ids")], device=device)
    token_types = torch.tensor(
        [as_int_list(infer_encoded["token_type_ids"], name="token_type_ids")],
        dtype=torch.long,
        device=device,
    )
    attention = torch.ones_like(input_ids)
    audio_features = infer_encoded["audio_input_features"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention,
            token_type_ids=token_types,
            audio_input_features=audio_features,
            use_cache=False,
            logits_to_keep=1,
        )
    if outputs.logits.shape != (1, 1, int(model.config.vocab_size)):
        raise RuntimeError(f"Swift-loaded audio prefill logits shape mismatch: {tuple(outputs.logits.shape)}")
    if not bool(torch.isfinite(outputs.logits).all().item()):
        raise RuntimeError("Swift-loaded audio prefill logits contain NaN or Inf")

    memory = {
        "device_index": device_index,
        "device_name": properties.name,
        "allocated_gib": torch.cuda.memory_allocated(device_index) / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved(device_index) / (1024**3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / (1024**3),
    }
    report = {
        "status": "ok",
        "packages": {name: version(name) for name in required_versions},
        "plugins": {
            "text": str(args.text_plugin_path.resolve()),
            "audio": str(args.audio_plugin_path.resolve()),
            "text_registration_preserved": True,
        },
        "registries": {
            "model": model_registry_name,
            "template": template_registry_name,
            "model_arch": arch_registry_name,
            "audio_model_type": AUDIO_MODEL_TYPE,
            "audio_template_type": AUDIO_TEMPLATE_TYPE,
            "audio_model_arch": AUDIO_MODEL_ARCH,
            "resolved_model_arch_type": f"{type(resolved_model_arch).__module__}.{type(resolved_model_arch).__name__}",
            "resolved_model_arch_repr": repr(resolved_model_arch),
            "arch_groups": arch_groups,
        },
        "processor": {
            "type": f"{type(processor).__module__}.{type(processor).__name__}",
            "tokenizer": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
            "feature_extractor": f"{type(processor.feature_extractor).__module__}.{type(processor.feature_extractor).__name__}",
        },
        "encoding": {"inference": infer_report, "training": train_report},
        "collator_shapes": {
            key: list(value.shape) for key, value in collated.items() if torch.is_tensor(value)
        },
        "model": {
            "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "load_seconds": load_seconds,
            "parameter_groups": parameters,
            "prefill_logits_shape": list(outputs.logits.shape),
            "prefill_finite": True,
        },
        "cuda_memory": memory,
    }
    atomic_write_json(args.output_report.resolve(), report)
    print("========== HRM AUDIO SWIFT REGISTRATION AUDIT ==========", flush=True)
    print(f"[registry] model={model_registry_name} template={template_registry_name} arch={arch_registry_name}", flush=True)
    print(f"[isolation] text_registration_preserved=True", flush=True)
    print(f"[arch] {json.dumps(arch_groups)}", flush=True)
    print(f"[infer-encode] {json.dumps(infer_report, ensure_ascii=False)}", flush=True)
    print(f"[train-encode] {json.dumps(train_report, ensure_ascii=False)}", flush=True)
    print(f"[collator] shapes={json.dumps(report['collator_shapes'])}", flush=True)
    print(f"[model] class={report['model']['class']} parameters={json.dumps(parameters)}", flush=True)
    print(f"[prefill] logits_shape={tuple(outputs.logits.shape)} finite=True", flush=True)
    print(f"[memory] {json.dumps(memory)}", flush=True)
    print(f"[result] status=OK output_report={args.output_report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
