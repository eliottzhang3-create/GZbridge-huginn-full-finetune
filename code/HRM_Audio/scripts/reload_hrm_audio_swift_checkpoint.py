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
from smoke_hrm_audio_swift_trainer import (
    audit_lora_runtime_hyperparameters,
    buffer_digest,
    frozen_parameter_groups,
    parameter_group_digest,
    runtime_contract,
)
from smoke_hrm_swift_trainer import logits_difference_report


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
    parser.add_argument("--expected-lora-rank", type=int, default=8)
    parser.add_argument("--expected-lora-alpha", type=int, default=16)
    parser.add_argument("--expected-lora-dropout", type=float, default=0.0)
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


def build_controlled_prefix_batch(
    wrapper: torch.nn.Module,
    inference_batch: dict[str, Any],
    audio_prefix: torch.Tensor,
) -> dict[str, Any]:
    input_ids = inference_batch["input_ids"]
    attention_mask = inference_batch["attention_mask"]
    token_type_ids = inference_batch["token_type_ids"]
    text_embeds = wrapper.get_input_embeddings()(input_ids)
    audio_prefix = audio_prefix.to(device=text_embeds.device, dtype=text_embeds.dtype)
    if audio_prefix.ndim != 3 or audio_prefix.shape[0] != input_ids.shape[0]:
        raise RuntimeError(
            "Controlled audio prefix batch mismatch: "
            f"prefix={tuple(audio_prefix.shape)} input_ids={tuple(input_ids.shape)}"
        )
    if audio_prefix.shape[-1] != text_embeds.shape[-1]:
        raise RuntimeError(
            "Controlled audio prefix hidden-size mismatch: "
            f"prefix={audio_prefix.shape[-1]} text={text_embeds.shape[-1]}"
        )
    prefix_length = audio_prefix.shape[1]
    prefix_attention = torch.ones(
        (input_ids.shape[0], prefix_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    prefix_types = torch.ones(
        (input_ids.shape[0], prefix_length),
        dtype=token_type_ids.dtype,
        device=token_type_ids.device,
    )
    logits_to_keep = inference_batch["logits_to_keep"]
    if torch.is_tensor(logits_to_keep):
        if logits_to_keep.ndim != 1 or logits_to_keep.dtype != torch.bool:
            raise RuntimeError(
                "Controlled logits_to_keep must be a one-dimensional boolean mask, "
                f"got shape={tuple(logits_to_keep.shape)} dtype={logits_to_keep.dtype}"
            )
        if logits_to_keep.numel() != input_ids.shape[1]:
            raise RuntimeError(
                "Controlled logits_to_keep/text length mismatch: "
                f"mask={logits_to_keep.numel()} text={input_ids.shape[1]}"
            )
        prefix_keep = torch.zeros(prefix_length, dtype=torch.bool, device=logits_to_keep.device)
        logits_to_keep = torch.cat([prefix_keep, logits_to_keep], dim=0)
    return {
        "inputs_embeds": torch.cat([audio_prefix, text_embeds], dim=1),
        "attention_mask": torch.cat([prefix_attention, attention_mask], dim=1),
        "token_type_ids": torch.cat([prefix_types, token_type_ids], dim=1),
        "logits_to_keep": logits_to_keep,
    }


def validate_audio_prefix_cross_instance(report: dict[str, Any]) -> dict[str, Any]:
    epsilon = float(torch.finfo(torch.bfloat16).eps)
    # This is a bounded smoke-test guardrail for the longer Whisper + aligner
    # BF16 graph. It is not used to prove checkpoint state equivalence; exact
    # persistent-state hashes and runtime-contract equality do that separately.
    thresholds = {
        "max_abs_diff_over_scale": 16.0 * epsilon,
        "mean_abs_diff_over_scale": epsilon,
        "cosine_similarity": 1.0 - epsilon,
    }
    failures = {
        key: {"actual": float(report[key]), "required": threshold}
        for key, threshold in thresholds.items()
        if (
            (key == "cosine_similarity" and float(report[key]) < threshold)
            or (key != "cosine_similarity" and float(report[key]) > threshold)
        )
    }
    if failures:
        raise RuntimeError(
            "Fresh-process Whisper+aligner audio-prefix drift exceeds the bounded BF16 envelope: "
            f"report={report} failures={failures}"
        )
    return {
        "name": "Whisper+aligner audio-prefix bounded BF16 drift",
        "role": "numerical_sanity_guard_not_state_equivalence",
        "dtype": "torch.bfloat16",
        "epsilon": epsilon,
        "thresholds": thresholds,
        "accepted": True,
    }


def validate_long_prefix_hrm_cross_instance(
    report: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    epsilon = float(torch.finfo(torch.bfloat16).eps)
    # Long PrefixLM attention plus recurrent HRM compounds BF16 allocation- and
    # kernel-level drift across processes. Exact state/runtime hashes establish
    # reload equivalence; this separate envelope only rejects numerically wild
    # outputs while preserving the measured drift in the report.
    thresholds = {
        "max_abs_diff_over_scale": 16.0 * epsilon,
        "mean_abs_diff_over_scale": epsilon,
        "cosine_similarity": 1.0 - epsilon,
        "top1_agreement": 0.95,
    }
    failures = {}
    for key, threshold in thresholds.items():
        actual = float(report[key])
        if key in {"cosine_similarity", "top1_agreement"}:
            failed = actual < threshold
        else:
            failed = actual > threshold
        if failed:
            failures[key] = {"actual": actual, "required": threshold}
    if failures:
        raise RuntimeError(
            f"{name} exceeds the bounded long-PrefixLM BF16 envelope: "
            f"report={report} failures={failures}"
        )
    return {
        "name": name,
        "role": "numerical_sanity_guard_not_state_equivalence",
        "dtype": "torch.bfloat16",
        "epsilon": epsilon,
        "thresholds": thresholds,
        "accepted": True,
    }


def main() -> None:
    args = parse_args()
    if args.expected_lora_rank <= 0:
        raise ValueError(f"Expected LoRA rank must be positive, got {args.expected_lora_rank}")
    if args.expected_lora_alpha <= 0:
        raise ValueError(f"Expected LoRA alpha must be positive, got {args.expected_lora_alpha}")
    if not 0.0 <= args.expected_lora_dropout < 1.0:
        raise ValueError(f"Expected LoRA dropout must be in [0, 1), got {args.expected_lora_dropout}")
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
    parameter_report = trainability_audit.audit_parameters(
        reloaded_model,
        wrapper,
        expected_lora_rank=args.expected_lora_rank,
    )
    lora_runtime_report = audit_lora_runtime_hyperparameters(
        reloaded_model,
        expected_rank=args.expected_lora_rank,
        expected_alpha=args.expected_lora_alpha,
        expected_dropout=args.expected_lora_dropout,
    )
    adapter_report = audit_adapter_state(reloaded_model, checkpoint / "adapter_model.safetensors")
    aligner_report = audit_aligner_state(reloaded_model, checkpoint / "vit.safetensors")

    payload = torch.load(reference_payload_path, map_location="cpu", weights_only=False)
    required_payload_keys = {
        "batch",
        "reference_logits",
        "reference_audio_prefix",
        "reference_controlled_logits",
        "reference_frozen_parameter_digests",
        "reference_buffer_digest",
        "reference_runtime_contract",
    }
    missing_payload_keys = sorted(required_payload_keys - set(payload))
    if missing_payload_keys:
        raise RuntimeError(f"Fresh reload reference payload is incomplete: missing={missing_payload_keys}")
    batch = {
        key: value.to(device=device) if torch.is_tensor(value) else value
        for key, value in payload["batch"].items()
    }
    if torch.is_floating_point(batch["audio_input_features"]):
        batch["audio_input_features"] = batch["audio_input_features"].to(dtype=torch.bfloat16)
    reference_logits = payload["reference_logits"]
    reference_audio_prefix = payload["reference_audio_prefix"]
    reference_controlled_logits = payload["reference_controlled_logits"]
    reloaded_model.eval()
    fresh_frozen_parameter_digests = {
        name: parameter_group_digest(entries)
        for name, entries in frozen_parameter_groups(
            reloaded_model,
            wrapper,
            expected_lora_rank=args.expected_lora_rank,
        ).items()
    }
    if fresh_frozen_parameter_digests != payload["reference_frozen_parameter_digests"]:
        raise RuntimeError(
            "Fresh-process frozen Whisper/HRM parameter digests differ: "
            f"reference={payload['reference_frozen_parameter_digests']} "
            f"fresh={fresh_frozen_parameter_digests}"
        )
    fresh_buffer_digest = buffer_digest(reloaded_model)
    if fresh_buffer_digest != payload["reference_buffer_digest"]:
        raise RuntimeError(
            "Fresh-process model buffers differ: "
            f"reference={payload['reference_buffer_digest']} fresh={fresh_buffer_digest}"
        )
    fresh_runtime_contract = runtime_contract(reloaded_model, wrapper)
    if fresh_runtime_contract != payload["reference_runtime_contract"]:
        raise RuntimeError(
            "Fresh-process runtime contract differs: "
            f"reference={payload['reference_runtime_contract']} fresh={fresh_runtime_contract}"
        )
    with torch.inference_mode():
        audio_prefix = wrapper.build_audio_prefix(batch["audio_input_features"])
        repeat_audio_prefix = wrapper.build_audio_prefix(batch["audio_input_features"])
        logits = reloaded_model(**batch, use_cache=False).logits.detach().cpu()
        repeat_logits = reloaded_model(**batch, use_cache=False).logits.detach().cpu()
        controlled_batch = build_controlled_prefix_batch(wrapper, batch, reference_audio_prefix)
        controlled_logits = reloaded_model(
            **controlled_batch,
            use_cache=False,
        ).logits.detach().cpu()
        repeat_controlled_logits = reloaded_model(
            **controlled_batch,
            use_cache=False,
        ).logits.detach().cpu()

    audio_prefix_self_repeat = logits_difference_report(
        audio_prefix.detach().cpu(),
        repeat_audio_prefix.detach().cpu(),
    )
    if not audio_prefix_self_repeat["exact"]:
        raise RuntimeError(f"Fresh reloaded audio prefix is not self-deterministic: {audio_prefix_self_repeat}")
    self_repeat = logits_difference_report(logits, repeat_logits)
    if not self_repeat["exact"]:
        raise RuntimeError(f"Fresh reloaded model is not self-deterministic: {self_repeat}")
    controlled_self_repeat = logits_difference_report(controlled_logits, repeat_controlled_logits)
    if not controlled_self_repeat["exact"]:
        raise RuntimeError(
            f"Fresh reloaded controlled HRM path is not self-deterministic: {controlled_self_repeat}"
        )

    audio_prefix_cross_instance = logits_difference_report(
        reference_audio_prefix,
        audio_prefix.detach().cpu(),
    )
    controlled_cross_instance = logits_difference_report(
        reference_controlled_logits,
        controlled_logits,
    )
    controlled_validation = validate_long_prefix_hrm_cross_instance(
        controlled_cross_instance,
        name="Controlled HRM recurrent/LoRA long-PrefixLM cross-process drift",
    )
    end_to_end_cross_instance = logits_difference_report(reference_logits, logits)
    audio_prefix_validation = validate_audio_prefix_cross_instance(audio_prefix_cross_instance)
    end_to_end_validation = validate_long_prefix_hrm_cross_instance(
        end_to_end_cross_instance,
        name="Full HRM audio long-PrefixLM cross-process drift",
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
        "expected_lora_rank": args.expected_lora_rank,
        "expected_lora_alpha": args.expected_lora_alpha,
        "expected_lora_dropout": args.expected_lora_dropout,
        "lora_runtime": lora_runtime_report,
        "aligner_load": aligner_load_report,
        "parameters": parameter_report,
        "adapter": adapter_report,
        "aligner": aligner_report,
        "exact_persistent_state": {
            "frozen_parameter_digests": fresh_frozen_parameter_digests,
            "buffer_digest": fresh_buffer_digest,
            "runtime_contract": fresh_runtime_contract,
            "accepted": True,
        },
        "audio_prefix": {
            "self_repeat": audio_prefix_self_repeat,
            "cross_instance": audio_prefix_cross_instance,
            "validation": audio_prefix_validation,
        },
        "controlled_hrm": {
            "self_repeat": controlled_self_repeat,
            "cross_instance": controlled_cross_instance,
            "validation": controlled_validation,
        },
        "end_to_end_logits": {
            "self_repeat": self_repeat,
            "cross_instance": end_to_end_cross_instance,
            "validation": end_to_end_validation,
        },
        "memory": memory,
    }
    atomic_write_json(output_report, report)
    print("========== HRM AUDIO FRESH-PROCESS CHECKPOINT RELOAD ==========" , flush=True)
    print(f"[checkpoint] {checkpoint}", flush=True)
    print(f"[adapter] {adapter_report}", flush=True)
    print(f"[aligner] {aligner_report}", flush=True)
    print(f"[trainables] total={parameter_report['trainable']} groups={parameter_report['groups']}", flush=True)
    print(f"[lora-runtime] {lora_runtime_report}", flush=True)
    print(f"[frozen-parameter-digests] {fresh_frozen_parameter_digests}", flush=True)
    print(f"[buffer-digest] {fresh_buffer_digest}", flush=True)
    print(f"[runtime-contract] {fresh_runtime_contract}", flush=True)
    print(f"[audio-prefix] self_repeat={audio_prefix_self_repeat}", flush=True)
    print(f"[audio-prefix] cross_instance={audio_prefix_cross_instance}", flush=True)
    print(f"[controlled-hrm] self_repeat={controlled_self_repeat}", flush=True)
    print(f"[controlled-hrm] cross_instance={controlled_cross_instance}", flush=True)
    print(f"[end-to-end-logits] self_repeat={self_repeat}", flush=True)
    print(f"[end-to-end-logits] cross_instance={end_to_end_cross_instance}", flush=True)
    print(f"[memory] {memory}", flush=True)
    print(f"[result] status=OK output_report={output_report}", flush=True)


if __name__ == "__main__":
    main()
