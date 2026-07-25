#!/usr/bin/env python3
"""Audit HRM audio prefill, KV-cache decode, and deterministic generate integration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from transformers import AutoTokenizer


PROMPT = "<|im_start|><|quad_end|><|object_ref_end|>What is 1 + 1?<|im_end|>"
EXPECTED_AUDIO_PREFIX = 34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrm-model-path", type=Path, required=True)
    parser.add_argument("--whisper-model-path", type=Path, required=True)
    parser.add_argument("--wrapper-model-path", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_wrapper_package(wrapper_path: Path):
    package_name = "hrm_text_audio_v1_generation_audit"
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


def cache_length(cache: Any) -> int | None:
    if cache is None:
        return None
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        raise TypeError(f"HRM cache has no get_seq_length(): {type(cache)}")
    return int(getter())


def tensor_summary(value: Any) -> dict[str, Any] | None:
    if not torch.is_tensor(value):
        return None
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }
    if value.numel() <= 4096:
        result["unique"] = value.detach().cpu().unique().tolist()
    return result


class GenerationCapture:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.outer_calls: list[dict[str, Any]] = []
        self.inner_calls: list[dict[str, Any]] = []
        self.audio_prefix_calls: list[dict[str, Any]] = []
        self.audio_encoder_calls = 0
        self.compressor_calls = 0
        self.projector_calls = 0
        self.handles: list[Any] = []
        self.original_build_audio_prefix = model.build_audio_prefix

    def reset(self) -> None:
        self.outer_calls.clear()
        self.inner_calls.clear()
        self.audio_prefix_calls.clear()
        self.audio_encoder_calls = 0
        self.compressor_calls = 0
        self.projector_calls = 0

    def install(self) -> None:
        def audited_build_audio_prefix(_model, audio_input_features, audio_attention_mask=None):
            prefix = self.original_build_audio_prefix(audio_input_features, audio_attention_mask)
            self.audio_prefix_calls.append(
                {
                    "features": tensor_summary(audio_input_features),
                    "prefix": tensor_summary(prefix),
                }
            )
            return prefix

        def outer_pre_hook(_module, _args, kwargs):
            self.outer_calls.append(
                {
                    "input_ids": tensor_summary(kwargs.get("input_ids")),
                    "inputs_embeds": tensor_summary(kwargs.get("inputs_embeds")),
                    "attention_mask": tensor_summary(kwargs.get("attention_mask")),
                    "position_ids": tensor_summary(kwargs.get("position_ids")),
                    "token_type_ids": tensor_summary(kwargs.get("token_type_ids")),
                    "cache_position": tensor_summary(kwargs.get("cache_position")),
                    "cache_length_before": cache_length(kwargs.get("past_key_values")),
                    "has_audio": kwargs.get("audio_input_features") is not None,
                }
            )

        def inner_pre_hook(_module, _args, kwargs):
            self.inner_calls.append(
                {
                    "input_ids": tensor_summary(kwargs.get("input_ids")),
                    "inputs_embeds": tensor_summary(kwargs.get("inputs_embeds")),
                    "attention_mask": tensor_summary(kwargs.get("attention_mask")),
                    "position_ids": tensor_summary(kwargs.get("position_ids")),
                    "token_type_ids": tensor_summary(kwargs.get("token_type_ids")),
                    "cache_position": tensor_summary(kwargs.get("cache_position")),
                    "cache_length_before": cache_length(kwargs.get("past_key_values")),
                }
            )

        def count_encoder(_module, _args, _kwargs):
            self.audio_encoder_calls += 1

        def count_compressor(_module, _args, _kwargs):
            self.compressor_calls += 1

        def count_projector(_module, _args, _kwargs):
            self.projector_calls += 1

        self.model.build_audio_prefix = MethodType(audited_build_audio_prefix, self.model)
        self.handles = [
            self.model.register_forward_pre_hook(outer_pre_hook, with_kwargs=True),
            self.model.model.register_forward_pre_hook(inner_pre_hook, with_kwargs=True),
            self.model.audio_encoder.register_forward_pre_hook(count_encoder, with_kwargs=True),
            self.model.temporal_compressor.register_forward_pre_hook(count_compressor, with_kwargs=True),
            self.model.audio_projector.register_forward_pre_hook(count_projector, with_kwargs=True),
        ]

    def remove(self) -> None:
        self.model.build_audio_prefix = self.original_build_audio_prefix
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def counts(self) -> dict[str, int]:
        return {
            "audio_prefix": len(self.audio_prefix_calls),
            "audio_encoder": self.audio_encoder_calls,
            "compressor": self.compressor_calls,
            "projector": self.projector_calls,
            "outer_forward": len(self.outer_calls),
            "inner_forward": len(self.inner_calls),
        }


def require_one_audio_execution(capture: GenerationCapture, *, stage: str) -> None:
    counts = capture.counts()
    for name in ("audio_prefix", "audio_encoder", "compressor", "projector"):
        if counts[name] != 1:
            raise RuntimeError(f"{stage}: expected one {name} execution, got {counts}")


def validate_prefill_inner(call: dict[str, Any], *, prompt_length: int, hidden_size: int) -> None:
    expected_length = EXPECTED_AUDIO_PREFIX + prompt_length
    if call["input_ids"] is not None:
        raise RuntimeError(f"Audio prefill reached native HRM through input_ids: {call}")
    embeds = call["inputs_embeds"]
    if embeds is None or embeds["shape"] != [1, expected_length, hidden_size]:
        raise RuntimeError(f"Audio prefill embedding shape mismatch: {call}")
    attention = call["attention_mask"]
    if attention is None or attention["shape"] != [1, expected_length] or attention.get("unique") != [1]:
        raise RuntimeError(f"Audio prefill attention mask mismatch: {call}")
    token_types = call["token_type_ids"]
    if token_types is None or token_types["shape"] != [1, expected_length] or token_types.get("unique") != [1]:
        raise RuntimeError(f"Audio prefill PrefixLM token types mismatch: {call}")


def validate_cached_inner(call: dict[str, Any]) -> None:
    input_ids = call["input_ids"]
    if input_ids is None or input_ids["shape"] != [1, 1]:
        raise RuntimeError(f"Cached decode must receive one input token: {call}")
    if call["inputs_embeds"] is not None:
        raise RuntimeError(f"Cached decode unexpectedly received inputs_embeds: {call}")
    cache_before = call["cache_length_before"]
    if cache_before is None or cache_before <= 0:
        raise RuntimeError(f"Cached decode did not receive initialized cache: {call}")
    token_types = call["token_type_ids"]
    if token_types is None or token_types["shape"] != [1, 1] or token_types.get("unique") != [0]:
        raise RuntimeError(f"Cached response token_type_ids must be zero: {call}")
    positions = call["position_ids"]
    if positions is None or positions["shape"] != [1, 1] or positions.get("unique") != [cache_before]:
        raise RuntimeError(f"Cached position_ids do not continue from cache length: {call}")
    attention = call["attention_mask"]
    if attention is not None and attention["shape"] != [1, cache_before + 1]:
        raise RuntimeError(f"Cached attention length mismatch: {call}")


def main() -> None:
    args = parse_args()
    if args.min_new_tokens < 2 or args.max_new_tokens < args.min_new_tokens:
        raise ValueError("Require 2 <= min_new_tokens <= max_new_tokens")
    expected_versions = {"transformers": "5.9.0", "torch": "2.11.0+cu128"}
    mismatches = {
        name: {"expected": expected, "actual": version(name)}
        for name, expected in expected_versions.items()
        if version(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Unexpected generation-audit environment: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM audio generation audit requires CUDA")

    paths = {
        "hrm_model": args.hrm_model_path.resolve(),
        "whisper_model": args.whisper_model_path.resolve(),
        "wrapper_model": args.wrapper_model_path.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} path: {path}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"Generation audit requires CUDA, got {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    package = import_wrapper_package(paths["wrapper_model"])
    config = package.HrmTextAudioConfig.from_pretrained(paths["wrapper_model"], local_files_only=True)
    print("========== HRM AUDIO GENERATION + CACHE AUDIT ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[packages] transformers={version('transformers')} torch={version('torch')}", flush=True)
    print(f"[paths] {json.dumps({name: str(path) for name, path in paths.items()})}", flush=True)
    print(
        f"[cuda] device={device} name={properties.name!r} total_gib={properties.total_memory / (1024**3):.3f}",
        flush=True,
    )
    print(
        f"[settings] min_new_tokens={args.min_new_tokens} max_new_tokens={args.max_new_tokens} "
        "do_sample=False dtype=bfloat16 attention=sdpa",
        flush=True,
    )

    load_started = time.perf_counter()
    model = package.HrmTextAudioForConditionalGeneration.from_hrm_text_pretrained(
        paths["hrm_model"],
        audio_encoder_path=paths["whisper_model"],
        config=config,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_started
    if model.audio_prefix_length != EXPECTED_AUDIO_PREFIX:
        raise RuntimeError(f"Unexpected audio prefix length: {model.audio_prefix_length}")

    tokenizer = AutoTokenizer.from_pretrained(paths["hrm_model"], local_files_only=True, use_fast=True)
    tokenized = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"].to(device)
    text_attention = tokenized["attention_mask"].to(device)
    prompt_token_types = torch.ones_like(input_ids)
    prompt_length = int(input_ids.shape[1])
    combined_length = EXPECTED_AUDIO_PREFIX + prompt_length
    feature_values = torch.linspace(
        -1.0,
        1.0,
        steps=int(config.audio_feature_size) * 3000,
        dtype=torch.bfloat16,
        device=device,
    )
    audio_features = feature_values.view(1, int(config.audio_feature_size), 3000)
    generation_inputs = {
        "input_ids": input_ids,
        "attention_mask": text_attention,
        "token_type_ids": prompt_token_types,
        "audio_input_features": audio_features,
    }
    print(f"[prompt] text={PROMPT!r} tokens={prompt_length} combined_tokens={combined_length}", flush=True)

    capture = GenerationCapture(model)
    capture.install()
    try:
        capture.reset()
        with torch.inference_mode():
            prefill = model(**generation_inputs, use_cache=True, logits_to_keep=1)
        require_one_audio_execution(capture, stage="manual-prefill")
        if len(capture.inner_calls) != 1:
            raise RuntimeError(f"Manual prefill native call count mismatch: {capture.counts()}")
        validate_prefill_inner(capture.inner_calls[0], prompt_length=prompt_length, hidden_size=int(config.hidden_size))
        if prefill.logits.shape != (1, 1, int(config.vocab_size)) or not bool(torch.isfinite(prefill.logits).all()):
            raise RuntimeError(f"Manual prefill logits invalid: {tuple(prefill.logits.shape)}")
        prefill_cache_length = cache_length(prefill.past_key_values)
        if prefill_cache_length != combined_length:
            raise RuntimeError(
                f"Audio prefill cache length mismatch: expected={combined_length} actual={prefill_cache_length}"
            )
        first_token = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        print(
            f"[manual-prefill] cache_length={prefill_cache_length} first_token={int(first_token.item())} "
            f"counts={json.dumps(capture.counts())}",
            flush=True,
        )

        manual_calls_before = capture.counts()
        with torch.inference_mode():
            cached = model(
                input_ids=first_token,
                attention_mask=torch.ones((1, combined_length + 1), dtype=torch.long, device=device),
                position_ids=torch.tensor([[combined_length]], dtype=torch.long, device=device),
                token_type_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
                past_key_values=prefill.past_key_values,
                use_cache=True,
                logits_to_keep=1,
                audio_input_features=audio_features,
            )
        if capture.counts()["audio_prefix"] != manual_calls_before["audio_prefix"]:
            raise RuntimeError("Manual cached decode encoded audio a second time")
        if len(capture.inner_calls) != 2:
            raise RuntimeError(f"Manual cached native call count mismatch: {capture.counts()}")
        validate_cached_inner(capture.inner_calls[1])
        cached_length = cache_length(cached.past_key_values)
        if cached_length != combined_length + 1:
            raise RuntimeError(f"Manual cached length mismatch: expected={combined_length + 1} actual={cached_length}")
        second_token = torch.argmax(cached.logits[:, -1, :], dim=-1, keepdim=True)
        manual_token_ids = [int(first_token.item()), int(second_token.item())]
        print(
            f"[manual-cached] cache_length={cached_length} second_token={manual_token_ids[1]} "
            f"audio_reencoded=False",
            flush=True,
        )

        capture.reset()
        torch.cuda.synchronize(device_index)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **generation_inputs,
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=int(model.config.eos_token_id),
                pad_token_id=int(model.config.pad_token_id),
            )
        torch.cuda.synchronize(device_index)
        generation_seconds = time.perf_counter() - generation_started
        require_one_audio_execution(capture, stage="model.generate")
        if len(capture.inner_calls) < 2:
            raise RuntimeError(f"model.generate did not exercise cached decode: {capture.counts()}")
        validate_prefill_inner(capture.inner_calls[0], prompt_length=prompt_length, hidden_size=int(config.hidden_size))
        for call in capture.inner_calls[1:]:
            validate_cached_inner(call)
    finally:
        capture.remove()

    if not torch.equal(generated[:, :prompt_length], input_ids):
        raise RuntimeError("model.generate did not preserve the text prompt ids")
    generated_ids = generated[0, prompt_length:]
    if not args.min_new_tokens <= generated_ids.numel() <= args.max_new_tokens:
        raise RuntimeError(f"Unexpected generated token count: {generated_ids.numel()}")
    if generated_ids[:2].tolist() != manual_token_ids:
        raise RuntimeError(
            f"Manual/generate greedy token mismatch: manual={manual_token_ids} "
            f"generate={generated_ids[:2].tolist()}"
        )
    generated_raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    generated_clean = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"[generated-raw] {generated_raw}", flush=True)
    print(f"[generated-clean] {generated_clean}", flush=True)
    print(
        f"[generation] tokens={generated_ids.numel()} seconds={generation_seconds:.6f} "
        f"counts={json.dumps(capture.counts())}",
        flush=True,
    )

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
        "paths": {name: str(path) for name, path in paths.items()},
        "packages": {"transformers": version("transformers"), "torch": version("torch")},
        "load_seconds": load_seconds,
        "prompt": {"text": PROMPT, "tokens": prompt_length, "combined_tokens": combined_length},
        "manual": {
            "prefill_cache_length": prefill_cache_length,
            "cached_length": cached_length,
            "first_two_token_ids": manual_token_ids,
        },
        "generation": {
            "min_new_tokens": args.min_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "output_ids": generated[0].tolist(),
            "generated_ids": generated_ids.tolist(),
            "generated_raw": generated_raw,
            "generated_clean": generated_clean,
            "seconds": generation_seconds,
            "execution_counts": capture.counts(),
            "outer_calls": capture.outer_calls,
            "inner_calls": capture.inner_calls,
            "audio_prefix_calls": capture.audio_prefix_calls,
        },
        "cuda_memory": memory,
    }
    atomic_write_json(args.output_report.resolve(), report)
    print(f"[memory] {json.dumps(memory)}", flush=True)
    print(f"[result] status=OK output_report={args.output_report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
