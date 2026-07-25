#!/usr/bin/env python3
"""Load HRM-Text through Swift and audit deterministic PrefixLM inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


DEFAULT_MODEL_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text"
MODEL_TYPE = "hrm_text_native"
TEMPLATE_PROMPTS = {
    "hrm_text_direct": "<|im_start|><|object_ref_start|>{question}<|im_end|>",
    "hrm_text_synth_cot": "<|im_start|><|quad_end|><|object_ref_end|>{question}<|im_end|>",
}
EXPECTED_PARAMETER_COUNT = 1_182_795_264


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--template-type", choices=sorted(TEMPLATE_PROMPTS), default="hrm_text_synth_cot")
    parser.add_argument("--question", default="What is 1 + 1?")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_plugin(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Plugin not found: {path}")
    module_name = "hrm_text_swift_model_inference_audit"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def tokenizer_from_processor(processor: Any):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def as_int_list(value: Any, *, name: str) -> list[int]:
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be list/tuple/tensor, got {type(value)}")
    return [int(item) for item in value]


def tensor_summary(value: Any) -> dict[str, Any] | None:
    if not torch.is_tensor(value):
        return None
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }
    if value.numel() > 0 and value.numel() <= 4096:
        result["unique"] = [int(item) for item in torch.unique(value.detach()).cpu().tolist()]
    return result


def count_parameters(model: torch.nn.Module) -> tuple[int, Counter[str], Counter[str]]:
    total = 0
    dtypes: Counter[str] = Counter()
    devices: Counter[str] = Counter()
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        dtypes[str(parameter.dtype)] += count
        devices[str(parameter.device)] += count
    return total, dtypes, devices


def main() -> None:
    args = parse_args()
    if version("ms-swift") != "4.4.2" or version("transformers") != "5.9.0":
        raise RuntimeError(
            f"Unexpected environment: ms-swift={version('ms-swift')} transformers={version('transformers')}"
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    question = args.question.strip()
    if not question:
        raise ValueError("--question must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM Swift inference audit requires CUDA")

    try:
        from swift import get_model_processor, get_template
    except ImportError:
        from swift.model import get_model_processor
        from swift.template import get_template

    model_path = args.model_path.resolve()
    required_files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing_files = [name for name in required_files if not (model_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Incomplete HRM-Text snapshot at {model_path}: missing={missing_files}")

    plugin_path = args.plugin_path.resolve()
    plugin = import_plugin(plugin_path)
    if getattr(plugin, "MODEL_TYPE", None) != MODEL_TYPE:
        raise RuntimeError(f"Plugin model type mismatch: expected={MODEL_TYPE} actual={plugin.MODEL_TYPE!r}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This audit requires a CUDA device, got {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    print("========== HRM SWIFT MODEL + INFERENCE AUDIT ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[packages] ms-swift={version('ms-swift')} transformers={version('transformers')} torch={version('torch')}", flush=True)
    print(f"[model-path] {model_path}", flush=True)
    print(f"[plugin-path] {plugin_path}", flush=True)
    print(
        f"[cuda] device={device} index={device_index} name={properties.name!r} "
        f"total_gib={properties.total_memory / (1024**3):.3f}",
        flush=True,
    )
    print(
        f"[settings] model_type={MODEL_TYPE} template={args.template_type} "
        f"dtype=bfloat16 attention=sdpa max_new_tokens={args.max_new_tokens} do_sample=False",
        flush=True,
    )

    print("[stage] swift-load=get_model_processor", flush=True)
    load_started = time.perf_counter()
    model, processor = get_model_processor(
        str(model_path),
        model_type=MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    if model is None:
        raise RuntimeError("Swift get_model_processor(load_model=True) returned model=None")
    model.eval()
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_started

    tokenizer = tokenizer_from_processor(processor)
    parameter_count, dtype_counts, device_counts = count_parameters(model)
    if model.__class__.__name__ != "HrmTextForCausalLM":
        raise RuntimeError(f"Unexpected model class: {model.__class__.__module__}.{model.__class__.__name__}")
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected parameter count: expected={EXPECTED_PARAMETER_COUNT} actual={parameter_count}")
    if set(dtype_counts) != {"torch.bfloat16"}:
        raise RuntimeError(f"Expected all parameters in BF16, got {dict(dtype_counts)}")
    if set(device_counts) != {str(device)}:
        raise RuntimeError(f"Expected all parameters on {device}, got {dict(device_counts)}")
    if not bool(getattr(model.config, "prefix_lm", False)):
        raise RuntimeError("Swift-loaded HRM config does not enable prefix_lm")
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    if attention_implementation != "sdpa":
        raise RuntimeError(f"Expected SDPA attention, got {attention_implementation!r}")
    if tokenizer.eos_token_id != model.config.eos_token_id:
        raise RuntimeError(
            f"Tokenizer/model EOS mismatch: tokenizer={tokenizer.eos_token_id} model={model.config.eos_token_id}"
        )
    print(
        f"[model] class={model.__class__.__module__}.{model.__class__.__name__} "
        f"parameters={parameter_count} load_seconds={load_seconds:.6f}",
        flush=True,
    )
    print(f"[model] dtype_counts={dict(dtype_counts)} device_counts={dict(device_counts)}", flush=True)

    template = get_template(
        template_type=args.template_type,
        processor=processor,
        max_length=512,
        use_chat_template=True,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("transformers")
    sample = {"messages": [{"role": "user", "content": question}]}
    encoded = template.encode(sample)
    swift_input_ids = as_int_list(encoded["input_ids"], name="input_ids")
    swift_token_type_ids = as_int_list(encoded.get("token_type_ids"), name="token_type_ids")
    if len(swift_input_ids) != len(swift_token_type_ids):
        raise RuntimeError(
            f"Swift input/token-type length mismatch: input={len(swift_input_ids)} type={len(swift_token_type_ids)}"
        )
    if set(swift_token_type_ids) != {1}:
        raise RuntimeError(f"Inference prompt must have all token_type_ids=1, got {swift_token_type_ids}")

    native_prompt = TEMPLATE_PROMPTS[args.template_type].format(question=question)
    native_encoded = tokenizer(native_prompt, add_special_tokens=False)
    native_input_ids = as_int_list(native_encoded["input_ids"], name="native_input_ids")
    if swift_input_ids != native_input_ids:
        raise RuntimeError(
            f"Swift/native prompt token mismatch: swift={swift_input_ids} native={native_input_ids}"
        )
    decoded_prompt = tokenizer.decode(swift_input_ids, skip_special_tokens=False)
    if decoded_prompt != native_prompt:
        raise RuntimeError(f"Decoded Swift prompt mismatch: expected={native_prompt!r} actual={decoded_prompt!r}")

    prompt_length = len(swift_input_ids)
    if prompt_length + args.max_new_tokens > model.config.max_position_embeddings:
        raise RuntimeError(
            f"Generation exceeds context: prompt={prompt_length} max_new={args.max_new_tokens} "
            f"limit={model.config.max_position_embeddings}"
        )
    swift_inputs = {
        "input_ids": torch.tensor([swift_input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, prompt_length), dtype=torch.long, device=device),
        "token_type_ids": torch.tensor([swift_token_type_ids], dtype=torch.long, device=device),
    }
    native_inputs = {
        "input_ids": torch.tensor([native_input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, prompt_length), dtype=torch.long, device=device),
        "token_type_ids": torch.ones((1, prompt_length), dtype=torch.long, device=device),
    }
    print(f"[prompt] {native_prompt}", flush=True)
    print(f"[prompt] tokens={prompt_length} ids={swift_input_ids}", flush=True)

    with torch.inference_mode():
        swift_prefill = model(**swift_inputs, use_cache=False, logits_to_keep=1).logits
        native_prefill = model(**native_inputs, use_cache=False, logits_to_keep=1).logits
    if swift_prefill.shape != (1, 1, model.config.vocab_size):
        raise RuntimeError(f"Unexpected prefill logits shape: {tuple(swift_prefill.shape)}")
    if not bool(torch.isfinite(swift_prefill).all().item()):
        raise RuntimeError("Swift prefill logits contain NaN or Inf")
    prefill_max_abs_diff = float((swift_prefill.float() - native_prefill.float()).abs().max().item())
    if prefill_max_abs_diff > 1e-6:
        raise RuntimeError(f"Swift/native prefill logits differ: max_abs_diff={prefill_max_abs_diff}")
    print(
        f"[prefill] shape={tuple(swift_prefill.shape)} finite=True "
        f"swift_native_max_abs_diff={prefill_max_abs_diff}",
        flush=True,
    )

    forward_calls: list[dict[str, Any]] = []

    def capture_forward(_module, _args, kwargs):
        if len(forward_calls) >= args.max_new_tokens + 2:
            return
        forward_calls.append(
            {
                "input_ids": tensor_summary(kwargs.get("input_ids")),
                "inputs_embeds": tensor_summary(kwargs.get("inputs_embeds")),
                "attention_mask": tensor_summary(kwargs.get("attention_mask")),
                "token_type_ids": tensor_summary(kwargs.get("token_type_ids")),
                "has_past_key_values": kwargs.get("past_key_values") is not None,
            }
        )

    hook = model.register_forward_pre_hook(capture_forward, with_kwargs=True)
    torch.cuda.synchronize(device_index)
    generation_started = time.perf_counter()
    try:
        with torch.inference_mode():
            swift_output_ids = model.generate(
                **swift_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
            )
    finally:
        hook.remove()
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generation_started

    if not forward_calls:
        raise RuntimeError("Forward hook captured no model.generate calls")
    first_token_types = forward_calls[0]["token_type_ids"]
    if first_token_types is None or first_token_types.get("shape") != [1, prompt_length]:
        raise RuntimeError(f"Generate did not forward the full PrefixLM token_type_ids: {first_token_types}")
    if first_token_types.get("unique") != [1]:
        raise RuntimeError(f"Generate prefill token_type_ids are not all one: {first_token_types}")

    with torch.inference_mode():
        native_output_ids = model.generate(
            **native_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )
    if not torch.equal(swift_output_ids, native_output_ids):
        raise RuntimeError(
            "Deterministic Swift/native generation mismatch: "
            f"swift={swift_output_ids[0].tolist()} native={native_output_ids[0].tolist()}"
        )
    if not torch.equal(swift_output_ids[:, :prompt_length], swift_inputs["input_ids"]):
        raise RuntimeError("Generated sequence does not preserve the Swift prompt prefix")

    generated_ids = swift_output_ids[0, prompt_length:]
    if generated_ids.numel() == 0:
        raise RuntimeError("HRM generated zero new tokens")
    generated_raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    generated_clean = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stopped_on_eos = int(generated_ids[-1].item()) == int(model.config.eos_token_id)
    tokens_per_second = float(generated_ids.numel()) / generation_seconds
    print(f"[generated-raw] {generated_raw}", flush=True)
    print(f"[generated-clean] {generated_clean}", flush=True)
    print(
        f"[generation] tokens={generated_ids.numel()} seconds={generation_seconds:.6f} "
        f"tokens_per_second={tokens_per_second:.4f} stopped_on_eos={stopped_on_eos}",
        flush=True,
    )
    print(f"[forward-audit] calls={len(forward_calls)} first={json.dumps(forward_calls[0], ensure_ascii=False)}", flush=True)

    report = {
        "status": "ok",
        "packages": {
            "ms-swift": version("ms-swift"),
            "transformers": version("transformers"),
            "torch": version("torch"),
        },
        "model_path": str(model_path),
        "plugin_path": str(plugin_path),
        "settings": {
            "model_type": MODEL_TYPE,
            "template_type": args.template_type,
            "question": question,
            "max_new_tokens": args.max_new_tokens,
            "dtype": "torch.bfloat16",
            "attention_implementation": attention_implementation,
            "do_sample": False,
        },
        "model": {
            "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "parameter_count": parameter_count,
            "dtype_counts": dict(dtype_counts),
            "device_counts": dict(device_counts),
            "prefix_lm": bool(model.config.prefix_lm),
            "load_seconds": load_seconds,
        },
        "tokenizer": {
            "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
            "vocab_size": len(tokenizer),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "prompt": {
            "sample": sample,
            "text": native_prompt,
            "decoded": decoded_prompt,
            "swift_input_ids": swift_input_ids,
            "native_input_ids": native_input_ids,
            "token_type_ids": swift_token_type_ids,
            "swift_native_ids_equal": swift_input_ids == native_input_ids,
        },
        "prefill": {
            "logits_shape": list(swift_prefill.shape),
            "finite": bool(torch.isfinite(swift_prefill).all().item()),
            "swift_native_max_abs_diff": prefill_max_abs_diff,
        },
        "generation": {
            "swift_native_ids_equal": torch.equal(swift_output_ids, native_output_ids),
            "output_ids": swift_output_ids[0].tolist(),
            "generated_ids": generated_ids.tolist(),
            "generated_raw": generated_raw,
            "generated_clean": generated_clean,
            "generated_tokens": int(generated_ids.numel()),
            "stopped_on_eos": stopped_on_eos,
            "seconds": generation_seconds,
            "tokens_per_second": tokens_per_second,
            "forward_calls": forward_calls,
        },
        "cuda_memory": {
            "device_index": device_index,
            "device_name": properties.name,
            "allocated_gib": torch.cuda.memory_allocated(device_index) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(device_index) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / (1024**3),
        },
    }
    atomic_write_json(args.output_report, report)
    print(f"[memory] {json.dumps(report['cuda_memory'], ensure_ascii=False)}", flush=True)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
