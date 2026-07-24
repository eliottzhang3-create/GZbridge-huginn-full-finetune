#!/usr/bin/env python3
"""Run deterministic native-Transformers generation with the HRM PrefixLM format."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text"

# Replace this list with the seven evaluation questions. The model is loaded
# once and the questions are generated sequentially to keep memory predictable.
QUESTIONS = [
    "What is 1 + 1?",
]

CONDITION_TOKENS = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", default="synth,cot")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def condition_prefix(condition: str) -> str:
    tags = [tag.strip() for tag in condition.split(",") if tag.strip()]
    if not tags:
        raise ValueError("At least one HRM condition tag is required")
    unknown = [tag for tag in tags if tag not in CONDITION_TOKENS]
    if unknown:
        raise ValueError(f"Unknown HRM condition tags: {unknown}; valid={sorted(CONDITION_TOKENS)}")
    return "".join(CONDITION_TOKENS[tag] for tag in tags)


def build_prompt(question: str, condition: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")
    return f"<|im_start|>{condition_prefix(condition)}{question}<|im_end|>"


def main() -> None:
    args = parse_args()
    if not QUESTIONS:
        raise RuntimeError("QUESTIONS is empty")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM-Text generation requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path.resolve()
    required_files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing_files = [name for name in required_files if not (model_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Incomplete HRM-Text snapshot at {model_path}: missing={missing_files}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This generation script requires a CUDA device, got {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    device_properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    print("========== HRM-TEXT PREFIXLM GENERATION ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[path] model={model_path}", flush=True)
    print(
        f"[cuda] device={device} index={device_index} name={device_properties.name!r} "
        f"total_gib={device_properties.total_memory / (1024**3):.3f}",
        flush=True,
    )
    print(
        f"[generation-config] condition={args.condition!r} max_new_tokens={args.max_new_tokens} "
        "do_sample=False attention=sdpa dtype=bfloat16",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    print("[stage] load=AutoModelForCausalLM", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
    ).eval()
    torch.cuda.synchronize(device_index)

    if model.__class__.__name__ != "HrmTextForCausalLM":
        raise RuntimeError(f"Unexpected model class: {model.__class__.__module__}.{model.__class__.__name__}")
    if not bool(getattr(model.config, "prefix_lm", False)):
        raise RuntimeError("Loaded HRM-Text config does not enable PrefixLM")
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError(
            f"Expected SDPA attention, got {getattr(model.config, '_attn_implementation', None)!r}"
        )
    if tokenizer.eos_token_id != model.config.eos_token_id:
        raise RuntimeError(
            f"Tokenizer/model EOS mismatch: tokenizer={tokenizer.eos_token_id}, model={model.config.eos_token_id}"
        )

    results: list[dict[str, Any]] = []
    for question_index, question in enumerate(QUESTIONS, start=1):
        prompt = build_prompt(question, args.condition)
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        token_type_ids = torch.ones_like(input_ids, device=device)
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens + args.max_new_tokens > model.config.max_position_embeddings:
            raise RuntimeError(
                f"Question {question_index} exceeds context: prompt={prompt_tokens}, "
                f"max_new={args.max_new_tokens}, limit={model.config.max_position_embeddings}"
            )
        if not torch.all(token_type_ids == 1):
            raise RuntimeError("PrefixLM token_type_ids must mark every prompt token as prefix")

        print(f"\n========== QUESTION {question_index}/{len(QUESTIONS)} ==========", flush=True)
        print(f"[question] {question}", flush=True)
        print(f"[prompt] {prompt}", flush=True)
        print(
            f"[tokens] prompt_tokens={prompt_tokens} token_type_ids_unique="
            f"{torch.unique(token_type_ids).tolist()}",
            flush=True,
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        with torch.inference_mode():
            prefill = model(**model_inputs, use_cache=False, logits_to_keep=1)
        if prefill.logits.shape[:2] != (1, 1) or not torch.isfinite(prefill.logits).all():
            raise RuntimeError(
                f"Question {question_index} prefill logits invalid: shape={tuple(prefill.logits.shape)}"
            )
        print(
            f"[prefill] logits_shape={tuple(prefill.logits.shape)} "
            f"finite={bool(torch.isfinite(prefill.logits).all().item())}",
            flush=True,
        )

        torch.cuda.synchronize(device_index)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
            )
        torch.cuda.synchronize(device_index)
        elapsed_seconds = time.perf_counter() - started

        generated_ids = output_ids[0, prompt_tokens:]
        generated_tokens = int(generated_ids.numel())
        if generated_tokens == 0:
            raise RuntimeError(f"Question {question_index} generated zero tokens")
        stopped_on_eos = int(generated_ids[-1].item()) == int(model.config.eos_token_id)
        generated_raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
        generated_clean = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_raw = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        tokens_per_second = generated_tokens / elapsed_seconds if elapsed_seconds > 0 else None

        print(f"[generated-raw] {generated_raw}", flush=True)
        print(f"[generated-clean] {generated_clean}", flush=True)
        print(
            f"[timing] elapsed_seconds={elapsed_seconds:.6f} generated_tokens={generated_tokens} "
            f"tokens_per_second={tokens_per_second:.4f} stopped_on_eos={stopped_on_eos}",
            flush=True,
        )

        results.append(
            {
                "index": question_index,
                "question": question,
                "condition": args.condition,
                "condition_prefix": condition_prefix(args.condition),
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "token_type_ids_unique": torch.unique(token_type_ids).tolist(),
                "prefill_logits_shape": list(prefill.logits.shape),
                "prefill_logits_finite": bool(torch.isfinite(prefill.logits).all().item()),
                "generated_tokens": generated_tokens,
                "stopped_on_eos": stopped_on_eos,
                "elapsed_seconds": elapsed_seconds,
                "tokens_per_second": tokens_per_second,
                "generated_raw": generated_raw,
                "generated_clean": generated_clean,
                "full_raw": full_raw,
            }
        )

    report = {
        "status": "ok",
        "model_path": str(model_path),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "dtype": "torch.bfloat16",
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "prefix_lm": bool(model.config.prefix_lm),
        "condition": args.condition,
        "max_new_tokens": args.max_new_tokens,
        "question_count": len(QUESTIONS),
        "tokenizer": {
            "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "results": results,
        "cuda_memory": {
            "device_index": device_index,
            "device_name": device_properties.name,
            "allocated_gib": torch.cuda.memory_allocated(device_index) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(device_index) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / (1024**3),
        },
    }
    atomic_write_json(args.output_report, report)
    print(f"\n[memory] {json.dumps(report['cuda_memory'], ensure_ascii=False)}", flush=True)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
