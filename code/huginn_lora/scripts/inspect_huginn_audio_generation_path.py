#!/usr/bin/env python3
"""Validate Huginn audio prefill, RoPE lengths, and one manual cached decode step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from generate_clotho_caption_samples_swift import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_DIR,
    build_prompt,
    import_plugin,
    load_clotho_groups,
    load_generation_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--eval-manifest", default="test_expand.jsonl")
    parser.add_argument("--plugin-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--second-index", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


class PreludeShapeCapture:
    def __init__(self) -> None:
        self.hidden_shape: tuple[int, ...] | None = None
        self.freqs_shape: tuple[int, ...] | None = None

    def reset(self) -> None:
        self.hidden_shape = None
        self.freqs_shape = None

    def hook(self, _module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        self.hidden_shape = tuple(args[0].shape)
        self.freqs_shape = tuple(args[1].shape)


def prepare_inputs(plugin: Any, processor: Any, audio_path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    sample_rate = int(getattr(feature_extractor, "sampling_rate", plugin.DEFAULT_SAMPLE_RATE))
    waveform = plugin.load_audio_file(audio_path, sample_rate, plugin.DEFAULT_MAX_AUDIO_SECONDS)
    feature_inputs = feature_extractor([waveform], sampling_rate=sample_rate, return_tensors="pt")
    tokenized = tokenizer(build_prompt(plugin), return_tensors="pt", add_special_tokens=True)
    inputs = {
        "input_ids": tokenized["input_ids"].to(device),
        "attention_mask": tokenized["attention_mask"].to(device),
        "audio_input_features": feature_inputs["input_features"].to(device=device, dtype=torch.bfloat16),
    }
    metadata = {
        "audio_path": str(audio_path),
        "audio_seconds_after_truncation": len(waveform) / float(sample_rate),
        "text_prompt_token_count": int(inputs["input_ids"].shape[1]),
        "feature_shape": tuple(inputs["audio_input_features"].shape),
    }
    return inputs, metadata


def forward_prefill(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    capture: PreludeShapeCapture,
    use_cache: bool,
) -> tuple[Any, dict[str, Any]]:
    capture.reset()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=use_cache)
    logits = outputs.logits
    if logits is None:
        raise RuntimeError("Direct audio prefill returned logits=None")
    details = {
        "use_cache": use_cache,
        "logits_shape": tuple(logits.shape),
        "prelude_hidden_shape": capture.hidden_shape,
        "prelude_freqs_shape": capture.freqs_shape,
        "audio_prefix_token_count": int(logits.shape[1] - inputs["input_ids"].shape[1]),
        "cache_sequence_length": (
            int(outputs.past_key_values.get_seq_length()) if outputs.past_key_values is not None else None
        ),
    }
    return outputs, details


def manual_cached_token(
    model: torch.nn.Module,
    prefill_outputs: Any,
    capture: PreludeShapeCapture,
    device: torch.device,
) -> dict[str, Any]:
    cache = prefill_outputs.past_key_values
    if cache is None:
        raise RuntimeError("Cached prefill did not return past_key_values")
    cache_length_before = int(cache.get_seq_length())
    next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
    position = torch.full((1, 1), cache_length_before, device=device, dtype=torch.long)
    capture.reset()
    with torch.inference_mode():
        outputs = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            cache_position=position.squeeze(0),
        )
    if outputs.logits is None:
        raise RuntimeError("Manual cached token returned logits=None")
    cache_length_after = int(outputs.past_key_values.get_seq_length())
    return {
        "input_token_id": int(next_token.item()),
        "cache_length_before": cache_length_before,
        "cache_length_after": cache_length_after,
        "logits_shape": tuple(outputs.logits.shape),
        "prelude_hidden_shape": capture.hidden_shape,
        "prelude_freqs_shape": capture.freqs_shape,
    }


def logits_difference(first: torch.Tensor, second: torch.Tensor) -> dict[str, float | bool]:
    first_last = first[:, -1, :].float()
    second_last = second[:, -1, :].float()
    return {
        "allclose": bool(torch.allclose(first_last, second_last)),
        "max_abs_difference": float((first_last - second_last).abs().max().item()),
        "mean_abs_difference": float((first_last - second_last).abs().mean().item()),
        "cosine_similarity": float(F.cosine_similarity(first_last, second_last).item()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_clotho_groups(args.dataset_dir, args.eval_manifest)
    requested = [args.first_index, args.second_index]
    if any(index < 0 or index >= len(groups) for index in requested):
        raise ValueError(f"Requested indices {requested} are outside 0..{len(groups) - 1}")
    first_audio, first_references = groups[args.first_index]
    second_audio, second_references = groups[args.second_index]

    print("========== HUGINN AUDIO GENERATION PATH INSPECT ==========")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] first_index={args.first_index} second_index={args.second_index}")
    plugin = import_plugin(args.plugin_path) if args.plugin_path else import_plugin(
        "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
        "code/huginn_lora/plugins/huginn_audio_swift.py"
    )
    model, processor, restore = load_generation_model(plugin, args.checkpoint, device)
    print(f"[restore] {json.dumps(restore, ensure_ascii=False)}")

    capture = PreludeShapeCapture()
    hook_handle = model.transformer.prelude[0].register_forward_pre_hook(capture.hook)
    try:
        first_inputs, first_metadata = prepare_inputs(plugin, processor, first_audio, device)
        first_no_cache, first_no_cache_details = forward_prefill(model, first_inputs, capture, use_cache=False)
        print(f"[prefill-no-cache-first] metadata={json.dumps(first_metadata, ensure_ascii=False)}")
        print(f"[prefill-no-cache-first] details={json.dumps(first_no_cache_details, ensure_ascii=False)}")

        second_inputs, second_metadata = prepare_inputs(plugin, processor, second_audio, device)
        second_no_cache, second_no_cache_details = forward_prefill(model, second_inputs, capture, use_cache=False)
        difference = logits_difference(first_no_cache.logits, second_no_cache.logits)
        print(f"[prefill-no-cache-second] metadata={json.dumps(second_metadata, ensure_ascii=False)}")
        print(f"[prefill-no-cache-second] details={json.dumps(second_no_cache_details, ensure_ascii=False)}")
        print(f"[audio-logit-difference] {json.dumps(difference, ensure_ascii=False)}")

        cached_prefill, cached_prefill_details = forward_prefill(model, first_inputs, capture, use_cache=True)
        cached_token_details = manual_cached_token(model, cached_prefill, capture, device)
        print(f"[prefill-cache-first] details={json.dumps(cached_prefill_details, ensure_ascii=False)}")
        print(f"[manual-cached-token] details={json.dumps(cached_token_details, ensure_ascii=False)}")
    finally:
        hook_handle.remove()

    payload = {
        "checkpoint": args.checkpoint,
        "restore": restore,
        "first": {
            "metadata": first_metadata,
            "references": first_references,
            "prefill_no_cache": first_no_cache_details,
            "prefill_cache": cached_prefill_details,
            "manual_cached_token": cached_token_details,
        },
        "second": {
            "metadata": second_metadata,
            "references": second_references,
            "prefill_no_cache": second_no_cache_details,
        },
        "audio_logit_difference": difference,
    }
    output_path = output_dir / "huginn_audio_generation_path_inspect.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("========== HUGINN AUDIO GENERATION PATH INSPECT DONE ==========")
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
