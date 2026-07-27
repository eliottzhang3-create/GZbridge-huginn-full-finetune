#!/usr/bin/env python3
"""Generate deterministic Clotho-v2 caption samples with HRM-Text audio Swift checkpoints.

This is the independent HRM-Text counterpart of the Huginn Clotho sample
generator. It uses the HRM Swift checkpoint restore path and the verified
manual audio-prefill/cache decoder. Generic multimodal generate is intentionally
not used because the 34-token HRM audio prefix must be included in cache
positions and response PrefixLM token types.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_mmau_test_mini_hrm_audio_swift import (  # noqa: E402
    cache_length,
    import_plugin,
    load_checkpoint,
)


DEFAULT_DATASET_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn"
DEFAULT_EVAL_MANIFEST = "test_expand.jsonl"
DEFAULT_WRAPPER_MODEL_PATH = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/hrm-text-audio-v1"
)
DEFAULT_PLUGIN_PATH = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/HRM_Audio/plugins/hrm_text_audio_swift.py"
)
DEFAULT_OUTPUT_ROOT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/"
    "clotho_caption_samples"
)
DEFAULT_CHECKPOINTS = [
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/"
    "audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-2802",
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/"
    "audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-5604",
]
PROMPT = "Listen to the audio and describe it."
PROMPT_VERSION = "hrm_audio_clotho_direct_v1"
GENERATION_PATH = "hrm_audio_manual_prefill_cache"
DEFAULT_SAMPLE_COUNT = 3
DEFAULT_SEED = 74
DEFAULT_MAX_NEW_TOKENS = 64
EXPECTED_AUDIO_PREFIX = 34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("HRM_CLOTHO_CAPTION_DATASET_DIR", DEFAULT_DATASET_DIR),
    )
    parser.add_argument(
        "--eval-manifest",
        default=os.environ.get("HRM_CLOTHO_CAPTION_MANIFEST", DEFAULT_EVAL_MANIFEST),
    )
    parser.add_argument(
        "--wrapper-model-path",
        default=os.environ.get("HRM_AUDIO_WRAPPER_MODEL_PATH", DEFAULT_WRAPPER_MODEL_PATH),
    )
    parser.add_argument(
        "--plugin-path",
        default=os.environ.get("HRM_CLOTHO_CAPTION_PLUGIN_PATH", DEFAULT_PLUGIN_PATH),
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("HRM_CLOTHO_CAPTION_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=int(os.environ.get("HRM_CLOTHO_CAPTION_SAMPLE_COUNT", str(DEFAULT_SAMPLE_COUNT))),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("HRM_CLOTHO_CAPTION_SEED", str(DEFAULT_SEED))),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(
            os.environ.get("HRM_CLOTHO_CAPTION_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))
        ),
    )
    parser.add_argument("--device", default=os.environ.get("HRM_CLOTHO_CAPTION_DEVICE", "cuda:0"))
    parser.add_argument("--expected-lora-rank", type=int, default=16)
    parser.add_argument("--expected-lora-alpha", type=int, default=32)
    parser.add_argument("--expected-lora-dropout", type=float, default=0.0)
    return parser.parse_args()


def as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def record_captions(record: dict[str, Any]) -> list[str]:
    captions: list[str] = []
    for key in ("references", "captions", "caption_list", "ref_captions", "caption", "text"):
        captions.extend(as_text_list(record.get(key)))
    if not captions and isinstance(record.get("messages"), list):
        captions.extend(
            message.get("content", "").strip()
            for message in record["messages"]
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        )
    return list(dict.fromkeys(captions))


def load_clotho_groups(dataset_dir: Path, manifest_name: str) -> list[tuple[Path, list[str]]]:
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Clotho manifest not found: {manifest_path}")
    grouped: dict[Path, list[str]] = defaultdict(list)
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError(f"{manifest_path}:{line_number} is not a JSON object")
        raw_path = record.get("audio_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{manifest_path}:{line_number} has no audio_path")
        audio_path = Path(raw_path)
        if not audio_path.is_absolute():
            audio_path = dataset_dir / audio_path
        captions = record_captions(record)
        if not captions:
            raise ValueError(f"{manifest_path}:{line_number} has no reference caption")
        grouped[audio_path].extend(captions)

    groups = [(path, list(dict.fromkeys(captions))) for path, captions in sorted(grouped.items())]
    for audio_path, _ in groups:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Clotho audio file not found: {audio_path}")
    if not groups:
        raise ValueError(f"No Clotho audio groups found in {manifest_path}")
    return groups


def find_hrm_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    matches = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "HrmTextAudioForConditionalGeneration"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one HRM audio wrapper, found {len(matches)}")
    return matches[0]


class GenerationExecutionAudit:
    """Verify that audio encoding occurs once and cached decode never re-encodes it."""

    def __init__(self, wrapper: torch.nn.Module):
        self.wrapper = wrapper
        self.audio_encoder_calls = 0
        self.compressor_calls = 0
        self.projector_calls = 0
        self.handles: list[Any] = []

    def reset(self) -> None:
        self.audio_encoder_calls = 0
        self.compressor_calls = 0
        self.projector_calls = 0

    def install(self) -> None:
        def count_audio_encoder(_module: Any, _args: Any, _kwargs: Any) -> None:
            self.audio_encoder_calls += 1

        def count_compressor(_module: Any, _args: Any, _kwargs: Any) -> None:
            self.compressor_calls += 1

        def count_projector(_module: Any, _args: Any, _kwargs: Any) -> None:
            self.projector_calls += 1

        self.handles = [
            self.wrapper.audio_encoder.register_forward_pre_hook(count_audio_encoder, with_kwargs=True),
            self.wrapper.temporal_compressor.register_forward_pre_hook(count_compressor, with_kwargs=True),
            self.wrapper.audio_projector.register_forward_pre_hook(count_projector, with_kwargs=True),
        ]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def report(self) -> dict[str, int]:
        return {
            "audio_encoder": self.audio_encoder_calls,
            "temporal_compressor": self.compressor_calls,
            "audio_projector": self.projector_calls,
        }


def prepare_audio_features(
    plugin: Any,
    processor: Any,
    audio_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    waveform = plugin.load_audio_16k(audio_path)
    feature_extractor = processor.feature_extractor
    sample_rate = int(feature_extractor.sampling_rate)
    features = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        max_length=sample_rate * 30,
        truncation=True,
    )["input_features"]
    if tuple(features.shape) != (1, 80, 3000):
        raise RuntimeError(
            f"Unexpected HRM Clotho Whisper feature shape: {tuple(features.shape)}"
        )
    return features.to(device=device, dtype=torch.bfloat16), len(waveform) / float(sample_rate)


def token_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return set()


def build_prompt() -> str:
    return f"<|im_start|><|object_ref_start|>{PROMPT}<|im_end|>"


def generate_caption(
    model: torch.nn.Module,
    wrapper: torch.nn.Module,
    processor: Any,
    plugin: Any,
    audio_path: Path,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    audio_features, audio_seconds = prepare_audio_features(plugin, processor, audio_path, device)
    prompt = build_prompt()
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = tokenized["attention_mask"].to(device=device, dtype=torch.long)
    token_type_ids = torch.ones_like(input_ids, dtype=torch.long, device=device)
    prompt_token_count = int(input_ids.shape[1])
    stop_token_ids = (
        token_ids(getattr(tokenizer, "eos_token_id", None))
        | token_ids(getattr(wrapper.config, "eos_token_id", None))
        | token_ids(getattr(model, "eos_token_id", None))
    )
    if not stop_token_ids:
        raise RuntimeError("HRM tokenizer/model exposes no EOS token id")

    capture = GenerationExecutionAudit(wrapper)
    capture.install()
    capture.reset()
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                audio_input_features=audio_features,
                use_cache=True,
                logits_to_keep=1,
            )
            if outputs.logits is None or outputs.past_key_values is None:
                raise RuntimeError("HRM audio prefill did not return logits and cache")
            cache = outputs.past_key_values
            prefill_cache_length = cache_length(cache)
            expected_cache_length = EXPECTED_AUDIO_PREFIX + prompt_token_count
            if prefill_cache_length != expected_cache_length:
                raise RuntimeError(
                    f"HRM Clotho prefill cache mismatch: expected={expected_cache_length} "
                    f"actual={prefill_cache_length}"
                )

            generated_ids: list[int] = []
            stop_reason = "max_new_tokens"
            for _ in range(max_new_tokens):
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                next_id = int(next_token.item())
                if next_id in stop_token_ids:
                    stop_reason = f"stop_token:{next_id}"
                    break
                generated_ids.append(next_id)
                if len(generated_ids) == max_new_tokens:
                    break

                current_cache_length = cache_length(cache)
                outputs = model(
                    input_ids=next_token,
                    attention_mask=torch.ones(
                        (1, current_cache_length + 1), dtype=torch.long, device=device
                    ),
                    position_ids=torch.tensor(
                        [[current_cache_length]], dtype=torch.long, device=device
                    ),
                    token_type_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
                    past_key_values=cache,
                    cache_position=torch.tensor(
                        [current_cache_length], dtype=torch.long, device=device
                    ),
                    use_cache=True,
                    logits_to_keep=1,
                )
                if outputs.logits is None or outputs.past_key_values is None:
                    raise RuntimeError("HRM cached Clotho decode did not return logits and cache")
                cache = outputs.past_key_values
    finally:
        execution_report = capture.report()
        capture.remove()

    if execution_report != {
        "audio_encoder": 1,
        "temporal_compressor": 1,
        "audio_projector": 1,
    }:
        raise RuntimeError(
            "HRM Clotho generation must encode audio exactly once; "
            f"execution={execution_report}"
        )
    generated_raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    generated_clean = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "prompt": prompt,
        "prompt_version": PROMPT_VERSION,
        "audio_seconds_after_truncation": audio_seconds,
        "audio_feature_shape": list(audio_features.shape),
        "audio_prefix_token_count": prefill_cache_length - prompt_token_count,
        "text_prompt_token_count": prompt_token_count,
        "prefill_cache_length": prefill_cache_length,
        "final_cache_length": cache_length(cache),
        "stop_token_ids": sorted(stop_token_ids),
        "stop_reason": stop_reason,
        "generated_token_count": len(generated_ids),
        "generated_token_ids": generated_ids,
        "generated_raw": generated_raw,
        "generated_caption": generated_clean,
        "execution_counts": execution_report,
    }


def checkpoint_slug(checkpoint: Path) -> str:
    return f"{checkpoint.parent.name}_{checkpoint.name}"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("HRM Clotho generation requires CUDA")
    if args.sample_count <= 0 or args.max_new_tokens <= 0:
        raise ValueError("sample-count and max-new-tokens must be positive")
    if args.expected_lora_rank <= 0 or args.expected_lora_alpha <= 0:
        raise ValueError("expected LoRA rank and alpha must be positive")
    if not 0.0 <= args.expected_lora_dropout < 1.0:
        raise ValueError("expected LoRA dropout must be in [0, 1)")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"HRM Clotho generation requires CUDA, got {device}")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    wrapper_model_path = Path(args.wrapper_model_path).expanduser().resolve()
    plugin_path = Path(args.plugin_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    checkpoints = [
        Path(value).expanduser().resolve() for value in (args.checkpoint or DEFAULT_CHECKPOINTS)
    ]
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Clotho dataset directory is missing: {dataset_dir}")
    if not wrapper_model_path.is_dir():
        raise FileNotFoundError(f"HRM audio wrapper directory is missing: {wrapper_model_path}")
    if not plugin_path.is_file():
        raise FileNotFoundError(f"HRM audio Swift plugin is missing: {plugin_path}")

    groups = load_clotho_groups(dataset_dir, args.eval_manifest)
    selected_indices = sorted(
        random.Random(args.seed).sample(range(len(groups)), min(args.sample_count, len(groups)))
    )
    selected_groups = [groups[index] for index in selected_indices]
    print("========== HRM AUDIO CLOTHO CAPTION GENERATION PLAN ==========", flush=True)
    print(f"[dataset_dir] {dataset_dir}", flush=True)
    print(f"[eval_manifest] {args.eval_manifest}", flush=True)
    print(f"[available_audio_groups] {len(groups)}", flush=True)
    print(f"[selected_indices] {selected_indices}", flush=True)
    print(f"[sample_count] {len(selected_groups)} seed={args.seed}", flush=True)
    print(f"[max_new_tokens] {args.max_new_tokens}", flush=True)
    print(f"[generation_path] {GENERATION_PATH}", flush=True)
    print(f"[checkpoints] {json.dumps([str(path) for path in checkpoints])}", flush=True)

    plugin = import_plugin(plugin_path)
    for checkpoint in checkpoints:
        output_dir = output_root / checkpoint_slug(checkpoint)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("========== HRM AUDIO CLOTHO CAPTION GENERATION ==========", flush=True)
        print(f"[checkpoint] {checkpoint}", flush=True)
        print(f"[output_dir] {output_dir}", flush=True)
        model, processor, restore = load_checkpoint(
            checkpoint,
            wrapper_model_path,
            plugin_path,
            device,
            args.expected_lora_rank,
            args.expected_lora_alpha,
            args.expected_lora_dropout,
        )
        wrapper = find_hrm_wrapper(model)
        if int(getattr(wrapper, "audio_prefix_length", -1)) != EXPECTED_AUDIO_PREFIX:
            raise RuntimeError(
                f"Unexpected HRM audio prefix length: {wrapper.audio_prefix_length}"
            )
        print(f"[restore] {json.dumps(restore, ensure_ascii=False)}", flush=True)

        samples: list[dict[str, Any]] = []
        for sample_number, (audio_path, references) in enumerate(selected_groups, start=1):
            generated = generate_caption(
                model,
                wrapper,
                processor,
                plugin,
                audio_path,
                args.max_new_tokens,
                device,
            )
            sample = {
                "sample_number": sample_number,
                "selected_group_index": selected_indices[sample_number - 1],
                "audio_path": str(audio_path),
                "reference_count": len(references),
                "references": references,
                **generated,
            }
            samples.append(sample)
            print(f"========== HRM SAMPLE {sample_number} ==========", flush=True)
            print(f"[audio] path={audio_path}", flush=True)
            print(
                f"[audio] seconds_after_truncation={generated['audio_seconds_after_truncation']:.3f}",
                flush=True,
            )
            print(
                "[generation] "
                f"prompt_tokens={generated['text_prompt_token_count']} "
                f"audio_prefix_tokens={generated['audio_prefix_token_count']} "
                f"cache={generated['prefill_cache_length']}->{generated['final_cache_length']} "
                f"stop_reason={generated['stop_reason']}",
                flush=True,
            )
            print(f"[generation] token_count={generated['generated_token_count']}", flush=True)
            print(f"[generation] caption={generated['generated_caption']}", flush=True)
            for reference_number, reference in enumerate(references, start=1):
                print(f"[reference {reference_number}] {reference}", flush=True)

        payload = {
            "checkpoint": str(checkpoint),
            "dataset_dir": str(dataset_dir),
            "eval_manifest": args.eval_manifest,
            "selected_indices": selected_indices,
            "sample_count": len(samples),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "generation_path": GENERATION_PATH,
            "prompt_version": PROMPT_VERSION,
            "restore": restore,
            "samples": samples,
        }
        output_path = output_dir / "clotho_caption_samples.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[output] {output_path}", flush=True)
        del model, processor, wrapper
        gc.collect()
        torch.cuda.empty_cache()
    print("========== HRM AUDIO CLOTHO CAPTION GENERATION DONE ==========", flush=True)


if __name__ == "__main__":
    main()
