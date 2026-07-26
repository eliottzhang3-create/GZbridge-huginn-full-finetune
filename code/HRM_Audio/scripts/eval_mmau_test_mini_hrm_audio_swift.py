#!/usr/bin/env python3
"""Evaluate HRM-Text audio Swift checkpoints on MMAU test-mini.

This evaluator is intentionally independent from the Huginn evaluation line.
For every MMAU item it computes the mean per-token conditional log-probability
of each complete answer choice under the HRM audio wrapper, then selects the
highest-scoring choice. The audio prefix is encoded once per choice prefill;
subsequent candidate tokens use the native HRM cache path.

The evaluator is resumable. Each checkpoint gets its own output directory,
run_config.json, append-only JSONL result file, prediction file, and summary.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_DATASET_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU test_mini/test_mini.parquet"
DEFAULT_WRAPPER_MODEL_PATH = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/hrm-text-audio-v1"
)
DEFAULT_PLUGIN_PATH = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/HRM_Audio/plugins/hrm_text_audio_swift.py"
)
DEFAULT_OUTPUT_ROOT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/mmau_test_mini"
)
DEFAULT_CHECKPOINTS = [
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/"
    "audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-2802",
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/hrm_text/"
    "audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-5604",
]

MODEL_TYPE = "hrm_text_audio_whisper"
TEMPLATE_CONTRACT = "<|im_start|><|object_ref_start|>{query}<|im_end|>"
PROMPT_VERSION = "mmau_hrm_audio_direct_v1"
SCORING_VERSION = "mean_per_token_conditional_logprob_v1"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 30.0
EXPECTED_AUDIO_FEATURE_SHAPE = (1, 80, 3000)


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in (None, "") else int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint path; repeat for multiple checkpoints. Defaults to the two formal HRM checkpoints.",
    )
    parser.add_argument("--dataset-path", default=os.environ.get("MMAU_TEST_MINI_PATH", DEFAULT_DATASET_PATH))
    parser.add_argument("--plugin-path", default=os.environ.get("MMAU_PLUGIN_PATH", DEFAULT_PLUGIN_PATH))
    parser.add_argument(
        "--wrapper-model-path",
        default=os.environ.get("HRM_AUDIO_WRAPPER_MODEL_PATH", DEFAULT_WRAPPER_MODEL_PATH),
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("MMAU_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT),
        help="Root directory; each checkpoint is written below its own deterministic subdirectory.",
    )
    parser.add_argument("--start-offset", type=int, default=int(os.environ.get("MMAU_START_OFFSET", "0")))
    parser.add_argument("--max-samples", type=int, default=_optional_int_env("MMAU_MAX_SAMPLES"))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("MMAU_LOG_EVERY", "10")))
    parser.add_argument("--device", default=os.environ.get("MMAU_DEVICE", "cuda:0"))
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=float(os.environ.get("MMAU_MAX_AUDIO_SECONDS", str(DEFAULT_MAX_AUDIO_SECONDS))),
    )
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--expected-lora-rank", type=int, default=16)
    parser.add_argument("--expected-lora-alpha", type=int, default=32)
    parser.add_argument("--expected-lora-dropout", type=float, default=0.0)
    return parser.parse_args()


def import_plugin(path: Path) -> Any:
    """Load the HRM-only Swift plugin exactly once in this process."""
    module_name = "hrm_audio_mmau_eval_swift_plugin"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    if not path.is_file():
        raise FileNotFoundError(f"HRM audio Swift plugin is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import HRM audio Swift plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(
    checkpoint: Path,
    wrapper_model_path: Path,
    plugin_path: Path,
    device: torch.device,
    expected_lora_rank: int,
    expected_lora_alpha: int,
    expected_lora_dropout: float,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Load base + aligner sidecar + Swift LoRA using the verified HRM route."""
    if device.type != "cuda":
        raise ValueError(f"HRM audio MMAU evaluation requires CUDA, got {device}")
    required = [
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
        checkpoint / "vit.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is incomplete: {missing}")

    import_plugin(plugin_path)
    from reload_hrm_audio_swift_checkpoint import (
        audit_adapter_state,
        audit_aligner_state,
        force_expected_policy,
        load_aligner_sidecar,
    )

    try:
        from swift import get_model_processor
    except ImportError:
        from swift.model import get_model_processor
    from swift.tuners import Swift

    adapter_config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    expected_config = {
        "r": expected_lora_rank,
        "lora_alpha": expected_lora_alpha,
        "lora_dropout": expected_lora_dropout,
    }
    config_mismatches = {}
    for key, expected in expected_config.items():
        actual = adapter_config.get(key)
        if isinstance(expected, float):
            equal = actual is not None and abs(float(actual) - expected) <= 1e-12
        else:
            equal = actual == expected
        if not equal:
            config_mismatches[key] = {"expected": expected, "actual": actual}
    if config_mismatches:
        raise RuntimeError(f"LoRA adapter config mismatch: {config_mismatches}")

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    base_model, processor = get_model_processor(
        str(wrapper_model_path),
        model_type=MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    if base_model is None or processor is None:
        raise RuntimeError("Swift HRM audio base model or processor is None")

    # Training stores aligner weights in vit.safetensors and LoRA weights in
    # adapter_model.safetensors. Restore them in the same order as the
    # validated HRM fresh-process reload audit.
    aligner_load_report = load_aligner_sidecar(base_model, checkpoint / "vit.safetensors")
    model = Swift.from_pretrained(base_model, str(checkpoint), is_trainable=False)
    wrapper = force_expected_policy(model)
    model.config.use_cache = True
    wrapper.config.use_cache = True
    if hasattr(wrapper, "model") and hasattr(wrapper.model, "config"):
        wrapper.model.config.use_cache = True
    model.eval()
    wrapper.eval()

    adapter_report = audit_adapter_state(model, checkpoint / "adapter_model.safetensors")
    aligner_report = audit_aligner_state(model, checkpoint / "vit.safetensors")
    wrapper.audio_encoder.requires_grad_(False)
    runtime_report = {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "wrapper_class": f"{type(wrapper).__module__}.{type(wrapper).__qualname__}",
        "device": str(device),
        "dtype": "torch.bfloat16",
        "attention": "sdpa",
        "recurrence": {
            "H_cycles": int(getattr(wrapper.config, "H_cycles", -1)),
            "L_cycles": int(getattr(wrapper.config, "L_cycles", -1)),
            "L_bp_cycles": list(getattr(wrapper.config, "L_bp_cycles", [])),
        },
        "audio_prefix_length": int(getattr(wrapper, "audio_prefix_length", -1)),
        "aligner_load": aligner_load_report,
        "adapter_exact": adapter_report,
        "aligner_exact": aligner_report,
        "adapter_config": adapter_config,
        "expected_policy": {
            "audio_encoder_frozen": not any(
                parameter.requires_grad for parameter in wrapper.audio_encoder.parameters()
            ),
            "text_backbone_frozen_except_lora": True,
            "aligner_restored": True,
            "lora_rank": expected_lora_rank,
            "lora_alpha": expected_lora_alpha,
            "lora_dropout": expected_lora_dropout,
        },
    }
    return model, processor, runtime_report


def iter_rows(parquet_path: Path, start_offset: int, end_offset: int) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            if row_index >= end_offset:
                return
            if row_index >= start_offset:
                yield row_index, row
            row_index += 1


def parse_metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("other_attributes")
    if isinstance(raw, str):
        metadata = json.loads(raw)
    elif isinstance(raw, dict):
        metadata = dict(raw)
    else:
        raise TypeError(f"MMAU other_attributes must be JSON text or object, got {type(raw)}")
    if not isinstance(metadata, dict):
        raise TypeError("MMAU other_attributes must decode to an object")
    sample_id = metadata.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"MMAU row has no valid metadata.id: {metadata}")
    return metadata


def extract_audio_bytes(row: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    context = row.get("context")
    if not isinstance(context, dict):
        raise TypeError("MMAU context must be a mapping containing embedded audio bytes")
    payload = context.get("bytes")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"MMAU context.bytes must be bytes, got {type(payload)} for {metadata['id']}")
    return bytes(payload)


def decode_audio_with_ffmpeg(audio_bytes: bytes, source_label: str, sample_rate: int) -> torch.Tensor:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to decode embedded MMAU audio bytes")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for MMAU audio {source_label}: {error}") from exc
    if not completed.stdout:
        raise RuntimeError(f"ffmpeg produced empty audio for MMAU sample {source_label}")
    waveform = torch.frombuffer(bytearray(completed.stdout), dtype=torch.float32).clone()
    if waveform.numel() == 0:
        raise RuntimeError(f"Decoded MMAU audio is empty for {source_label}")
    if not bool(torch.isfinite(waveform).all().item()):
        raise RuntimeError(f"Decoded MMAU audio contains NaN/Inf for {source_label}")
    return waveform


def prepare_audio_features(
    processor: Any,
    audio_bytes: bytes,
    source_label: str,
    device: torch.device,
    max_audio_seconds: float,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    if max_audio_seconds <= 0:
        raise ValueError(f"max_audio_seconds must be positive, got {max_audio_seconds}")
    waveform = decode_audio_with_ffmpeg(audio_bytes, source_label, DEFAULT_SAMPLE_RATE)
    max_samples = int(DEFAULT_SAMPLE_RATE * max_audio_seconds)
    waveform = waveform[:max_samples].contiguous()
    feature_extractor = processor.feature_extractor
    features = feature_extractor(
        waveform.cpu().numpy(),
        sampling_rate=DEFAULT_SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=max_samples,
        truncation=True,
    )["input_features"]
    if tuple(features.shape) != EXPECTED_AUDIO_FEATURE_SHAPE:
        raise RuntimeError(
            f"Unexpected Whisper feature shape for {source_label}: "
            f"expected={EXPECTED_AUDIO_FEATURE_SHAPE} actual={tuple(features.shape)}"
        )
    features = features.to(device=device, dtype=torch.bfloat16)
    return features, waveform.numel() / DEFAULT_SAMPLE_RATE, {
        "decoded_samples": int(waveform.numel()),
        "decoded_seconds": waveform.numel() / DEFAULT_SAMPLE_RATE,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
    }


def build_prompt(instruction: str, choices: list[str]) -> str:
    choices_text = "\n".join(choices)
    query = (
        "Listen to the audio and answer the multiple-choice question. "
        "Answer with exactly one complete option from the choices.\n\n"
        f"Question: {instruction}\n"
        f"Choices:\n{choices_text}\n"
        "Answer:"
    )
    return f"<|im_start|><|object_ref_start|>{query}<|im_end|>"


def tokenize_candidate(tokenizer: Any, prompt: str, choice: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    full_ids = tokenizer(prompt + choice, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    prompt_length = int(prompt_ids.shape[0])
    if full_ids.shape[0] <= prompt_length or not torch.equal(full_ids[:prompt_length], prompt_ids):
        raise RuntimeError(f"Tokenizer changed the HRM MMAU prompt boundary for choice {choice!r}")
    return prompt_ids, full_ids[prompt_length:]


def cache_length(cache: Any) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        raise RuntimeError(f"HRM cache does not expose get_seq_length(): {type(cache)}")
    return int(getter())


def score_choice(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    audio_features: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    if candidate_ids.numel() == 0:
        raise ValueError("MMAU answer choice tokenization produced an empty candidate")
    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    candidate_ids = candidate_ids.to(device=device, dtype=torch.long)
    text_length = int(prompt_ids.shape[0])
    text_attention = torch.ones((1, text_length), dtype=torch.long, device=device)
    prompt_types = torch.ones((1, text_length), dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=prompt_ids.unsqueeze(0),
            attention_mask=text_attention,
            token_type_ids=prompt_types,
            audio_input_features=audio_features,
            use_cache=True,
            logits_to_keep=1,
        )
        if outputs.logits is None or outputs.past_key_values is None:
            raise RuntimeError("HRM audio prefill did not return logits and past_key_values")
        cache = outputs.past_key_values
        prefill_cache_length = cache_length(cache)
        expected_cache_length = int(getattr(model, "audio_prefix_length", 34)) + text_length
        if prefill_cache_length != expected_cache_length:
            raise RuntimeError(
                f"HRM audio prefill cache length mismatch: expected={expected_cache_length} "
                f"actual={prefill_cache_length}"
            )
        token_logprobs: list[float] = []
        for index, token_id in enumerate(candidate_ids.tolist()):
            log_probs = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
            token_logprobs.append(float(log_probs[0, int(token_id)].item()))
            if index + 1 == candidate_ids.numel():
                break
            current_cache_length = cache_length(cache)
            token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
            outputs = model(
                input_ids=token,
                attention_mask=torch.ones(
                    (1, current_cache_length + 1), dtype=torch.long, device=device
                ),
                position_ids=torch.tensor([[current_cache_length]], dtype=torch.long, device=device),
                token_type_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
                past_key_values=cache,
                cache_position=torch.tensor([current_cache_length], dtype=torch.long, device=device),
                use_cache=True,
                logits_to_keep=1,
            )
            if outputs.logits is None or outputs.past_key_values is None:
                raise RuntimeError("HRM audio cached candidate decode did not return logits and cache")
            cache = outputs.past_key_values
    total_logprob = sum(token_logprobs)
    return {
        "token_count": len(token_logprobs),
        "total_logprob": total_logprob,
        "mean_logprob": total_logprob / len(token_logprobs),
        "prefill_cache_length": prefill_cache_length,
        "final_cache_length": cache_length(cache),
    }


def evaluate_row(
    row: dict[str, Any],
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    max_audio_seconds: float,
) -> dict[str, Any]:
    instruction = row.get("instruction")
    choices = row.get("choices")
    answer = row.get("answer")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("MMAU instruction is empty")
    if not isinstance(choices, list) or len(choices) < 2 or not all(
        isinstance(choice, str) and choice.strip() for choice in choices
    ):
        raise ValueError("MMAU choices must contain at least two non-empty strings")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("MMAU answer is empty")

    metadata = parse_metadata(row)
    audio_bytes = extract_audio_bytes(row, metadata)
    audio_features, used_seconds, audio_report = prepare_audio_features(
        processor,
        audio_bytes,
        str(metadata["id"]),
        device,
        max_audio_seconds,
    )
    prompt = build_prompt(instruction, choices)
    tokenizer = processor.tokenizer
    prompt_ids, _ = tokenize_candidate(tokenizer, prompt, choices[0])
    choice_scores = []
    for choice in choices:
        _, candidate_ids = tokenize_candidate(tokenizer, prompt, choice)
        score = score_choice(model, prompt_ids, candidate_ids, audio_features, device)
        choice_scores.append({"choice": choice, **score})
    predicted_index = max(range(len(choice_scores)), key=lambda index: choice_scores[index]["mean_logprob"])
    prediction = choices[predicted_index]
    return {
        "metadata": metadata,
        "instruction": instruction,
        "choices": choices,
        "answer": answer,
        "prediction": prediction,
        "prediction_index": predicted_index,
        "correct_exact_choice": prediction == answer,
        "choice_scores": choice_scores,
        "prompt": prompt,
        "prompt_token_count": int(prompt_ids.numel()),
        "audio": {"embedded_bytes": len(audio_bytes), "used_seconds": used_seconds, **audio_report},
    }


def normalize_tokens(value: Any) -> set[str]:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return set(re.findall(r"\b\w+\b", str(value).lower()))


def official_string_match(answer: str, prediction: str, choices: list[str]) -> bool:
    prediction_tokens = normalize_tokens(prediction)
    answer_tokens = normalize_tokens(answer)
    if not prediction_tokens:
        return False
    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = normalize_tokens(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(incorrect_tokens)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    results = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"MMAU result at {path}:{line_number} is not an object")
        results.append(value)
    return results


def append_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def metric_bucket(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, int | float]]:
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for result in results:
        value = result.get("metadata", {}).get(key, "unknown")
        buckets[str(value)][0] += int(bool(result["official_match"]))
        buckets[str(value)][1] += 1
    return {
        value: {"correct": correct, "total": total, "accuracy": correct / total}
        for value, (correct, total) in sorted(buckets.items())
    }


def checkpoint_slug(checkpoint: Path) -> str:
    raw = f"{checkpoint.parent.name}_{checkpoint.name}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def dataset_size(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def prepare_output(
    output_dir: Path,
    *,
    checkpoint: Path,
    dataset_path: Path,
    plugin_path: Path,
    wrapper_model_path: Path,
    start_offset: int,
    end_offset: int,
    max_audio_seconds: float,
    expected_lora_rank: int,
    expected_lora_alpha: int,
    expected_lora_dropout: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "evaluator": "HRM_Audio/scripts/eval_mmau_test_mini_hrm_audio_swift.py",
        "evaluation_version": "2026-07-26.hrmaudio.mmau.v1",
        "checkpoint": str(checkpoint),
        "dataset_path": str(dataset_path),
        "dataset_size": dataset_size(dataset_path),
        "plugin_path": str(plugin_path),
        "wrapper_model_path": str(wrapper_model_path),
        "requested_range": [start_offset, end_offset],
        "max_audio_seconds": max_audio_seconds,
        "model_type": MODEL_TYPE,
        "template_contract": TEMPLATE_CONTRACT,
        "prompt_version": PROMPT_VERSION,
        "scoring_version": SCORING_VERSION,
        "audio_decode": "ffmpeg_pcm_f32le_mono_16k",
        "expected_lora": {
            "rank": expected_lora_rank,
            "alpha": expected_lora_alpha,
            "dropout": expected_lora_dropout,
        },
    }
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                f"Existing HRM MMAU output has a different run_config: {config_path}. "
                "Choose a new output root."
            )
    else:
        config_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_dir / "mmau_test_mini_results.jsonl"


def write_summary(output_dir: Path, results: list[dict[str, Any]], runtime_report: dict[str, Any]) -> dict[str, Any]:
    exact_correct = sum(int(bool(result["correct_exact_choice"])) for result in results)
    official_correct = sum(int(bool(result["official_match"])) for result in results)
    predictions = [
        {
            "id": result["metadata"]["id"],
            "dataset_row_index": result.get("dataset_row_index"),
            "answer": result["answer"],
            "prediction": result["prediction"],
            "correct_exact_choice": result["correct_exact_choice"],
            "official_match": result["official_match"],
        }
        for result in results
    ]
    predictions_path = output_dir / "mmau_test_mini_predictions.json"
    summary = {
        "status": "OK",
        "completed_sample_count": len(results),
        "official_string_match_correct": official_correct,
        "official_string_match_accuracy": official_correct / len(results) if results else 0.0,
        "exact_choice_correct": exact_correct,
        "exact_choice_accuracy": exact_correct / len(results) if results else 0.0,
        "task_metrics": metric_bucket(results, "task"),
        "difficulty_metrics": metric_bucket(results, "difficulty"),
        "subcategory_metrics": metric_bucket(results, "sub-category"),
        "predictions_path": str(predictions_path),
        "runtime_restore": runtime_report,
    }
    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "mmau_test_mini_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path, dataset_path: Path) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"HRM checkpoint directory is missing: {checkpoint}")
    dataset_rows = dataset_size(dataset_path)
    end_offset = dataset_rows if args.max_samples is None else min(dataset_rows, args.start_offset + args.max_samples)
    if args.start_offset < 0 or args.start_offset >= end_offset:
        raise ValueError(f"Requested MMAU range is empty: [{args.start_offset}, {end_offset})")

    output_dir = Path(args.output_root) / checkpoint_slug(checkpoint)
    results_path = prepare_output(
        output_dir,
        checkpoint=checkpoint,
        dataset_path=dataset_path,
        plugin_path=Path(args.plugin_path).resolve(),
        wrapper_model_path=Path(args.wrapper_model_path).resolve(),
        start_offset=args.start_offset,
        end_offset=end_offset,
        max_audio_seconds=args.max_audio_seconds,
        expected_lora_rank=args.expected_lora_rank,
        expected_lora_alpha=args.expected_lora_alpha,
        expected_lora_dropout=args.expected_lora_dropout,
    )
    existing_results = read_jsonl(results_path)
    completed_ids = {result.get("metadata", {}).get("id") for result in existing_results}
    device = torch.device(args.device)
    print("========== HRM AUDIO MMAU TEST-MINI EVAL ==========", flush=True)
    print(f"[checkpoint] {checkpoint}", flush=True)
    print(f"[dataset] path={dataset_path} rows={dataset_rows} range=[{args.start_offset}, {end_offset})", flush=True)
    print(f"[output] {output_dir} resumed={len(completed_ids)}", flush=True)

    model, processor, runtime_report = load_checkpoint(
        checkpoint,
        Path(args.wrapper_model_path).resolve(),
        Path(args.plugin_path).resolve(),
        device,
        args.expected_lora_rank,
        args.expected_lora_alpha,
        args.expected_lora_dropout,
    )
    print(f"[restore] {json.dumps(runtime_report, ensure_ascii=False)}", flush=True)

    started = time.monotonic()
    processed_now = 0
    skipped = 0
    requested_count = end_offset - args.start_offset
    with results_path.open("a", encoding="utf-8") as handle:
        for row_index, row in iter_rows(dataset_path, args.start_offset, end_offset):
            metadata = parse_metadata(row)
            sample_id = metadata["id"]
            if sample_id in completed_ids:
                skipped += 1
                continue
            result = evaluate_row(row, model, processor, device, args.max_audio_seconds)
            result["dataset_row_index"] = row_index
            result["official_match"] = official_string_match(
                result["answer"], result["prediction"], result["choices"]
            )
            append_jsonl(handle, result)
            existing_results.append(result)
            completed_ids.add(sample_id)
            processed_now += 1
            if args.print_samples:
                print(
                    f"[sample] row={row_index} id={sample_id} prediction={result['prediction']!r} "
                    f"answer={result['answer']!r} exact={result['correct_exact_choice']} "
                    f"official={result['official_match']}",
                    flush=True,
                )
            if processed_now % args.log_every == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = processed_now / elapsed
                completed = len(existing_results)
                official = sum(int(bool(item.get("official_match"))) for item in existing_results)
                remaining = max(requested_count - skipped - processed_now, 0)
                print(
                    f"[progress] processed_now={processed_now} skipped={skipped} completed={completed} "
                    f"official_accuracy={official / completed:.4f} seconds_per_sample={1 / rate:.2f} "
                    f"eta_seconds={remaining / rate:.0f}",
                    flush=True,
                )

    summary = write_summary(output_dir, existing_results, runtime_report)
    print(
        f"[summary] checkpoint={checkpoint.name} completed={summary['completed_sample_count']} "
        f"official_accuracy={summary['official_string_match_accuracy']:.4f} "
        f"exact_accuracy={summary['exact_choice_accuracy']:.4f}",
        flush=True,
    )
    print(f"[summary] {output_dir / 'mmau_test_mini_summary.json'}", flush=True)
    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("HRM audio MMAU evaluation requires CUDA")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive when provided")
    if args.expected_lora_rank <= 0 or args.expected_lora_alpha <= 0:
        raise ValueError("expected LoRA rank and alpha must be positive")
    if not 0.0 <= args.expected_lora_dropout < 1.0:
        raise ValueError("expected LoRA dropout must be in [0, 1)")

    checkpoints = [Path(value).expanduser().resolve() for value in (args.checkpoint or DEFAULT_CHECKPOINTS)]
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    wrapper_model_path = Path(args.wrapper_model_path).expanduser().resolve()
    plugin_path = Path(args.plugin_path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"MMAU test-mini parquet is missing: {dataset_path}")
    if not wrapper_model_path.is_dir():
        raise FileNotFoundError(f"HRM audio wrapper model is missing: {wrapper_model_path}")
    if not plugin_path.is_file():
        raise FileNotFoundError(f"HRM audio Swift plugin is missing: {plugin_path}")
    if args.start_offset < 0:
        raise ValueError("start-offset must be non-negative")

    args.dataset_path = str(dataset_path)
    args.wrapper_model_path = str(wrapper_model_path)
    args.plugin_path = str(plugin_path)
    print("========== HRM AUDIO MMAU TEST-MINI EVAL PLAN ==========", flush=True)
    print(f"[checkpoints] {json.dumps([str(path) for path in checkpoints], ensure_ascii=False)}", flush=True)
    print(f"[dataset] {dataset_path}", flush=True)
    print(f"[wrapper] {wrapper_model_path}", flush=True)
    print(f"[plugin] {plugin_path}", flush=True)
    print(f"[device] {args.device}", flush=True)
    print(f"[range] start={args.start_offset} max_samples={args.max_samples}", flush=True)
    print("[scoring] mean per-token conditional log-probability over complete choices", flush=True)

    summaries = []
    for checkpoint in checkpoints:
        summaries.append(evaluate_checkpoint(args, checkpoint, dataset_path))
    print("========== HRM AUDIO MMAU TEST-MINI EVAL DONE ==========", flush=True)
    for checkpoint, summary in zip(checkpoints, summaries):
        print(
            f"[final] checkpoint={checkpoint.name} completed={summary['completed_sample_count']} "
            f"official_accuracy={summary['official_string_match_accuracy']:.4f} "
            f"exact_accuracy={summary['exact_choice_accuracy']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
