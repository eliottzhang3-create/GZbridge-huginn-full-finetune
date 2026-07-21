#!/usr/bin/env python3
"""Run a small, exact-choice likelihood smoke evaluation on MMAU test-mini."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from generate_clotho_caption_samples_swift import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PLUGIN_PATH,
    import_plugin,
    load_generation_model,
)


DEFAULT_DATASET_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU test_mini/test_mini.parquet"
HUGINN_STOP_TOKEN_IDS = {65504, 65505, 65508}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--plugin-path", default=DEFAULT_PLUGIN_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None, help="Fixed Huginn recurrence count; default uses config.mean_recurrence.")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_rows(parquet_path: Path, offset: int, count: int) -> list[dict[str, Any]]:
    if offset < 0 or count <= 0:
        raise ValueError("sample_offset must be non-negative and sample_count must be positive")
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    seen = 0
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if seen >= offset:
                rows.append(row)
                if len(rows) == count:
                    return rows
            seen += 1
    if not rows:
        raise ValueError(f"No MMAU rows found at offset {offset}; dataset has {seen} rows")
    return rows


def parse_attributes(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("other_attributes")
    if not isinstance(raw, str):
        raise TypeError(f"MMAU other_attributes must be a JSON string, got {type(raw)}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("MMAU other_attributes JSON is not an object")
    return payload


def extract_embedded_audio(row: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    attributes = parse_attributes(row)
    sample_id = attributes.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("MMAU row has no valid id")
    context = row.get("context")
    if not isinstance(context, dict) or not isinstance(context.get("bytes"), bytes):
        raise TypeError("MMAU context must contain embedded audio bytes")
    wav_bytes = context["bytes"]
    return wav_bytes, {
        "id": sample_id,
        **attributes,
        "embedded_audio_bytes": len(wav_bytes),
        "embedded_audio_magic_hex": wav_bytes[:12].hex(),
    }


def build_prompt(plugin: Any, instruction: str, choices: list[str]) -> str:
    choices_text = "\n".join(choices)
    user_content = (
        "Listen to the audio and answer the multiple-choice question. "
        "Answer with exactly one complete option from the choices.\n\n"
        f"Question: {instruction}\n"
        f"Choices:\n{choices_text}\n"
        "Answer:"
    )
    return (
        "<|begin_header|>system<|end_header|>\n\n"
        f"{plugin.DEFAULT_SYSTEM_PROMPT}<|end_turn|>"
        "<|begin_header|>user<|end_header|>\n\n"
        f"{user_content}<|end_turn|>"
        "<|begin_header|>Huginn<|end_header|>\n\n"
    )


def tokenize_candidate(tokenizer: Any, prompt: str, choice: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    full_ids = tokenizer(prompt + choice, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    if full_ids.shape[0] <= prompt_ids.shape[0] or not torch.equal(full_ids[:prompt_ids.shape[0]], prompt_ids):
        raise RuntimeError(f"Tokenizer changed the prompt boundary for MMAU choice: {choice!r}")
    return prompt_ids, full_ids[prompt_ids.shape[0]:]


def prepare_audio_inputs(
    plugin: Any,
    processor: Any,
    audio_bytes: bytes,
    source_label: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    if getattr(plugin, "MODEL_TYPE", None) == "huginn_losatok_raven":
        waveform = plugin.decode_audio_bytes_16k(audio_bytes, source_label)
        values = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)
        return {
            "audio_input_values": values,
            "audio_attention_mask": torch.ones_like(values, dtype=torch.long),
        }, waveform.numel() / float(plugin.DEFAULT_SAMPLE_RATE)
    feature_extractor = processor.feature_extractor
    sample_rate = int(getattr(feature_extractor, "sampling_rate", plugin.DEFAULT_SAMPLE_RATE))
    waveform = plugin.decode_audio_with_ffmpeg_bytes(audio_bytes, source_label, sample_rate)
    waveform = plugin.trim_audio(waveform, sample_rate, plugin.DEFAULT_MAX_AUDIO_SECONDS)
    features = feature_extractor([waveform], sampling_rate=sample_rate, return_tensors="pt")["input_features"]
    return {"audio_input_features": features.to(device=device, dtype=torch.bfloat16)}, len(waveform) / float(sample_rate)


def score_choice(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    audio_inputs: dict[str, torch.Tensor],
    device: torch.device,
    num_steps: int | None = None,
) -> dict[str, Any]:
    if candidate_ids.numel() == 0:
        raise ValueError("Candidate token sequence is empty")
    attention_mask = torch.ones_like(prompt_ids, device=device)
    prefill_kwargs: dict[str, Any] = {
        "input_ids": prompt_ids.unsqueeze(0).to(device),
        "attention_mask": attention_mask.unsqueeze(0),
        "use_cache": True,
        **audio_inputs,
    }
    if num_steps is not None:
        prefill_kwargs["num_steps"] = num_steps
    with torch.inference_mode():
        outputs = model(**prefill_kwargs)
        if outputs.logits is None or outputs.past_key_values is None:
            raise RuntimeError("MMAU audio prefill did not return logits and cache")
        cache = outputs.past_key_values
        prefill_cache_length = int(cache.get_seq_length())
        token_logprobs: list[float] = []
        for token_index, token_id in enumerate(candidate_ids.tolist()):
            log_probs = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
            token_logprobs.append(float(log_probs[0, token_id].item()))
            if token_index + 1 == candidate_ids.numel():
                break
            token = torch.tensor([[token_id]], device=device, dtype=torch.long)
            cache_position = torch.tensor([cache.get_seq_length()], device=device, dtype=torch.long)
            decode_kwargs: dict[str, Any] = {
                "input_ids": token,
                "past_key_values": cache,
                "use_cache": True,
                "cache_position": cache_position,
            }
            if num_steps is not None:
                decode_kwargs["num_steps"] = num_steps
            outputs = model(**decode_kwargs)
            if outputs.logits is None or outputs.past_key_values is None:
                raise RuntimeError("MMAU cached candidate decode did not return logits and cache")
            cache = outputs.past_key_values
    total_logprob = sum(token_logprobs)
    return {
        "token_count": len(token_logprobs),
        "total_logprob": total_logprob,
        "mean_logprob": total_logprob / len(token_logprobs),
        "prefill_cache_length": prefill_cache_length,
    }


def evaluate_row(
    row: dict[str, Any],
    plugin: Any,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    num_steps: int | None = None,
) -> dict[str, Any]:
    instruction = row.get("instruction")
    choices = row.get("choices")
    answer = row.get("answer")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("MMAU instruction is empty")
    if not isinstance(choices, list) or not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise ValueError("MMAU choices are invalid")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("MMAU test-mini answer is empty")

    audio_bytes, metadata = extract_embedded_audio(row)
    audio_inputs, used_seconds = prepare_audio_inputs(plugin, processor, audio_bytes, metadata["id"], device)
    prompt = build_prompt(plugin, instruction, choices)
    tokenizer = processor.tokenizer
    prompt_ids, _ = tokenize_candidate(tokenizer, prompt, choices[0])
    choice_scores = []
    for choice in choices:
        _, candidate_ids = tokenize_candidate(tokenizer, prompt, choice)
        score = score_choice(model, prompt_ids, candidate_ids, audio_inputs, device, num_steps=num_steps)
        choice_scores.append({"choice": choice, **score})

    predicted_index = max(range(len(choice_scores)), key=lambda index: choice_scores[index]["mean_logprob"])
    prediction = choices[predicted_index]
    return {
        "metadata": metadata,
        "instruction": instruction,
        "choices": choices,
        "answer": answer,
        "prediction": prediction,
        "correct_exact_choice": prediction == answer,
        "audio_seconds_after_truncation": used_seconds,
        "prompt_token_count": int(prompt_ids.numel()),
        "choice_scores": choice_scores,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("num_steps must be positive when provided")
    parquet_path = Path(args.dataset_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"MMAU test-mini parquet not found: {parquet_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(parquet_path, args.sample_offset, args.sample_count)
    device = torch.device(args.device)

    print("========== MMAU TEST-MINI SWIFT SMOKE ==========")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] dataset_path={parquet_path}")
    print(f"[config] sample_offset={args.sample_offset} sample_count={len(rows)}")
    print(f"[config] num_steps={args.num_steps if args.num_steps is not None else 'config.mean_recurrence'}")
    plugin = import_plugin(args.plugin_path)
    model, processor, restore = load_generation_model(plugin, args.checkpoint, device)
    print(f"[restore] {json.dumps(restore, ensure_ascii=False)}")
    if not restore["aligner_restore"]["restored_boundary_embeddings"]:
        print("[warning] audio_bos/audio_eos were not found in the checkpoint; model initialization values are in use.")

    results: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=args.sample_offset):
        result = evaluate_row(row, plugin, model, processor, device, num_steps=args.num_steps)
        results.append(result)
        print(f"========== MMAU SAMPLE {row_index} ==========")
        print(f"[sample] id={result['metadata']['id']} task={result['metadata'].get('task')} difficulty={result['metadata'].get('difficulty')}")
        print(f"[sample] audio_seconds_after_truncation={result['audio_seconds_after_truncation']:.3f} prompt_tokens={result['prompt_token_count']}")
        print(f"[sample] question={result['instruction']}")
        for choice_index, score in enumerate(result["choice_scores"]):
            print(
                f"[choice {choice_index}] text={score['choice']!r} token_count={score['token_count']} "
                f"mean_logprob={score['mean_logprob']:.6f} total_logprob={score['total_logprob']:.6f}"
            )
        print(f"[sample] prediction={result['prediction']!r} answer={result['answer']!r} exact_correct={result['correct_exact_choice']}")

    correct_count = sum(result["correct_exact_choice"] for result in results)
    payload = {
        "checkpoint": args.checkpoint,
        "dataset_path": str(parquet_path),
        "sample_offset": args.sample_offset,
        "sample_count": len(results),
        "num_steps": args.num_steps,
        "scoring": "mean per-token conditional log probability of each complete option",
        "accuracy_exact_choice": correct_count / len(results),
        "correct_count": correct_count,
        "results": results,
    }
    output_path = output_dir / "mmau_test_mini_smoke_results.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("========== MMAU TEST-MINI SWIFT SMOKE DONE ==========")
    print(f"[summary] exact_choice_accuracy={correct_count / len(results):.4f} correct={correct_count}/{len(results)}")
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
