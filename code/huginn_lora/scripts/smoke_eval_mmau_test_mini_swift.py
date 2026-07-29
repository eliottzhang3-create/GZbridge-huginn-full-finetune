#!/usr/bin/env python3
"""Run a small official-style generative evaluation on MMAU test-mini."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch

from generate_clotho_caption_samples_swift import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PLUGIN_PATH,
    import_plugin,
    load_generation_model,
)


DEFAULT_DATASET_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU test_mini/test_mini.parquet"
HUGINN_STOP_TOKEN_IDS = {65504, 65505, 65508}
OFFICIAL_OPTION_ORDER_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--plugin-path", default=DEFAULT_PLUGIN_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None, help="Fixed Huginn recurrence count; default uses config.mean_recurrence.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
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
    if len(choices) > 26:
        raise ValueError("MMAU choices cannot exceed 26 options")
    choices_text = "\n".join(
        f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
    )
    user_content = (
        "Listen to the audio and answer the multiple-choice question. "
        "Select exactly one of the choices. You may answer with the option letter "
        "or the complete option text.\n\n"
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


def generate_response(
    model: torch.nn.Module,
    processor: Any,
    plugin: Any,
    prompt: str,
    audio_inputs: dict[str, torch.Tensor],
    device: torch.device,
    max_new_tokens: int,
    num_steps: int | None = None,
) -> dict[str, Any]:
    """Generate one greedy response with the manual audio-prefix cache path.

    Huginn's audio prefix is part of the KV-cache sequence length.  The
    generic Transformers ``generate`` path does not preserve that custom
    position handling, so MMAU uses the same explicit prefill/decode path as
    the repository's caption-generation evaluator.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    tokenizer = processor.tokenizer
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    stop_token_ids = {
        token_id
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(model.config, "eos_token_id", None),
            *HUGINN_STOP_TOKEN_IDS,
        )
        if token_id is not None
    }
    prefill_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
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
        generated_ids: list[int] = []
        stop_reason = "max_new_tokens"
        for _ in range(max_new_tokens):
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_token.item())
            if token_id in stop_token_ids:
                stop_reason = f"stop_token:{token_id}"
                break
            generated_ids.append(token_id)
            if len(generated_ids) == max_new_tokens:
                break
            cache_position = torch.tensor([cache.get_seq_length()], device=device, dtype=torch.long)
            decode_kwargs: dict[str, Any] = {
                "input_ids": next_token,
                "past_key_values": cache,
                "use_cache": True,
                "cache_position": cache_position,
            }
            if num_steps is not None:
                decode_kwargs["num_steps"] = num_steps
            outputs = model(**decode_kwargs)
            if outputs.logits is None or outputs.past_key_values is None:
                raise RuntimeError("MMAU cached decode did not return logits and cache")
            cache = outputs.past_key_values

    generated_tensor = torch.tensor(generated_ids, device=device, dtype=torch.long)
    return {
        "response": tokenizer.decode(generated_tensor, skip_special_tokens=True).strip(),
        "prompt_token_count": int(input_ids.shape[1]),
        "prefill_cache_length": prefill_cache_length,
        "generated_token_count": len(generated_ids),
        "stop_reason": stop_reason,
    }


def normalize_tokens(value: Any) -> set[str]:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return set(re.findall(r"\b\w+\b", str(value).lower()))


def official_string_match(answer: str, prediction: str, choices: list[str]) -> bool:
    """The official MMAU evaluator's token-set matching rule."""
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


def extract_choice_response(response: str, displayed_choices: list[str]) -> str:
    """Map a generated letter/option response back to displayed choice text."""
    cleaned = response.strip()
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_patterns = (
        r"^\s*(?:answer\s*(?:is|:)?\s*)?\(?([A-Z])\)?(?:[\s.):-]|$)",
        r"\b(?:option|choice|answer)\s*(?:is|:)?\s*\(?([A-Z])\)?\b",
    )
    for pattern in label_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            index = labels.find(match.group(1).upper())
            if 0 <= index < len(displayed_choices):
                return displayed_choices[index]

    matches = [
        choice
        for choice in displayed_choices
        if official_string_match(choice, cleaned, displayed_choices)
    ]
    if len(matches) == 1:
        return matches[0]
    exact_matches = [choice for choice in displayed_choices if cleaned.casefold() == choice.casefold()]
    if len(exact_matches) == 1:
        return exact_matches[0]
    return cleaned


def evaluate_row(
    row: dict[str, Any],
    plugin: Any,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    num_steps: int | None = None,
    max_new_tokens: int = 64,
    option_order_count: int = OFFICIAL_OPTION_ORDER_COUNT,
    seed: int = 0,
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
    if option_order_count != OFFICIAL_OPTION_ORDER_COUNT:
        raise ValueError(
            f"Official MMAU evaluation requires {OFFICIAL_OPTION_ORDER_COUNT} option orderings, "
            f"got {option_order_count}"
        )
    sample_id = str(metadata["id"])
    order_rng = random.Random(f"mmau-official:{seed}:{sample_id}")
    orderings: list[list[int]] = []
    generated_runs: list[dict[str, Any]] = []
    for _ in range(option_order_count):
        order = list(range(len(choices)))
        order_rng.shuffle(order)
        orderings.append(order)

    for order in orderings:
        displayed_choices = [choices[index] for index in order]
        prompt = build_prompt(plugin, instruction, displayed_choices)
        generated = generate_response(
            model,
            processor,
            plugin,
            prompt,
            audio_inputs,
            device,
            max_new_tokens=max_new_tokens,
            num_steps=num_steps,
        )
        extracted = extract_choice_response(generated["response"], displayed_choices)
        generated_runs.append(
            {
                "displayed_choice_indices": order,
                "displayed_choices": displayed_choices,
                "prompt": prompt,
                "response": generated["response"],
                "extracted_choice": extracted,
                **{key: value for key, value in generated.items() if key != "response"},
            }
        )

    vote_counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for run_index, run in enumerate(generated_runs):
        choice = run["extracted_choice"]
        vote_counts[choice] = vote_counts.get(choice, 0) + 1
        first_seen.setdefault(choice, run_index)
    prediction = max(vote_counts, key=lambda choice: (vote_counts[choice], -first_seen[choice]))
    return {
        "metadata": metadata,
        "instruction": instruction,
        "choices": choices,
        "answer": answer,
        "prediction": prediction,
        "correct_exact_choice": prediction == answer,
        "audio_seconds_after_truncation": used_seconds,
        "option_order_count": option_order_count,
        "option_order_seed": seed,
        "vote_counts": vote_counts,
        "generated_runs": generated_runs,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("num_steps must be positive when provided")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
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
    print(
        f"[config] generation=greedy_manual_audio_cache max_new_tokens={args.max_new_tokens} "
        f"official_option_order_count={OFFICIAL_OPTION_ORDER_COUNT} seed={args.seed}"
    )
    plugin = import_plugin(args.plugin_path)
    model, processor, restore = load_generation_model(plugin, args.checkpoint, device)
    print(f"[restore] {json.dumps(restore, ensure_ascii=False)}")
    if not restore["aligner_restore"]["restored_boundary_embeddings"]:
        print("[warning] audio_bos/audio_eos were not found in the checkpoint; model initialization values are in use.")

    results: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=args.sample_offset):
        result = evaluate_row(
            row,
            plugin,
            model,
            processor,
            device,
            num_steps=args.num_steps,
            max_new_tokens=args.max_new_tokens,
            option_order_count=OFFICIAL_OPTION_ORDER_COUNT,
            seed=args.seed,
        )
        result["official_match"] = official_string_match(
            result["answer"], result["prediction"], result["choices"]
        )
        results.append(result)
        print(f"========== MMAU SAMPLE {row_index} ==========")
        print(f"[sample] id={result['metadata']['id']} task={result['metadata'].get('task')} difficulty={result['metadata'].get('difficulty')}")
        print(
            f"[sample] audio_seconds_after_truncation={result['audio_seconds_after_truncation']:.3f} "
            f"prompt_tokens={result['generated_runs'][0]['prompt_token_count']}"
        )
        print(f"[sample] question={result['instruction']}")
        for run_index, run in enumerate(result["generated_runs"]):
            print(
                f"[order {run_index}] displayed_indices={run['displayed_choice_indices']} "
                f"response={run['response']!r} extracted={run['extracted_choice']!r} "
                f"stop_reason={run['stop_reason']}"
            )
        print(
            f"[sample] prediction={result['prediction']!r} answer={result['answer']!r} "
            f"official_match={result['official_match']} exact_correct={result['correct_exact_choice']}"
        )

    official_correct_count = sum(result["official_match"] for result in results)
    exact_correct_count = sum(result["correct_exact_choice"] for result in results)
    payload = {
        "checkpoint": args.checkpoint,
        "dataset_path": str(parquet_path),
        "sample_offset": args.sample_offset,
        "sample_count": len(results),
        "num_steps": args.num_steps,
        "scoring": "greedy generated response with official MMAU string_match",
        "generation": "manual audio-prefix KV-cache greedy decode",
        "option_order_count": OFFICIAL_OPTION_ORDER_COUNT,
        "option_order_seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "accuracy_official_string_match": official_correct_count / len(results),
        "official_string_match_correct_count": official_correct_count,
        "accuracy_exact_choice": exact_correct_count / len(results),
        "exact_choice_correct_count": exact_correct_count,
        "results": results,
    }
    output_path = output_dir / "mmau_test_mini_smoke_results.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("========== MMAU TEST-MINI SWIFT SMOKE DONE ==========")
    print(
        f"[summary] official_string_match_accuracy={official_correct_count / len(results):.4f} "
        f"correct={official_correct_count}/{len(results)} "
        f"exact_choice_accuracy={exact_correct_count / len(results):.4f}"
    )
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
