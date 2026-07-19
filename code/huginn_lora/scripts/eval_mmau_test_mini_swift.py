#!/usr/bin/env python3
"""Resumable full MMAU test-mini evaluation with Huginn audio choice likelihoods."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch

from generate_clotho_caption_samples_swift import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PLUGIN_PATH,
    import_plugin,
    load_generation_model,
)
from smoke_eval_mmau_test_mini_swift import (
    DEFAULT_DATASET_PATH,
    evaluate_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--plugin-path", default=DEFAULT_PLUGIN_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Default: evaluate to the end of test-mini.")
    parser.add_argument("--num-steps", type=int, default=None, help="Fixed Huginn recurrence count; default uses config.mean_recurrence.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument(
        "--fsdp-export-dir",
        default=None,
        help="Optional shared cache directory for a merged FSDP SHARDED_STATE_DICT checkpoint.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def iter_rows(parquet_path: Path, start_offset: int, end_offset: int) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    index = 0
    for batch in parquet_file.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            if index >= end_offset:
                return
            if index >= start_offset:
                yield index, row
            index += 1


def normalize_tokens(value: Any) -> set[str]:
    import re

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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def metric_bucket(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, int | float]]:
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for result in results:
        value = result["metadata"].get(key, "unknown")
        buckets[str(value)][0] += int(result["official_match"])
        buckets[str(value)][1] += 1
    return {
        value: {"correct": correct, "total": total, "accuracy": correct / total}
        for value, (correct, total) in sorted(buckets.items())
    }


def write_summary(
    output_dir: Path,
    results: list[dict[str, Any]],
    checkpoint: str,
    dataset_path: str,
    num_steps: int | None,
) -> dict[str, Any]:
    correct = sum(result["official_match"] for result in results)
    exact_correct = sum(result["correct_exact_choice"] for result in results)
    predictions = [
        {"id": result["metadata"]["id"], "model_prediction": result["prediction"]}
        for result in results
    ]
    predictions_path = output_dir / "mmau_test_mini_predictions.json"
    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "checkpoint": checkpoint,
        "dataset_path": dataset_path,
        "num_steps": num_steps,
        "completed_sample_count": len(results),
        "official_string_match_accuracy": correct / len(results) if results else 0.0,
        "official_string_match_correct": correct,
        "exact_choice_accuracy": exact_correct / len(results) if results else 0.0,
        "exact_choice_correct": exact_correct,
        "task_metrics": metric_bucket(results, "task"),
        "difficulty_metrics": metric_bucket(results, "difficulty"),
        "subcategory_metrics": metric_bucket(results, "sub-category"),
        "predictions_path": str(predictions_path),
    }
    summary_path = output_dir / "mmau_test_mini_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def prepare_output_dir(output_dir: Path, checkpoint: str, dataset_path: str, num_steps: int | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    expected = {
        "checkpoint": checkpoint,
        "dataset_path": dataset_path,
        "num_steps": num_steps,
        "scoring": "mean per-token conditional log probability",
    }
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                f"Existing output directory has a different run_config: {existing}. Choose a new output directory."
            )
    else:
        config_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_dir / "mmau_test_mini_results.jsonl"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.start_offset < 0 or args.log_every <= 0 or (args.num_steps is not None and args.num_steps <= 0):
        raise ValueError("start_offset must be non-negative, log_every must be positive, and num_steps must be positive when provided")
    parquet_path = Path(args.dataset_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"MMAU test-mini parquet not found: {parquet_path}")

    import pyarrow.parquet as pq

    dataset_size = pq.ParquetFile(parquet_path).metadata.num_rows
    end_offset = dataset_size if args.max_samples is None else min(dataset_size, args.start_offset + args.max_samples)
    if args.start_offset >= end_offset:
        raise ValueError(f"Requested range [{args.start_offset}, {end_offset}) is empty")

    output_dir = Path(args.output_dir)
    results_path = prepare_output_dir(output_dir, args.checkpoint, str(parquet_path), args.num_steps)
    existing_results = read_jsonl(results_path)
    completed_ids = {result["metadata"]["id"] for result in existing_results}
    device = torch.device(args.device)

    print("========== MMAU TEST-MINI SWIFT FULL EVAL ==========")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] dataset_path={parquet_path}")
    print(f"[config] requested_range=[{args.start_offset}, {end_offset}) total_dataset_rows={dataset_size}")
    print(f"[config] num_steps={args.num_steps if args.num_steps is not None else 'config.mean_recurrence'}")
    print(f"[config] output_dir={output_dir} resumed_completed={len(completed_ids)}")
    plugin = import_plugin(args.plugin_path)
    model, processor, restore = load_generation_model(
        plugin,
        args.checkpoint,
        device,
        args.fsdp_export_dir,
    )
    print(f"[restore] {json.dumps(restore, ensure_ascii=False)}")
    if not restore["aligner_restore"]["restored_boundary_embeddings"]:
        print("[warning] audio_bos/audio_eos were not found in the checkpoint; model initialization values are in use.")

    started_at = time.monotonic()
    processed_now = 0
    skipped = 0
    requested_count = end_offset - args.start_offset
    with results_path.open("a", encoding="utf-8") as handle:
        for row_index, row in iter_rows(parquet_path, args.start_offset, end_offset):
            attributes = json.loads(row["other_attributes"])
            sample_id = attributes["id"]
            if sample_id in completed_ids:
                skipped += 1
                continue
            result = evaluate_row(row, plugin, model, processor, device, num_steps=args.num_steps)
            result["dataset_row_index"] = row_index
            result["official_match"] = official_string_match(result["answer"], result["prediction"], result["choices"])
            append_jsonl(handle, result)
            existing_results.append(result)
            completed_ids.add(sample_id)
            processed_now += 1

            if args.print_samples:
                print(
                    f"[sample] row={row_index} id={sample_id} prediction={result['prediction']!r} "
                    f"answer={result['answer']!r} official_match={result['official_match']}"
                )
            if processed_now % args.log_every == 0:
                elapsed = time.monotonic() - started_at
                rate = processed_now / elapsed
                remaining = max(requested_count - skipped - processed_now, 0)
                correct = sum(item["official_match"] for item in existing_results)
                print(
                    f"[progress] processed_now={processed_now} skipped={skipped} "
                    f"completed_total={len(existing_results)} accuracy={correct / len(existing_results):.4f} "
                    f"seconds_per_sample={1.0 / rate:.2f} eta_seconds={remaining / rate:.0f}",
                    flush=True,
                )

    summary = write_summary(output_dir, existing_results, args.checkpoint, str(parquet_path), args.num_steps)
    print("========== MMAU TEST-MINI SWIFT FULL EVAL DONE ==========")
    print(
        f"[summary] completed={summary['completed_sample_count']} "
        f"official_accuracy={summary['official_string_match_accuracy']:.4f} "
        f"exact_choice_accuracy={summary['exact_choice_accuracy']:.4f}"
    )
    print(f"[summary] predictions_path={summary['predictions_path']}")
    print(f"[summary] summary_path={output_dir / 'mmau_test_mini_summary.json'}")


if __name__ == "__main__":
    main()
