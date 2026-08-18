#!/usr/bin/env python3
"""Measure the real Ouro BAT DataLoader/audio-rendering path only.

No Ouro forward, backward, optimizer, Spatial-AST inference, or Q-Former
inference is executed here.  The template and collator are the production
ones, so each measured ``next(loader)`` includes the same lazy work used by
training:

    JSONL row -> template encoding -> AudioSet read -> RIR read/crop/pad
    -> binaural convolution -> fixed-length batch collation.

The four worker configurations (0, 2, 4, 8) are run sequentially in one
submitted single-card job.  The reported global batch size is the training
equivalent of local batch 8 on eight ranks; this script itself is one process
and performs no distributed communication.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from bat.configs.training import BAT_TRAINING


MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
WORKER_CONFIGS = (0, 2, 4, 8)
LOCAL_BATCH_SIZE = 8
ASSUMED_WORLD_SIZE = 8
STATIC_SEQUENCE_LENGTH = 176


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B"),
    )
    parser.add_argument(
        "--plugin-path",
        type=Path,
        default=Path(
            "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
            "code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py"
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument(
        "--total-batches",
        type=int,
        default=24,
        help=(
            "Total batches consumed for each worker configuration, including "
            "warmup batches. With local batch size 8, 24 batches consume 1536 rows."
        ),
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    from importlib.metadata import version

    try:
        return version(name)
    except Exception as exc:
        return f"<unavailable:{type(exc).__name__}>"


def import_plugin(path: Path):
    spec = importlib.util.spec_from_file_location("ouro_bat_dataloader_profile_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path, start_index: int, count: int) -> tuple[list[dict[str, Any]], float]:
    if start_index < 0 or count <= 0:
        raise ValueError("start-index must be non-negative and count must be positive")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index:
                continue
            if len(rows) >= count:
                break
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Manifest line {index} is not an object")
            rows.append(value)
    if len(rows) != count:
        raise RuntimeError(
            f"Manifest does not contain enough rows: requested={count} "
            f"start={start_index} got={len(rows)}"
        )
    return rows, time.perf_counter() - started


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty timing list")
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": float(len(ordered)),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "mean_seconds": sum(ordered) / len(ordered),
        "p50_seconds": percentile(0.50),
        "p95_seconds": percentile(0.95),
    }


def source_summary(record: Any) -> dict[str, str] | None:
    if not isinstance(record, dict):
        return None
    return {
        key: str(record.get(key, ""))
        for key in (
            "audio_id",
            "reverb_id",
            "audio_id2",
            "reverb_id2",
            "question_id",
            "question_type",
        )
    }


def validate_batch(batch: dict[str, Any], expected_batch_size: int) -> dict[str, Any]:
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    waveforms = batch.get("audio_waveforms")
    records = batch.get("bat_audio_records")
    if not all(torch.is_tensor(value) for value in (input_ids, labels, attention_mask, waveforms)):
        raise RuntimeError(f"Production batch is missing tensor fields: {sorted(batch)}")
    if tuple(input_ids.shape) != (expected_batch_size, STATIC_SEQUENCE_LENGTH):
        raise RuntimeError(f"Unexpected input_ids shape: {tuple(input_ids.shape)}")
    if tuple(labels.shape) != tuple(input_ids.shape) or tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise RuntimeError("Production input/labels/attention shapes are not aligned")
    if tuple(waveforms.shape) != (expected_batch_size, 2, 320000):
        raise RuntimeError(f"Unexpected audio waveform shape: {tuple(waveforms.shape)}")
    if not bool(torch.isfinite(waveforms.float()).all().item()):
        raise RuntimeError("Production audio waveform batch contains NaN or Inf")
    if not isinstance(records, list) or len(records) != expected_batch_size:
        raise RuntimeError("Production audio provenance is missing or misaligned")
    if not bool((labels[:, :64] == -100).all().item()):
        raise RuntimeError("Audio prefix labels are not fully masked")
    rms = torch.sqrt(torch.mean(waveforms.float() ** 2, dim=(1, 2)))
    if not bool((rms > 0).all().item()):
        raise RuntimeError("Production audio batch contains silent examples")
    return {
        "input_ids_shape": list(input_ids.shape),
        "labels_shape": list(labels.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "audio_waveforms_shape": list(waveforms.shape),
        "audio_prefix_label_ignore_count": int((labels[:, :64] == -100).sum().item()),
        "waveform_rms_min": float(rms.min().item()),
        "waveform_rms_max": float(rms.max().item()),
        "source_preview": [source_summary(record) for record in records[:4]],
    }


def close_loader(loader: DataLoader, iterator: Any) -> None:
    """Ask persistent worker processes to exit before the next configuration."""
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    del iterator
    del loader


def main() -> None:
    args = parse_args()
    BAT_TRAINING.validate()
    if args.warmup_batches < 0 or args.total_batches <= args.warmup_batches:
        raise ValueError("total-batches must be greater than warmup-batches >= 0")
    if args.start_index < 0 or args.prefetch_factor <= 0:
        raise ValueError("start-index must be non-negative and prefetch-factor must be positive")
    output = args.output_report.expanduser().resolve()
    if str(output).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {output}")
    for path in (args.model_path, args.plugin_path, args.dataset):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite profiling report: {output}")

    os.environ.setdefault("BAT_AUDIO_AUDIT", "1")
    os.environ.setdefault("BAT_MAX_SEQUENCE_LENGTH", str(STATIC_SEQUENCE_LENGTH))
    plugin = import_plugin(args.plugin_path.resolve())
    if plugin.MODEL_TYPE != MODEL_TYPE or plugin.TEMPLATE_TYPE != TEMPLATE_TYPE:
        raise RuntimeError("Ouro BAT plugin registration constants do not match the DataLoader profiler")

    total_batches = args.total_batches
    measure_batches = total_batches - args.warmup_batches
    rows, manifest_read_seconds = load_rows(
        args.dataset.resolve(),
        args.start_index,
        total_batches * LOCAL_BATCH_SIZE,
    )
    try:
        from swift import get_model_processor, get_template
    except ImportError:
        from swift import get_model_processor
        from swift.template import get_template

    print("========== OURO BAT DATALOADER-ONLY PROFILING ==========")
    print(f"[packages] ms-swift={package_version('ms-swift')} transformers={package_version('transformers')} torch={torch.__version__}")
    print(f"[dataset] {args.dataset} start_index={args.start_index} rows={len(rows)}")
    print(f"[batch] local_batch_size={LOCAL_BATCH_SIZE} assumed_world_size={ASSUMED_WORLD_SIZE} global_batch_size={LOCAL_BATCH_SIZE * ASSUMED_WORLD_SIZE}")
    print(
        f"[benchmark] total_batches={total_batches} warmup_batches={args.warmup_batches} "
        f"measure_batches={measure_batches} rows={total_batches * LOCAL_BATCH_SIZE} "
        f"workers={WORKER_CONFIGS}"
    )
    print("[scope] DataLoader/template/audio renderer only; no Ouro forward/backward/optimizer/Spatial-AST/Q-Former")

    setup_started = time.perf_counter()
    # The model is loaded on CPU only to obtain the exact registered processor
    # and template contract.  It is never called and is excluded from all
    # batch timing measurements.
    model, processor = get_model_processor(
        str(args.model_path.resolve()),
        model_type=MODEL_TYPE,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    del model
    template = get_template(
        template_type=TEMPLATE_TYPE,
        processor=processor,
        max_length=512,
        use_chat_template=True,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("train")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Ouro tokenizer has no pad/eos token")
    setup_seconds = time.perf_counter() - setup_started

    # Reuse the exact helper implementation used by the existing pure-pipeline
    # profiler, including Template.encode and _data_collator_mm_data.
    from profile_bat_ouro_pipeline import EncodedBATDataset, make_collator

    dataset = EncodedBATDataset(rows, template)
    results: list[dict[str, Any]] = []
    for num_workers in WORKER_CONFIGS:
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": LOCAL_BATCH_SIZE,
            "shuffle": False,
            "drop_last": True,
            "num_workers": num_workers,
            "pin_memory": args.pin_memory,
            "collate_fn": make_collator(
                template,
                int(pad_token_id),
                static_sequence_length=STATIC_SEQUENCE_LENGTH,
            ),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
            loader_kwargs["persistent_workers"] = args.persistent_workers
        loader = DataLoader(**loader_kwargs)
        iterator = iter(loader)
        all_times: list[float] = []
        measured_times: list[float] = []
        measured_completion_seconds: list[float] = []
        batch_audits: list[dict[str, Any]] = []
        first_batch_seconds = None
        worker_started = time.perf_counter()
        measurement_started = None
        try:
            for batch_index in range(total_batches):
                started = time.perf_counter()
                batch = next(iterator)
                batch_ready = time.perf_counter()
                elapsed = batch_ready - started
                if first_batch_seconds is None:
                    first_batch_seconds = elapsed
                audit = validate_batch(batch, LOCAL_BATCH_SIZE)
                all_times.append(elapsed)
                if batch_index >= args.warmup_batches:
                    if measurement_started is None:
                        measurement_started = started
                    measured_times.append(elapsed)
                    measured_completion_seconds.append(
                        batch_ready - measurement_started
                    )
                    if len(batch_audits) < 2:
                        batch_audits.append(audit)
            worker_elapsed = time.perf_counter() - worker_started
        finally:
            close_loader(loader, iterator)
        if len(measured_times) != measure_batches:
            raise RuntimeError(
                f"Worker={num_workers} measured {len(measured_times)} batches, "
                f"expected {measure_batches}"
            )
        measured_wall_seconds = sum(measured_times)
        if measured_wall_seconds <= 0:
            raise RuntimeError(f"Worker={num_workers} has non-positive measured wall time")
        batches_per_second = measure_batches / measured_wall_seconds
        samples_per_second = measure_batches * LOCAL_BATCH_SIZE / measured_wall_seconds
        # This is the direct requested interpretation: how many local batches
        # would pass in one second at the measured steady rate.  Keep the
        # existing names as aliases for compatibility with prior reports.
        batches_per_one_second = batches_per_second
        samples_per_one_second = samples_per_second

        # Also retain an observation-window statistic.  It counts actual
        # completions in each one-second wall-clock window rather than
        # extrapolating a rate.  This makes aggressive DataLoader prefetching
        # visible in the report (especially for workers=8).
        observation_seconds = measured_completion_seconds[-1]
        window_count = max(1, math.ceil(observation_seconds))
        one_second_window_batch_counts = []
        for window_index in range(window_count):
            window_start = float(window_index)
            window_end = window_start + 1.0
            count = sum(
                window_start <= completion < window_end
                for completion in measured_completion_seconds
            )
            one_second_window_batch_counts.append(int(count))
        result = {
            "num_workers": num_workers,
            "prefetch_factor": args.prefetch_factor if num_workers > 0 else None,
            "persistent_workers": bool(args.persistent_workers) if num_workers > 0 else False,
            "pin_memory": bool(args.pin_memory),
            "local_batch_size": LOCAL_BATCH_SIZE,
            "assumed_world_size": ASSUMED_WORLD_SIZE,
            "global_batch_size": LOCAL_BATCH_SIZE * ASSUMED_WORLD_SIZE,
            "warmup_batches": args.warmup_batches,
            "total_batches": total_batches,
            "total_rows_consumed": total_batches * LOCAL_BATCH_SIZE,
            "measured_batches": measure_batches,
            "first_batch_seconds": first_batch_seconds,
            "all_batch_wait": summarize(all_times),
            "measured_batch_wait": summarize(measured_times),
            "measured_wall_seconds": measured_wall_seconds,
            "batches_per_one_second": batches_per_one_second,
            "samples_per_one_second": samples_per_one_second,
            "steady_batches_per_second": batches_per_one_second,
            "steady_samples_per_second": samples_per_one_second,
            "one_second_observation_window": {
                "window_seconds": 1.0,
                "observation_seconds": observation_seconds,
                "window_count": window_count,
                "actual_batch_counts": one_second_window_batch_counts,
                "max_actual_batches_in_one_second": max(one_second_window_batch_counts),
            },
            "worker_configuration_wall_seconds": worker_elapsed,
            "batch_audits": batch_audits,
        }
        results.append(result)
        print(
            f"[workers={num_workers}] first={first_batch_seconds:.4f}s "
            f"p50={result['measured_batch_wait']['p50_seconds']:.4f}s "
            f"p95={result['measured_batch_wait']['p95_seconds']:.4f}s "
            f"batches_per_1s={result['batches_per_one_second']:.4f} "
            f"samples_per_1s={result['samples_per_one_second']:.4f} "
            f"observed_1s_max_batches={result['one_second_observation_window']['max_actual_batches_in_one_second']}",
            flush=True,
        )

    renderer = getattr(template, "audio_renderer", None)
    report = {
        "status": "ok",
        "scope": {
            "model_forward_executed": False,
            "model_backward_executed": False,
            "optimizer_step_executed": False,
            "spatial_ast_executed": False,
            "qformer_executed": False,
            "ddp_communication": False,
        },
        "route": "stage3_ab_cde",
        "curriculum": False,
        "manifest": {
            "path": str(args.dataset.resolve()),
            "start_index": args.start_index,
            "rows_loaded": len(rows),
            "manifest_read_seconds": manifest_read_seconds,
        },
        "setup": {
            "model_loaded_for_registration_only": True,
            "model_device": "cpu",
            "setup_seconds": setup_seconds,
            "template_type": TEMPLATE_TYPE,
            "static_sequence_length": STATIC_SEQUENCE_LENGTH,
            "audio_prefix_tokens": 64,
            "audio_root": None if renderer is None else str(renderer.audio_root),
            "reverb_root": None if renderer is None else str(renderer.reverb_root),
        },
        "batch_contract": {
            "local_batch_size": LOCAL_BATCH_SIZE,
            "assumed_world_size": ASSUMED_WORLD_SIZE,
            "global_batch_size": LOCAL_BATCH_SIZE * ASSUMED_WORLD_SIZE,
            "input_ids_shape": [LOCAL_BATCH_SIZE, STATIC_SEQUENCE_LENGTH],
            "audio_waveforms_shape": [LOCAL_BATCH_SIZE, 2, 320000],
            "audio_prefix_labels": "-100",
            "shuffle": False,
            "drop_last": True,
        },
        "worker_results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[report] {output}")
    print("[status] ok")


if __name__ == "__main__":
    main()
