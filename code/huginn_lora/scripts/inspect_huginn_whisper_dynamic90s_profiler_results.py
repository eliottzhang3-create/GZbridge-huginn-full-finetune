#!/usr/bin/env python3
"""Validate and aggregate four-rank dynamic-90s Torch Profiler output."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SUMMARY_VERSION = "huginn_whisper_dynamic90s_torch_profiler_v1"
POOL_NAMES = (
    "wavcaps_no_bbc_aac",
    "audiocaps_v2_aac",
    "clotho_v2_aac",
    "gigaspeech_l_asr",
)
REQUIRED_MODULES = {
    "model_total",
    "whisper_encoder",
    "whisper_encoder_layer",
    "audio_aligner",
    "huginn_prelude",
    "huginn_recurrent_core",
    "huginn_coda",
    "huginn_sandwich_block",
    "huginn_attention",
    "huginn_mlp",
    "lm_head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler-dir", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--formal-max-steps", type=int, default=17700)
    return parser.parse_args()


def load_summary(root: Path, rank: int) -> dict[str, Any]:
    path = root / f"profiler_summary_rank{rank}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing rank profiler summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("summary_version") != SUMMARY_VERSION:
        raise RuntimeError(f"Profiler summary version mismatch at {path}: {payload.get('summary_version')!r}")
    if int(payload.get("rank", -1)) != rank:
        raise RuntimeError(f"Profiler rank mismatch at {path}: {payload.get('rank')}")
    return payload


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "p50": float(percentile(values, 0.50)),
        "p95": float(percentile(values, 0.95)),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def phase_values(payload: dict[str, Any], section: str, phase: str) -> list[float]:
    return [float(row["elapsed_us"]) for row in payload[section] if row.get("phase") == phase]


def top_events(
    payloads: list[dict[str, Any]],
    category: str | None,
    limit: int = 20,
    sort_by: str = "cuda_time_total_us",
) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, float]] = {}
    for payload in payloads:
        source_rows = (
            payload.get("events_by_cpu" if sort_by == "cpu_time_total_us" else "events", [])
            if category is None
            else payload.get("categorized_events", {}).get(category, [])
        )
        for row in source_rows:
            key = str(row["key"])
            entry = aggregate.setdefault(
                key,
                {"count": 0.0, "cpu_time_total_us": 0.0, "cuda_time_total_us": 0.0, "flops": 0.0},
            )
            entry["count"] += float(row.get("count", 0))
            entry["cpu_time_total_us"] += float(row.get("cpu_time_total_us", 0.0))
            entry["cuda_time_total_us"] += float(row.get("cuda_time_total_us", 0.0))
            entry["flops"] += float(row.get("flops", 0.0))
    rows = [{"key": key, **values} for key, values in aggregate.items()]
    secondary = "cpu_time_total_us" if sort_by == "cuda_time_total_us" else "cuda_time_total_us"
    rows.sort(key=lambda row: (row[sort_by], row[secondary]), reverse=True)
    return rows[:limit]


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or args.per_device_batch <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("World size, batch size, and gradient accumulation must be positive")
    profiler_dir = Path(args.profiler_dir).resolve()
    payloads = [load_summary(profiler_dir, rank) for rank in range(args.world_size)]

    expected_microbatches = args.max_steps * args.gradient_accumulation
    positions: list[int] = []
    pool_counts: Counter[int] = Counter()
    durations: list[float] = []
    prefixes: list[float] = []
    segment_counts: list[float] = []
    recurrence_totals: list[float] = []
    recurrence_no_grad: list[float] = []
    recurrence_with_grad: list[float] = []
    recurrent_core_calls: list[float] = []
    whisper_calls: list[float] = []
    aligner_calls: list[float] = []
    whisper_layer_calls: list[float] = []
    sandwich_block_calls: list[float] = []
    local_padding_ratios: list[float] = []
    per_rank_sequences: dict[int, list[int]] = {}
    rank_reports: dict[str, Any] = {}

    for rank, payload in enumerate(payloads):
        if int(payload.get("world_size", -1)) != args.world_size:
            raise RuntimeError(f"Rank {rank} world size mismatch: {payload.get('world_size')}")
        microbatches = payload.get("microbatches", [])
        optimizer_steps = payload.get("optimizer_steps", [])
        if len(microbatches) != expected_microbatches:
            raise RuntimeError(
                f"Rank {rank} microbatch count mismatch: expected={expected_microbatches} actual={len(microbatches)}"
            )
        if len(optimizer_steps) != args.max_steps:
            raise RuntimeError(
                f"Rank {rank} optimizer-step count mismatch: expected={args.max_steps} actual={len(optimizer_steps)}"
            )
        installed = set(payload.get("installed_modules", {}))
        missing_modules = sorted(REQUIRED_MODULES - installed)
        if missing_modules:
            raise RuntimeError(f"Rank {rank} did not install all module ranges: {missing_modules}")
        if int(payload.get("recurrence_sampler_modules", 0)) != 1:
            raise RuntimeError(f"Rank {rank} recurrence sampler instrumentation is incomplete")

        schedule = payload["schedule"]
        phase_counts = Counter(row["phase"] for row in microbatches)
        expected_phase_counts = {
            "wait": int(schedule["wait"]),
            "warmup": int(schedule["warmup"]),
            "active": int(schedule["active"]),
            "post_active": expected_microbatches
            - int(schedule["wait"])
            - int(schedule["warmup"])
            - int(schedule["active"]),
        }
        if dict(phase_counts) != {key: value for key, value in expected_phase_counts.items() if value}:
            raise RuntimeError(
                f"Rank {rank} profiler phase coverage mismatch: expected={expected_phase_counts} actual={dict(phase_counts)}"
            )
        trace_files = sorted((profiler_dir / f"rank{rank}").glob("*.pt.trace.json*"))
        if not trace_files:
            raise FileNotFoundError(f"Rank {rank} produced no Chrome/TensorBoard trace")

        rank_positions: list[int] = []
        rank_durations: list[float] = []
        rank_prefixes: list[float] = []
        rank_sequences: list[int] = []
        for row in microbatches:
            inputs = row.get("inputs", {})
            input_shape = inputs.get("audio_input_features_shape")
            if not input_shape or int(input_shape[0]) != args.per_device_batch:
                raise RuntimeError(f"Rank {rank} has invalid local audio batch shape: {input_shape}")
            row_positions = [int(value) for value in inputs.get("audio_training_global_positions", [])]
            row_pool_ids = [int(value) for value in inputs.get("audio_training_pool_ids", [])]
            row_durations = [float(value) for value in inputs.get("audio_training_effective_duration_seconds", [])]
            row_prefixes = [int(value) for value in inputs.get("estimated_prefix_tokens_per_sample", [])]
            row_segments = [int(value) for value in inputs.get("valid_segment_counts", [])]
            sequence_tokens = int(inputs.get("estimated_huginn_sequence_tokens", 0))
            expected = args.per_device_batch
            lengths = {
                "positions": len(row_positions),
                "pool_ids": len(row_pool_ids),
                "durations": len(row_durations),
                "prefixes": len(row_prefixes),
                "segments": len(row_segments),
            }
            if any(value != expected for value in lengths.values()):
                raise RuntimeError(f"Rank {rank} profiler input metadata is incomplete: {lengths}")
            if any(value <= 0.0 or value > 90.001 for value in row_durations):
                raise RuntimeError(f"Rank {rank} observed invalid effective audio duration: {row_durations}")
            if any(value < 2 or value > 752 for value in row_prefixes):
                raise RuntimeError(f"Rank {rank} observed invalid dynamic prefix length: {row_prefixes}")
            if any(value < 1 or value > 3 for value in row_segments):
                raise RuntimeError(f"Rank {rank} observed invalid Whisper segment count: {row_segments}")
            if sequence_tokens <= max(row_prefixes):
                raise RuntimeError(f"Rank {rank} observed invalid Huginn sequence length: {sequence_tokens}")
            recurrences = row.get("recurrences", [])
            if len(recurrences) != 1:
                raise RuntimeError(f"Rank {rank} microbatch did not capture exactly one recurrence sample: {recurrences}")
            recurrence = recurrences[0]
            if int(recurrence["total"]) != int(recurrence["no_grad"]) + int(recurrence["with_grad"]):
                raise RuntimeError(f"Rank {rank} recurrence accounting mismatch: {recurrence}")
            if int(recurrence["total"]) <= 0:
                raise RuntimeError(f"Rank {rank} observed a non-positive recurrence count: {recurrence}")
            module_calls = row.get("module_calls", {})
            core_call_count = sum(
                int(count) for name, count in module_calls.items() if name.startswith("huginn_recurrent_core:")
            )
            whisper_call_count = sum(
                int(count) for name, count in module_calls.items() if name.startswith("whisper_encoder:")
            )
            aligner_call_count = sum(
                int(count) for name, count in module_calls.items() if name.startswith("audio_aligner:")
            )
            whisper_layer_call_count = sum(
                int(count) for name, count in module_calls.items() if name.startswith("whisper_encoder_layer:")
            )
            sandwich_block_call_count = sum(
                int(count) for name, count in module_calls.items() if name.startswith("huginn_sandwich_block:")
            )
            if min(
                core_call_count,
                whisper_call_count,
                aligner_call_count,
                whisper_layer_call_count,
                sandwich_block_call_count,
            ) <= 0:
                raise RuntimeError(f"Rank {rank} module call accounting is incomplete: {module_calls}")

            rank_positions.extend(row_positions)
            rank_durations.extend(row_durations)
            rank_prefixes.extend(row_prefixes)
            rank_sequences.append(sequence_tokens)
            positions.extend(row_positions)
            pool_counts.update(row_pool_ids)
            durations.extend(row_durations)
            prefixes.extend(row_prefixes)
            segment_counts.extend(row_segments)
            recurrence_totals.append(float(recurrence["total"]))
            recurrence_no_grad.append(float(recurrence["no_grad"]))
            recurrence_with_grad.append(float(recurrence["with_grad"]))
            recurrent_core_calls.append(float(core_call_count))
            whisper_calls.append(float(whisper_call_count))
            aligner_calls.append(float(aligner_call_count))
            whisper_layer_calls.append(float(whisper_layer_call_count))
            sandwich_block_calls.append(float(sandwich_block_call_count))
            local_padding_ratios.append(
                1.0 - (sum(row_prefixes) / (len(row_prefixes) * max(row_prefixes)))
            )

        expected_rank_samples = expected_microbatches * args.per_device_batch
        if len(rank_positions) != expected_rank_samples:
            raise RuntimeError(
                f"Rank {rank} consumed sample count mismatch: expected={expected_rank_samples} actual={len(rank_positions)}"
            )
        post_dispatch = phase_values(payload, "dispatch_rows", "post_active")
        post_compute_steps = phase_values(payload, "optimizer_steps", "post_active")
        rank_reports[str(rank)] = {
            "trace_files": [str(path) for path in trace_files],
            "phase_counts": dict(phase_counts),
            "post_active_dispatch_us": distribution(post_dispatch),
            "post_active_optimizer_compute_us": distribution(post_compute_steps),
            "post_active_estimated_total_step_us": (
                (sum(post_compute_steps) / len(post_compute_steps))
                + args.gradient_accumulation * (sum(post_dispatch) / len(post_dispatch))
                if post_dispatch and post_compute_steps
                else 0.0
            ),
            "duration_seconds": distribution(rank_durations),
            "prefix_tokens": distribution(rank_prefixes),
            "huginn_sequence_tokens": distribution([float(value) for value in rank_sequences]),
            "max_memory_allocated_gib": float(payload["memory"]["max_memory_allocated_bytes"]) / (1024**3),
            "max_memory_reserved_gib": float(payload["memory"]["max_memory_reserved_bytes"]) / (1024**3),
            "module_call_counts": payload.get("module_call_counts", {}),
        }
        per_rank_sequences[rank] = rank_sequences

    expected_global_samples = args.world_size * expected_microbatches * args.per_device_batch
    if len(positions) != expected_global_samples:
        raise RuntimeError(f"Global consumed sample count mismatch: expected={expected_global_samples} actual={len(positions)}")
    if len(set(positions)) != len(positions):
        duplicates = [value for value, count in Counter(positions).items() if count > 1]
        raise RuntimeError(f"Profiler forward window contains duplicate global positions: {duplicates[:20]}")
    expected_positions = list(range(min(positions), min(positions) + expected_global_samples))
    if sorted(positions) != expected_positions:
        raise RuntimeError(
            f"Profiler forward positions are not one contiguous window: actual={min(positions)}..{max(positions)} "
            f"expected={expected_positions[0]}..{expected_positions[-1]}"
        )
    if set(pool_counts) != set(range(len(POOL_NAMES))):
        raise RuntimeError(f"Profiler window did not cover all four pools: {dict(pool_counts)}")

    cross_rank_sequence_ratios: list[float] = []
    cross_rank_sequence_spreads: list[float] = []
    for microbatch_index in range(expected_microbatches):
        values = [per_rank_sequences[rank][microbatch_index] for rank in range(args.world_size)]
        cross_rank_sequence_ratios.append(max(values) / min(values))
        cross_rank_sequence_spreads.append(float(max(values) - min(values)))

    rank_step_estimates = [float(report["post_active_estimated_total_step_us"]) for report in rank_reports.values()]
    steady_step_us = max(rank_step_estimates)
    formal_days = steady_step_us / 1_000_000.0 * args.formal_max_steps / 86400.0
    module_events = top_events(payloads, "module")
    communication_events = top_events(payloads, "communication")
    matrix_events = top_events(payloads, "matrix")
    attention_events = top_events(payloads, "attention")
    convolution_events = top_events(payloads, "convolution")
    data_events = top_events(payloads, "data")
    trainer_events = top_events(payloads, "trainer")
    top_cuda_events = top_events(payloads, None, sort_by="cuda_time_total_us")
    top_cpu_events = top_events(payloads, None, sort_by="cpu_time_total_us")

    report = {
        "validation_passed": True,
        "summary_version": SUMMARY_VERSION,
        "world_size": args.world_size,
        "per_device_batch": args.per_device_batch,
        "gradient_accumulation": args.gradient_accumulation,
        "profile_max_steps": args.max_steps,
        "formal_max_steps": args.formal_max_steps,
        "global_samples_profiled": expected_global_samples,
        "global_position_range": [min(positions), max(positions)],
        "pool_counts": {POOL_NAMES[index]: pool_counts[index] for index in range(len(POOL_NAMES))},
        "audio_duration_seconds": distribution(durations),
        "dynamic_prefix_tokens": distribution(prefixes),
        "whisper_segment_counts": distribution(segment_counts),
        "recurrence": {
            "total": distribution(recurrence_totals),
            "no_grad": distribution(recurrence_no_grad),
            "with_grad": distribution(recurrence_with_grad),
            "actual_core_calls": distribution(recurrent_core_calls),
            "core_recompute_multiplier": distribution(
                [calls / recurrence for calls, recurrence in zip(recurrent_core_calls, recurrence_totals)]
            ),
        },
        "checkpoint_recompute_calls_per_microbatch": {
            "whisper_encoder": distribution(whisper_calls),
            "whisper_encoder_layers_total": distribution(whisper_layer_calls),
            "audio_aligner": distribution(aligner_calls),
            "huginn_sandwich_blocks_total": distribution(sandwich_block_calls),
        },
        "length_imbalance": {
            "local_batch_prefix_padding_ratio": distribution(local_padding_ratios),
            "cross_rank_sequence_max_over_min": distribution(cross_rank_sequence_ratios),
            "cross_rank_sequence_token_spread": distribution(cross_rank_sequence_spreads),
        },
        "rank_reports": rank_reports,
        "post_active_estimated_steady_step_seconds": steady_step_us / 1_000_000.0,
        "post_active_projected_formal_days": formal_days,
        "projection_warning": (
            "Projection uses a short, instrumented real-data run. The post-active phase is lower-overhead than the active "
            "trace but is not a substitute for a longer uninstrumented throughput run."
        ),
        "top_module_events": module_events,
        "top_communication_events": communication_events,
        "top_matrix_events": matrix_events,
        "top_attention_events": attention_events,
        "top_convolution_events": convolution_events,
        "top_data_events": data_events,
        "top_trainer_events": trainer_events,
        "top_cuda_events": top_cuda_events,
        "top_cpu_events": top_cpu_events,
    }
    if not math.isfinite(formal_days) or formal_days <= 0:
        raise RuntimeError(f"Invalid projected formal duration: {formal_days}")
    output = Path(args.output_report).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[profile-aggregate] samples={expected_global_samples} positions={min(positions)}..{max(positions)}")
    print(f"[profile-aggregate] pool_counts={report['pool_counts']}")
    print(
        f"[profile-aggregate] duration_seconds={report['audio_duration_seconds']} "
        f"prefix_tokens={report['dynamic_prefix_tokens']}"
    )
    print(f"[profile-aggregate] recurrence={report['recurrence']}")
    print(f"[profile-aggregate] checkpoint_recompute={report['checkpoint_recompute_calls_per_microbatch']}")
    print(f"[profile-aggregate] length_imbalance={report['length_imbalance']}")
    for rank, rank_report in rank_reports.items():
        print(
            f"[profile-rank] rank={rank} post_step_us={rank_report['post_active_estimated_total_step_us']:.3f} "
            f"max_allocated_gib={rank_report['max_memory_allocated_gib']:.3f}"
        )
    print(
        f"[profile-projection] steady_step_seconds={report['post_active_estimated_steady_step_seconds']:.6f} "
        f"formal_steps={args.formal_max_steps} projected_days={formal_days:.6f}"
    )
    print(f"[profile-aggregate] report={output}")
    print("========== HUGINN WHISPER DYNAMIC90S TORCH PROFILER RESULTS PASSED ==========")


if __name__ == "__main__":
    main()
