"""Opt-in full-path profiler for Huginn Whisper dynamic-90s FSDP2 training.

This module is loaded as part of a dedicated external Swift plugin. It leaves
the model and dataset semantics unchanged while recording the real Trainer
path, including data dispatch, coarse FSDP module calls, recurrence sampling,
CUDA/NCCL operators, memory, and per-microbatch audio shapes.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import MethodType
from typing import Any


def _enabled() -> bool:
    return os.environ.get("HUGINN_TORCH_PROFILER_ENABLED", "").strip().lower() in {"1", "true", "yes"}


if _enabled():
    import torch
    from transformers import Trainer, TrainerCallback

    SUMMARY_VERSION = "huginn_whisper_dynamic90s_torch_profiler_v1"
    RANK = int(os.environ.get("RANK", "0"))
    OUTPUT_DIR = Path(os.environ.get("HUGINN_TORCH_PROFILER_OUTPUT_DIR", "outputs/torch_profiler"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR = OUTPUT_DIR / f"rank{RANK}"
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    WAIT = int(os.environ.get("HUGINN_TORCH_PROFILER_WAIT", "4"))
    WARMUP = int(os.environ.get("HUGINN_TORCH_PROFILER_WARMUP", "4"))
    ACTIVE = int(os.environ.get("HUGINN_TORCH_PROFILER_ACTIVE", "8"))
    WITH_STACK = os.environ.get("HUGINN_TORCH_PROFILER_WITH_STACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    MAX_EVENT_ROWS = int(os.environ.get("HUGINN_TORCH_PROFILER_MAX_EVENT_ROWS", "1000"))

    if min(WAIT, WARMUP, ACTIVE) < 0 or ACTIVE <= 0:
        raise ValueError(f"Invalid profiler schedule: wait={WAIT} warmup={WARMUP} active={ACTIVE}")
    if MAX_EVENT_ROWS <= 0:
        raise ValueError(f"HUGINN_TORCH_PROFILER_MAX_EVENT_ROWS must be positive: {MAX_EVENT_ROWS}")

    CURRENT_PROFILER: Any = None
    CURRENT_MODULE_COUNTS: Counter[str] | None = None
    CURRENT_RECURRENCES: list[dict[str, int]] | None = None
    DISPATCH_ROWS: list[dict[str, Any]] = []
    MICROBATCH_ROWS: list[dict[str, Any]] = []
    OPTIMIZER_STEP_ROWS: list[dict[str, Any]] = []
    MODULE_CALL_COUNTS: dict[str, Counter[str]] = defaultdict(Counter)

    PROFILED_MODULE_CLASSES = {
        "HuginnAudioForConditionalGeneration": "model_total",
        "WhisperEncoderFSDPUnit": "whisper_encoder",
        "WhisperEncoderLayer": "whisper_encoder_layer",
        "AudioAlignerFSDPUnit": "audio_aligner",
        "HuginnPreludeFSDPUnit": "huginn_prelude",
        "HuginnRecurrentCoreFSDPUnit": "huginn_recurrent_core",
        "HuginnCodaFSDPUnit": "huginn_coda",
        "SandwichBlock": "huginn_sandwich_block",
        "CausalSelfAttention": "huginn_attention",
        "GatedMLP": "huginn_mlp",
    }
    REQUIRED_MODULE_LABELS = tuple(PROFILED_MODULE_CLASSES.values()) + ("lm_head",)

    def _phase(profiler: Any = None) -> str:
        profiler = CURRENT_PROFILER if profiler is None else profiler
        try:
            step_num = int(getattr(profiler, "step_num", 0))
        except (TypeError, ValueError):
            step_num = 0
        if step_num < WAIT:
            return "wait"
        if step_num < WAIT + WARMUP:
            return "warmup"
        if step_num < WAIT + WARMUP + ACTIVE:
            return "active"
        return "post_active"

    def _wall_stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "total_us": 0.0, "mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0, "max_us": 0.0}
        ordered = sorted(values)

        def percentile(fraction: float) -> float:
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        return {
            "count": len(values),
            "total_us": float(sum(values)),
            "mean_us": float(sum(values) / len(values)),
            "p50_us": float(percentile(0.50)),
            "p95_us": float(percentile(0.95)),
            "max_us": float(max(values)),
        }

    def _stats_by_phase(rows: list[dict[str, Any]], value_key: str) -> dict[str, dict[str, float | int]]:
        return {
            phase: _wall_stats([float(row[value_key]) for row in rows if row.get("phase") == phase])
            for phase in ("wait", "warmup", "active", "post_active", "mixed")
        }

    def _cpu_values(value: Any) -> list[Any] | None:
        if not torch.is_tensor(value) or value.device.type != "cpu":
            return None
        return value.detach().reshape(-1).tolist()

    def _input_mapping(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        for candidate in list(args) + list(kwargs.values()):
            if isinstance(candidate, dict) and (
                "input_ids" in candidate or "audio_input_features" in candidate
            ):
                return candidate
        return {}

    def _summarize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key in ("input_ids", "attention_mask", "audio_input_features", "audio_segment_feature_lengths", "audio_segment_mask"):
            value = inputs.get(key)
            if torch.is_tensor(value):
                summary[f"{key}_shape"] = list(value.shape)

        feature_lengths = inputs.get("audio_segment_feature_lengths")
        segment_mask = inputs.get("audio_segment_mask")
        lengths_values = _cpu_values(feature_lengths)
        mask_values = _cpu_values(segment_mask)
        if torch.is_tensor(feature_lengths) and lengths_values is not None:
            rows = feature_lengths.shape[0] if feature_lengths.ndim == 2 else 0
            cols = feature_lengths.shape[1] if feature_lengths.ndim == 2 else 0
            if rows and cols:
                lengths = [lengths_values[index * cols : (index + 1) * cols] for index in range(rows)]
                if mask_values is None:
                    masks = [[value > 0 for value in row] for row in lengths]
                else:
                    masks = [mask_values[index * cols : (index + 1) * cols] for index in range(rows)]
                summary["segment_feature_lengths"] = lengths
                summary["valid_segment_counts"] = [sum(bool(value) for value in row) for row in masks]
                # The model uses valid_encoder_frames=feature_frames//2 and a
                # kernel/stride-6 compressor, then adds audio BOS/EOS.
                summary["estimated_prefix_tokens_per_sample"] = [
                    2 + sum((int(length) // 2) // 6 for length, valid in zip(row_lengths, row_mask) if valid)
                    for row_lengths, row_mask in zip(lengths, masks)
                ]
                summary["local_valid_whisper_segments"] = sum(summary["valid_segment_counts"])
                summary["local_batch_padded_prefix_tokens"] = max(
                    summary["estimated_prefix_tokens_per_sample"],
                    default=0,
                )

        for key in (
            "audio_training_global_positions",
            "audio_training_pool_ids",
            "audio_training_record_indices",
            "audio_training_pool_occurrence_indices",
            "audio_training_pool_epochs",
            "audio_training_effective_duration_seconds",
        ):
            values = _cpu_values(inputs.get(key))
            if values is not None:
                summary[key] = values

        attention_values = _cpu_values(inputs.get("attention_mask"))
        attention_mask = inputs.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_values is not None and attention_mask.ndim == 2:
            width = int(attention_mask.shape[1])
            summary["text_tokens_per_sample"] = [
                int(sum(attention_values[index * width : (index + 1) * width]))
                for index in range(int(attention_mask.shape[0]))
            ]
        input_ids = inputs.get("input_ids")
        if torch.is_tensor(input_ids) and input_ids.ndim == 2 and "local_batch_padded_prefix_tokens" in summary:
            summary["estimated_huginn_sequence_tokens"] = (
                int(summary["local_batch_padded_prefix_tokens"]) + int(input_ids.shape[1])
            )
        return summary

    def _patch_recurrence_sampler(model: torch.nn.Module) -> int:
        patched = 0
        seen: set[int] = set()
        for module in model.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            original = getattr(module, "randomized_iteration_sampler", None)
            if not callable(original) or getattr(original, "_huginn_profiled", False):
                continue

            def sampled(self, *args, __original=original, **kwargs):
                result = __original(*args, **kwargs)
                if CURRENT_RECURRENCES is not None and isinstance(result, tuple) and len(result) == 2:
                    no_grad, with_grad = result
                    CURRENT_RECURRENCES.append(
                        {
                            "no_grad": int(no_grad),
                            "with_grad": int(with_grad),
                            "total": int(no_grad) + int(with_grad),
                        }
                    )
                return result

            sampled._huginn_profiled = True
            module.randomized_iteration_sampler = MethodType(sampled, module)
            patched += 1
        return patched

    def _patch_profiled_modules(model: torch.nn.Module) -> dict[str, list[str]]:
        installed: dict[str, list[str]] = defaultdict(list)
        seen: set[int] = set()
        for name, module in model.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            label = PROFILED_MODULE_CLASSES.get(type(module).__name__)
            if label is None and name.endswith("lm_head"):
                label = "lm_head"
            if label is None or getattr(module.forward, "_huginn_profiled", False):
                continue
            original = module.forward

            def profiled_forward(self, *args, __original=original, __label=label, **kwargs):
                phase = _phase()
                grad_mode = "grad" if torch.is_grad_enabled() else "no_grad"
                MODULE_CALL_COUNTS[__label][f"{phase}:{grad_mode}"] += 1
                if CURRENT_MODULE_COUNTS is not None:
                    CURRENT_MODULE_COUNTS[f"{__label}:{grad_mode}"] += 1
                if phase == "active":
                    with torch.profiler.record_function(f"huginn.module.{__label}.forward.{grad_mode}"):
                        return __original(*args, **kwargs)
                return __original(*args, **kwargs)

            profiled_forward._huginn_profiled = True
            module.forward = MethodType(profiled_forward, module)
            installed[label].append(name)
        return dict(installed)

    def _event_value(event: Any, *names: str) -> float:
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _event_row(event: Any) -> dict[str, Any]:
        return {
            "key": str(event.key),
            "count": int(getattr(event, "count", 0)),
            "cpu_time_total_us": _event_value(event, "cpu_time_total"),
            "self_cpu_time_total_us": _event_value(event, "self_cpu_time_total"),
            "cuda_time_total_us": _event_value(event, "cuda_time_total", "device_time_total"),
            "self_cuda_time_total_us": _event_value(event, "self_cuda_time_total", "self_device_time_total"),
            "flops": _event_value(event, "flops"),
            "input_shapes": str(getattr(event, "input_shapes", "")),
        }

    def _event_category(key: str) -> str | None:
        lowered = key.lower()
        if "huginn.module." in lowered:
            return "module"
        if "dataloader_dispatch" in lowered:
            return "data"
        if "trainer_training_step" in lowered or "optimizer_step" in lowered:
            return "trainer"
        if any(value in lowered for value in ("nccl", "all_gather", "allgather", "reduce_scatter", "broadcast")):
            return "communication"
        if any(value in lowered for value in ("scaled_dot_product", "flex_attention", "flash", "attention")):
            return "attention"
        if any(value in lowered for value in ("mm", "matmul", "gemm", "linear")):
            return "matrix"
        if "conv" in lowered:
            return "convolution"
        return None

    def _write_summary(profiler: Any, installed_modules: dict[str, list[str]], recurrence_modules: int, runtime_us: float) -> None:
        events = [_event_row(event) for event in profiler.key_averages(group_by_input_shape=True)]
        events.sort(key=lambda row: (row["cuda_time_total_us"], row["cpu_time_total_us"]), reverse=True)
        events_by_cpu = sorted(
            events,
            key=lambda row: (row["cpu_time_total_us"], row["cuda_time_total_us"]),
            reverse=True,
        )
        categorized: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in events:
            category = _event_category(row["key"])
            if category is not None:
                categorized[category].append(row)
        for rows in categorized.values():
            rows.sort(key=lambda row: (row["cuda_time_total_us"], row["cpu_time_total_us"]), reverse=True)

        memory = {}
        if torch.cuda.is_available():
            memory = {
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            }
        summary = {
            "summary_version": SUMMARY_VERSION,
            "rank": RANK,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "schedule": {"wait": WAIT, "warmup": WARMUP, "active": ACTIVE},
            "profiler_step_unit": "one Trainer.training_step microbatch",
            "with_stack": WITH_STACK,
            "runtime_us_including_profiler": runtime_us,
            "installed_modules": installed_modules,
            "recurrence_sampler_modules": recurrence_modules,
            "module_call_counts": {name: dict(counts) for name, counts in MODULE_CALL_COUNTS.items()},
            "required_module_labels": list(REQUIRED_MODULE_LABELS),
            "dispatch_wall_time_us": _stats_by_phase(DISPATCH_ROWS, "elapsed_us"),
            "training_step_wall_time_us": _stats_by_phase(MICROBATCH_ROWS, "elapsed_us"),
            "optimizer_step_wall_time_us": _stats_by_phase(OPTIMIZER_STEP_ROWS, "elapsed_us"),
            "dispatch_rows": DISPATCH_ROWS,
            "microbatches": MICROBATCH_ROWS,
            "optimizer_steps": OPTIMIZER_STEP_ROWS,
            "memory": memory,
            "event_count_total": len(events),
            "event_summary_limit": MAX_EVENT_ROWS,
            "events": events[:MAX_EVENT_ROWS],
            "events_by_cpu": events_by_cpu[:MAX_EVENT_ROWS],
            "categorized_events": {
                category: rows[:MAX_EVENT_ROWS]
                for category, rows in categorized.items()
            },
        }
        path = OUTPUT_DIR / f"profiler_summary_rank{RANK}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[torch-profiler] rank={RANK} summary={path} events={len(events)} "
            f"microbatches={len(MICROBATCH_ROWS)} max_allocated={memory.get('max_memory_allocated_bytes', 0)}",
            flush=True,
        )
        print(
            profiler.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
                row_limit=40,
            ),
            flush=True,
        )

    class OptimizerStepWallCallback(TrainerCallback):
        def __init__(self):
            self.started_ns: int | None = None
            self.start_microbatch = 0

        def on_step_begin(self, args, state, control, **kwargs):
            self.started_ns = time.perf_counter_ns()
            self.start_microbatch = len(MICROBATCH_ROWS)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.started_ns is None:
                return control
            phases = {row["phase"] for row in MICROBATCH_ROWS[self.start_microbatch :]}
            phase = next(iter(phases)) if len(phases) == 1 else "mixed"
            OPTIMIZER_STEP_ROWS.append(
                {
                    "global_step": int(state.global_step),
                    "phase": phase,
                    "elapsed_us": (time.perf_counter_ns() - self.started_ns) / 1000.0,
                    "microbatch_start": self.start_microbatch,
                    "microbatch_end": len(MICROBATCH_ROWS),
                }
            )
            self.started_ns = None
            return control

    def _patch_dispatcher() -> None:
        try:
            from accelerate.data_loader import DataLoaderDispatcher
        except Exception as exc:
            print(f"[torch-profiler] DataLoaderDispatcher unavailable: {type(exc).__name__}: {exc}", flush=True)
            return
        original_iter = getattr(DataLoaderDispatcher, "__iter__", None)
        if original_iter is None or getattr(original_iter, "_huginn_profiled", False):
            return

        def profiled_iter(self):
            iterator = original_iter(self)
            while True:
                phase = _phase()
                start_ns = time.perf_counter_ns()
                try:
                    with torch.profiler.record_function(f"huginn.rank{RANK}.dataloader_dispatch_next"):
                        batch = next(iterator)
                except StopIteration:
                    return
                DISPATCH_ROWS.append(
                    {
                        "index": len(DISPATCH_ROWS),
                        "phase": phase,
                        "elapsed_us": (time.perf_counter_ns() - start_ns) / 1000.0,
                    }
                )
                yield batch

        profiled_iter._huginn_profiled = True
        DataLoaderDispatcher.__iter__ = profiled_iter

    if not getattr(Trainer._inner_training_loop, "_huginn_profiled", False):
        original_inner_training_loop = Trainer._inner_training_loop
        original_training_step = Trainer.training_step

        def profiled_training_step(self, *args, **kwargs):
            global CURRENT_MODULE_COUNTS, CURRENT_RECURRENCES
            phase = _phase()
            inputs = _input_mapping(args, kwargs)
            row = {
                "index": len(MICROBATCH_ROWS),
                "phase": phase,
                "inputs": _summarize_inputs(inputs),
            }
            CURRENT_MODULE_COUNTS = Counter()
            CURRENT_RECURRENCES = []
            start_ns = time.perf_counter_ns()
            with torch.profiler.record_function(f"huginn.rank{RANK}.trainer_training_step"):
                result = original_training_step(self, *args, **kwargs)
            row["elapsed_us"] = (time.perf_counter_ns() - start_ns) / 1000.0
            row["module_calls"] = dict(CURRENT_MODULE_COUNTS)
            row["recurrences"] = list(CURRENT_RECURRENCES)
            if torch.cuda.is_available():
                row["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
                row["cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved())
                row["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            MICROBATCH_ROWS.append(row)
            CURRENT_MODULE_COUNTS = None
            CURRENT_RECURRENCES = None
            if CURRENT_PROFILER is not None:
                CURRENT_PROFILER.step()
            return result

        def profiled_inner_training_loop(self, *args, **kwargs):
            global CURRENT_PROFILER
            installed_modules = _patch_profiled_modules(self.model)
            recurrence_modules = _patch_recurrence_sampler(self.model)
            missing = sorted(set(REQUIRED_MODULE_LABELS) - set(installed_modules))
            if missing:
                raise RuntimeError(
                    f"Profiler could not instrument required modules: missing={missing} installed={installed_modules}"
                )
            if recurrence_modules != 1:
                raise RuntimeError(f"Expected exactly one recurrence sampler module, found {recurrence_modules}")
            if not any(isinstance(callback, OptimizerStepWallCallback) for callback in self.callback_handler.callbacks):
                self.add_callback(OptimizerStepWallCallback())

            schedule = torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1)
            handler = torch.profiler.tensorboard_trace_handler(str(TRACE_DIR), worker_name=f"rank{RANK}")
            runtime_start_ns = time.perf_counter_ns()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
                on_trace_ready=handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=WITH_STACK,
                with_flops=True,
            ) as profiler:
                CURRENT_PROFILER = profiler
                try:
                    result = original_inner_training_loop(self, *args, **kwargs)
                finally:
                    CURRENT_PROFILER = None
            runtime_us = (time.perf_counter_ns() - runtime_start_ns) / 1000.0
            _write_summary(profiler, installed_modules, recurrence_modules, runtime_us)
            return result

        profiled_training_step._huginn_profiled = True
        profiled_inner_training_loop._huginn_profiled = True
        Trainer.training_step = profiled_training_step
        Trainer._inner_training_loop = profiled_inner_training_loop
        _patch_dispatcher()
        print(
            f"[torch-profiler] installed Whisper dynamic90s hooks rank={RANK} "
            f"schedule=wait{WAIT},warmup{WARMUP},active{ACTIVE} stack={WITH_STACK} trace_dir={TRACE_DIR}",
            flush=True,
        )
