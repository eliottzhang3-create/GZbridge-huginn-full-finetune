"""Opt-in Torch Profiler hooks for the real Swift Trainer/FSDP2 path.

The hooks are enabled only when ``HUGINN_TORCH_PROFILER_ENABLED=1``.  They
profile the Trainer's actual inner loop, so CPU data wait, WebDataset/FLAC
decode, DataLoaderDispatcher activity, CUDA kernels, NCCL collectives, and
memory events can appear in the per-rank traces.  No checkpoint or model state
is written by this module.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _enabled() -> bool:
    return os.environ.get("HUGINN_TORCH_PROFILER_ENABLED", "").strip().lower() in {"1", "true", "yes"}


if _enabled():
    import torch
    from transformers import Trainer

    _RANK = int(os.environ.get("RANK", "0"))
    _OUTPUT_DIR = Path(os.environ.get("HUGINN_TORCH_PROFILER_OUTPUT_DIR", "outputs/torch_profiler"))
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _TRACE_DIR = _OUTPUT_DIR / f"rank{_RANK}"
    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _WAIT = int(os.environ.get("HUGINN_TORCH_PROFILER_WAIT", "1"))
    _WARMUP = int(os.environ.get("HUGINN_TORCH_PROFILER_WARMUP", "1"))
    _ACTIVE = int(os.environ.get("HUGINN_TORCH_PROFILER_ACTIVE", "4"))
    _patched = False
    _CURRENT_PROFILER = None
    # These are deliberately kept separate from the CUDA-sorted top-200 event
    # list.  DataLoaderDispatcher is primarily a CPU-side event and can
    # otherwise disappear from the compact JSON summary even though it is
    # present in the full Chrome trace.
    _DISPATCH_WALL_US = {"wait": [], "warmup": [], "active": [], "post_active": []}
    _TRAINING_STEP_WALL_US = {"wait": [], "warmup": [], "active": [], "post_active": []}

    def _event_value(event: Any, *names: str) -> float:
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _profiler_phase(profiler: Any) -> str:
        """Return the schedule phase for the next operation.

        ``profiler.step()`` is called after each Trainer.training_step in this
        plugin, so ``step_num`` is a micro-batch counter.  The dispatcher
        fetches the next batch before training_step, hence this phase is the
        phase in which that fetch is recorded.
        """
        try:
            step_num = int(getattr(profiler, "step_num", 0))
        except (TypeError, ValueError):
            step_num = 0
        if step_num < _WAIT:
            return "wait"
        if step_num < _WAIT + _WARMUP:
            return "warmup"
        if step_num < _WAIT + _WARMUP + _ACTIVE:
            return "active"
        return "post_active"

    def _record_wall_time(target: dict[str, list[float]], phase: str, elapsed_us: float) -> None:
        # The profiler smoke is short, but keep a hard cap so an accidentally
        # long profiling run cannot grow Python-side lists without bound.
        values = target.setdefault(phase, [])
        if len(values) < 10000:
            values.append(float(elapsed_us))

    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def _wall_stats_by_phase(values_by_phase: dict[str, list[float]]) -> dict[str, Any]:
        result = {}
        for phase, values in values_by_phase.items():
            if not values:
                result[phase] = {
                    "count": 0,
                    "total_us": 0.0,
                    "mean_us": 0.0,
                    "p50_us": 0.0,
                    "p95_us": 0.0,
                    "max_us": 0.0,
                }
                continue
            result[phase] = {
                "count": len(values),
                "total_us": float(sum(values)),
                "mean_us": float(sum(values) / len(values)),
                "p50_us": float(_percentile(values, 0.50)),
                "p95_us": float(_percentile(values, 0.95)),
                "max_us": float(max(values)),
            }
        return result

    def _is_selected_event(key: str) -> bool:
        key_lower = key.lower()
        return any(
            pattern in key_lower
            for pattern in (
                "dataloader_dispatch",
                "trainer_training_step",
                "profilerstep",
                "fsdp::all_gather",
                "record_param_comms",
                "nccl",
                "allgather",
                "all_gather",
                "reduce_scatter",
                "broadcast",
            )
        )

    def _write_summary(profiler: Any) -> None:
        events = []
        for event in profiler.key_averages(group_by_input_shape=True):
            events.append(
                {
                    "key": str(event.key),
                    "count": int(getattr(event, "count", 0)),
                    "cpu_time_total_us": _event_value(event, "cpu_time_total"),
                    "self_cpu_time_total_us": _event_value(event, "self_cpu_time_total"),
                    "cuda_time_total_us": _event_value(event, "cuda_time_total", "device_time_total"),
                    "self_cuda_time_total_us": _event_value(
                        event, "self_cuda_time_total", "self_device_time_total"
                    ),
                    "input_shapes": str(getattr(event, "input_shapes", "")),
                }
            )
        events.sort(key=lambda item: item["cuda_time_total_us"], reverse=True)
        selected_events = [item for item in events if _is_selected_event(item["key"])]
        selected_events.sort(
            key=lambda item: (
                item["key"].lower().find("dataloader_dispatch") < 0,
                -item["cpu_time_total_us"],
            )
        )
        if torch.cuda.is_available():
            memory = {
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            }
        else:
            memory = {}
        summary = {
            "rank": _RANK,
            "wait": _WAIT,
            "warmup": _WARMUP,
            "active": _ACTIVE,
            "profiler_step_unit": "Trainer.training_step (one micro-batch)",
            "event_count_total": len(events),
            "event_summary_limit": 200,
            "events": events[:200],
            # Unlike ``events``, this list is not truncated by CUDA ranking.
            # It is intentionally restricted to data/FSDP/NCCL/training
            # events so the summary remains small and directly actionable.
            "selected_events": selected_events,
            "data_loader_dispatch_wall_time_us": _wall_stats_by_phase(_DISPATCH_WALL_US),
            "training_step_wall_time_us": _wall_stats_by_phase(_TRAINING_STEP_WALL_US),
            "memory": memory,
        }
        path = _OUTPUT_DIR / f"profiler_summary_rank{_RANK}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[torch-profiler] rank={_RANK} summary={path} "
            f"events={len(events)} max_allocated={memory.get('max_memory_allocated_bytes', 0)}",
            flush=True,
        )
        print(
            profiler.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
                row_limit=30,
            ),
            flush=True,
        )
        print(
            f"[torch-profiler] rank={_RANK} dispatch_active_stats="
            f"{summary['data_loader_dispatch_wall_time_us']['active']} "
            f"training_step_active_stats={summary['training_step_wall_time_us']['active']}",
            flush=True,
        )

    def _patch_dispatcher() -> None:
        try:
            from accelerate.data_loader import DataLoaderDispatcher
        except Exception as exc:  # noqa: BLE001 - retain a visible diagnostic
            print(f"[torch-profiler] DataLoaderDispatcher patch unavailable: {type(exc).__name__}: {exc}", flush=True)
            return
        original_iter = getattr(DataLoaderDispatcher, "__iter__", None)
        if original_iter is None or getattr(original_iter, "_huginn_torch_profiler", False):
            return

        def profiled_iter(self):
            iterator = original_iter(self)
            while True:
                # Resolve the profiler dynamically rather than only at
                # iterator construction time.  Swift/Transformers can create
                # the iterator before the Trainer enters the profiler context.
                profiler = _CURRENT_PROFILER
                phase = _profiler_phase(profiler) if profiler is not None else "post_active"
                start_ns = time.perf_counter_ns()
                try:
                    if profiler is None:
                        batch = next(iterator)
                    else:
                        with torch.profiler.record_function(f"huginn.rank{_RANK}.dataloader_dispatch_next"):
                            batch = next(iterator)
                except StopIteration:
                    return
                elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
                if profiler is not None:
                    _record_wall_time(_DISPATCH_WALL_US, phase, elapsed_us)
                yield batch

        profiled_iter._huginn_torch_profiler = True
        DataLoaderDispatcher.__iter__ = profiled_iter
        print("[torch-profiler] installed Accelerate DataLoaderDispatcher timing hook", flush=True)

    if not getattr(Trainer._inner_training_loop, "_huginn_torch_profiler", False):
        _original_inner_training_loop = Trainer._inner_training_loop
        _original_training_step = Trainer.training_step

        def profiled_training_step(self, *args, **kwargs):
            profiler = getattr(self, "_huginn_active_profiler", None)
            phase = _profiler_phase(profiler) if profiler is not None else "post_active"
            start_ns = time.perf_counter_ns()
            with torch.profiler.record_function(f"huginn.rank{_RANK}.trainer_training_step"):
                result = _original_training_step(self, *args, **kwargs)
            if profiler is not None:
                elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
                _record_wall_time(_TRAINING_STEP_WALL_US, phase, elapsed_us)
            if profiler is not None:
                profiler.step()
            return result

        def profiled_inner_training_loop(self, *args, **kwargs):
            global _CURRENT_PROFILER
            schedule = torch.profiler.schedule(wait=_WAIT, warmup=_WARMUP, active=_ACTIVE, repeat=1)
            handler = torch.profiler.tensorboard_trace_handler(str(_TRACE_DIR), worker_name=f"rank{_RANK}")
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=schedule,
                on_trace_ready=handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as profiler:
                _CURRENT_PROFILER = profiler
                self._huginn_active_profiler = profiler
                try:
                    result = _original_inner_training_loop(self, *args, **kwargs)
                finally:
                    self._huginn_active_profiler = None
                    _CURRENT_PROFILER = None
            _write_summary(profiler)
            return result

        profiled_training_step._huginn_torch_profiler = True
        profiled_inner_training_loop._huginn_torch_profiler = True
        Trainer.training_step = profiled_training_step
        Trainer._inner_training_loop = profiled_inner_training_loop
        _patch_dispatcher()
        _patched = True
        print(
            f"[torch-profiler] installed Trainer/FSDP2 hooks rank={_RANK} "
            f"schedule=wait{_WAIT},warmup{_WARMUP},active{_ACTIVE} trace_dir={_TRACE_DIR}",
            flush=True,
        )
