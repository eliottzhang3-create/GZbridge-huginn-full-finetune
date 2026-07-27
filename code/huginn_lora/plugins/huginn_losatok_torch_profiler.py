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

    def _event_value(event: Any, *names: str) -> float:
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return 0.0

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
            "events": events[:200],
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
            profiler = _CURRENT_PROFILER
            if profiler is None:
                yield from iterator
                return
            while True:
                try:
                    with torch.profiler.record_function(f"huginn.rank{_RANK}.dataloader_dispatch_next"):
                        batch = next(iterator)
                except StopIteration:
                    return
                yield batch

        profiled_iter._huginn_torch_profiler = True
        DataLoaderDispatcher.__iter__ = profiled_iter
        print("[torch-profiler] installed Accelerate DataLoaderDispatcher timing hook", flush=True)

    if not getattr(Trainer._inner_training_loop, "_huginn_torch_profiler", False):
        _original_inner_training_loop = Trainer._inner_training_loop
        _original_training_step = Trainer.training_step

        def profiled_training_step(self, *args, **kwargs):
            profiler = getattr(self, "_huginn_active_profiler", None)
            with torch.profiler.record_function(f"huginn.rank{_RANK}.trainer_training_step"):
                result = _original_training_step(self, *args, **kwargs)
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
