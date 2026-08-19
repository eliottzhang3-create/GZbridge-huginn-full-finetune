"""Low-frequency runtime resource monitoring for long BAT training jobs.

The monitor intentionally appends JSONL records instead of repeatedly replacing
one JSON file.  This is friendlier to CloudStorFS and leaves a useful tail even
when a rank exits through a native signal before Python can finalize training.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def _proc_status() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmSize", "VmPeak", "Threads"}:
                fields = value.strip().split()
                if fields and fields[0].isdigit():
                    # Linux Vm* values are kB; Threads is a count.
                    values[key] = int(fields[0]) * (1024 if key != "Threads" else 1)
    except Exception:
        pass
    return values


def _fd_count() -> int | None:
    try:
        return len(list(Path("/proc/self/fd").iterdir()))
    except Exception:
        return None


def _mapped_region_count() -> int | None:
    try:
        return sum(1 for _ in Path("/proc/self/maps").open("r", encoding="utf-8"))
    except Exception:
        return None


def _df(path: str) -> str | None:
    try:
        result = subprocess.run(
            ["df", "-h", "-P", path],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()
        return lines[-1] if lines else None
    except Exception:
        return None


def collect_runtime_stats(*, step: int, cache_root: str | None = None) -> dict[str, Any]:
    status = _proc_status()
    result: dict[str, Any] = {
        "timestamp": time.time(),
        "step": int(step),
        "pid": os.getpid(),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rss_bytes": status.get("VmRSS"),
        "hwm_bytes": status.get("VmHWM"),
        "virtual_memory_bytes": status.get("VmSize"),
        "peak_virtual_memory_bytes": status.get("VmPeak"),
        "thread_count": status.get("Threads"),
        "open_fd_count": _fd_count(),
        "mapped_region_count": _mapped_region_count(),
        "filesystems": {
            "/dev/shm": _df("/dev/shm"),
            "/tmp": _df("/tmp"),
        },
    }
    if cache_root:
        result["cache_root"] = cache_root
        result["filesystems"]["cache_root"] = _df(cache_root)

    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            result["cuda"] = {
                "device": device,
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
    except Exception as exc:
        result["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return result


class BATRuntimeMonitorCallback(TrainerCallback):
    """A Transformers-compatible callback with sparse append-only telemetry."""

    def __init__(self, output_dir: Path, *, interval_steps: int = 500, cache_root: str | None = None):
        self.output_dir = Path(output_dir)
        self.interval_steps = max(1, int(interval_steps))
        self.cache_root = cache_root
        self.path = self.output_dir / f"runtime_monitor_rank{os.environ.get('RANK', '0')}.jsonl"

    def _record(self, step: int, event: str) -> None:
        payload = collect_runtime_stats(step=step, cache_root=self.cache_root)
        payload["event"] = event
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def on_train_begin(self, args, state, control, **kwargs):
        self._record(int(getattr(state, "global_step", 0)), "train_begin")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0))
        if step > 0 and step % self.interval_steps == 0:
            self._record(step, "periodic")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._record(int(getattr(state, "global_step", 0)), "train_end")
        return control

    def on_save(self, args, state, control, **kwargs):
        self._record(int(getattr(state, "global_step", 0)), "checkpoint_save")
        return control
