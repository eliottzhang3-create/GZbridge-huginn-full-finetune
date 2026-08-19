"""Small dependency-light helpers shared by BAT failure-isolation audits."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


def require_private_absolute(path: Path) -> Path:
    path = path.expanduser()
    normalized = str(path).replace("\\", "/")
    if not path.is_absolute() or normalized.startswith("/hpc_stor03/public"):
        raise ValueError(f"Output must be a private absolute path: {path}")
    return path


def write_json(path: Path, payload: dict[str, Any], *, atomic: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if not atomic:
        path.write_text(text, encoding="utf-8")
        return
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except FileNotFoundError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            yield line_number, value


def process_stats() -> dict[str, Any]:
    result: dict[str, Any] = {
        "pid": os.getpid(),
        "rss_bytes": None,
        "open_fd_count": None,
        "mapped_region_count": None,
    }
    try:
        result["rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        pass
    try:
        result["open_fd_count"] = len(list(Path("/proc/self/fd").iterdir()))
    except Exception:
        pass
    try:
        result["mapped_region_count"] = sum(1 for _ in Path("/proc/self/maps").open("r", encoding="utf-8"))
    except Exception:
        pass
    return result


def filesystem_stats(paths: Iterable[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        try:
            completed = subprocess.run(
                ["df", "-h", "-P", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            result[str(path)] = completed.stdout.strip().splitlines()[-1]
        except Exception as exc:
            result[str(path)] = f"unavailable:{type(exc).__name__}:{exc}"
    return result
