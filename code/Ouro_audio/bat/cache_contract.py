"""Runtime checks that the training Dataset uses the job-local Arrow cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _cache_files_from(value: Any) -> list[Path]:
    files: list[Path] = []
    for item in getattr(value, "cache_files", []) or []:
        if isinstance(item, dict) and item.get("filename"):
            files.append(Path(str(item["filename"])).expanduser().resolve())
    return files


def dataset_cache_files(dataset: Any) -> list[Path]:
    """Collect Arrow cache files from a Dataset and common wrapper objects."""
    found: list[Path] = []
    visited: set[int] = set()
    pending: list[Any] = [dataset]
    while pending:
        value = pending.pop()
        if value is None or id(value) in visited:
            continue
        visited.add(id(value))
        found.extend(_cache_files_from(value))
        for name in ("dataset", "train_dataset", "base_dataset"):
            child = getattr(value, name, None)
            if child is not None and child is not value:
                pending.append(child)
        children = getattr(value, "datasets", None)
        if isinstance(children, (list, tuple)):
            pending.extend(children)
    unique: dict[str, Path] = {str(path): path for path in found}
    return list(unique.values())


def assert_local_arrow_cache(dataset: Any, cache_root: str) -> dict[str, Any]:
    root = Path(cache_root).expanduser().resolve()
    files = dataset_cache_files(dataset)
    if not files:
        raise RuntimeError("Training dataset exposes no Arrow cache_files; cannot prove local cache reuse")
    outside: list[str] = []
    for path in files:
        try:
            path.relative_to(root)
        except ValueError:
            outside.append(str(path))
    if outside:
        raise RuntimeError(
            "Training dataset uses Arrow files outside BAT_LOCAL_ARROW_CACHE: "
            f"root={root} outside={outside[:8]}"
        )
    return {
        "status": "ok",
        "cache_root": str(root),
        "cache_files": [str(path) for path in files],
        "cache_file_count": len(files),
        "cache_bytes": sum(path.stat().st_size for path in files if path.is_file()),
    }
