"""Render deterministic indexed audio-pool selections as Swift SFT rows."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator

from .indexed_atomic_mixture import (
    GLOBAL_POOL_WEIGHTS,
    POOL_ORDER,
    DeterministicHierarchicalMixture,
    IndexedJsonlPool,
)


SYSTEM_PROMPT = "You are a helpful assistant that can understand audio and respond accurately."
TASK_PROMPTS = {
    "AAC": "Listen to the audio and describe what can be heard.",
    "ASR": "Listen to the audio and transcribe the spoken words.",
}
EXPECTED_TASKS = {
    "wavcaps_no_bbc_aac": "AAC",
    "audiocaps_v2_aac": "AAC",
    "clotho_v2_aac": "AAC",
    "gigaspeech_l_asr": "ASR",
}
EXPECTED_CONTRACT_VERSION = "huginn_whisper_dynamic30s_data_v2"


def load_pool_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path).expanduser().resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"Dynamic-90s pool registry is missing: {registry_path}")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Pool registry must be a JSON object: {registry_path}")
    if payload.get("contract_version") != EXPECTED_CONTRACT_VERSION:
        raise ValueError(
            "Pool registry contract is incompatible with the current single-30s route: "
            f"actual={payload.get('contract_version')!r} expected={EXPECTED_CONTRACT_VERSION!r}"
        )
    if payload.get("duration_policy") != "retain_all_then_cap_at30s":
        raise ValueError(
            "Pool registry uses an obsolete duration policy: "
            f"{payload.get('duration_policy')!r}"
        )
    pools = payload.get("pools")
    if not isinstance(pools, dict) or set(pools) != set(POOL_ORDER):
        raise ValueError(f"Unexpected registry pool set: {sorted(pools) if isinstance(pools, dict) else pools!r}")
    weights = payload.get("sampling_weights")
    if not isinstance(weights, dict) or set(weights) != set(GLOBAL_POOL_WEIGHTS):
        raise ValueError(f"Unexpected registry sampling weights: {weights!r}")
    for name, expected_weight in GLOBAL_POOL_WEIGHTS.items():
        if abs(float(weights[name]) - expected_weight) > 1e-12:
            raise ValueError(
                f"Registry weight mismatch for {name}: actual={weights[name]} expected={expected_weight}"
            )
        entry = pools[name]
        if not isinstance(entry, dict):
            raise ValueError(f"Registry pool entry is not an object: {name}")
        required = (
            "manifest_path",
            "index_path",
            "record_count",
            "task",
        )
        missing = [key for key in required if key not in entry]
        if missing:
            raise ValueError(f"Registry pool {name} is missing fields: {missing}")
        if str(entry["task"]) != EXPECTED_TASKS[name]:
            raise ValueError(
                f"Registry task mismatch for {name}: actual={entry['task']!r} expected={EXPECTED_TASKS[name]!r}"
            )
        if int(entry["record_count"]) <= 0:
            raise ValueError(f"Registry pool {name} has invalid record_count={entry['record_count']!r}")
        planning_hours = entry.get("planning_effective_duration_hours")
        if planning_hours is not None and float(planning_hours) <= 0:
            raise ValueError(
                f"Registry pool {name} has invalid planning effective hours="
                f"{planning_hours!r}"
            )
    return payload


def open_indexed_pools(registry: dict[str, Any], stack: ExitStack) -> dict[str, IndexedJsonlPool]:
    pools: dict[str, IndexedJsonlPool] = {}
    for name in POOL_ORDER:
        entry = registry["pools"][name]
        pools[name] = stack.enter_context(
            IndexedJsonlPool(
                name=name,
                manifest_path=entry["manifest_path"],
                index_path=entry["index_path"],
                record_count=int(entry["record_count"]),
            )
        )
    return pools


def validate_atomic_record(pool_name: str, record: dict[str, Any]) -> None:
    if record.get("schema_version") != "huginn_whisper_dynamic90s_atomic_v1":
        raise ValueError(f"Atomic schema mismatch in {pool_name}: {record.get('schema_version')!r}")
    if record.get("task") != EXPECTED_TASKS[pool_name]:
        raise ValueError(f"Atomic task mismatch in {pool_name}: {record.get('task')!r}")
    if not isinstance(record.get("uid"), str) or not record["uid"]:
        raise ValueError(f"Atomic record has no UID in {pool_name}")
    audio = record.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("path"), str) or not audio["path"]:
        raise ValueError(f"Atomic record has invalid audio in {pool_name}: {audio!r}")
    targets = record.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"Atomic record has no targets in {pool_name}")
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise ValueError(f"Atomic record has invalid targets in {pool_name}: {targets!r}")


def render_training_row(
    record: dict[str, Any],
    pool_name: str,
    global_position: int,
    record_index: int,
    pool_occurrence_index: int,
    pool_epoch: int,
    pool_epoch_offset: int,
    target_index: int,
) -> dict[str, Any]:
    validate_atomic_record(pool_name, record)
    task = str(record["task"])
    target = str(record["targets"][target_index]).strip()
    raw_audio = record["audio"]
    # Keep one stable Arrow schema across whole-file AAC and segment-level ASR.
    # The base audio loader treats null segment bounds as a whole-file reference.
    audio = {
        "path": str(raw_audio["path"]),
        "format": str(raw_audio.get("format", "")),
        "start_sec": float(raw_audio["start_sec"]) if raw_audio.get("start_sec") is not None else None,
        "end_sec": float(raw_audio["end_sec"]) if raw_audio.get("end_sec") is not None else None,
        "raw_duration_sec": (
            float(record["raw_duration_sec"]) if record.get("raw_duration_sec") is not None else None
        ),
        # These provenance fields survive the Swift/HF iterable boundary and
        # let the checkpoint smoke audit the exact samples encoded after a
        # cold resume. The production audio loader intentionally ignores them.
        "global_position": int(global_position),
        "pool_name": pool_name,
        "task": task,
        "uid": str(record["uid"]),
        "record_index": int(record_index),
        "pool_occurrence_index": int(pool_occurrence_index),
        "pool_epoch": int(pool_epoch),
        "pool_epoch_offset": int(pool_epoch_offset),
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK_PROMPTS[task]},
            {"role": "assistant", "content": target},
        ],
        "audios": [audio],
        "metadata": {
            "global_position": int(global_position),
            "pool_name": pool_name,
            "record_index": int(record_index),
            "pool_occurrence_index": int(pool_occurrence_index),
            "pool_epoch": int(pool_epoch),
            "pool_epoch_offset": int(pool_epoch_offset),
            "target_index": int(target_index),
            "uid": str(record["uid"]),
            "dataset": str(record.get("dataset", "")),
            "source": str(record.get("source", "")),
            "task": task,
            "raw_duration_sec": (
                float(record["raw_duration_sec"]) if record.get("raw_duration_sec") is not None else -1.0
            ),
        },
    }


def iter_dynamic90s_mixture_rows(
    registry_path: str | Path,
    seed: int,
    start_position: int = 0,
    max_samples: int | None = None,
    pool_occurrence_counts: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic rows beginning at an explicit global sample position."""
    if start_position < 0:
        raise ValueError(f"start_position must be non-negative, got {start_position}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive when set, got {max_samples}")
    registry = load_pool_registry(registry_path)
    pool_sizes = {
        name: int(registry["pools"][name]["record_count"])
        for name in POOL_ORDER
    }
    planner = DeterministicHierarchicalMixture(pool_sizes=pool_sizes, seed=seed)
    with ExitStack() as stack:
        pools = open_indexed_pools(registry, stack)
        emitted = 0
        for selection in planner.iter_selections(start_position, pool_occurrence_counts):
            if max_samples is not None and emitted >= max_samples:
                break
            global_position = selection.global_position
            record = pools[selection.pool_name].record(selection.record_index)
            targets = record.get("targets")
            if not isinstance(targets, list) or not targets:
                raise ValueError(
                    f"Selected record has no targets: pool={selection.pool_name} index={selection.record_index}"
                )
            target_index = planner.target_index(global_position, len(targets))
            yield render_training_row(
                record=record,
                pool_name=selection.pool_name,
                global_position=global_position,
                record_index=selection.record_index,
                pool_occurrence_index=selection.pool_occurrence_index,
                pool_epoch=selection.pool_epoch,
                pool_epoch_offset=selection.pool_epoch_offset,
                target_index=target_index,
            )
            emitted += 1
