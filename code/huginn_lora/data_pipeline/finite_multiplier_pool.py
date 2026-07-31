"""Finite, globally shuffled multiplier pool for Huginn dynamic-30s training."""

from __future__ import annotations

import bisect
import json
import mmap
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .dynamic90s_mixture_rows import SYSTEM_PROMPT, TASK_PROMPTS
from .indexed_atomic_mixture import IndexedJsonlPool, splitmix64


CONTRACT_VERSION = "huginn_whisper_dynamic30s_multiplier_pool_v1"
SAMPLER_VERSION = "finite_global_shuffle_multiplier_v1"
STATISTICS_VERSION = "huginn_dynamic30s_multiplier_training_statistics_v1"
INDEX_FORMAT = "little-endian uint64 without header"
GLOBAL_BATCH_SIZE = 32

COMPONENT_ORDER = (
    "gigaspeech_m_asr",
    "audiocaps_v2_aac",
    "clotho_v2_aac",
    "wavcaps_audioset_aac",
    "wavcaps_soundbible_aac",
    "wavcaps_freesound_quarter_aac",
)
POOL_ORDER = (
    "wavcaps_no_bbc_aac",
    "audiocaps_v2_aac",
    "clotho_v2_aac",
    "gigaspeech_l_asr",
)
EXPECTED_MULTIPLIERS = {
    "gigaspeech_m_asr": 1,
    "audiocaps_v2_aac": 3,
    "clotho_v2_aac": 3,
    "wavcaps_audioset_aac": 2,
    "wavcaps_soundbible_aac": 2,
    "wavcaps_freesound_quarter_aac": 1,
}
EXPECTED_TASKS = {
    "gigaspeech_m_asr": "ASR",
    "audiocaps_v2_aac": "AAC",
    "clotho_v2_aac": "AAC",
    "wavcaps_audioset_aac": "AAC",
    "wavcaps_soundbible_aac": "AAC",
    "wavcaps_freesound_quarter_aac": "AAC",
}
EXPECTED_POOL_NAMES = {
    "gigaspeech_m_asr": "gigaspeech_l_asr",
    "audiocaps_v2_aac": "audiocaps_v2_aac",
    "clotho_v2_aac": "clotho_v2_aac",
    "wavcaps_audioset_aac": "wavcaps_no_bbc_aac",
    "wavcaps_soundbible_aac": "wavcaps_no_bbc_aac",
    "wavcaps_freesound_quarter_aac": "wavcaps_no_bbc_aac",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Multiplier-pool JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Multiplier-pool JSON must be an object: {path}")
    return payload


def load_multiplier_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path).expanduser().resolve()
    payload = _load_json(registry_path)
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"Multiplier contract mismatch: actual={payload.get('contract_version')!r} "
            f"expected={CONTRACT_VERSION!r}"
        )
    if payload.get("sampler_version") != SAMPLER_VERSION:
        raise ValueError(f"Multiplier sampler mismatch: {payload.get('sampler_version')!r}")
    if payload.get("duration_policy") != "retain_all_then_cap_at30s":
        raise ValueError(f"Multiplier duration policy mismatch: {payload.get('duration_policy')!r}")
    if int(payload.get("global_batch_size", -1)) != GLOBAL_BATCH_SIZE:
        raise ValueError(f"Multiplier global batch mismatch: {payload.get('global_batch_size')!r}")
    components = payload.get("components")
    if not isinstance(components, dict) or tuple(components) != COMPONENT_ORDER:
        raise ValueError(f"Multiplier component order mismatch: {list(components or {})}")
    pools = payload.get("pools")
    if not isinstance(pools, dict) or tuple(pools) != POOL_ORDER:
        raise ValueError(f"Multiplier aggregate pool order mismatch: {list(pools or {})}")

    expected_virtual_start = 0
    aggregate_offsets = {name: 0 for name in POOL_ORDER}
    for name in COMPONENT_ORDER:
        entry = components[name]
        selected_count = int(entry.get("selected_record_count", 0))
        multiplier = int(entry.get("multiplier", 0))
        expanded_count = int(entry.get("expanded_record_count", 0))
        pool_name = str(entry.get("pool_name", ""))
        if selected_count <= 0 or multiplier != EXPECTED_MULTIPLIERS[name]:
            raise ValueError(f"Invalid multiplier component cardinality: {name}={entry}")
        if expanded_count != selected_count * multiplier:
            raise ValueError(f"Invalid expanded count for {name}: {entry}")
        if entry.get("task") != EXPECTED_TASKS[name] or pool_name != EXPECTED_POOL_NAMES[name]:
            raise ValueError(f"Invalid task/pool mapping for {name}: {entry}")
        if int(entry.get("virtual_start", -1)) != expected_virtual_start:
            raise ValueError(f"Non-contiguous virtual range for {name}: {entry}")
        expected_virtual_start += expanded_count
        if int(entry.get("virtual_end", -1)) != expected_virtual_start:
            raise ValueError(f"Invalid virtual end for {name}: {entry}")
        if int(entry.get("aggregate_pool_offset", -1)) != aggregate_offsets[pool_name]:
            raise ValueError(f"Invalid aggregate pool offset for {name}: {entry}")
        aggregate_offsets[pool_name] += expanded_count
        for key in ("manifest_path", "index_path"):
            if not Path(entry[key]).is_file():
                raise FileNotFoundError(f"Component {name} source artifact is missing: {entry[key]}")
        selection_path = entry.get("selection_index_path")
        if selection_path is not None:
            selection = Path(selection_path)
            if not selection.is_file() or selection.stat().st_size != selected_count * 8:
                raise ValueError(f"Invalid selection index for {name}: {selection}")

    total_records = int(payload.get("total_records", -1))
    if total_records != expected_virtual_start or total_records % GLOBAL_BATCH_SIZE:
        raise ValueError(
            f"Multiplier total must equal virtual coverage and divide global batch: "
            f"total={total_records} virtual={expected_virtual_start}"
        )
    schedule_path = Path(payload.get("schedule_path", ""))
    if not schedule_path.is_file() or schedule_path.stat().st_size != total_records * 8:
        raise ValueError(f"Multiplier schedule size mismatch: {schedule_path}")
    if int(payload.get("max_steps", -1)) != total_records // GLOBAL_BATCH_SIZE:
        raise ValueError(f"Multiplier max_steps mismatch: {payload.get('max_steps')!r}")
    expected_pool_sizes = {name: int(pools[name]["record_count"]) for name in POOL_ORDER}
    if expected_pool_sizes != aggregate_offsets:
        raise ValueError(
            f"Aggregate pool sizes differ from component exposure: "
            f"pools={expected_pool_sizes} components={aggregate_offsets}"
        )
    return payload


class UInt64Index:
    def __init__(self, path: str | Path, count: int) -> None:
        self.path = Path(path)
        self.count = int(count)
        if self.count <= 0 or self.path.stat().st_size != self.count * 8:
            raise ValueError(f"Invalid uint64 index: path={self.path} count={self.count}")
        self._handle = None
        self._mmap = None

    def __enter__(self) -> "UInt64Index":
        self._handle = self.path.open("rb")
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._handle is not None:
            self._handle.close()
        self._mmap = None
        self._handle = None

    def __getitem__(self, index: int) -> int:
        if self._mmap is None:
            raise RuntimeError(f"Index is not open: {self.path}")
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return int(struct.unpack_from("<Q", self._mmap, index * 8)[0])


@dataclass(frozen=True)
class MultiplierSelection:
    global_position: int
    schedule_slot: int
    component_name: str
    pool_name: str
    task: str
    replica_id: int
    selection_offset: int
    record_index: int
    pool_occurrence_index: int


class FiniteMultiplierPool:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path).expanduser().resolve()
        self.registry = load_multiplier_registry(self.registry_path)
        self.seed = int(self.registry["seed"])
        self.total_records = int(self.registry["total_records"])
        self.component_starts = [
            int(self.registry["components"][name]["virtual_start"])
            for name in COMPONENT_ORDER
        ]
        self._stack: ExitStack | None = None
        self._schedule: UInt64Index | None = None
        self._pools: dict[str, IndexedJsonlPool] = {}
        self._selection_indices: dict[str, UInt64Index] = {}

    def __enter__(self) -> "FiniteMultiplierPool":
        self._stack = ExitStack()
        self._schedule = self._stack.enter_context(
            UInt64Index(self.registry["schedule_path"], self.total_records)
        )
        for name in COMPONENT_ORDER:
            entry = self.registry["components"][name]
            self._pools[name] = self._stack.enter_context(
                IndexedJsonlPool(
                    name=name,
                    manifest_path=entry["manifest_path"],
                    index_path=entry["index_path"],
                    record_count=int(entry["base_record_count"]),
                )
            )
            selection_path = entry.get("selection_index_path")
            if selection_path is not None:
                self._selection_indices[name] = self._stack.enter_context(
                    UInt64Index(selection_path, int(entry["selected_record_count"]))
                )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._schedule = None
        self._pools = {}
        self._selection_indices = {}

    def selection(self, global_position: int) -> MultiplierSelection:
        if self._schedule is None:
            raise RuntimeError("FiniteMultiplierPool must be entered as a context manager")
        if global_position < 0 or global_position >= self.total_records:
            raise IndexError(global_position)
        slot = self._schedule[global_position]
        component_index = bisect.bisect_right(self.component_starts, slot) - 1
        if component_index < 0:
            raise RuntimeError(f"Schedule slot is outside component ranges: {slot}")
        component_name = COMPONENT_ORDER[component_index]
        entry = self.registry["components"][component_name]
        if slot >= int(entry["virtual_end"]):
            raise RuntimeError(f"Schedule slot is outside {component_name}: {slot}")
        local_slot = slot - int(entry["virtual_start"])
        selected_count = int(entry["selected_record_count"])
        replica_id, selection_offset = divmod(local_slot, selected_count)
        selection_index = self._selection_indices.get(component_name)
        record_index = selection_index[selection_offset] if selection_index else selection_offset
        return MultiplierSelection(
            global_position=global_position,
            schedule_slot=slot,
            component_name=component_name,
            pool_name=str(entry["pool_name"]),
            task=str(entry["task"]),
            replica_id=replica_id,
            selection_offset=selection_offset,
            record_index=record_index,
            pool_occurrence_index=int(entry["aggregate_pool_offset"]) + local_slot,
        )

    def record(self, selection: MultiplierSelection) -> dict[str, Any]:
        return self._pools[selection.component_name].record(selection.record_index)

    def pool_counts_before(self, position: int) -> dict[str, int]:
        if position < 0 or position > self.total_records:
            raise ValueError(f"Position outside finite schedule: {position}")
        counts = {name: 0 for name in POOL_ORDER}
        for global_position in range(position):
            counts[self.selection(global_position).pool_name] += 1
        return counts


def _validate_atomic_record(record: dict[str, Any], selection: MultiplierSelection) -> None:
    if record.get("task") != selection.task:
        raise ValueError(
            f"Task mismatch for {selection.component_name}: "
            f"record={record.get('task')!r} expected={selection.task!r}"
        )
    if not isinstance(record.get("uid"), str) or not record["uid"]:
        raise ValueError(f"Atomic record has no UID: {selection}")
    audio = record.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("path"), str):
        raise ValueError(f"Atomic record has invalid audio: {selection}")
    targets = record.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"Atomic record has no targets: {selection}")


def render_multiplier_row(
    record: dict[str, Any],
    selection: MultiplierSelection,
    seed: int,
) -> dict[str, Any]:
    _validate_atomic_record(record, selection)
    targets = record["targets"]
    target_index = splitmix64(seed ^ selection.global_position ^ 0xA5A5A5A5A5A5A5A5) % len(targets)
    target = str(targets[target_index]).strip()
    raw_audio = record["audio"]
    audio = {
        "path": str(raw_audio["path"]),
        "format": str(raw_audio.get("format", "")),
        "start_sec": float(raw_audio["start_sec"]) if raw_audio.get("start_sec") is not None else None,
        "end_sec": float(raw_audio["end_sec"]) if raw_audio.get("end_sec") is not None else None,
        "raw_duration_sec": (
            float(record["raw_duration_sec"]) if record.get("raw_duration_sec") is not None else None
        ),
        "global_position": selection.global_position,
        "pool_name": selection.pool_name,
        "task": selection.task,
        "uid": str(record["uid"]),
        "record_index": selection.record_index,
        "pool_occurrence_index": selection.pool_occurrence_index,
        "pool_epoch": selection.replica_id,
        "pool_epoch_offset": selection.selection_offset,
        "component_name": selection.component_name,
        "replica_id": selection.replica_id,
        "schedule_slot": selection.schedule_slot,
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK_PROMPTS[selection.task]},
            {"role": "assistant", "content": target},
        ],
        "audios": [audio],
        "metadata": {
            "global_position": selection.global_position,
            "schedule_slot": selection.schedule_slot,
            "component_name": selection.component_name,
            "pool_name": selection.pool_name,
            "task": selection.task,
            "replica_id": selection.replica_id,
            "record_index": selection.record_index,
            "selection_offset": selection.selection_offset,
            "pool_occurrence_index": selection.pool_occurrence_index,
            "target_index": int(target_index),
            "uid": str(record["uid"]),
            "raw_duration_sec": (
                float(record["raw_duration_sec"]) if record.get("raw_duration_sec") is not None else -1.0
            ),
        },
    }


def iter_multiplier_rows(
    registry_path: str | Path,
    start_position: int = 0,
    max_samples: int | None = None,
) -> Iterator[dict[str, Any]]:
    with FiniteMultiplierPool(registry_path) as pool:
        if start_position < 0 or start_position > pool.total_records:
            raise ValueError(f"Invalid multiplier start position: {start_position}")
        remaining = pool.total_records - start_position
        limit = remaining if max_samples is None else min(remaining, int(max_samples))
        if limit < 0:
            raise ValueError(f"Invalid multiplier max_samples: {max_samples}")
        for global_position in range(start_position, start_position + limit):
            selection = pool.selection(global_position)
            yield render_multiplier_row(pool.record(selection), selection, pool.seed)
