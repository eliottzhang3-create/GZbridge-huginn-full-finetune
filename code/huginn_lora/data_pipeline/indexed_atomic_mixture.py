"""Indexed JSONL access and deterministic hierarchical mixture selection."""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


MASK64 = (1 << 64) - 1
SAMPLER_VERSION = "deterministic_hierarchical_no_replacement_v2"
POOL_ORDER = (
    "wavcaps_no_bbc_aac",
    "audiocaps_v2_aac",
    "clotho_v2_aac",
    "gigaspeech_l_asr",
)
GLOBAL_POOL_WEIGHTS = {
    "wavcaps_no_bbc_aac": 0.36,
    "audiocaps_v2_aac": 0.18,
    "clotho_v2_aac": 0.06,
    "gigaspeech_l_asr": 0.40,
}
POOL_STREAM_IDS = {name: index + 1 for index, name in enumerate(POOL_ORDER)}


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def unit_interval(value: int) -> float:
    return value / float(1 << 64)


@dataclass(frozen=True)
class MixtureSelection:
    global_position: int
    pool_name: str
    record_index: int
    pool_occurrence_index: int
    pool_epoch: int
    pool_epoch_offset: int


def _feistel_permute(value: int, domain_bits: int, key: int) -> int:
    """Apply a deterministic permutation over an even-bit power-of-two domain."""
    if domain_bits <= 0 or domain_bits % 2:
        raise ValueError(f"domain_bits must be a positive even integer, got {domain_bits}")
    half_bits = domain_bits // 2
    half_mask = (1 << half_bits) - 1
    left = int(value) >> half_bits
    right = int(value) & half_mask
    for round_index in range(6):
        round_key = splitmix64(key ^ ((round_index + 1) * 0xD6E8FEB86659FD93 & MASK64))
        round_value = splitmix64(round_key ^ right) & half_mask
        left, right = right, left ^ round_value
    return (left << half_bits) | right


def permute_bounded_index(index: int, size: int, key: int) -> int:
    """Cycle-walk a keyed Feistel permutation into ``[0, size)``.

    The mapping is a bijection for every key, so each dataset epoch covers
    every atomic record exactly once without materializing a permutation.
    """
    if size <= 0 or index < 0 or index >= size:
        raise ValueError(f"Invalid bounded permutation request: index={index} size={size}")
    if size == 1:
        return 0
    domain_bits = max(2, (int(size) - 1).bit_length())
    if domain_bits % 2:
        domain_bits += 1
    value = int(index)
    while True:
        value = _feistel_permute(value, domain_bits, int(key) & MASK64)
        if value < size:
            return value


class IndexedJsonlPool:
    """Random access to a JSONL manifest using its uint64 offset sidecar."""

    def __init__(self, name: str, manifest_path: str | Path, index_path: str | Path, record_count: int) -> None:
        self.name = name
        self.manifest_path = Path(manifest_path)
        self.index_path = Path(index_path)
        self.record_count = int(record_count)
        if self.record_count <= 0:
            raise ValueError(f"Pool {name} must have a positive record count")
        if not self.manifest_path.is_file() or not self.index_path.is_file():
            raise FileNotFoundError(
                f"Pool files are missing: name={name} manifest={self.manifest_path} index={self.index_path}"
            )
        expected_index_size = self.record_count * 8
        actual_index_size = self.index_path.stat().st_size
        if actual_index_size != expected_index_size:
            raise ValueError(
                f"Index size mismatch for {name}: expected={expected_index_size} actual={actual_index_size}"
            )
        self._manifest_handle = None
        self._index_handle = None
        self._index_map = None

    def _ensure_open(self) -> None:
        if self._manifest_handle is not None:
            return
        self._manifest_handle = self.manifest_path.open("rb")
        self._index_handle = self.index_path.open("rb")
        self._index_map = mmap.mmap(self._index_handle.fileno(), length=0, access=mmap.ACCESS_READ)

    def offset(self, record_index: int) -> int:
        if record_index < 0 or record_index >= self.record_count:
            raise IndexError(f"Pool {self.name} record index is out of range: {record_index}")
        self._ensure_open()
        assert self._index_map is not None
        return struct.unpack_from("<Q", self._index_map, record_index * 8)[0]

    def record(self, record_index: int) -> dict[str, Any]:
        offset = self.offset(record_index)
        assert self._manifest_handle is not None
        self._manifest_handle.seek(offset)
        line = self._manifest_handle.readline()
        if not line:
            raise ValueError(f"Pool {self.name} index points beyond JSONL data: index={record_index} offset={offset}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Pool {self.name} record is not an object: index={record_index}")
        return payload

    def close(self) -> None:
        if self._index_map is not None:
            self._index_map.close()
        if self._index_handle is not None:
            self._index_handle.close()
        if self._manifest_handle is not None:
            self._manifest_handle.close()
        self._index_map = None
        self._index_handle = None
        self._manifest_handle = None

    def __enter__(self) -> "IndexedJsonlPool":
        self._ensure_open()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_manifest_handle"] = None
        state["_index_handle"] = None
        state["_index_map"] = None
        return state


class DeterministicHierarchicalMixture:
    """Deterministic hierarchical weighting with per-pool no-replacement epochs."""

    def __init__(self, pool_sizes: dict[str, int], seed: int = 20260730) -> None:
        if set(pool_sizes) != set(POOL_ORDER):
            raise ValueError(f"Unexpected pool set: {sorted(pool_sizes)}")
        if any(int(size) <= 0 for size in pool_sizes.values()):
            raise ValueError(f"Every pool size must be positive: {pool_sizes}")
        self.pool_sizes = {name: int(pool_sizes[name]) for name in POOL_ORDER}
        self.seed = int(seed) & MASK64
        self.sampler_version = SAMPLER_VERSION

    def _draw(self, global_position: int, stream: int) -> int:
        if global_position < 0:
            raise ValueError(f"global_position must be non-negative, got {global_position}")
        position_key = (int(global_position) * 0xD2B74407B1CE6E93) & MASK64
        stream_key = (int(stream) * 0x9E3779B185EBCA87) & MASK64
        return splitmix64(self.seed ^ position_key ^ stream_key)

    def pool_for_position(self, global_position: int) -> str:
        task_draw = unit_interval(self._draw(global_position, 1))
        if task_draw >= 0.60:
            return "gigaspeech_l_asr"
        aac_draw = unit_interval(self._draw(global_position, 2))
        if aac_draw < 0.60:
            return "wavcaps_no_bbc_aac"
        if aac_draw < 0.90:
            return "audiocaps_v2_aac"
        return "clotho_v2_aac"

    def pool_occurrence_counts_before(self, global_position: int) -> dict[str, int]:
        """Count deterministic pool selections in ``[0, global_position)``."""
        if global_position < 0:
            raise ValueError(f"global_position must be non-negative, got {global_position}")
        counts = {name: 0 for name in POOL_ORDER}
        for position in range(int(global_position)):
            counts[self.pool_for_position(position)] += 1
        return counts

    def record_index_for_occurrence(self, pool_name: str, pool_occurrence_index: int) -> tuple[int, int, int]:
        """Map one pool-local occurrence to a unique record in its current epoch."""
        if pool_name not in self.pool_sizes or pool_occurrence_index < 0:
            raise ValueError(
                f"Invalid pool occurrence request: pool={pool_name!r} occurrence={pool_occurrence_index}"
            )
        pool_size = self.pool_sizes[pool_name]
        pool_epoch, pool_epoch_offset = divmod(int(pool_occurrence_index), pool_size)
        pool_key = splitmix64(
            self.seed
            ^ (POOL_STREAM_IDS[pool_name] * 0xA24BAED4963EE407 & MASK64)
            ^ (pool_epoch * 0x9FB21C651E98DF25 & MASK64)
        )
        record_index = permute_bounded_index(pool_epoch_offset, pool_size, pool_key)
        return record_index, pool_epoch, pool_epoch_offset

    def selection(
        self,
        global_position: int,
        pool_occurrence_counts: dict[str, int] | None = None,
    ) -> MixtureSelection:
        """Select one record, reconstructing prior pool counts when needed."""
        counts = (
            self.pool_occurrence_counts_before(global_position)
            if pool_occurrence_counts is None
            else {name: int(pool_occurrence_counts[name]) for name in POOL_ORDER}
        )
        if sum(counts.values()) != global_position:
            raise ValueError(
                "Pool occurrence counts do not match global_position: "
                f"position={global_position} counts={counts}"
            )
        pool_name = self.pool_for_position(global_position)
        pool_occurrence_index = counts[pool_name]
        record_index, pool_epoch, pool_epoch_offset = self.record_index_for_occurrence(
            pool_name,
            pool_occurrence_index,
        )
        return MixtureSelection(
            global_position=int(global_position),
            pool_name=pool_name,
            record_index=record_index,
            pool_occurrence_index=pool_occurrence_index,
            pool_epoch=pool_epoch,
            pool_epoch_offset=pool_epoch_offset,
        )

    def iter_selections(
        self,
        start_position: int = 0,
        pool_occurrence_counts: dict[str, int] | None = None,
    ) -> Iterator[MixtureSelection]:
        """Yield an efficient sequential stream from an arbitrary exact position."""
        if start_position < 0:
            raise ValueError(f"start_position must be non-negative, got {start_position}")
        counts = (
            self.pool_occurrence_counts_before(start_position)
            if pool_occurrence_counts is None
            else {name: int(pool_occurrence_counts[name]) for name in POOL_ORDER}
        )
        if set(counts) != set(POOL_ORDER) or any(value < 0 for value in counts.values()):
            raise ValueError(f"Invalid pool occurrence counts: {counts}")
        if sum(counts.values()) != start_position:
            raise ValueError(
                "Pool occurrence counts do not match start_position: "
                f"position={start_position} counts={counts}"
            )
        global_position = int(start_position)
        while True:
            pool_name = self.pool_for_position(global_position)
            pool_occurrence_index = counts[pool_name]
            record_index, pool_epoch, pool_epoch_offset = self.record_index_for_occurrence(
                pool_name,
                pool_occurrence_index,
            )
            yield MixtureSelection(
                global_position=global_position,
                pool_name=pool_name,
                record_index=record_index,
                pool_occurrence_index=pool_occurrence_index,
                pool_epoch=pool_epoch,
                pool_epoch_offset=pool_epoch_offset,
            )
            counts[pool_name] += 1
            global_position += 1

    def target_index(self, global_position: int, target_count: int) -> int:
        if target_count <= 0:
            raise ValueError(f"target_count must be positive, got {target_count}")
        return self._draw(global_position, 4) % int(target_count)

    def positions_for_rank(
        self,
        rank: int,
        world_size: int,
        start_position: int = 0,
    ) -> Iterator[int]:
        if world_size <= 0 or rank < 0 or rank >= world_size or start_position < 0:
            raise ValueError(
                f"Invalid distributed position request: rank={rank} world_size={world_size} start={start_position}"
            )
        position = start_position + ((rank - start_position) % world_size)
        while True:
            yield position
            position += world_size
