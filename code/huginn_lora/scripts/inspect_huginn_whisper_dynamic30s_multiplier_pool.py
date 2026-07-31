#!/usr/bin/env python3
"""Audit the finite multiplier pool, source filters, and global permutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import TASK_PROMPTS  # noqa: E402
from data_pipeline.finite_multiplier_pool import (  # noqa: E402
    COMPONENT_ORDER,
    EXPECTED_MULTIPLIERS,
    FiniteMultiplierPool,
    UInt64Index,
    load_multiplier_registry,
    render_multiplier_row,
)
from scripts.prepare_huginn_whisper_dynamic30s_multiplier_pool import (  # noqa: E402
    aligned_quarter_count,
)


DEFAULT_REGISTRY = (
    REPO_ROOT
    / "data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m"
    / "multiplier_pool_registry.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--progress-every", type=int, default=1000000)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"Manifest contains an empty row: path={path} index={index}")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Manifest row is not an object: path={path} index={index}")
            yield index, payload


def selection_values(entry: dict[str, Any]) -> list[int] | None:
    selection_path = entry.get("selection_index_path")
    if selection_path is None:
        return None
    count = int(entry["selected_record_count"])
    with UInt64Index(selection_path, count) as index:
        values = [index[offset] for offset in range(count)]
    if len(set(values)) != count:
        raise AssertionError(f"Selection index contains duplicates: {selection_path}")
    if any(value < 0 or value >= int(entry["base_record_count"]) for value in values):
        raise AssertionError(f"Selection index contains out-of-range values: {selection_path}")
    return values


def validate_source_filters(registry: dict[str, Any]) -> dict[str, Any]:
    components = registry["components"]
    selected = {
        name: selection_values(components[name])
        for name in COMPONENT_ORDER
    }
    wav_names = (
        "wavcaps_audioset_aac",
        "wavcaps_soundbible_aac",
        "wavcaps_freesound_quarter_aac",
    )
    wav_manifest = Path(components[wav_names[0]]["manifest_path"])
    wav_selected_sets = {name: set(selected[name] or []) for name in wav_names}
    expected_sources = {
        "wavcaps_audioset_aac": "AudioSet_SL",
        "wavcaps_soundbible_aac": "SoundBible",
        "wavcaps_freesound_quarter_aac": "FreeSound",
    }
    wav_full_source_counts: Counter[str] = Counter()
    wav_selected_counts: Counter[str] = Counter()
    for record_index, record in iter_jsonl(wav_manifest):
        source = str(record.get("source", ""))
        wav_full_source_counts[source] += 1
        for name in wav_names:
            if record_index not in wav_selected_sets[name]:
                continue
            if source != expected_sources[name]:
                raise AssertionError(
                    f"WavCaps selection source mismatch: component={name} index={record_index} source={source}"
                )
            wav_selected_counts[name] += 1
    for name in wav_names:
        if wav_selected_counts[name] != int(components[name]["selected_record_count"]):
            raise AssertionError(f"WavCaps selection was not fully validated: {name}")
    if int(components["wavcaps_audioset_aac"]["selected_record_count"]) != wav_full_source_counts["AudioSet_SL"]:
        raise AssertionError("AudioSet-SL selection does not cover the complete source")
    if int(components["wavcaps_soundbible_aac"]["selected_record_count"]) != wav_full_source_counts["SoundBible"]:
        raise AssertionError("SoundBible selection does not cover the complete source")

    giga_entry = components["gigaspeech_m_asr"]
    giga_selected = set(selected["gigaspeech_m_asr"] or [])
    observed_m = set()
    for record_index, record in iter_jsonl(Path(giga_entry["manifest_path"])):
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        is_m = "{M}" in (metadata.get("subsets") or [])
        if is_m:
            observed_m.add(record_index)
            if record.get("task") != "ASR":
                raise AssertionError(f"GigaSpeech-M row is not ASR: {record_index}")
    if observed_m != giga_selected:
        missing = sorted(observed_m - giga_selected)[:10]
        extra = sorted(giga_selected - observed_m)[:10]
        raise AssertionError(
            f"GigaSpeech-M selection is not complete: missing={missing} extra={extra}"
        )
    return {
        "wavcaps_full_source_counts": dict(wav_full_source_counts),
        "wavcaps_selected_counts": dict(wav_selected_counts),
        "gigaspeech_m_selected_count": len(giga_selected),
    }


def validate_schedule(registry_path: Path, progress_every: int) -> dict[str, Any]:
    registry = load_multiplier_registry(registry_path)
    total = int(registry["total_records"])
    seen = bytearray(total)
    component_counts: Counter[str] = Counter()
    replica_counts: Counter[tuple[str, int]] = Counter()
    pool_counts: Counter[str] = Counter()
    first_positions: dict[str, int] = {}
    digest = hashlib.sha256()
    with FiniteMultiplierPool(registry_path) as pool:
        for position in range(total):
            selection = pool.selection(position)
            slot = selection.schedule_slot
            if seen[slot]:
                raise AssertionError(f"Global schedule repeats slot {slot} at position {position}")
            seen[slot] = 1
            component_counts[selection.component_name] += 1
            replica_counts[(selection.component_name, selection.replica_id)] += 1
            pool_counts[selection.pool_name] += 1
            first_positions.setdefault(selection.component_name, position)
            digest.update(int(slot).to_bytes(8, "little"))
            if progress_every > 0 and (position + 1) % progress_every == 0:
                print(f"[schedule-audit] positions={position + 1}/{total}", flush=True)
        if not all(seen):
            missing = [index for index, value in enumerate(seen) if not value][:10]
            raise AssertionError(f"Global schedule omits virtual slots: {missing}")

        row_samples: dict[str, Any] = {}
        for name in COMPONENT_ORDER:
            position = first_positions[name]
            selection = pool.selection(position)
            record = pool.record(selection)
            row = render_multiplier_row(record, selection, pool.seed)
            expected_prompt = TASK_PROMPTS[registry["components"][name]["task"]]
            if row["messages"][1] != {"role": "user", "content": expected_prompt}:
                raise AssertionError(f"Task prompt mismatch for {name}: {row['messages']}")
            if row["metadata"]["component_name"] != name:
                raise AssertionError(f"Component provenance mismatch for {name}: {row['metadata']}")
            row_samples[name] = {
                "position": position,
                "uid": row["metadata"]["uid"],
                "task": row["metadata"]["task"],
                "replica_id": row["metadata"]["replica_id"],
                "record_index": row["metadata"]["record_index"],
            }

    for name in COMPONENT_ORDER:
        entry = registry["components"][name]
        selected_count = int(entry["selected_record_count"])
        multiplier = EXPECTED_MULTIPLIERS[name]
        if component_counts[name] != selected_count * multiplier:
            raise AssertionError(f"Expanded component count mismatch: {name}")
        for replica_id in range(multiplier):
            if replica_counts[(name, replica_id)] != selected_count:
                raise AssertionError(
                    f"Replica coverage mismatch: component={name} replica={replica_id}"
                )
    expected_pool_counts = {
        name: int(registry["pools"][name]["record_count"])
        for name in registry["pools"]
    }
    if dict(pool_counts) != expected_pool_counts:
        raise AssertionError(f"Aggregate pool counts mismatch: {pool_counts} != {expected_pool_counts}")
    if digest.hexdigest() != registry["schedule_sha256"]:
        raise AssertionError(
            f"Schedule digest mismatch: scanned={digest.hexdigest()} registry={registry['schedule_sha256']}"
        )
    return {
        "component_counts": dict(component_counts),
        "replica_counts": {
            f"{name}:replica-{replica}": count
            for (name, replica), count in sorted(replica_counts.items())
        },
        "pool_counts": dict(pool_counts),
        "schedule_sha256": digest.hexdigest(),
        "row_samples": row_samples,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_self_test() -> None:
    for source_count in (101, 1000, 10001):
        target, selected = aligned_quarter_count(source_count, fixed_expanded_count=12345)
        if (12345 + selected) % 32 or abs(selected - target) > 16:
            raise AssertionError(
                f"Quarter alignment self-test failed: source={source_count} target={target} selected={selected}"
            )
    with tempfile.TemporaryDirectory(prefix="huginn-multiplier-audit-") as temporary:
        path = Path(temporary) / "identity.bin"
        path.write_bytes(b"A\nB\n")
        first = sha256_file(path)
        path.write_bytes(b"B\nA\n")
        if sha256_file(path) == first:
            raise AssertionError("Order-sensitive digest self-test failed")
    print("[self-test] quarter_alignment=true order_sensitive_digest=true")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    registry_path = args.registry.expanduser().resolve()
    registry = load_multiplier_registry(registry_path)
    if sha256_file(Path(registry["schedule_path"])) != registry["schedule_sha256"]:
        raise AssertionError("Schedule file SHA256 differs from the frozen registry")
    if sha256_file(Path(registry["source_registry_path"])) != registry["source_registry_sha256"]:
        raise AssertionError("Source registry identity changed after multiplier preparation")
    source_filter_report = validate_source_filters(registry)
    schedule_report = validate_schedule(registry_path, args.progress_every)
    report = {
        "gate": "huginn_whisper_dynamic30s_multiplier_pool_audit_v1",
        "validation_passed": True,
        "registry": str(registry_path),
        "contract_version": registry["contract_version"],
        "sampler_version": registry["sampler_version"],
        "seed": int(registry["seed"]),
        "total_records": int(registry["total_records"]),
        "max_steps": int(registry["max_steps"]),
        "global_batch_size": int(registry["global_batch_size"]),
        "source_filter_audit": source_filter_report,
        "schedule_audit": schedule_report,
        "audio_decode": False,
        "audio_copy": False,
    }
    output_report = (
        args.output_report.expanduser().resolve()
        if args.output_report
        else registry_path.with_name("multiplier_pool_audit.json")
    )
    write_json_atomic(output_report, report)
    print(
        f"[multiplier-audit] records={registry['total_records']} max_steps={registry['max_steps']} "
        f"schedule_sha256={schedule_report['schedule_sha256']}",
        flush=True,
    )
    print(f"[multiplier-audit] report={output_report}", flush=True)
    print("========== HUGINN WHISPER DYNAMIC30S MULTIPLIER POOL AUDIT PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
