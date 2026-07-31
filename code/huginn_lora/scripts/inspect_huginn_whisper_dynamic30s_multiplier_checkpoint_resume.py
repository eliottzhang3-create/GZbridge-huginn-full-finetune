#!/usr/bin/env python3
"""Audit real-data FSDP4 cold resume on the finite multiplier schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
SCRIPTS_ROOT = HUGINN_LORA_ROOT / "scripts"
for path in (HUGINN_LORA_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_pipeline.finite_multiplier_pool import (  # noqa: E402
    POOL_ORDER,
    SAMPLER_VERSION,
    STATISTICS_VERSION,
    FiniteMultiplierPool,
)
from inspect_huginn_whisper_dynamic90s_checkpoint_resume_markers import (  # noqa: E402
    read_data_records,
    read_forward_records,
    validate_phase_markers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-audit-dir", type=Path, required=True)
    parser.add_argument("--resume-audit-dir", type=Path, required=True)
    parser.add_argument("--data-audit-dir", type=Path, required=True)
    parser.add_argument("--forward-audit-dir", type=Path, required=True)
    parser.add_argument("--save-stats-state", type=Path, required=True)
    parser.add_argument("--resume-stats-state", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--save-step", type=int, default=4)
    parser.add_argument("--resume-step", type=int, default=6)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def collapse_template_records(
    records: list[dict[str, Any]],
    expected_positions: list[int],
    phase: str,
    max_prefetch: int,
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    by_position: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_position.setdefault(int(record["global_position"]), []).append(record)
    observed_positions = sorted(by_position)
    expected_set = set(expected_positions)
    if not expected_set.issubset(by_position):
        raise AssertionError(
            f"Multiplier {phase} template audit missed consumed positions: "
            f"missing={sorted(expected_set - set(by_position))}"
        )
    prefetched = [position for position in observed_positions if position not in expected_set]
    if prefetched != list(range(expected_positions[-1] + 1, expected_positions[-1] + 1 + len(prefetched))):
        raise AssertionError(f"Multiplier {phase} prefetch is not a contiguous tail: {prefetched}")
    if len(prefetched) > max_prefetch:
        raise AssertionError(f"Multiplier {phase} prefetch exceeds bound {max_prefetch}: {prefetched}")
    fields = (
        "pool_name",
        "task",
        "uid",
        "record_index",
        "pool_occurrence_index",
        "pool_epoch",
        "pool_epoch_offset",
        "component_name",
        "replica_id",
        "schedule_slot",
    )
    collapsed: dict[int, dict[str, Any]] = {}
    for position in expected_positions:
        duplicates = by_position[position]
        first = duplicates[0]
        for duplicate in duplicates[1:]:
            if any(first.get(field) != duplicate.get(field) for field in fields):
                raise AssertionError(
                    f"Multiplier {phase} duplicate encode changed provenance at {position}: {duplicates}"
                )
        collapsed[position] = first
    return collapsed, prefetched


def audit_window(
    pool: FiniteMultiplierPool,
    data_records: list[dict[str, Any]],
    forward_records: list[dict[str, Any]],
    phase: str,
    start_position: int,
    end_position: int,
    world_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, float], dict[str, Any]]:
    expected_positions = list(range(start_position, end_position))
    actual_positions = sorted(int(record["global_position"]) for record in forward_records)
    if actual_positions != expected_positions:
        raise AssertionError(
            f"Multiplier {phase} actual forward window mismatch: "
            f"actual={actual_positions} expected={expected_positions}"
        )
    rank_counts = Counter(int(record["rank"]) for record in forward_records)
    expected_rank_count = len(expected_positions) // world_size
    if rank_counts != Counter({rank: expected_rank_count for rank in range(world_size)}):
        raise AssertionError(f"Multiplier {phase} rank aggregation mismatch: {rank_counts}")
    templates, prefetched = collapse_template_records(
        data_records,
        expected_positions,
        phase,
        max_prefetch=2 * world_size,
    )
    forward_by_position = {int(record["global_position"]): record for record in forward_records}
    entries: list[dict[str, Any]] = []
    pool_counts = {name: 0 for name in POOL_ORDER}
    pool_durations = {name: 0.0 for name in POOL_ORDER}
    component_counts: Counter[str] = Counter()
    replica_counts: Counter[str] = Counter()
    for position in expected_positions:
        selection = pool.selection(position)
        atomic = pool.record(selection)
        template = templates[position]
        forward = forward_by_position[position]
        expected = {
            "global_position": position,
            "schedule_slot": selection.schedule_slot,
            "component_name": selection.component_name,
            "pool_name": selection.pool_name,
            "task": selection.task,
            "uid": str(atomic["uid"]),
            "record_index": selection.record_index,
            "pool_occurrence_index": selection.pool_occurrence_index,
            "pool_epoch": selection.replica_id,
            "pool_epoch_offset": selection.selection_offset,
            "replica_id": selection.replica_id,
        }
        actual_template = {key: template.get(key) for key in expected}
        if actual_template != expected:
            raise AssertionError(
                f"Multiplier {phase} template provenance mismatch at {position}: "
                f"actual={actual_template} expected={expected}"
            )
        expected_forward = {
            "pool_name": selection.pool_name,
            "record_index": selection.record_index,
            "pool_occurrence_index": selection.pool_occurrence_index,
            "pool_epoch": selection.replica_id,
        }
        actual_forward = {key: forward.get(key) for key in expected_forward}
        if actual_forward != expected_forward:
            raise AssertionError(
                f"Multiplier {phase} forward provenance mismatch at {position}: "
                f"actual={actual_forward} expected={expected_forward}"
            )
        duration = float(forward["effective_duration_seconds"])
        if not 0.0 < duration <= 30.000001:
            raise AssertionError(f"Invalid effective duration at {position}: {duration}")
        pool_counts[selection.pool_name] += 1
        pool_durations[selection.pool_name] += duration
        component_counts[selection.component_name] += 1
        replica_counts[f"{selection.component_name}:replica-{selection.replica_id}"] += 1
        entries.append(expected)
    return entries, pool_counts, pool_durations, {
        "rank_counts": dict(rank_counts),
        "component_counts": dict(component_counts),
        "replica_counts": dict(replica_counts),
        "prefetched_positions": prefetched,
    }


def validate_statistics_state(
    path: Path,
    expected_step: int,
    expected_entries: list[dict[str, Any]],
    expected_durations: dict[str, float],
    registry: dict[str, Any],
) -> dict[str, Any]:
    state = load_json(path)
    total_samples = len(expected_entries)
    if (
        state.get("statistics_version") != STATISTICS_VERSION
        or state.get("sampler_version") != SAMPLER_VERSION
        or int(state.get("sampler_seed", -1)) != int(registry["seed"])
        or int(state.get("global_step", -1)) != expected_step
        or int(state.get("total_samples", -1)) != total_samples
        or int(state.get("next_global_position", -1)) != total_samples
    ):
        raise AssertionError(f"Multiplier statistics header mismatch at {path}: {state}")
    expected_counts = Counter(str(entry["pool_name"]) for entry in expected_entries)
    pools = state.get("pools")
    if not isinstance(pools, dict) or tuple(pools) != POOL_ORDER:
        raise AssertionError(f"Multiplier statistics pool set mismatch at {path}: {pools}")
    for name in POOL_ORDER:
        entry = pools[name]
        if int(entry["sample_count"]) != expected_counts[name]:
            raise AssertionError(f"Multiplier statistics count mismatch at {path}: pool={name}")
        if int(entry["pool_size"]) != int(registry["pools"][name]["record_count"]):
            raise AssertionError(f"Multiplier statistics pool size mismatch at {path}: pool={name}")
        if abs(float(entry["effective_duration_seconds"]) - expected_durations[name]) > 1e-5:
            raise AssertionError(f"Multiplier statistics duration mismatch at {path}: pool={name}")
    if abs(float(state["total_effective_duration_seconds"]) - sum(expected_durations.values())) > 1e-5:
        raise AssertionError(f"Multiplier statistics total duration mismatch at {path}")
    return state


def digest_entries(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.world_size != 4 or not 0 < args.save_step < args.resume_step:
        raise ValueError("Expected world_size=4 and 0 < save_step < resume_step")
    save_pids, save_launches = validate_phase_markers(
        args.save_audit_dir,
        "save",
        0,
        args.save_step,
        args.world_size,
    )
    resume_pids, resume_launches = validate_phase_markers(
        args.resume_audit_dir,
        "resume",
        args.save_step,
        args.resume_step,
        args.world_size,
    )
    if save_launches & resume_launches:
        raise AssertionError("Multiplier save and resume used the same process-group launch ID")
    registry = load_json(args.registry)
    save_end = args.save_step * args.world_size
    resume_end = args.resume_step * args.world_size
    with FiniteMultiplierPool(args.registry) as pool:
        save_entries, save_counts, save_durations, save_details = audit_window(
            pool,
            read_data_records(args.data_audit_dir, "save", args.world_size),
            read_forward_records(args.forward_audit_dir, "save", args.world_size),
            "save",
            0,
            save_end,
            args.world_size,
        )
        resume_entries, resume_counts, resume_durations, resume_details = audit_window(
            pool,
            read_data_records(args.data_audit_dir, "resume", args.world_size),
            read_forward_records(args.forward_audit_dir, "resume", args.world_size),
            "resume",
            save_end,
            resume_end,
            args.world_size,
        )
    combined_entries = save_entries + resume_entries
    if len({int(entry["schedule_slot"]) for entry in combined_entries}) != len(combined_entries):
        raise AssertionError("Multiplier checkpoint smoke repeated a global schedule slot")
    cumulative_durations = {
        name: save_durations[name] + resume_durations[name]
        for name in POOL_ORDER
    }
    save_state = validate_statistics_state(
        args.save_stats_state,
        args.save_step,
        save_entries,
        save_durations,
        registry,
    )
    resume_state = validate_statistics_state(
        args.resume_stats_state,
        args.resume_step,
        combined_entries,
        cumulative_durations,
        registry,
    )
    run_delta = resume_state.get("run_delta", {})
    for name in POOL_ORDER:
        if int(run_delta.get("sample_counts", {}).get(name, -1)) != resume_counts[name]:
            raise AssertionError(f"Multiplier resume sample delta mismatch: pool={name}")
        if abs(
            float(run_delta.get("effective_duration_seconds", {}).get(name, -1.0))
            - resume_durations[name]
        ) > 1e-5:
            raise AssertionError(f"Multiplier resume duration delta mismatch: pool={name}")
    report = {
        "gate": "huginn_whisper_dynamic30s_multiplier_checkpoint_resume_v1",
        "validation_passed": True,
        "save_step": args.save_step,
        "resume_step": args.resume_step,
        "save_processes": {"pids": sorted(save_pids), "launches": sorted(save_launches)},
        "resume_processes": {"pids": sorted(resume_pids), "launches": sorted(resume_launches)},
        "save": {
            "pool_counts": save_counts,
            "pool_durations": save_durations,
            "details": save_details,
            "provenance_sha256": digest_entries(save_entries),
        },
        "resume_delta": {
            "pool_counts": resume_counts,
            "pool_durations": resume_durations,
            "details": resume_details,
            "provenance_sha256": digest_entries(resume_entries),
        },
        "cumulative": {
            "samples": len(combined_entries),
            "pool_durations": cumulative_durations,
            "provenance_sha256": digest_entries(combined_entries),
        },
        "save_statistics_state": save_state,
        "resume_statistics_state": resume_state,
    }
    write_json_atomic(args.output_report, report)
    print(
        f"[multiplier-resume] save_positions=0..{save_end - 1} "
        f"resume_positions={save_end}..{resume_end - 1} repeated_schedule_slots=0",
        flush=True,
    )
    print(
        f"[multiplier-resume] save_digest={report['save']['provenance_sha256']} "
        f"resume_digest={report['resume_delta']['provenance_sha256']} "
        f"cumulative_digest={report['cumulative']['provenance_sha256']}",
        flush=True,
    )
    print(f"[multiplier-resume] report={args.output_report.resolve()}", flush=True)
    print("========== HUGINN WHISPER DYNAMIC30S MULTIPLIER CHECKPOINT RESUME PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
