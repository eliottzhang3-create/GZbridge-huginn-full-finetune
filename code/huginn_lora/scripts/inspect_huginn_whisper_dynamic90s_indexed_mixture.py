"""Validate indexed full pools and the deterministic FSDP4 mixture planner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code/huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.indexed_atomic_mixture import (  # noqa: E402
    GLOBAL_POOL_WEIGHTS,
    POOL_ORDER,
    SAMPLER_VERSION,
    DeterministicHierarchicalMixture,
    IndexedJsonlPool,
    splitmix64,
)


DEFAULT_CONTRACT = HUGINN_LORA_ROOT / "configs/huginn_whisper_dynamic90s_data_contract_v1.json"
DEFAULT_REGISTRY = REPO_ROOT / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/pool_registry.json"
DEFAULT_FULL_REPORT = REPO_ROOT / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/full_pool_report.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/sampler"
EXPECTED_TASKS = {
    "wavcaps_no_bbc_aac": "AAC",
    "audiocaps_v2_aac": "AAC",
    "clotho_v2_aac": "AAC",
    "gigaspeech_l_asr": "ASR",
}
EXPECTED_SOURCES = {
    "wavcaps_no_bbc_aac": {"FreeSound", "AudioSet_SL", "SoundBible"},
    "audiocaps_v2_aac": {"AudioCaps-v2"},
    "clotho_v2_aac": {"Clotho-v2"},
    "gigaspeech_l_asr": {"audiobook", "podcast", "youtube"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--full_report", default=str(DEFAULT_FULL_REPORT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--simulation_draws", type=int, default=1000000)
    parser.add_argument("--schedule_records", type=int, default=4096)
    parser.add_argument("--random_access_probes_per_pool", type=int, default=64)
    parser.add_argument("--global_tolerance", type=float, default=0.003)
    parser.add_argument("--rank_tolerance", type=float, default=0.008)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def compare_weights(actual: dict[str, Any], expected: dict[str, float], label: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label} keys mismatch: actual={sorted(actual)} expected={sorted(expected)}")
    mismatches = {
        key: {"actual": float(actual[key]), "expected": expected[key]}
        for key in expected
        if abs(float(actual[key]) - expected[key]) > 1e-12
    }
    if mismatches:
        raise ValueError(f"{label} values mismatch: {mismatches}")


def validate_record(pool_name: str, record: dict[str, Any]) -> None:
    if record.get("schema_version") != "huginn_whisper_dynamic90s_atomic_v1":
        raise ValueError(f"Schema mismatch in {pool_name}: {record.get('schema_version')!r}")
    if record.get("task") != EXPECTED_TASKS[pool_name]:
        raise ValueError(f"Task mismatch in {pool_name}: {record.get('task')!r}")
    if record.get("source") not in EXPECTED_SOURCES[pool_name]:
        raise ValueError(f"Source mismatch in {pool_name}: {record.get('source')!r}")
    if not isinstance(record.get("uid"), str) or not record["uid"]:
        raise ValueError(f"Empty UID in {pool_name}")
    targets = record.get("targets")
    if not isinstance(targets, list) or not targets or any(not isinstance(target, str) or not target for target in targets):
        raise ValueError(f"Invalid targets in {pool_name}: {targets!r}")
    audio = record.get("audio")
    if not isinstance(audio, dict) or not audio.get("path") or not audio.get("format"):
        raise ValueError(f"Invalid audio reference in {pool_name}: {audio!r}")
    if pool_name == "gigaspeech_l_asr":
        if "start_sec" not in audio or "end_sec" not in audio or float(audio["end_sec"]) <= float(audio["start_sec"]):
            raise ValueError(f"Invalid GigaSpeech segment audio reference: {audio!r}")
    if "effective_audio_tokens" in record:
        raise ValueError(f"Pool {pool_name} unexpectedly precomputed effective_audio_tokens")
    raw_duration = record.get("raw_duration_sec")
    if raw_duration is not None and (float(raw_duration) <= 0 or float(raw_duration) > 90.0):
        raise ValueError(f"Pool {pool_name} contains an ineligible duration: {raw_duration}")


def probe_indices(record_count: int, probe_count: int, seed: int) -> list[int]:
    indices = {0, record_count // 2, record_count - 1}
    for probe in range(probe_count):
        indices.add(splitmix64(seed ^ probe) % record_count)
    return sorted(indices)


def ratios(counts: Counter[str], total: int) -> dict[str, float]:
    return {name: counts[name] / total for name in POOL_ORDER}


def maximum_error(observed: dict[str, float]) -> float:
    return max(abs(observed[name] - GLOBAL_POOL_WEIGHTS[name]) for name in POOL_ORDER)


def audit_no_replacement_epochs(
    planner: DeterministicHierarchicalMixture,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for pool_name in POOL_ORDER:
        pool_size = planner.pool_sizes[pool_name]
        first_epoch_order: list[int] = []
        changed_positions = 0
        for epoch in range(2):
            seen = bytearray(pool_size)
            for epoch_offset in range(pool_size):
                occurrence = epoch * pool_size + epoch_offset
                record_index, observed_epoch, observed_offset = planner.record_index_for_occurrence(
                    pool_name,
                    occurrence,
                )
                if observed_epoch != epoch or observed_offset != epoch_offset:
                    raise AssertionError(
                        f"Pool epoch coordinates changed: pool={pool_name} occurrence={occurrence} "
                        f"actual=({observed_epoch},{observed_offset}) expected=({epoch},{epoch_offset})"
                    )
                if seen[record_index]:
                    raise AssertionError(
                        f"No-replacement epoch repeated a record: pool={pool_name} "
                        f"epoch={epoch} record_index={record_index}"
                    )
                seen[record_index] = 1
                if epoch == 0:
                    first_epoch_order.append(record_index)
                elif record_index != first_epoch_order[epoch_offset]:
                    changed_positions += 1
            if sum(seen) != pool_size:
                raise AssertionError(
                    f"No-replacement epoch did not cover its complete pool: "
                    f"pool={pool_name} epoch={epoch} covered={sum(seen)} size={pool_size}"
                )
        if pool_size > 1 and changed_positions == 0:
            raise AssertionError(f"Pool epoch 1 did not reshuffle relative to epoch 0: {pool_name}")
        reports[pool_name] = {
            "record_count": pool_size,
            "audited_epochs": 2,
            "zero_duplicates_per_epoch": True,
            "complete_coverage_per_epoch": True,
            "epoch_1_changed_positions": changed_positions,
            "passed": True,
        }
        print(
            f"[no-replacement] pool={pool_name} records={pool_size} epochs=2 "
            f"changed_positions={changed_positions} passed=true",
            flush=True,
        )
    return reports


def main() -> None:
    args = parse_args()
    if (
        args.world_size <= 0
        or args.simulation_draws <= 0
        or args.schedule_records <= 0
        or args.random_access_probes_per_pool <= 0
        or args.global_tolerance <= 0
        or args.rank_tolerance <= 0
    ):
        raise ValueError("All numeric gate arguments must be positive")

    contract = load_json(Path(args.contract))
    sampling = contract.get("sampling_contract", {})
    if sampling.get("unit") != "hierarchical_sample_draw_probability_with_per_pool_no_replacement_epochs_v2":
        raise ValueError(f"Unexpected sampler unit: {sampling.get('unit')!r}")
    compare_weights(sampling.get("global_pool_weights", {}), GLOBAL_POOL_WEIGHTS, "contract weights")
    if sampling.get("task_weights") != {"AAC": 0.6, "ASR": 0.4}:
        raise ValueError(f"Unexpected task weights: {sampling.get('task_weights')}")
    if sampling.get("aac_source_weights") != {
        "wavcaps_no_bbc_aac": 0.6,
        "audiocaps_v2_aac": 0.3,
        "clotho_v2_aac": 0.1,
    }:
        raise ValueError(f"Unexpected AAC source weights: {sampling.get('aac_source_weights')}")

    registry = load_json(Path(args.registry))
    full_report = load_json(Path(args.full_report))
    if not full_report.get("validation_passed"):
        raise ValueError("Full atomic-pool report has not passed")
    compare_weights(registry.get("sampling_weights", {}), GLOBAL_POOL_WEIGHTS, "registry weights")
    registry_pools = registry.get("pools", {})
    if set(registry_pools) != set(POOL_ORDER):
        raise ValueError(f"Registry pool set mismatch: {sorted(registry_pools)}")
    for pool_name, entry in registry_pools.items():
        planning_hours = float(entry.get("planning_effective_duration_hours", -1.0))
        if planning_hours <= 0:
            raise ValueError(f"Pool {pool_name} has invalid planning effective hours: {planning_hours}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mixture_sampler_report.json"
    schedule_path = output_dir / "mixture_schedule.pilot.jsonl"
    if not args.overwrite and (report_path.exists() or schedule_path.exists()):
        raise FileExistsError(f"Refusing to overwrite sampler outputs under {output_dir}")

    print("========== HUGINN WHISPER DYNAMIC30S INDEXED MIXTURE START ==========", flush=True)
    print("[scope] audio_read=false audio_decode=false token_accounting=false", flush=True)
    print(
        f"[sampler] seed={args.seed} world_size={args.world_size} "
        f"simulation_draws={args.simulation_draws}",
        flush=True,
    )

    pools: dict[str, IndexedJsonlPool] = {}
    random_access_report: dict[str, Any] = {}
    try:
        for pool_name in POOL_ORDER:
            entry = registry_pools[pool_name]
            pool = IndexedJsonlPool(
                pool_name,
                entry["manifest_path"],
                entry["index_path"],
                int(entry["record_count"]),
            )
            pools[pool_name] = pool
            indices = probe_indices(pool.record_count, args.random_access_probes_per_pool, args.seed)
            uid_counts: Counter[str] = Counter()
            source_counts: Counter[str] = Counter()
            for record_index in indices:
                record = pool.record(record_index)
                validate_record(pool_name, record)
                uid_counts[record["uid"]] += 1
                source_counts[record["source"]] += 1
            duplicate_probe_uids = [uid for uid, count in uid_counts.items() if count > 1]
            if duplicate_probe_uids:
                raise ValueError(f"Duplicate UIDs in random-access probes for {pool_name}: {duplicate_probe_uids[:10]}")
            random_access_report[pool_name] = {
                "record_count": pool.record_count,
                "probe_count": len(indices),
                "probe_source_counts": dict(sorted(source_counts.items())),
                "first_offset": pool.offset(0),
                "last_offset": pool.offset(pool.record_count - 1),
                "passed": True,
            }
            print(
                f"[index] pool={pool_name} records={pool.record_count} probes={len(indices)} passed=true",
                flush=True,
            )

        planner = DeterministicHierarchicalMixture(
            {pool_name: pool.record_count for pool_name, pool in pools.items()},
            seed=args.seed,
        )
        no_replacement_report = audit_no_replacement_epochs(planner)
        global_counts: Counter[str] = Counter()
        rank_counts = [Counter() for _ in range(args.world_size)]
        task_counts: Counter[str] = Counter()
        aac_counts: Counter[str] = Counter()
        for position in range(args.simulation_draws):
            pool_name = planner.pool_for_position(position)
            global_counts[pool_name] += 1
            rank_counts[position % args.world_size][pool_name] += 1
            task = EXPECTED_TASKS[pool_name]
            task_counts[task] += 1
            if task == "AAC":
                aac_counts[pool_name] += 1

        observed_global = ratios(global_counts, args.simulation_draws)
        global_error = maximum_error(observed_global)
        if global_error > args.global_tolerance:
            raise AssertionError(
                f"Global mixture error exceeds tolerance: error={global_error} tolerance={args.global_tolerance}"
            )
        observed_task = {
            "AAC": task_counts["AAC"] / args.simulation_draws,
            "ASR": task_counts["ASR"] / args.simulation_draws,
        }
        aac_total = task_counts["AAC"]
        observed_aac = {
            "wavcaps_no_bbc_aac": aac_counts["wavcaps_no_bbc_aac"] / aac_total,
            "audiocaps_v2_aac": aac_counts["audiocaps_v2_aac"] / aac_total,
            "clotho_v2_aac": aac_counts["clotho_v2_aac"] / aac_total,
        }

        rank_reports = []
        for rank, counts in enumerate(rank_counts):
            rank_total = sum(counts.values())
            observed = ratios(counts, rank_total)
            error = maximum_error(observed)
            if error > args.rank_tolerance:
                raise AssertionError(
                    f"Rank mixture error exceeds tolerance: rank={rank} error={error} tolerance={args.rank_tolerance}"
                )
            rank_reports.append(
                {
                    "rank": rank,
                    "draw_count": rank_total,
                    "counts": dict(counts),
                    "ratios": observed,
                    "max_absolute_error": error,
                }
            )

        deterministic_positions = [0, 1, 2, 3, 17, 1024, 123456]
        first_pass = [planner.selection(position) for position in deterministic_positions]
        repeat_planner = DeterministicHierarchicalMixture(planner.pool_sizes, seed=args.seed)
        second_pass = [repeat_planner.selection(position) for position in deterministic_positions]
        if first_pass != second_pass:
            raise AssertionError("No-replacement mixture selection is not reproducible for the same seed")
        resume_probe_count = args.world_size * 8
        resume_starts = sorted(
            {
                0,
                1,
                17,
                min(4096, max(0, args.simulation_draws - resume_probe_count)),
                min(123456, max(0, args.simulation_draws - resume_probe_count)),
            }
        )
        resumed_tail: list[Any] = []
        for resume_start in resume_starts:
            uninterrupted_tail = list(
                islice(planner.iter_selections(0), resume_start, resume_start + resume_probe_count)
            )
            resumed_tail = list(islice(planner.iter_selections(resume_start), resume_probe_count))
            if resumed_tail != uninterrupted_tail:
                raise AssertionError(
                    f"Arbitrary-position resume diverged at start={resume_start}: "
                    f"uninterrupted={uninterrupted_tail[:3]} resumed={resumed_tail[:3]}"
                )
        resume_start = resume_starts[-1]
        for rank in range(args.world_size):
            positions = [
                selection.global_position
                for selection in resumed_tail
                if selection.global_position % args.world_size == rank
            ]
            if len(positions) != 8:
                raise AssertionError(f"Rank resume position coverage failed: rank={rank} positions={positions}")

        schedule_records: list[dict[str, Any]] = []
        for selection in islice(planner.iter_selections(0), args.schedule_records):
            position = selection.global_position
            record = pools[selection.pool_name].record(selection.record_index)
            validate_record(selection.pool_name, record)
            target_index = planner.target_index(position, len(record["targets"]))
            schedule_records.append(
                {
                    "global_position": position,
                    "rank": position % args.world_size,
                    "pool": selection.pool_name,
                    "record_index": selection.record_index,
                    "pool_occurrence_index": selection.pool_occurrence_index,
                    "pool_epoch": selection.pool_epoch,
                    "pool_epoch_offset": selection.pool_epoch_offset,
                    "uid": record["uid"],
                    "task": record["task"],
                    "source": record["source"],
                    "target_index": target_index,
                    "target_count": len(record["targets"]),
                }
            )

        report = {
            "gate": "huginn_whisper_dynamic30s_indexed_mixture_no_replacement_v2",
            "validation_passed": True,
            "contract_version": registry.get("contract_version"),
            "duration_policy": "discard_gt90s_then_cap_at30s",
            "sampler_version": SAMPLER_VERSION,
            "seed": args.seed,
            "world_size": args.world_size,
            "simulation_draws": args.simulation_draws,
            "sampler_unit": "hierarchical_sample_draw_probability_with_per_pool_no_replacement_epochs_v2",
            "no_replacement_epoch_audit": no_replacement_report,
            "expected_global_pool_weights": GLOBAL_POOL_WEIGHTS,
            "observed_global_pool_counts": dict(global_counts),
            "observed_global_pool_ratios": observed_global,
            "global_max_absolute_error": global_error,
            "global_tolerance": args.global_tolerance,
            "observed_task_ratios": observed_task,
            "observed_aac_internal_ratios": observed_aac,
            "rank_tolerance": args.rank_tolerance,
            "rank_reports": rank_reports,
            "random_access": random_access_report,
            "deterministic_resume_positions": deterministic_positions,
            "audited_resume_starts": resume_starts,
            "pilot_schedule_path": str(schedule_path),
            "pilot_schedule_records": args.schedule_records,
            "audio_read": False,
            "audio_decode": False,
            "token_accounting": False,
            "realized_token_accounting": "deferred to training-time logging",
        }
        write_jsonl_atomic(schedule_path, schedule_records)
        write_json_atomic(report_path, report)
        print(
            f"[mixture] counts={dict(global_counts)} ratios={observed_global} "
            f"max_error={global_error:.6f}",
            flush=True,
        )
        print(f"[mixture] task_ratios={observed_task} aac_internal_ratios={observed_aac}", flush=True)
        print(f"[mixture] schedule={schedule_path}", flush=True)
        print(f"[mixture] report={report_path}", flush=True)
    finally:
        for pool in pools.values():
            pool.close()

    print("========== HUGINN WHISPER DYNAMIC30S INDEXED MIXTURE PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
