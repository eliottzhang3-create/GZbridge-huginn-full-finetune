#!/usr/bin/env python3
"""Audit cold-process FSDP4 resume markers and exact mixture positions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import load_pool_registry, open_indexed_pools  # noqa: E402
from data_pipeline.indexed_atomic_mixture import (  # noqa: E402
    POOL_ORDER,
    SAMPLER_VERSION,
    DeterministicHierarchicalMixture,
)


TRAINING_STATS_VERSION = "huginn_dynamic90s_training_statistics_v1"


EXPECTED_TRAINABLE_TENSORS = {
    "lora": 66,
    "aligner": 14,
    "audio_encoder": 0,
    "huginn_base": 0,
    "other": 0,
}
EXPECTED_UNIT_TRAINABLE_TENSORS = {
    "WhisperEncoderFSDPUnit": 0,
    "AudioAlignerFSDPUnit": 14,
    "HuginnPreludeFSDPUnit": 16,
    "HuginnRecurrentCoreFSDPUnit": 34,
    "HuginnCodaFSDPUnit": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-audit-dir", type=Path, required=True)
    parser.add_argument("--resume-audit-dir", type=Path, required=True)
    parser.add_argument("--data-audit-dir", type=Path, required=True)
    parser.add_argument("--forward-audit-dir", type=Path, required=True)
    parser.add_argument("--save-stats-state", type=Path, required=True)
    parser.add_argument("--resume-stats-state", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--save-step", type=int, default=4)
    parser.add_argument("--resume-step", type=int, default=6)
    parser.add_argument("--world-size", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty checkpoint marker: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint marker is not an object: {path}")
    return payload


def validate_phase_markers(
    audit_dir: Path,
    phase: str,
    start_step: int,
    end_step: int,
    world_size: int,
) -> tuple[set[int], set[str]]:
    pids: set[int] = set()
    launch_ids: set[str] = set()
    expected_updates = end_step - start_step
    for rank in range(world_size):
        fsdp = read_json(audit_dir / f"fsdp-rank-{rank}.json")
        if (
            fsdp.get("kind") != "fsdp"
            or fsdp.get("stage") != "checkpoint"
            or fsdp.get("rank") != rank
            or fsdp.get("world_size") != world_size
            or fsdp.get("trainable_tensors") != EXPECTED_TRAINABLE_TENSORS
            or int(fsdp.get("dtensor_trainable_count", -1)) != 80
        ):
            raise AssertionError(f"Invalid {phase} FSDP marker for rank {rank}: {fsdp}")
        units = fsdp.get("fsdp_units")
        if not isinstance(units, dict) or set(units) != set(EXPECTED_UNIT_TRAINABLE_TENSORS):
            raise AssertionError(f"Invalid {phase} FSDP units for rank {rank}: {units}")
        for name, expected_trainables in EXPECTED_UNIT_TRAINABLE_TENSORS.items():
            unit = units[name]
            if (
                int(unit.get("parameter_count", 0)) <= 0
                or int(unit.get("dtensor_parameter_count", -1)) != int(unit["parameter_count"])
                or int(unit.get("trainable_parameter_count", -1)) != expected_trainables
            ):
                raise AssertionError(f"Invalid {phase} unit {name} rank {rank}: {unit}")

        start = read_json(audit_dir / f"checkpoint-start-rank-{rank}.json")
        if (
            start.get("kind") != "checkpoint_start"
            or start.get("phase") != phase
            or start.get("rank") != rank
            or start.get("global_step") != start_step
            or not start.get("optimizer_type")
        ):
            raise AssertionError(f"Invalid {phase} start marker for rank {rank}: {start}")
        pids.add(int(start["pid"]))
        launch_ids.add(str(start.get("launch_id", "")))
        if phase == "resume":
            if (
                int(start.get("optimizer_state_count", 0)) <= 0
                or start.get("optimizer_step_min") != start_step
                or start.get("optimizer_step_max") != start_step
                or start.get("scheduler_last_epoch") != start_step
                or not isinstance(start.get("learning_rates"), list)
                or any(float(value) <= 0 for value in start["learning_rates"])
            ):
                raise AssertionError(f"Resume optimizer/scheduler state mismatch for rank {rank}: {start}")

        end = read_json(audit_dir / f"checkpoint-end-rank-{rank}.json")
        if (
            end.get("kind") != "checkpoint_end"
            or end.get("phase") != phase
            or end.get("rank") != rank
            or end.get("start_global_step") != start_step
            or end.get("global_step") != end_step
            or end.get("new_optimizer_steps") != expected_updates
            or end.get("finite_loss_log_count") != expected_updates
            or end.get("finite_grad_norm_log_count") != expected_updates
            or end.get("audio_batch_count") != expected_updates
            or end.get("audio_sample_count") != expected_updates
            or int(end.get("realized_audio_tokens", 0)) <= 0
        ):
            raise AssertionError(f"Invalid {phase} end marker for rank {rank}: {end}")
        if phase == "resume":
            rng = read_json(audit_dir / f"rng-restore-rank-{rank}.json")
            if (
                rng.get("kind") != "rng_restore"
                or rng.get("phase") != "resume"
                or rng.get("rank") != rank
                or not all(rng.get("checks", {}).values())
            ):
                raise AssertionError(f"Invalid resume RNG marker for rank {rank}: {rng}")
        print(
            f"[checkpoint-marker] phase={phase} rank={rank} pid={start['pid']} "
            f"step_window=[{start_step},{end_step}] dtensor_parameters={fsdp['dtensor_parameter_count']} "
            f"finite_losses={end['finite_loss_log_count']} finite_grad_norms={end['finite_grad_norm_log_count']} "
            f"audio_samples={end['audio_sample_count']} audio_tokens={end['realized_audio_tokens']}"
        )
    if len(pids) != world_size:
        raise AssertionError(f"Phase {phase} expected one process per rank, observed pids={sorted(pids)}")
    if len(launch_ids) != 1 or "" in launch_ids:
        raise AssertionError(f"Phase {phase} has invalid launch IDs: {sorted(launch_ids)}")
    return pids, launch_ids


def read_data_records(data_dir: Path, phase: str, world_size: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(data_dir.glob(f"data-{phase}-rank-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No data-position audit files for phase={phase} under {data_dir}")
    observed_ranks: set[int] = set()
    for path in paths:
        try:
            rank = int(path.stem.rsplit("-rank-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid data-position audit filename: {path}") from exc
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Data-position audit rank is outside world_size={world_size}: {path}")
        observed_ranks.add(rank)
        rank_records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(record.get("rank") != rank or record.get("phase") != phase for record in rank_records):
            raise AssertionError(f"Invalid rank/phase data audit in {path}: {rank_records}")
        records.extend(rank_records)
    print(f"[data-dispatch] phase={phase} encoder_source_ranks={sorted(observed_ranks)}")
    return records


def validate_data_window(
    records: list[dict[str, Any]],
    phase: str,
    start_position: int,
    end_position: int,
    planner: DeterministicHierarchicalMixture,
    pools: dict[str, Any],
    max_prefetched_positions: int,
) -> Counter[str]:
    expected_positions = list(range(start_position, end_position))
    actual_positions = [int(record["global_position"]) for record in records]
    position_counts = Counter(actual_positions)
    unique_positions = sorted(position_counts)
    contiguous_positions = list(range(start_position, start_position + len(unique_positions)))
    if unique_positions != contiguous_positions:
        raise AssertionError(
            f"Phase {phase} template-encoded positions are not contiguous from {start_position}: "
            f"actual={unique_positions}"
        )
    if unique_positions[: len(expected_positions)] != expected_positions:
        raise AssertionError(
            f"Phase {phase} consumed data prefix mismatch: "
            f"expected={expected_positions} actual_prefix={unique_positions[:len(expected_positions)]}"
        )
    prefetched_positions = unique_positions[len(expected_positions):]
    if len(prefetched_positions) > max_prefetched_positions:
        raise AssertionError(
            f"Phase {phase} encoded too many unconsumed prefetch positions: "
            f"maximum={max_prefetched_positions} actual={prefetched_positions}"
        )
    multiplicities = set(position_counts.values())
    if len(multiplicities) != 1 or not multiplicities.issubset({1, 2}):
        raise AssertionError(
            f"Phase {phase} has inconsistent template-encode multiplicities: {dict(position_counts)}"
        )
    # Swift 4.1.3 may invoke template encoding twice per raw streaming row
    # (length/preparation plus actual collation). This is instrumentation
    # duplication, not a second model sample: the per-rank prefix counters
    # above independently prove the exact number of model-consumed samples.
    # Every duplicate must carry identical provenance before it is collapsed.
    unique_records: dict[int, dict[str, Any]] = {}
    provenance_fields = (
        "pool_name",
        "task",
        "uid",
        "record_index",
        "pool_occurrence_index",
        "pool_epoch",
        "pool_epoch_offset",
    )
    for record in records:
        position = int(record["global_position"])
        previous = unique_records.get(position)
        if previous is None:
            unique_records[position] = record
            continue
        previous_provenance = {key: previous.get(key) for key in provenance_fields}
        current_provenance = {key: record.get(key) for key in provenance_fields}
        if current_provenance != previous_provenance:
            raise AssertionError(
                f"Phase {phase} duplicate provenance mismatch at position {position}: "
                f"first={previous_provenance} duplicate={current_provenance}"
            )
    pool_counts: Counter[str] = Counter()
    prefetched_pool_counts: Counter[str] = Counter()
    for position in unique_positions:
        record = unique_records[position]
        position = int(record["global_position"])
        selection = planner.selection(position)
        atomic = pools[selection.pool_name].record(selection.record_index)
        expected = {
            "pool_name": selection.pool_name,
            "task": atomic["task"],
            "uid": atomic["uid"],
            "record_index": selection.record_index,
            "pool_occurrence_index": selection.pool_occurrence_index,
            "pool_epoch": selection.pool_epoch,
            "pool_epoch_offset": selection.pool_epoch_offset,
        }
        actual = {key: record.get(key) for key in expected}
        if actual != expected:
            raise AssertionError(
                f"Phase {phase} data provenance mismatch at position {position}: actual={actual} expected={expected}"
            )
        if position < end_position:
            pool_counts[selection.pool_name] += 1
        else:
            prefetched_pool_counts[selection.pool_name] += 1
    print(
        f"[data-window] phase={phase} positions={start_position}..{end_position - 1} "
        f"unique_records={len(unique_records)} raw_encode_records={len(records)} "
        f"encode_multiplicity={next(iter(multiplicities))} pool_counts={dict(pool_counts)} "
        f"unconsumed_prefetch_positions={prefetched_positions} "
        f"unconsumed_prefetch_pool_counts={dict(prefetched_pool_counts)}"
    )
    return pool_counts


def read_forward_records(
    audit_dir: Path,
    phase: str,
    world_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    observed_ranks: set[int] = set()
    for rank in range(world_size):
        path = audit_dir / f"forward-{phase}-rank-{rank}.jsonl"
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing forward-consumption audit: {path}")
        rank_records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(record.get("rank") != rank or record.get("phase") != phase for record in rank_records):
            raise AssertionError(f"Invalid forward-consumption records in {path}: {rank_records}")
        observed_ranks.add(rank)
        records.extend(rank_records)
    if observed_ranks != set(range(world_size)):
        raise AssertionError(f"Forward-consumption audit missed ranks: {observed_ranks}")
    return records


def validate_forward_window(
    records: list[dict[str, Any]],
    phase: str,
    start_position: int,
    end_position: int,
    world_size: int,
    planner: DeterministicHierarchicalMixture,
) -> tuple[Counter[str], dict[str, float]]:
    expected_positions = list(range(start_position, end_position))
    actual_positions = sorted(int(record["global_position"]) for record in records)
    if actual_positions != expected_positions or len(set(actual_positions)) != len(actual_positions):
        raise AssertionError(
            f"Phase {phase} actual forward positions mismatch: "
            f"expected={expected_positions} actual={actual_positions}"
        )
    rank_counts = Counter(int(record["rank"]) for record in records)
    expected_per_rank = (end_position - start_position) // world_size
    if rank_counts != Counter({rank: expected_per_rank for rank in range(world_size)}):
        raise AssertionError(
            f"Phase {phase} forward records were not aggregated from all ranks: {dict(rank_counts)}"
        )
    pool_counts: Counter[str] = Counter()
    pool_durations: dict[str, float] = {name: 0.0 for name in POOL_ORDER}
    for record in records:
        position = int(record["global_position"])
        selection = planner.selection(position)
        expected = {
            "pool_name": selection.pool_name,
            "record_index": selection.record_index,
            "pool_occurrence_index": selection.pool_occurrence_index,
            "pool_epoch": selection.pool_epoch,
        }
        actual = {key: record.get(key) for key in expected}
        if actual != expected:
            raise AssertionError(
                f"Phase {phase} actual forward sampler provenance mismatch at {position}: "
                f"actual={actual} expected={expected}"
            )
        duration = float(record["effective_duration_seconds"])
        if not (0.0 < duration <= 90.001):
            raise AssertionError(f"Phase {phase} invalid effective duration at {position}: {duration}")
        pool_counts[selection.pool_name] += 1
        pool_durations[selection.pool_name] += duration
    print(
        f"[forward-window] phase={phase} positions={start_position}..{end_position - 1} "
        f"records={len(records)} rank_counts={dict(rank_counts)} pool_counts={dict(pool_counts)} "
        f"effective_hours={sum(pool_durations.values()) / 3600.0:.9f}"
    )
    return pool_counts, pool_durations


def validate_statistics_state(
    path: Path,
    expected_step: int,
    expected_records: list[dict[str, Any]],
    expected_seed: int,
) -> dict[str, Any]:
    state = read_json(path)
    if (
        state.get("statistics_version") != TRAINING_STATS_VERSION
        or state.get("sampler_version") != SAMPLER_VERSION
        or int(state.get("sampler_seed", -1)) != expected_seed
        or int(state.get("global_step", -1)) != expected_step
        or int(state.get("world_size", -1)) != 4
    ):
        raise AssertionError(f"Invalid cumulative training statistics header at {path}: {state}")
    expected_counts = Counter(str(record["pool_name"]) for record in expected_records)
    expected_durations = {
        name: sum(
            float(record["effective_duration_seconds"])
            for record in expected_records
            if record["pool_name"] == name
        )
        for name in POOL_ORDER
    }
    if int(state.get("total_samples", -1)) != len(expected_records):
        raise AssertionError(f"Statistics total sample mismatch at {path}: {state}")
    if int(state.get("next_global_position", -1)) != len(expected_records):
        raise AssertionError(f"Statistics next position mismatch at {path}: {state}")
    pools = state.get("pools")
    if not isinstance(pools, dict) or set(pools) != set(POOL_ORDER):
        raise AssertionError(f"Statistics pools mismatch at {path}: {pools}")
    for name in POOL_ORDER:
        if int(pools[name].get("sample_count", -1)) != expected_counts[name]:
            raise AssertionError(f"Statistics pool sample mismatch at {path}: pool={name} state={pools[name]}")
        actual_duration = float(pools[name].get("effective_duration_seconds", -1.0))
        if abs(actual_duration - expected_durations[name]) > 1e-5:
            raise AssertionError(
                f"Statistics pool duration mismatch at {path}: pool={name} "
                f"actual={actual_duration} expected={expected_durations[name]}"
            )
        expected_sample_ratio = expected_counts[name] / len(expected_records)
        if abs(float(pools[name].get("sample_ratio", -1.0)) - expected_sample_ratio) > 1e-12:
            raise AssertionError(f"Statistics pool sample ratio mismatch at {path}: pool={name}")
    total_duration = sum(expected_durations.values())
    if abs(float(state.get("total_effective_duration_seconds", -1.0)) - total_duration) > 1e-5:
        raise AssertionError(f"Statistics total duration mismatch at {path}: {state}")
    for name in POOL_ORDER:
        expected_duration_ratio = expected_durations[name] / total_duration
        if abs(float(pools[name].get("duration_ratio", -1.0)) - expected_duration_ratio) > 1e-9:
            raise AssertionError(f"Statistics pool duration ratio mismatch at {path}: pool={name}")
    print(
        f"[statistics-state] path={path} step={expected_step} samples={len(expected_records)} "
        f"effective_hours={total_duration / 3600.0:.9f}"
    )
    return state


def main() -> None:
    args = parse_args()
    if args.world_size != 4 or not (0 < args.save_step < args.resume_step):
        raise ValueError("Expected world_size=4 and 0 < save_step < resume_step")
    save_pids, save_launch_ids = validate_phase_markers(
        args.save_audit_dir, "save", 0, args.save_step, args.world_size
    )
    resume_pids, resume_launch_ids = validate_phase_markers(
        args.resume_audit_dir,
        "resume",
        args.save_step,
        args.resume_step,
        args.world_size,
    )
    if save_launch_ids & resume_launch_ids:
        raise AssertionError(
            "Save and resume did not use distinct launch IDs: "
            f"overlap={sorted(save_launch_ids & resume_launch_ids)}"
        )

    registry = load_pool_registry(args.registry)
    pool_sizes = {name: int(registry["pools"][name]["record_count"]) for name in POOL_ORDER}
    planner = DeterministicHierarchicalMixture(pool_sizes=pool_sizes, seed=args.seed)
    with ExitStack() as stack:
        pools = open_indexed_pools(registry, stack)
        save_records = read_data_records(args.data_audit_dir, "save", args.world_size)
        resume_records = read_data_records(args.data_audit_dir, "resume", args.world_size)
        save_counts = validate_data_window(
            save_records,
            "save",
            0,
            args.save_step * args.world_size,
            planner,
            pools,
            max_prefetched_positions=2 * args.world_size,
        )
        resume_counts = validate_data_window(
            resume_records,
            "resume",
            args.save_step * args.world_size,
            args.resume_step * args.world_size,
            planner,
            pools,
            max_prefetched_positions=2 * args.world_size,
        )
        save_forward_records = read_forward_records(args.forward_audit_dir, "save", args.world_size)
        resume_forward_records = read_forward_records(args.forward_audit_dir, "resume", args.world_size)
        save_forward_counts, save_forward_durations = validate_forward_window(
            save_forward_records,
            "save",
            0,
            args.save_step * args.world_size,
            args.world_size,
            planner,
        )
        resume_forward_counts, resume_forward_durations = validate_forward_window(
            resume_forward_records,
            "resume",
            args.save_step * args.world_size,
            args.resume_step * args.world_size,
            args.world_size,
            planner,
        )
        if save_forward_counts != save_counts or resume_forward_counts != resume_counts:
            raise AssertionError(
                "Template consumed-prefix counts differ from actual forward counts: "
                f"save_template={save_counts} save_forward={save_forward_counts} "
                f"resume_template={resume_counts} resume_forward={resume_forward_counts}"
            )
        combined_forward_records = save_forward_records + resume_forward_records
        epoch_record_keys = [
            (str(record["pool_name"]), int(record["pool_epoch"]), int(record["record_index"]))
            for record in combined_forward_records
        ]
        if len(epoch_record_keys) != len(set(epoch_record_keys)):
            duplicates = [key for key, count in Counter(epoch_record_keys).items() if count > 1]
            raise AssertionError(f"No-replacement sampler repeated records across checkpoint: {duplicates}")
        if any(int(record["pool_epoch"]) != 0 for record in combined_forward_records):
            raise AssertionError("The short checkpoint smoke unexpectedly crossed a pool epoch boundary")
        raw_record_keys = [
            (str(record["pool_name"]), int(record["record_index"]))
            for record in combined_forward_records
        ]
        if len(raw_record_keys) != len(set(raw_record_keys)):
            duplicates = [key for key, count in Counter(raw_record_keys).items() if count > 1]
            raise AssertionError(f"Checkpoint smoke repeated raw pool records: {duplicates}")
        save_state = validate_statistics_state(
            args.save_stats_state,
            args.save_step,
            save_forward_records,
        )
        resume_state = validate_statistics_state(
            args.resume_stats_state,
            args.resume_step,
            combined_forward_records,
            args.seed,
        )
        run_delta = resume_state.get("run_delta", {})
        delta_counts = run_delta.get("sample_counts", {})
        delta_durations = run_delta.get("effective_duration_seconds", {})
        for name in POOL_ORDER:
            if int(delta_counts.get(name, -1)) != resume_forward_counts[name]:
                raise AssertionError(
                    f"Resume statistics sample delta mismatch for {name}: "
                    f"actual={delta_counts.get(name)} expected={resume_forward_counts[name]}"
                )
            if abs(float(delta_durations.get(name, -1.0)) - resume_forward_durations[name]) > 1e-5:
                raise AssertionError(
                    f"Resume statistics duration delta mismatch for {name}: "
                    f"actual={delta_durations.get(name)} expected={resume_forward_durations[name]}"
                )
        if int(save_state["next_global_position"]) != args.save_step * args.world_size:
            raise AssertionError(f"Save statistics did not persist the exact resume position: {save_state}")
    print(
        f"[process-groups] save_launch={sorted(save_launch_ids)} resume_launch={sorted(resume_launch_ids)} "
        f"save_pids={sorted(save_pids)} resume_pids={sorted(resume_pids)} distinct_launches=true"
    )
    print(f"[mixture] save_pool_counts={dict(save_counts)} resume_pool_counts={dict(resume_counts)}")
    print(
        "[no-replacement] cross_checkpoint_duplicates=0 "
        f"save_effective_hours={sum(save_forward_durations.values()) / 3600.0:.9f} "
        f"resume_delta_effective_hours={sum(resume_forward_durations.values()) / 3600.0:.9f} "
        "prefetch_counted_in_statistics=false four_rank_aggregation=true"
    )
    print("========== HUGINN WHISPER DYNAMIC90S CHECKPOINT RESUME MARKERS PASSED ==========")


if __name__ == "__main__":
    main()
