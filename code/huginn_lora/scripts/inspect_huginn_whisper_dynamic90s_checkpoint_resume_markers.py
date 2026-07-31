#!/usr/bin/env python3
"""Audit cold-process FSDP4 resume markers and exact mixture positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


TRAINING_STATS_VERSION = "huginn_dynamic30s_training_statistics_v2"


EXPECTED_TRAINABLE_TENSORS = {
    "lora": 66,
    "aligner": 14,
    "huginn_base": 0,
    "other": 0,
}
EXPECTED_UNIT_TRAINABLE_TENSORS = {
    "WhisperEncoderFSDPUnit": None,
    "AudioAlignerFSDPUnit": 14,
    "HuginnPreludeFSDPUnit": 16,
    "HuginnRecurrentCoreFSDPUnit": 34,
    "HuginnCodaFSDPUnit": 16,
}
EXPECTED_RESHARD_AFTER_FORWARD = {
    "WhisperEncoderFSDPUnit": True,
    "AudioAlignerFSDPUnit": True,
    "HuginnPreludeFSDPUnit": True,
    "HuginnRecurrentCoreFSDPUnit": False,
    "HuginnCodaFSDPUnit": True,
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
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--artifact-fingerprint", type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty checkpoint marker: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint marker is not an object: {path}")
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


def validate_optimizer_groups(
    marker: dict[str, Any],
    trainables: dict[str, Any],
    *,
    phase: str,
    rank: int,
) -> None:
    audits = marker.get("optimizer_group_audit")
    if not isinstance(audits, list) or not audits:
        raise AssertionError(f"Missing {phase} optimizer-group audit for rank {rank}: {marker}")
    observed = {name: 0 for name in ("lora", "aligner", "audio_encoder", "huginn_base", "other")}
    for audit in audits:
        if abs(float(audit.get("learning_rate", -1.0)) - 1e-4) > 1e-12:
            raise AssertionError(f"Invalid {phase} optimizer LR for rank {rank}: {audits}")
        counts = audit.get("parameter_counts")
        if not isinstance(counts, dict):
            raise AssertionError(f"Missing {phase} optimizer parameter counts for rank {rank}: {audits}")
        for name in observed:
            observed[name] += int(counts.get(name, 0))
    expected = {name: int(trainables.get(name, 0)) for name in observed}
    if observed != expected:
        raise AssertionError(
            f"Invalid {phase} optimizer ownership for rank {rank}: "
            f"observed={observed} expected={expected}"
        )


def validate_lora_runtime(marker: dict[str, Any], *, phase: str, rank: int) -> None:
    audit = marker.get("lora_runtime_audit")
    expected = {
        "tensor_count": 66,
        "target_module_count": 33,
        "direct_lora_layer_count": 33,
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "restricted_to_huginn_transformer": True,
    }
    if not isinstance(audit, dict) or any(audit.get(key) != value for key, value in expected.items()):
        raise AssertionError(f"Invalid {phase} LoRA runtime audit for rank {rank}: {audit}")
    adapters = audit.get("adapters")
    if not isinstance(adapters, dict) or not adapters:
        raise AssertionError(f"Missing {phase} PEFT adapter audit for rank {rank}: {audit}")
    for adapter_name, config in adapters.items():
        if config != {"rank": 8, "alpha": 16.0, "dropout": 0.05}:
            raise AssertionError(
                f"Invalid {phase} PEFT adapter {adapter_name!r} for rank {rank}: {config}"
            )


def validate_reshard_audit(
    audits: Any,
    *,
    phase: str,
    rank: int,
    context: str,
) -> None:
    if not isinstance(audits, dict) or set(audits) != set(EXPECTED_RESHARD_AFTER_FORWARD):
        raise AssertionError(
            f"Invalid {phase} {context} reshard unit set for rank {rank}: {audits}"
        )
    for class_name, expected in EXPECTED_RESHARD_AFTER_FORWARD.items():
        audit = audits[class_name]
        if not isinstance(audit, dict):
            raise AssertionError(
                f"Invalid {phase} {context} reshard payload for rank {rank} "
                f"unit={class_name}: {audit}"
            )
        if "reshard_after_forward" in audit:
            audit = audit["reshard_after_forward"]
        if not isinstance(audit, dict):
            raise AssertionError(
                f"Invalid {phase} {context} nested reshard payload for rank {rank} "
                f"unit={class_name}: {audit}"
            )
        explicit = {
            bool(candidate["value"])
            for candidate in audit.get("candidates", [])
            if "value" in candidate
        }
        if (
            audit.get("effective") is not expected
            or explicit != {expected}
            or audit.get("has_fsdp_state") is not True
        ):
            raise AssertionError(
                f"Invalid {phase} {context} reshard state for rank {rank} "
                f"unit={class_name}: expected={expected} audit={audit}"
            )


def validate_checkpointing_contract(marker: dict[str, Any], *, phase: str, rank: int) -> None:
    if (
        marker.get("vit_gradient_checkpointing_arg") is not False
        or marker.get("whisper_internal_gradient_checkpointing") is not False
        or marker.get("whisper_gradient_checkpoint_modules") != []
        or marker.get("whisper_outer_activation_checkpointed") is not True
        or marker.get("whisper_double_checkpoint_candidate") is not False
    ):
        raise AssertionError(f"Invalid {phase} checkpointing contract for rank {rank}: {marker}")
    outer = marker.get("whisper_outer_checkpoint_wrappers", [])
    if (
        len(outer) != 1
        or not outer[0].get("path", "").endswith("audio_encoder.encoder")
        or "WhisperEncoder" not in outer[0].get("inner_mro", [])
    ):
        raise AssertionError(f"Invalid {phase} outer Whisper wrapper for rank {rank}: {outer}")
    validate_reshard_audit(
        marker.get("fsdp_reshard_after_forward"),
        phase=phase,
        rank=rank,
        context="effective",
    )
    runtime = marker.get("recurrent_core_runtime_audit")
    if (
        not isinstance(runtime, dict)
        or runtime.get("policy") != "recurrent_core_false_all_other_units_true"
        or int(runtime.get("world_size", -1)) != 4
    ):
        raise AssertionError(f"Invalid {phase} recurrent-core runtime audit for rank {rank}: {runtime}")
    before = runtime.get("before")
    expected_before = {name: True for name in EXPECTED_RESHARD_AFTER_FORWARD}
    if not isinstance(before, dict) or set(before) != set(expected_before):
        raise AssertionError(f"Invalid {phase} pre-mutation reshard audit for rank {rank}: {before}")
    for class_name in expected_before:
        audit = before[class_name].get("reshard_after_forward", {})
        if audit.get("effective") is not True:
            raise AssertionError(
                f"{phase} rank {rank} {class_name} was not initially reshard=true: {audit}"
            )
    validate_reshard_audit(
        runtime.get("after"),
        phase=phase,
        rank=rank,
        context="post-mutation",
    )


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
        ):
            raise AssertionError(f"Invalid {phase} FSDP marker for rank {rank}: {fsdp}")
        trainables = fsdp.get("trainable_tensors")
        if (
            not isinstance(trainables, dict)
            or int(trainables.get("lora", -1)) != EXPECTED_TRAINABLE_TENSORS["lora"]
            or int(trainables.get("aligner", -1)) != EXPECTED_TRAINABLE_TENSORS["aligner"]
            or int(trainables.get("audio_encoder", 0)) <= 0
            or int(trainables.get("huginn_base", -1)) != 0
            or int(trainables.get("other", -1)) != 0
            or int(fsdp.get("dtensor_trainable_count", -1))
            != sum(int(value) for value in trainables.values())
        ):
            raise AssertionError(f"Invalid {phase} Whisper-unfrozen trainable split for rank {rank}: {fsdp}")
        units = fsdp.get("fsdp_units")
        if not isinstance(units, dict) or set(units) != set(EXPECTED_UNIT_TRAINABLE_TENSORS):
            raise AssertionError(f"Invalid {phase} FSDP units for rank {rank}: {units}")
        for name, expected_trainables in EXPECTED_UNIT_TRAINABLE_TENSORS.items():
            unit = units[name]
            if (
                int(unit.get("parameter_count", 0)) <= 0
                or int(unit.get("dtensor_parameter_count", -1)) != int(unit["parameter_count"])
            ):
                raise AssertionError(f"Invalid {phase} unit {name} rank {rank}: {unit}")
            actual_trainables = int(unit.get("trainable_parameter_count", -1))
            if expected_trainables is None:
                if actual_trainables != int(unit["parameter_count"]):
                    raise AssertionError(f"Whisper unit is not fully trainable in {phase} rank {rank}: {unit}")
            elif actual_trainables != expected_trainables:
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
        validate_checkpointing_contract(start, phase=phase, rank=rank)
        validate_lora_runtime(start, phase=phase, rank=rank)
        checkpoint_wrappers = start.get("checkpoint_wrappers", [])
        required_wrapper_suffixes = (
            "transformer.prelude.0",
            "transformer.prelude.1",
            "transformer.core_block.0",
            "transformer.core_block.1",
            "transformer.core_block.2",
            "transformer.core_block.3",
            "transformer.core_block.adapter",
            "transformer.coda.0",
            "transformer.coda.1",
            "audio_encoder.encoder",
            "audio_aligner.temporal_compressor",
            "audio_aligner.audio_projector",
            "audio_aligner.audio_boundary_embeddings",
        )
        missing_wrappers = [
            suffix
            for suffix in required_wrapper_suffixes
            if not any(wrapper.get("path", "").endswith(suffix) for wrapper in checkpoint_wrappers)
        ]
        if missing_wrappers:
            raise AssertionError(
                f"Missing {phase} activation-checkpoint wrappers for rank {rank}: {missing_wrappers}"
            )
        validate_optimizer_groups(start, trainables, phase=phase, rank=rank)
        if phase == "resume":
            if (
                int(start.get("optimizer_state_count", -1)) != sum(int(value) for value in trainables.values())
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
        gradients = end.get("gradient_audit")
        if not isinstance(gradients, dict):
            raise AssertionError(f"Missing {phase} gradient audit for rank {rank}: {end}")
        for group in ("lora", "aligner", "audio_encoder"):
            if int(gradients.get(group, {}).get("nonzero_gradient_tensors", 0)) <= 0:
                raise AssertionError(f"No nonzero {group} gradients in {phase} rank {rank}: {gradients}")
        for group in ("huginn_base", "other"):
            if int(gradients.get(group, {}).get("gradient_tensors", -1)) != 0:
                raise AssertionError(f"Forbidden {group} gradients in {phase} rank {rank}: {gradients}")
        validate_checkpointing_contract(end, phase=phase, rank=rank)
        validate_lora_runtime(end, phase=phase, rank=rank)
        loss_contract = end.get("loss_contract", {})
        if (
            not (2 < int(loss_contract.get("prefix_length", 0)) <= 127)
            or loss_contract.get("prefix_labels_all_ignored") is not True
            or loss_contract.get("shift_length_valid") is not True
            or int(loss_contract.get("supervised_shift_tokens", 0)) <= 0
            or loss_contract.get("response_only_contiguous_suffix") is not True
            or not loss_contract.get("response_spans")
        ):
            raise AssertionError(
                f"Invalid {phase} shifted-NTP loss contract for rank {rank}: {loss_contract}"
            )
        if (
            float(end.get("train_wall_seconds", 0.0)) <= 0.0
            or float(end.get("peak_memory_allocated_gib", 99.0)) >= 29.0
            or float(end.get("peak_memory_reserved_gib", 99.0)) >= 30.0
        ):
            raise AssertionError(f"Invalid {phase} runtime/memory audit for rank {rank}: {end}")
        validate_optimizer_groups(end, trainables, phase=phase, rank=rank)
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
            f"optimizer_states={start.get('optimizer_state_count')} "
            f"learning_rates={start.get('learning_rates')} "
            f"finite_losses={end['finite_loss_log_count']} finite_grad_norms={end['finite_grad_norm_log_count']} "
            f"audio_samples={end['audio_sample_count']} audio_tokens={end['realized_audio_tokens']} "
            f"train_wall_seconds={end['train_wall_seconds']:.3f} "
            f"peak_reserved_gib={end['peak_memory_reserved_gib']:.3f}"
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
        if not (0.0 < duration <= 30.001):
            raise AssertionError(f"Phase {phase} invalid effective duration at {position}: {duration}")
        pool_counts[selection.pool_name] += 1
        pool_durations[selection.pool_name] += duration
    print(
        f"[forward-window] phase={phase} positions={start_position}..{end_position - 1} "
        f"records={len(records)} rank_counts={dict(rank_counts)} pool_counts={dict(pool_counts)} "
        f"effective_hours={sum(pool_durations.values()) / 3600.0:.9f}"
    )
    return pool_counts, pool_durations


def consumed_template_records(records: list[dict[str, Any]], positions: set[int]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    identity_fields = (
        "pool_name",
        "uid",
        "record_index",
        "pool_occurrence_index",
        "pool_epoch",
        "pool_epoch_offset",
    )
    for record in records:
        position = int(record["global_position"])
        if position not in positions:
            continue
        previous = selected.get(position)
        if previous is None:
            selected[position] = record
            continue
        if any(previous.get(field) != record.get(field) for field in identity_fields):
            raise AssertionError(
                f"Template provenance changed across duplicate encodes at position {position}: "
                f"first={previous} duplicate={record}"
            )
    if set(selected) != positions:
        raise AssertionError(
            f"Template provenance does not cover actual forward positions: "
            f"expected={sorted(positions)} actual={sorted(selected)}"
        )
    return selected


def provenance_entries(
    forward_records: list[dict[str, Any]],
    template_records: list[dict[str, Any]],
    planner: DeterministicHierarchicalMixture,
    pools: dict[str, Any],
) -> list[dict[str, Any]]:
    positions = {int(record["global_position"]) for record in forward_records}
    templates = consumed_template_records(template_records, positions)
    entries: list[dict[str, Any]] = []
    for forward in sorted(forward_records, key=lambda record: int(record["global_position"])):
        position = int(forward["global_position"])
        selection = planner.selection(position)
        atomic = pools[selection.pool_name].record(selection.record_index)
        template = templates[position]
        entry = {
            "global_position": position,
            "pool_name": str(forward["pool_name"]),
            "record_index": int(forward["record_index"]),
            "uid": str(template["uid"]),
            "pool_occurrence_index": int(forward["pool_occurrence_index"]),
            "pool_epoch": int(forward["pool_epoch"]),
            "pool_epoch_offset": int(template["pool_epoch_offset"]),
        }
        expected = {
            "global_position": position,
            "pool_name": selection.pool_name,
            "record_index": selection.record_index,
            "uid": str(atomic["uid"]),
            "pool_occurrence_index": selection.pool_occurrence_index,
            "pool_epoch": selection.pool_epoch,
            "pool_epoch_offset": selection.pool_epoch_offset,
        }
        if entry != expected:
            raise AssertionError(
                f"Actual forward provenance identity mismatch at position {position}: "
                f"actual={entry} expected={expected}"
            )
        entries.append(entry)
    return entries


def provenance_digest(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: int(value["global_position"])):
        rendered = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(rendered.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_per_rank(
    forward_records: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    entry_by_position = {int(entry["global_position"]): entry for entry in entries}
    summary: dict[str, Any] = {}
    for rank in range(world_size):
        records = sorted(
            (record for record in forward_records if int(record["rank"]) == rank),
            key=lambda record: int(record["global_position"]),
        )
        rank_entries = [entry_by_position[int(record["global_position"])] for record in records]
        positions = [int(record["global_position"]) for record in records]
        pool_counts = Counter(str(record["pool_name"]) for record in records)
        pool_durations = {
            name: sum(
                float(record["effective_duration_seconds"])
                for record in records
                if str(record["pool_name"]) == name
            )
            for name in POOL_ORDER
        }
        summary[str(rank)] = {
            "sample_count": len(records),
            "positions": positions,
            "position_min": min(positions) if positions else None,
            "position_max": max(positions) if positions else None,
            "pool_sample_counts": {name: int(pool_counts[name]) for name in POOL_ORDER},
            "pool_effective_duration_seconds": pool_durations,
            "effective_duration_seconds": sum(pool_durations.values()),
            "provenance_sha256": provenance_digest(rank_entries),
        }
    if sum(int(entry["sample_count"]) for entry in summary.values()) != len(forward_records):
        raise AssertionError(f"Per-rank statistics do not sum to the forward window: {summary}")
    return summary


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
    enhanced_values = (
        args.save_checkpoint,
        args.resume_checkpoint,
        args.artifact_fingerprint,
        args.output_report,
    )
    if any(value is not None for value in enhanced_values) and not all(
        value is not None for value in enhanced_values
    ):
        raise ValueError(
            "Enhanced checkpoint audit requires --save-checkpoint, --resume-checkpoint, "
            "--artifact-fingerprint, and --output-report together"
        )
    enhanced_enabled = all(value is not None for value in enhanced_values)
    if enhanced_enabled:
        for checkpoint in (args.save_checkpoint, args.resume_checkpoint):
            if not checkpoint.is_dir() or not (checkpoint / "pytorch_model_fsdp_0").is_dir():
                raise FileNotFoundError(f"Enhanced smoke checkpoint is incomplete: {checkpoint}")
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
        if enhanced_enabled:
            save_entries = provenance_entries(
                save_forward_records,
                save_records,
                planner,
                pools,
            )
            resume_entries = provenance_entries(
                resume_forward_records,
                resume_records,
                planner,
                pools,
            )
            combined_entries = sorted(
                save_entries + resume_entries,
                key=lambda entry: int(entry["global_position"]),
            )
            provenance_digests = {
                "algorithm": "sha256_ordered_canonical_jsonl_v1",
                "save": provenance_digest(save_entries),
                "resume_delta": provenance_digest(resume_entries),
                "cumulative": provenance_digest(combined_entries),
            }
            per_rank_statistics = {
                "save": summarize_per_rank(
                    save_forward_records,
                    save_entries,
                    args.world_size,
                ),
                "resume_delta": summarize_per_rank(
                    resume_forward_records,
                    resume_entries,
                    args.world_size,
                ),
                "cumulative": summarize_per_rank(
                    save_forward_records + resume_forward_records,
                    combined_entries,
                    args.world_size,
                ),
            }
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
            args.seed,
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
    if enhanced_enabled:
        artifact_fingerprint = read_json(args.artifact_fingerprint)
        if (
            artifact_fingerprint.get("gate")
            != "huginn_whisper_dynamic30s_smoke_data_identity_v1"
            or artifact_fingerprint.get("hash_algorithm") != "sha256"
        ):
            raise AssertionError(f"Invalid data artifact fingerprint: {artifact_fingerprint}")
        checkpoint_fingerprint_paths = {
            "save": args.save_checkpoint / "smoke_data_artifact_fingerprint.json",
            "resume": args.resume_checkpoint / "smoke_data_artifact_fingerprint.json",
        }
        for checkpoint_role, checkpoint_fingerprint_path in checkpoint_fingerprint_paths.items():
            checkpoint_fingerprint = read_json(checkpoint_fingerprint_path)
            if checkpoint_fingerprint != artifact_fingerprint:
                raise AssertionError(
                    f"{checkpoint_role} checkpoint data identity does not match the frozen "
                    f"smoke fingerprint: checkpoint={checkpoint_fingerprint_path}"
                )
        save_sidecar = {
            "gate": "huginn_whisper_dynamic30s_smoke_checkpoint_rank_statistics_v1",
            "checkpoint_role": "save",
            "global_step": args.save_step,
            "provenance_digest": provenance_digests["save"],
            "per_rank": per_rank_statistics["save"],
            "data_artifact_fingerprint": artifact_fingerprint,
        }
        resume_sidecar = {
            "gate": "huginn_whisper_dynamic30s_smoke_checkpoint_rank_statistics_v1",
            "checkpoint_role": "resume",
            "global_step": args.resume_step,
            "provenance_digests": provenance_digests,
            "per_rank_resume_delta": per_rank_statistics["resume_delta"],
            "per_rank_cumulative": per_rank_statistics["cumulative"],
            "data_artifact_fingerprint": artifact_fingerprint,
        }
        save_sidecar_path = args.save_checkpoint / "smoke_rank_statistics.json"
        resume_sidecar_path = args.resume_checkpoint / "smoke_rank_statistics.json"
        write_json_atomic(save_sidecar_path, save_sidecar)
        write_json_atomic(resume_sidecar_path, resume_sidecar)
        report = {
            "gate": "huginn_whisper_dynamic30s_enhanced_checkpoint_resume_smoke_v1",
            "validation_passed": True,
            "save_step": args.save_step,
            "resume_step": args.resume_step,
            "world_size": args.world_size,
            "provenance_digests": provenance_digests,
            "per_rank_statistics": per_rank_statistics,
            "data_artifact_fingerprint": artifact_fingerprint,
            "checkpoint_sidecars": {
                "save": str(save_sidecar_path.resolve()),
                "resume": str(resume_sidecar_path.resolve()),
            },
            "checkpoint_data_identity": {
                role: str(path.resolve())
                for role, path in checkpoint_fingerprint_paths.items()
            },
        }
        write_json_atomic(args.output_report, report)
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
    if enhanced_enabled:
        print(
            f"[provenance-digest] algorithm={provenance_digests['algorithm']} "
            f"save={provenance_digests['save']} resume_delta={provenance_digests['resume_delta']} "
            f"cumulative={provenance_digests['cumulative']}"
        )
        print(
            f"[rank-statistics] save={save_sidecar_path} resume={resume_sidecar_path} "
            f"report={args.output_report.resolve()}"
        )
    print("========== HUGINN WHISPER DYNAMIC30S CHECKPOINT RESUME MARKERS PASSED ==========")


if __name__ == "__main__":
    main()
