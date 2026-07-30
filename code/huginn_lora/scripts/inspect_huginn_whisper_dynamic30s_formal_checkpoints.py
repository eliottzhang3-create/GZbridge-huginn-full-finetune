#!/usr/bin/env python3
"""Audit all four fixed-step checkpoints from formal dynamic-30s training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_huginn_whisper_dynamic90s_fsdp_checkpoints import (
    compare_model_states,
    inspect_checkpoint,
    write_json_atomic,
)


PLAN_VERSION = "huginn_whisper_dynamic30s_fixed20k_formal_plan_v3"
STATISTICS_VERSION = "huginn_dynamic30s_training_statistics_v2"
SAMPLER_VERSION = "deterministic_hierarchical_no_replacement_v2"
SAMPLER_EPOCH_POLICY = "per_pool_no_replacement_reshuffle_after_full_coverage"
PLAN_FILENAME = "formal_training_plan.json"
STATISTICS_FILENAME = "audio_training_statistics.json"
EXPECTED_CHECKPOINT_STEPS = (5000, 10000, 15000, 20000)
POOL_ORDER = (
    "wavcaps_no_bbc_aac",
    "audiocaps_v2_aac",
    "clotho_v2_aac",
    "gigaspeech_l_asr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required formal-training JSON is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Formal-training JSON is not an object: {path}")
    return payload


def checkpoint_step(path: Path) -> int:
    name = path.resolve().name
    if not name.startswith("checkpoint-"):
        raise ValueError(f"Formal checkpoint directory has an invalid name: {path}")
    try:
        return int(name.split("-", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Formal checkpoint directory has an invalid step: {path}") from exc


def validate_plan(plan: dict[str, Any], world_size: int) -> None:
    pool_sizes = plan.get("pool_sizes")
    scheduled_counts = plan.get("scheduled_pool_counts")
    checkpoint_counts = plan.get("scheduled_pool_counts_by_checkpoint")
    if (
        plan.get("plan_version") != PLAN_VERSION
        or plan.get("step_policy") != "user_fixed_20000_steps_no_duration_estimation"
        or plan.get("sampler_version") != SAMPLER_VERSION
        or plan.get("sampler_epoch_policy") != SAMPLER_EPOCH_POLICY
        or int(plan.get("sampler_seed", -1)) < 0
        or plan.get("duration_policy") != "retain_all_then_cap_at30s"
        or float(plan.get("target_realized_hours_minimum", -1.0)) != 3000.0
        or plan.get("duration_estimate_used_for_max_steps") is not False
        or int(plan.get("world_size", -1)) != world_size
        or int(plan.get("per_device_train_batch_size", -1)) != 2
        or int(plan.get("gradient_accumulation_steps", -1)) != 4
        or int(plan.get("global_batch_size", -1)) != 32
        or int(plan.get("max_steps", -1)) != 20000
        or int(plan.get("halfway_step", -1)) != 10000
        or int(plan.get("checkpoint_interval", -1)) != 5000
        or tuple(plan.get("checkpoint_steps", ())) != EXPECTED_CHECKPOINT_STEPS
        or int(plan.get("checkpoint_count", -1)) != 4
        or int(plan.get("total_scheduled_samples", -1)) != 640000
        or not isinstance(pool_sizes, dict)
        or set(pool_sizes) != set(POOL_ORDER)
        or any(int(pool_sizes[name]) <= 0 for name in POOL_ORDER)
        or not isinstance(scheduled_counts, dict)
        or set(scheduled_counts) != set(POOL_ORDER)
        or sum(int(scheduled_counts[name]) for name in POOL_ORDER) != 640000
        or not isinstance(checkpoint_counts, dict)
        or set(checkpoint_counts) != {str(step) for step in EXPECTED_CHECKPOINT_STEPS}
    ):
        raise RuntimeError(f"Frozen formal-training plan contract mismatch: {plan}")
    previous_counts = {name: 0 for name in POOL_ORDER}
    for step in EXPECTED_CHECKPOINT_STEPS:
        counts = checkpoint_counts[str(step)]
        if (
            not isinstance(counts, dict)
            or set(counts) != set(POOL_ORDER)
            or sum(int(counts[name]) for name in POOL_ORDER) != step * 32
            or any(int(counts[name]) < previous_counts[name] for name in POOL_ORDER)
        ):
            raise RuntimeError(
                f"Frozen checkpoint-{step} sampler counts are invalid: {counts}"
            )
        previous_counts = {name: int(counts[name]) for name in POOL_ORDER}
    if previous_counts != {
        name: int(scheduled_counts[name])
        for name in POOL_ORDER
    }:
        raise RuntimeError(
            "Final scheduled pool counts differ from the checkpoint schedule: "
            f"final={scheduled_counts} checkpoint={previous_counts}"
        )


def validate_runtime_contract(
    runtime: dict[str, Any],
    plan: dict[str, Any],
    *,
    step: int,
    checkpoint_index: int,
) -> None:
    is_final = step == 20000
    role = "final" if is_final else "scheduled"
    expected = {
        "checkpoint_role": role,
        "checkpoint_step": step,
        "checkpoint_index": checkpoint_index,
        "plan_version": plan["plan_version"],
        "step_policy": plan["step_policy"],
        "sampler_version": plan["sampler_version"],
        "sampler_epoch_policy": plan["sampler_epoch_policy"],
        "sampler_seed": int(plan["sampler_seed"]),
        "duration_policy": plan["duration_policy"],
        "duration_estimate_used_for_max_steps": False,
        "target_realized_hours_minimum": 3000.0,
        "max_steps": 20000,
        "halfway_step": 10000,
        "checkpoint_interval": 5000,
        "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
        "checkpoint_count": 4,
        "global_batch_size": 32,
        "total_scheduled_samples": 640000,
    }
    expected_phase = "formal_final" if is_final else "formal_checkpoint"
    if (
        runtime.get("phase") != expected_phase
        or int(runtime.get("global_step", -1)) != step
        or runtime.get("formal_training") != expected
    ):
        raise RuntimeError(
            f"Formal runtime contract mismatch at checkpoint-{step}: "
            f"actual={runtime.get('formal_training')} expected={expected}"
        )


def validate_statistics(
    state: dict[str, Any],
    plan: dict[str, Any],
    *,
    step: int,
) -> dict[str, Any]:
    expected_samples = step * 32
    pools = state.get("pools")
    tasks = state.get("tasks")
    if (
        state.get("statistics_version") != STATISTICS_VERSION
        or state.get("sampler_version") != SAMPLER_VERSION
        or int(state.get("sampler_seed", -1)) != int(plan["sampler_seed"])
        or state.get("event") != "checkpoint"
        or int(state.get("global_step", -1)) != step
        or int(state.get("world_size", -1)) != 4
        or int(state.get("total_samples", -1)) != expected_samples
        or int(state.get("next_global_position", -1)) != expected_samples
        or float(state.get("total_effective_duration_seconds", -1.0)) <= 0.0
        or not isinstance(pools, dict)
        or set(pools) != set(POOL_ORDER)
        or not isinstance(tasks, dict)
        or set(tasks) != {"AAC", "ASR"}
    ):
        raise RuntimeError(f"Formal checkpoint-{step} statistics header mismatch: {state}")

    total_count = 0
    total_duration = 0.0
    for name in POOL_ORDER:
        entry = pools[name]
        count = int(entry.get("sample_count", -1))
        duration = float(entry.get("effective_duration_seconds", -1.0))
        pool_size = int(plan["pool_sizes"][name])
        if (
            count < 0
            or duration < 0.0
            or int(entry.get("pool_size", -1)) != pool_size
            or int(entry.get("completed_pool_epochs", -1)) != count // pool_size
            or int(entry.get("current_pool_epoch_offset", -1)) != count % pool_size
        ):
            raise RuntimeError(f"Formal checkpoint-{step} pool statistics mismatch for {name}: {entry}")
        total_count += count
        total_duration += duration
    if total_count != expected_samples or abs(
        total_duration - float(state["total_effective_duration_seconds"])
    ) > 1e-4:
        raise RuntimeError(
            f"Formal checkpoint-{step} statistics totals mismatch: "
            f"count={total_count}/{expected_samples} duration={total_duration}/"
            f"{state['total_effective_duration_seconds']}"
        )
    aac_count = sum(int(pools[name]["sample_count"]) for name in POOL_ORDER[:3])
    asr_count = int(pools[POOL_ORDER[3]]["sample_count"])
    if (
        int(tasks["AAC"].get("sample_count", -1)) != aac_count
        or int(tasks["ASR"].get("sample_count", -1)) != asr_count
        or aac_count + asr_count != expected_samples
    ):
        raise RuntimeError(f"Formal checkpoint-{step} AAC/ASR statistics mismatch: {tasks}")
    expected_pool_counts = {
        name: int(plan["scheduled_pool_counts_by_checkpoint"][str(step)][name])
        for name in POOL_ORDER
    }
    observed_pool_counts = {
        name: int(pools[name]["sample_count"])
        for name in POOL_ORDER
    }
    if observed_pool_counts != expected_pool_counts:
        raise RuntimeError(
            f"Formal checkpoint-{step} deterministic sampler position mismatch: "
            f"actual={observed_pool_counts} expected={expected_pool_counts}"
        )
    return state


def main() -> None:
    args = parse_args()
    if args.world_size != 4 or len(args.checkpoint) != 4:
        raise ValueError("Formal audit requires world_size=4 and exactly four checkpoints")
    plan = load_json(args.plan.resolve())
    validate_plan(plan, args.world_size)
    checkpoints_by_step: dict[int, Path] = {}
    for path in args.checkpoint:
        step = checkpoint_step(path)
        if step in checkpoints_by_step:
            raise ValueError(f"Duplicate formal checkpoint step {step}: {path}")
        checkpoints_by_step[step] = path.resolve()
    if tuple(sorted(checkpoints_by_step)) != EXPECTED_CHECKPOINT_STEPS:
        raise ValueError(f"Formal checkpoint set mismatch: {sorted(checkpoints_by_step)}")

    inspected: dict[int, dict[str, Any]] = {}
    statistics_by_step: dict[int, dict[str, Any]] = {}
    for checkpoint_index, step in enumerate(EXPECTED_CHECKPOINT_STEPS, start=1):
        expected_phase = "formal_final" if step == 20000 else "formal_checkpoint"
        inspected[step] = inspect_checkpoint(
            checkpoints_by_step[step],
            step,
            args.world_size,
            expected_phase,
        )
        checkpoint = checkpoints_by_step[step]
        embedded_plan = load_json(checkpoint / PLAN_FILENAME)
        if embedded_plan != plan:
            raise RuntimeError(f"checkpoint-{step} embedded plan differs from the launch plan")
        validate_runtime_contract(
            inspected[step]["training_runtime_contract"],
            plan,
            step=step,
            checkpoint_index=checkpoint_index,
        )
        statistics_by_step[step] = validate_statistics(
            load_json(checkpoint / STATISTICS_FILENAME),
            plan,
            step=step,
        )

    for previous_step, current_step in zip(EXPECTED_CHECKPOINT_STEPS, EXPECTED_CHECKPOINT_STEPS[1:]):
        previous = statistics_by_step[previous_step]
        current = statistics_by_step[current_step]
        for name in POOL_ORDER:
            if (
                int(current["pools"][name]["sample_count"])
                < int(previous["pools"][name]["sample_count"])
                or float(current["pools"][name]["effective_duration_seconds"])
                < float(previous["pools"][name]["effective_duration_seconds"])
            ):
                raise RuntimeError(
                    f"Cumulative statistics regressed from checkpoint-{previous_step} "
                    f"to checkpoint-{current_step} for {name}"
                )

    final_stats = statistics_by_step[20000]
    final_counts = {
        name: int(final_stats["pools"][name]["sample_count"])
        for name in POOL_ORDER
    }
    expected_counts = {
        name: int(plan["scheduled_pool_counts"][name])
        for name in POOL_ORDER
    }
    if final_counts != expected_counts:
        raise RuntimeError(
            f"Final no-replacement schedule counts mismatch: "
            f"actual={final_counts} expected={expected_counts}"
        )

    comparison = compare_model_states(inspected[5000], inspected[20000])
    realized_hours = float(final_stats["total_effective_duration_hours"])
    target_hours = 3000.0
    target_met = realized_hours > target_hours
    report = {
        "gate": "huginn_whisper_dynamic30s_fixed20k_formal_checkpoints_v1",
        "validation_passed": target_met,
        "target_realized_hours_minimum": target_hours,
        "final_realized_hours": realized_hours,
        "target_strictly_exceeded": target_met,
        "plan": plan,
        "checkpoints": {
            str(step): {
                key: value
                for key, value in inspected[step].items()
                if key not in {"state_metadata", "grouped_keys"}
            }
            for step in EXPECTED_CHECKPOINT_STEPS
        },
        "statistics": {
            str(step): statistics_by_step[step]
            for step in EXPECTED_CHECKPOINT_STEPS
        },
        "first_to_final_model_comparison": comparison,
    }
    write_json_atomic(args.output_report.resolve(), report)
    print(
        "[formal-checkpoints] steps=[5000,10000,15000,20000] "
        f"samples={final_stats['total_samples']} realized_hours={realized_hours:.6f} "
        f"target_hours={target_hours:.6f} target_exceeded={target_met}"
    )
    print(f"[formal-checkpoints] pool_counts={final_counts}")
    print(f"[formal-checkpoints] report={args.output_report.resolve()}")
    if not target_met:
        raise RuntimeError(
            "All four formal checkpoints were saved and audited, but final realized hours did not "
            f"strictly exceed {target_hours}: actual={realized_hours}"
        )
    print("========== HUGINN WHISPER DYNAMIC30S FIXED20K FORMAL CHECKPOINTS PASSED ==========")


if __name__ == "__main__":
    main()
