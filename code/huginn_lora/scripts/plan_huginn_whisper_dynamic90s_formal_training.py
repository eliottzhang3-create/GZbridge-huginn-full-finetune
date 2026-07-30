#!/usr/bin/env python3
"""Freeze the formal dynamic-90s training length from the registered data pools."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import load_pool_registry  # noqa: E402
from data_pipeline.indexed_atomic_mixture import (  # noqa: E402
    GLOBAL_POOL_WEIGHTS,
    POOL_ORDER,
    SAMPLER_VERSION,
    DeterministicHierarchicalMixture,
)


PLAN_VERSION = "huginn_whisper_dynamic90s_formal_plan_v1"
DEFAULT_SEED = 20260730
DEFAULT_TARGET_HOURS = 4000.0
DEFAULT_RESERVE_RATIO = 1.05
DEFAULT_STEP_ROUNDING = 100
DEFAULT_WORLD_SIZE = 4
DEFAULT_PER_DEVICE_BATCH = 2
DEFAULT_GRADIENT_ACCUMULATION = 4

# These are the fixed source-pool sizes supplied for this formal run. They are
# deliberately recorded in the plan because AAC atomic rows do not carry
# duration metadata. Actual decoded/capped duration remains the completion
# authority and is written into every checkpoint by the runtime callback.
POOL_SOURCE_HOURS = {
    "wavcaps_no_bbc_aac": 6500.0,
    "audiocaps_v2_aac": 136.0,
    "clotho_v2_aac": 24.0,
    "gigaspeech_l_asr": 2498.217,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-hours", type=float, default=DEFAULT_TARGET_HOURS)
    parser.add_argument("--reserve-ratio", type=float, default=DEFAULT_RESERVE_RATIO)
    parser.add_argument("--step-rounding", type=int, default=DEFAULT_STEP_ROUNDING)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--per-device-batch", type=int, default=DEFAULT_PER_DEVICE_BATCH)
    parser.add_argument("--gradient-accumulation", type=int, default=DEFAULT_GRADIENT_ACCUMULATION)
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError(f"seed must be non-negative, got {args.seed}")
    if args.target_hours <= 0:
        raise ValueError(f"target-hours must be positive, got {args.target_hours}")
    if args.reserve_ratio < 1.0:
        raise ValueError(f"reserve-ratio must be >= 1, got {args.reserve_ratio}")
    if args.step_rounding <= 0 or args.step_rounding % 2:
        raise ValueError("step-rounding must be a positive even integer")
    if min(args.world_size, args.per_device_batch, args.gradient_accumulation) <= 0:
        raise ValueError("world-size, per-device-batch, and gradient-accumulation must be positive")


def estimated_hours_for_steps(
    planner: DeterministicHierarchicalMixture,
    steps: int,
    global_batch: int,
    average_hours: dict[str, float],
) -> tuple[float, dict[str, int]]:
    sample_count = steps * global_batch
    counts = {name: 0 for name in POOL_ORDER}
    for position in range(sample_count):
        counts[planner.pool_for_position(position)] += 1
    hours = sum(counts[name] * average_hours[name] for name in POOL_ORDER)
    return hours, counts


def main() -> None:
    args = parse_args()
    validate_args(args)
    registry_path = args.registry.expanduser().resolve()
    registry = load_pool_registry(registry_path)
    pool_sizes = {name: int(registry["pools"][name]["record_count"]) for name in POOL_ORDER}
    average_hours = {name: POOL_SOURCE_HOURS[name] / pool_sizes[name] for name in POOL_ORDER}
    expected_hours_per_sample = sum(
        GLOBAL_POOL_WEIGHTS[name] * average_hours[name]
        for name in POOL_ORDER
    )
    global_batch = args.world_size * args.per_device_batch * args.gradient_accumulation
    planning_hours = args.target_hours * args.reserve_ratio
    unrounded_steps = planning_hours / (expected_hours_per_sample * global_batch)
    max_steps = math.ceil(unrounded_steps / args.step_rounding) * args.step_rounding

    planner = DeterministicHierarchicalMixture(pool_sizes=pool_sizes, seed=args.seed)
    estimated_hours, scheduled_counts = estimated_hours_for_steps(
        planner,
        max_steps,
        global_batch,
        average_hours,
    )
    while estimated_hours <= planning_hours:
        max_steps += args.step_rounding
        estimated_hours, scheduled_counts = estimated_hours_for_steps(
            planner,
            max_steps,
            global_batch,
            average_hours,
        )

    if max_steps % 2:
        raise RuntimeError(f"Rounded max_steps must be even, got {max_steps}")
    halfway_step = max_steps // 2
    total_samples = max_steps * global_batch
    scheduled_ratios = {
        name: scheduled_counts[name] / total_samples
        for name in POOL_ORDER
    }
    payload = {
        "plan_version": PLAN_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "sampler_seed": args.seed,
        "registry_path": str(registry_path),
        "registry_contract_version": registry.get("contract_version"),
        "target_realized_hours_minimum": args.target_hours,
        "planning_reserve_ratio": args.reserve_ratio,
        "planning_hours": planning_hours,
        "duration_estimate_is_completion_authority": False,
        "completion_authority": "checkpoint audio_training_statistics.json total_effective_duration_hours",
        "source_pool_hours_assumption": POOL_SOURCE_HOURS,
        "pool_sizes": pool_sizes,
        "pool_average_source_hours_per_record": average_hours,
        "configured_pool_weights": GLOBAL_POOL_WEIGHTS,
        "scheduled_pool_counts": scheduled_counts,
        "scheduled_pool_ratios": scheduled_ratios,
        "estimated_source_hours": estimated_hours,
        "world_size": args.world_size,
        "per_device_train_batch_size": args.per_device_batch,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "global_batch_size": global_batch,
        "step_rounding": args.step_rounding,
        "unrounded_steps_with_reserve": unrounded_steps,
        "max_steps": max_steps,
        "halfway_step": halfway_step,
        "total_scheduled_samples": total_samples,
        "checkpoint_steps": [halfway_step, max_steps],
    }
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Existing formal plan differs from the current frozen plan: {output_path}")
    else:
        write_json_atomic(output_path, payload)
    print(
        "[formal-plan] "
        f"max_steps={max_steps} halfway_step={halfway_step} global_batch={global_batch} "
        f"estimated_hours={estimated_hours:.6f} target_hours={args.target_hours:.3f} "
        f"planning_hours={planning_hours:.3f}"
    )
    print(f"[formal-plan] scheduled_pool_counts={scheduled_counts}")
    print(f"[formal-plan] scheduled_pool_ratios={scheduled_ratios}")
    print(f"[formal-plan] output={output_path}")


if __name__ == "__main__":
    main()
