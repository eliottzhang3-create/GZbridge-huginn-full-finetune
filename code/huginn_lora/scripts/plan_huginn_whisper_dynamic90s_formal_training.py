#!/usr/bin/env python3
"""Freeze the user-specified 20k-step dynamic-30s formal training plan."""

from __future__ import annotations

import argparse
import json
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


PLAN_VERSION = "huginn_whisper_dynamic30s_fixed20k_formal_plan_v3"
DEFAULT_SEED = 20260730
DEFAULT_MAX_STEPS = 20000
DEFAULT_CHECKPOINT_INTERVAL = 5000
DEFAULT_WORLD_SIZE = 4
DEFAULT_PER_DEVICE_BATCH = 2
DEFAULT_GRADIENT_ACCUMULATION = 4
TARGET_REALIZED_HOURS = 3000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
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
    if args.max_steps != DEFAULT_MAX_STEPS:
        raise ValueError(
            f"The active formal contract fixes max_steps={DEFAULT_MAX_STEPS}, got {args.max_steps}"
        )
    if args.checkpoint_interval != DEFAULT_CHECKPOINT_INTERVAL:
        raise ValueError(
            "The active formal contract fixes checkpoint_interval="
            f"{DEFAULT_CHECKPOINT_INTERVAL}, got {args.checkpoint_interval}"
        )
    if args.max_steps % args.checkpoint_interval != 0:
        raise ValueError("max_steps must be divisible by checkpoint_interval")
    if (args.world_size, args.per_device_batch, args.gradient_accumulation) != (4, 2, 4):
        raise ValueError(
            "The active formal contract requires world_size/per_device_batch/gradient_accumulation=4/2/4"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    registry_path = args.registry.expanduser().resolve()
    registry = load_pool_registry(registry_path)
    pool_sizes = {
        name: int(registry["pools"][name]["record_count"])
        for name in POOL_ORDER
    }
    global_batch = args.world_size * args.per_device_batch * args.gradient_accumulation
    total_samples = args.max_steps * global_batch
    checkpoint_steps = list(
        range(args.checkpoint_interval, args.max_steps + 1, args.checkpoint_interval)
    )
    if checkpoint_steps != [5000, 10000, 15000, 20000]:
        raise RuntimeError(f"Formal checkpoint schedule changed unexpectedly: {checkpoint_steps}")

    sampler = DeterministicHierarchicalMixture(pool_sizes=pool_sizes, seed=args.seed)
    scheduled_counts = {name: 0 for name in POOL_ORDER}
    scheduled_counts_by_checkpoint: dict[str, dict[str, int]] = {}
    checkpoint_positions = {
        step * global_batch: step
        for step in checkpoint_steps
    }
    for position in range(total_samples):
        scheduled_counts[sampler.pool_for_position(position)] += 1
        completed_samples = position + 1
        checkpoint_step = checkpoint_positions.get(completed_samples)
        if checkpoint_step is not None:
            scheduled_counts_by_checkpoint[str(checkpoint_step)] = dict(scheduled_counts)
    scheduled_ratios = {
        name: scheduled_counts[name] / total_samples
        for name in POOL_ORDER
    }
    payload = {
        "plan_version": PLAN_VERSION,
        "step_policy": "user_fixed_20000_steps_no_duration_estimation",
        "sampler_version": SAMPLER_VERSION,
        "sampler_epoch_policy": "per_pool_no_replacement_reshuffle_after_full_coverage",
        "sampler_seed": args.seed,
        "registry_path": str(registry_path),
        "registry_contract_version": registry.get("contract_version"),
        "duration_policy": "retain_all_then_cap_at30s",
        "target_realized_hours_minimum": TARGET_REALIZED_HOURS,
        "duration_estimate_used_for_max_steps": False,
        "completion_authority": (
            "final checkpoint audio_training_statistics.json "
            "total_effective_duration_hours"
        ),
        "pool_sizes": pool_sizes,
        "configured_pool_weights": GLOBAL_POOL_WEIGHTS,
        "scheduled_pool_counts": scheduled_counts,
        "scheduled_pool_counts_by_checkpoint": scheduled_counts_by_checkpoint,
        "scheduled_pool_ratios": scheduled_ratios,
        "world_size": args.world_size,
        "per_device_train_batch_size": args.per_device_batch,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "global_batch_size": global_batch,
        "max_steps": args.max_steps,
        "halfway_step": args.max_steps // 2,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_count": len(checkpoint_steps),
        "total_scheduled_samples": total_samples,
    }
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Existing formal plan differs from the fixed plan: {output_path}")
    else:
        write_json_atomic(output_path, payload)
    print(
        "[formal-plan] "
        f"max_steps={args.max_steps} checkpoint_steps={checkpoint_steps} "
        f"global_batch={global_batch} total_samples={total_samples} "
        "duration_estimation=false"
    )
    print(f"[formal-plan] scheduled_pool_counts={scheduled_counts}")
    print(f"[formal-plan] scheduled_pool_ratios={scheduled_ratios}")
    print(f"[formal-plan] output={output_path}")


if __name__ == "__main__":
    main()
