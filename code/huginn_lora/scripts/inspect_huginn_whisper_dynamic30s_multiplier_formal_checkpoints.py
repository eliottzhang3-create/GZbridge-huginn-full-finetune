#!/usr/bin/env python3
"""Audit retained formal checkpoints from the finite multiplier epoch."""

from __future__ import annotations

import argparse
import json
import sys
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
)
from inspect_huginn_whisper_dynamic30s_smoke_fsdp_checkpoints import (  # noqa: E402
    strict_optimizer_coverage,
)
from inspect_huginn_whisper_dynamic90s_fsdp_checkpoints import (  # noqa: E402
    compare_model_states,
    inspect_checkpoint,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def checkpoint_step(path: Path) -> int:
    name = path.resolve().name
    if not name.startswith("checkpoint-"):
        raise ValueError(f"Invalid checkpoint directory name: {path}")
    return int(name.split("-", 1)[1])


def main() -> None:
    args = parse_args()
    if args.world_size != 4:
        raise ValueError("Multiplier formal checkpoint audit requires world_size=4")
    plan = load_json(args.plan.expanduser().resolve())
    if (
        plan.get("plan_version")
        != "huginn_dynamic30s_240ms_multiplier_single_epoch_plan_v2"
        or plan.get("sampler_version") != SAMPLER_VERSION
        or plan.get("training_contract", {}).get("audio")
        != {
            "maximum_seconds": 30.0,
            "token_duration_ms": 240,
            "maximum_content_tokens": 125,
        }
    ):
        raise ValueError(f"Invalid multiplier formal plan: {plan}")
    max_steps = int(plan["max_steps"])
    declared_steps = {int(value) for value in plan["checkpoint_steps"]}
    checkpoints = sorted(
        ((checkpoint_step(path), path.expanduser().resolve()) for path in args.checkpoint),
        key=lambda item: item[0],
    )
    if len(checkpoints) > int(plan["save_total_limit"]):
        raise ValueError(f"Too many retained multiplier checkpoints: {checkpoints}")
    if not checkpoints or checkpoints[-1][0] != max_steps:
        raise ValueError(f"Final multiplier checkpoint-{max_steps} is missing: {checkpoints}")
    if any(step not in declared_steps for step, _path in checkpoints):
        raise ValueError(f"Undeclared multiplier checkpoint was retained: {checkpoints}")

    inspected: list[dict[str, Any]] = []
    optimizer_coverage: dict[str, Any] = {}
    for step, path in checkpoints:
        expected_phase = (
            "multiplier_formal_final" if step == max_steps else "multiplier_formal_checkpoint"
        )
        checkpoint = inspect_checkpoint(path, step, args.world_size, expected_phase)
        optimizer_coverage[str(step)] = strict_optimizer_coverage(checkpoint)
        checkpoint_plan = load_json(path / "multiplier_formal_training_plan.json")
        if checkpoint_plan != plan:
            raise RuntimeError(f"Multiplier plan identity changed in checkpoint-{step}")
        formal_contract = checkpoint["training_runtime_contract"].get("formal_training")
        if (
            not isinstance(formal_contract, dict)
            or int(formal_contract.get("checkpoint_step", -1)) != step
            or formal_contract.get("plan_version") != plan["plan_version"]
            or formal_contract.get("sampler_version") != SAMPLER_VERSION
            or formal_contract.get("schedule_sha256") != plan["schedule_sha256"]
        ):
            raise RuntimeError(f"Multiplier formal runtime contract mismatch at {path}: {formal_contract}")
        inspected.append(checkpoint)

    final_path = checkpoints[-1][1]
    final_stats = load_json(final_path / "audio_training_statistics.json")
    if (
        final_stats.get("statistics_version") != STATISTICS_VERSION
        or final_stats.get("sampler_version") != SAMPLER_VERSION
        or int(final_stats.get("global_step", -1)) != max_steps
        or int(final_stats.get("total_samples", -1)) != int(plan["total_records"])
        or int(final_stats.get("next_global_position", -1)) != int(plan["total_records"])
    ):
        raise RuntimeError(f"Final multiplier statistics header mismatch: {final_stats}")
    pools = final_stats.get("pools")
    if not isinstance(pools, dict) or tuple(pools) != POOL_ORDER:
        raise RuntimeError(f"Final multiplier statistics pool set mismatch: {pools}")
    for name in POOL_ORDER:
        expected_count = int(plan["pool_sizes"][name])
        entry = pools[name]
        if (
            int(entry.get("sample_count", -1)) != expected_count
            or int(entry.get("pool_size", -1)) != expected_count
            or int(entry.get("completed_pool_epochs", -1)) != 1
            or int(entry.get("current_pool_epoch_offset", -1)) != 0
            or float(entry.get("effective_duration_seconds", 0.0)) <= 0.0
        ):
            raise RuntimeError(f"Final multiplier pool did not complete exactly once: {name}={entry}")
    if float(final_stats.get("total_effective_duration_seconds", 0.0)) <= 0.0:
        raise RuntimeError("Final multiplier effective duration is empty")

    comparison = None
    if len(inspected) >= 2:
        comparison = compare_model_states(inspected[0], inspected[-1])
    report = {
        "gate": "huginn_whisper_dynamic30s_multiplier_formal_checkpoints_v1",
        "validation_passed": True,
        "plan": plan,
        "retained_checkpoint_steps": [step for step, _path in checkpoints],
        "retained_checkpoint_paths": [str(path) for _step, path in checkpoints],
        "optimizer_coverage": optimizer_coverage,
        "final_statistics": final_stats,
        "model_comparison_first_to_final": comparison,
    }
    write_json_atomic(args.output_report, report)
    print(
        f"[multiplier-formal] checkpoints={[step for step, _path in checkpoints]} "
        f"samples={final_stats['total_samples']} "
        f"hours={final_stats['total_effective_duration_seconds'] / 3600.0:.6f}",
        flush=True,
    )
    print(f"[multiplier-formal] report={args.output_report.resolve()}", flush=True)
    print("========== HUGINN WHISPER DYNAMIC30S MULTIPLIER FORMAL AUDIT PASSED ==========", flush=True)


if __name__ == "__main__":
    main()
