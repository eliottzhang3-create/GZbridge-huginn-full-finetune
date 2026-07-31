#!/usr/bin/env python3
"""Freeze the exact one-epoch formal plan for the multiplier schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.finite_multiplier_pool import (  # noqa: E402
    POOL_ORDER,
    SAMPLER_VERSION,
    load_multiplier_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if (
        args.world_size != 4
        or args.per_device_batch != 2
        or args.gradient_accumulation_steps != 4
        or args.checkpoint_interval <= 0
    ):
        raise ValueError("Multiplier formal training requires FSDP4 B2 GA4 and a positive checkpoint interval")
    registry_path = args.registry.expanduser().resolve()
    audit_path = args.pool_audit.expanduser().resolve()
    registry = load_multiplier_registry(registry_path)
    audit = load_json(audit_path)
    if (
        not audit.get("validation_passed")
        or audit.get("gate") != "huginn_whisper_dynamic30s_multiplier_pool_audit_v1"
        or Path(audit.get("registry", "")).resolve() != registry_path
    ):
        raise ValueError(f"Multiplier pool audit has not passed for {registry_path}: {audit_path}")
    schedule_path = Path(registry["schedule_path"])
    if sha256_file(schedule_path) != registry["schedule_sha256"]:
        raise ValueError("Multiplier schedule identity changed after the passed pool audit")
    global_batch = args.world_size * args.per_device_batch * args.gradient_accumulation_steps
    total_records = int(registry["total_records"])
    if total_records % global_batch:
        raise ValueError(f"Multiplier schedule does not divide global batch: {total_records} % {global_batch}")
    max_steps = total_records // global_batch
    checkpoint_steps = list(range(args.checkpoint_interval, max_steps, args.checkpoint_interval))
    checkpoint_steps.append(max_steps)
    if checkpoint_steps != sorted(set(checkpoint_steps)):
        raise AssertionError(f"Invalid multiplier checkpoint schedule: {checkpoint_steps}")
    plan = {
        "plan_version": "huginn_dynamic30s_multiplier_single_epoch_plan_v1",
        "step_policy": "finite_expanded_pool_exactly_one_global_epoch",
        "sampler_version": SAMPLER_VERSION,
        "sampler_epoch_policy": "one_frozen_global_permutation_no_wraparound",
        "seed": int(registry["seed"]),
        "duration_policy": "retain_all_then_cap_at30s",
        "world_size": args.world_size,
        "per_device_train_batch_size": args.per_device_batch,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_batch_size": global_batch,
        "total_records": total_records,
        "max_steps": max_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_count_before_retention": len(checkpoint_steps),
        "save_total_limit": 4,
        "schedule_path": str(schedule_path.resolve()),
        "schedule_sha256": registry["schedule_sha256"],
        "source_registry_path": registry["source_registry_path"],
        "source_registry_sha256": registry["source_registry_sha256"],
        "pool_sizes": {
            name: int(registry["pools"][name]["record_count"])
            for name in POOL_ORDER
        },
        "components": {
            name: {
                "selected_record_count": int(entry["selected_record_count"]),
                "multiplier": int(entry["multiplier"]),
                "expanded_record_count": int(entry["expanded_record_count"]),
                "task": entry["task"],
                "pool_name": entry["pool_name"],
                "nominal_expanded_hours": float(entry["nominal_expanded_hours"]),
            }
            for name, entry in registry["components"].items()
        },
        "nominal_total_expanded_hours": sum(
            float(entry["nominal_expanded_hours"])
            for entry in registry["components"].values()
        ),
        "training_contract": {
            "whisper_encoder_trainable": True,
            "audio_aligner_trainable": True,
            "huginn_lora_trainable": True,
            "huginn_native_backbone_trainable": False,
            "learning_rates": {"whisper": 1e-4, "aligner": 1e-4, "lora": 1e-4},
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
            "audio": {"maximum_seconds": 30.0, "token_duration_ms": 160},
            "loss": "assistant_response_only_shifted_next_token_prediction",
        },
    }
    write_json_atomic(args.output.expanduser().resolve(), plan)
    print(
        f"[multiplier-plan] records={total_records} global_batch={global_batch} "
        f"max_steps={max_steps} checkpoints={checkpoint_steps}",
        flush=True,
    )
    print(f"[multiplier-plan] output={args.output.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
