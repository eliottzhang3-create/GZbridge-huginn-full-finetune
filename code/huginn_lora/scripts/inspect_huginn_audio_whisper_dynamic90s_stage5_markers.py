#!/usr/bin/env python3
"""Verify all four rank markers from the 20-step Stage 5 stability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MAX_STEPS = 20
EXPECTED_TRAINABLE_TENSORS = {
    "lora": 66,
    "aligner": 14,
    "audio_encoder": 0,
    "huginn_base": 0,
    "other": 0,
}
EXPECTED_FIRST_STEP_PREFIX_TOKENS = {10, 252, 502, 752}
EXPECTED_UNIT_TRAINABLE_TENSORS = {
    "WhisperEncoderFSDPUnit": 0,
    "AudioAlignerFSDPUnit": 14,
    "HuginnPreludeFSDPUnit": 16,
    "HuginnRecurrentCoreFSDPUnit": 34,
    "HuginnCodaFSDPUnit": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args()


def read_marker(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty Stage 5 marker: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    observed_devices: set[int] = set()
    observed_prefix_tokens: set[int] = set()
    for rank in range(4):
        fsdp = read_marker(args.audit_dir / f"fsdp-rank-{rank}.json")
        if (
            fsdp.get("kind") != "fsdp"
            or fsdp.get("stage") != "stage5"
            or fsdp.get("rank") != rank
            or fsdp.get("world_size") != 4
        ):
            raise AssertionError(f"Invalid Stage 5 FSDP marker for rank {rank}: {fsdp}")
        if fsdp.get("trainable_tensors") != EXPECTED_TRAINABLE_TENSORS:
            raise AssertionError(f"Rank {rank} trainable split mismatch: {fsdp.get('trainable_tensors')}")
        if int(fsdp.get("dtensor_parameter_count", 0)) <= 0:
            raise AssertionError(f"Rank {rank} exposes no FSDP2 DTensor parameters")
        if int(fsdp.get("dtensor_trainable_count", -1)) != 80:
            raise AssertionError(f"Rank {rank} DTensor trainable count is not 80: {fsdp}")
        fsdp_units = fsdp.get("fsdp_units")
        if not isinstance(fsdp_units, dict) or set(fsdp_units) != set(EXPECTED_UNIT_TRAINABLE_TENSORS):
            raise AssertionError(f"Rank {rank} FSDP unit topology mismatch: {fsdp_units}")
        for unit_name, expected_trainables in EXPECTED_UNIT_TRAINABLE_TENSORS.items():
            unit = fsdp_units[unit_name]
            parameter_count = int(unit.get("parameter_count", 0))
            if parameter_count <= 0 or int(unit.get("dtensor_parameter_count", -1)) != parameter_count:
                raise AssertionError(f"Rank {rank} unit {unit_name} is not completely sharded: {unit}")
            if int(unit.get("trainable_parameter_count", -1)) != expected_trainables:
                raise AssertionError(f"Rank {rank} unit {unit_name} trainable split mismatch: {unit}")

        prefix_tokens = fsdp.get("valid_prefix_tokens")
        if not isinstance(prefix_tokens, list) or len(prefix_tokens) != 1:
            raise AssertionError(f"Rank {rank} expected one local sample, got {prefix_tokens}")
        observed_prefix_tokens.add(int(prefix_tokens[0]))
        observed_devices.add(int(fsdp["cuda_device"]))

        optimizer = read_marker(args.audit_dir / f"optimizer-step-rank-{rank}.json")
        if (
            optimizer.get("kind") != "optimizer_step"
            or optimizer.get("stage") != "stage5"
            or optimizer.get("rank") != rank
            or optimizer.get("world_size") != 4
            or optimizer.get("global_step") != EXPECTED_MAX_STEPS
            or optimizer.get("max_steps") != EXPECTED_MAX_STEPS
            or not optimizer.get("optimizer_type")
        ):
            raise AssertionError(f"Invalid Stage 5 optimizer marker for rank {rank}: {optimizer}")

        stability = read_marker(args.audit_dir / f"stability-rank-{rank}.json")
        if (
            stability.get("kind") != "stability"
            or stability.get("stage") != "stage5"
            or stability.get("rank") != rank
            or stability.get("world_size") != 4
            or stability.get("global_step") != EXPECTED_MAX_STEPS
            or stability.get("max_steps") != EXPECTED_MAX_STEPS
            or stability.get("finite_loss_log_count") != EXPECTED_MAX_STEPS
            or stability.get("finite_grad_norm_log_count") != EXPECTED_MAX_STEPS
            or not stability.get("optimizer_type")
        ):
            raise AssertionError(f"Invalid Stage 5 stability marker for rank {rank}: {stability}")
        print(
            f"[stage5-marker] rank={rank} cuda={fsdp['cuda_device']} "
            f"prefix_tokens={prefix_tokens[0]} dtensor_parameters={fsdp['dtensor_parameter_count']} "
            f"finite_losses={stability['finite_loss_log_count']} "
            f"finite_grad_norms={stability['finite_grad_norm_log_count']} "
            f"optimizer={optimizer['optimizer_type']} global_step={optimizer['global_step']}"
        )

    if observed_devices != {0, 1, 2, 3}:
        raise AssertionError(f"Expected CUDA devices 0-3, observed {sorted(observed_devices)}")
    if observed_prefix_tokens != EXPECTED_FIRST_STEP_PREFIX_TOKENS:
        raise AssertionError(
            "First distributed step did not cover the expected dynamic prefixes: "
            f"expected={sorted(EXPECTED_FIRST_STEP_PREFIX_TOKENS)} "
            f"actual={sorted(observed_prefix_tokens)}"
        )
    print("========== HUGINN WHISPER DYNAMIC90S STAGE 5 MARKERS PASSED ==========")


if __name__ == "__main__":
    main()
