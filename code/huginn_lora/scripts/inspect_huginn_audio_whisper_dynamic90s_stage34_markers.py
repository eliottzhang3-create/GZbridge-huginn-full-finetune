#!/usr/bin/env python3
"""Verify all four FSDP and optimizer-step markers from the Stage 3-4 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_TRAINABLE_TENSORS = {
    "lora": 66,
    "aligner": 14,
    "audio_encoder": 0,
    "huginn_base": 0,
    "other": 0,
}
EXPECTED_FIRST_STEP_PREFIX_TOKENS = {10, 252, 502, 752}
EXPECTED_FSDP_UNITS = {
    "WhisperEncoderFSDPUnit",
    "AudioAlignerFSDPUnit",
    "HuginnPreludeFSDPUnit",
    "HuginnRecurrentCoreFSDPUnit",
    "HuginnCodaFSDPUnit",
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
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args()


def read_marker(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty Stage 3-4 marker: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    observed_devices: set[int] = set()
    observed_prefix_tokens: set[int] = set()
    for rank in range(4):
        fsdp = read_marker(args.audit_dir / f"fsdp-rank-{rank}.json")
        if fsdp.get("kind") != "fsdp" or fsdp.get("rank") != rank or fsdp.get("world_size") != 4:
            raise AssertionError(f"Invalid FSDP marker for rank {rank}: {fsdp}")
        if fsdp.get("trainable_tensors") != EXPECTED_TRAINABLE_TENSORS:
            raise AssertionError(f"Rank {rank} trainable split mismatch: {fsdp.get('trainable_tensors')}")
        if int(fsdp.get("dtensor_parameter_count", 0)) <= 0:
            raise AssertionError(f"Rank {rank} exposes no FSDP2 DTensor parameters")
        if int(fsdp.get("dtensor_trainable_count", -1)) != 80:
            raise AssertionError(f"Rank {rank} DTensor trainable count is not 80: {fsdp}")
        fsdp_units = fsdp.get("fsdp_units")
        if not isinstance(fsdp_units, dict) or set(fsdp_units) != EXPECTED_FSDP_UNITS:
            raise AssertionError(f"Rank {rank} FSDP unit topology mismatch: {fsdp_units}")
        for unit_name, unit_audit in fsdp_units.items():
            parameter_count = int(unit_audit.get("parameter_count", 0))
            dtensor_count = int(unit_audit.get("dtensor_parameter_count", -1))
            if parameter_count <= 0 or dtensor_count != parameter_count:
                raise AssertionError(
                    f"Rank {rank} unit {unit_name} is not completely sharded: {unit_audit}"
                )
            trainable_count = int(unit_audit.get("trainable_parameter_count", -1))
            if trainable_count != EXPECTED_UNIT_TRAINABLE_TENSORS[unit_name]:
                raise AssertionError(
                    f"Rank {rank} unit {unit_name} trainable split mismatch: {unit_audit}"
                )
        prefix_tokens = fsdp.get("valid_prefix_tokens")
        if not isinstance(prefix_tokens, list) or len(prefix_tokens) != 1:
            raise AssertionError(f"Rank {rank} expected one local sample, got {prefix_tokens}")
        observed_prefix_tokens.add(int(prefix_tokens[0]))
        observed_devices.add(int(fsdp["cuda_device"]))

        optimizer = read_marker(args.audit_dir / f"optimizer-step-rank-{rank}.json")
        if (
            optimizer.get("kind") != "optimizer_step"
            or optimizer.get("rank") != rank
            or optimizer.get("world_size") != 4
            or optimizer.get("global_step") != 1
            or optimizer.get("max_steps") != 1
            or not optimizer.get("optimizer_type")
        ):
            raise AssertionError(f"Invalid optimizer-step marker for rank {rank}: {optimizer}")
        print(
            f"[stage34-marker] rank={rank} cuda={fsdp['cuda_device']} "
            f"prefix_tokens={prefix_tokens[0]} dtensor_parameters={fsdp['dtensor_parameter_count']} "
            f"fsdp_units={sorted(fsdp_units)} "
            f"optimizer={optimizer['optimizer_type']} global_step=1"
        )

    if observed_devices != {0, 1, 2, 3}:
        raise AssertionError(f"Expected CUDA devices 0-3, observed {sorted(observed_devices)}")
    if observed_prefix_tokens != EXPECTED_FIRST_STEP_PREFIX_TOKENS:
        raise AssertionError(
            "First distributed step did not cover the expected dynamic prefixes: "
            f"expected={sorted(EXPECTED_FIRST_STEP_PREFIX_TOKENS)} actual={sorted(observed_prefix_tokens)}"
        )
    print("========== HUGINN WHISPER DYNAMIC90S STAGE 3-4 MARKERS PASSED ==========")


if __name__ == "__main__":
    main()
