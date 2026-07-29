#!/usr/bin/env python3
"""Verify all rank markers from the real-mixture FSDP4 training gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import load_pool_registry  # noqa: E402
from data_pipeline.indexed_atomic_mixture import (  # noqa: E402
    POOL_ORDER,
    DeterministicHierarchicalMixture,
)


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
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    return parser.parse_args()


def read_marker(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty real-data marker: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Marker is not a JSON object: {path}")
    return payload


def validate_fsdp_marker(payload: dict, rank: int, world_size: int) -> int:
    if (
        payload.get("kind") != "fsdp"
        or payload.get("stage") != "realdata"
        or payload.get("rank") != rank
        or payload.get("world_size") != world_size
    ):
        raise AssertionError(f"Invalid real-data FSDP marker for rank {rank}: {payload}")
    if payload.get("trainable_tensors") != EXPECTED_TRAINABLE_TENSORS:
        raise AssertionError(f"Rank {rank} trainable split mismatch: {payload.get('trainable_tensors')}")
    if int(payload.get("dtensor_parameter_count", 0)) <= 0:
        raise AssertionError(f"Rank {rank} exposes no FSDP2 DTensor parameters")
    if int(payload.get("dtensor_trainable_count", -1)) != 80:
        raise AssertionError(f"Rank {rank} DTensor trainable count is not 80: {payload}")
    units = payload.get("fsdp_units")
    if not isinstance(units, dict) or set(units) != set(EXPECTED_UNIT_TRAINABLE_TENSORS):
        raise AssertionError(f"Rank {rank} FSDP unit topology mismatch: {units}")
    for unit_name, expected_trainables in EXPECTED_UNIT_TRAINABLE_TENSORS.items():
        unit = units[unit_name]
        parameter_count = int(unit.get("parameter_count", 0))
        if parameter_count <= 0 or int(unit.get("dtensor_parameter_count", -1)) != parameter_count:
            raise AssertionError(f"Rank {rank} unit {unit_name} is not fully sharded: {unit}")
        if int(unit.get("trainable_parameter_count", -1)) != expected_trainables:
            raise AssertionError(f"Rank {rank} unit {unit_name} trainable split mismatch: {unit}")
    prefix_tokens = payload.get("valid_prefix_tokens")
    if not isinstance(prefix_tokens, list) or len(prefix_tokens) != 1:
        raise AssertionError(f"Rank {rank} expected one first-step local sample: {prefix_tokens}")
    value = int(prefix_tokens[0])
    if value < 2 or value > 752:
        raise AssertionError(f"Rank {rank} first prefix is outside [2, 752]: {value}")
    return value


def main() -> None:
    args = parse_args()
    if args.max_steps <= 1 or args.world_size != 4 or args.per_device_batch_size != 1:
        raise ValueError("This gate requires max_steps>1, world_size=4, and per-device batch size 1")
    registry = load_pool_registry(args.registry)
    pool_sizes = {
        name: int(registry["pools"][name]["record_count"])
        for name in POOL_ORDER
    }
    planner = DeterministicHierarchicalMixture(pool_sizes=pool_sizes, seed=args.seed)
    global_samples = args.max_steps * args.world_size * args.per_device_batch_size
    pool_counts = Counter(planner.pool_for_position(position) for position in range(global_samples))
    if set(pool_counts) != set(POOL_ORDER):
        raise AssertionError(
            f"The real-data gate schedule does not cover all four pools: {dict(pool_counts)}"
        )
    task_counts = {
        "AAC": sum(pool_counts[name] for name in POOL_ORDER if name != "gigaspeech_l_asr"),
        "ASR": pool_counts["gigaspeech_l_asr"],
    }

    observed_devices: set[int] = set()
    aggregate_audio_samples = 0
    aggregate_audio_tokens = 0
    for rank in range(args.world_size):
        fsdp = read_marker(args.audit_dir / f"fsdp-rank-{rank}.json")
        first_prefix_tokens = validate_fsdp_marker(fsdp, rank, args.world_size)
        observed_devices.add(int(fsdp["cuda_device"]))

        optimizer = read_marker(args.audit_dir / f"optimizer-step-rank-{rank}.json")
        if (
            optimizer.get("kind") != "optimizer_step"
            or optimizer.get("stage") != "realdata"
            or optimizer.get("rank") != rank
            or optimizer.get("world_size") != args.world_size
            or optimizer.get("global_step") != args.max_steps
            or optimizer.get("max_steps") != args.max_steps
            or not optimizer.get("optimizer_type")
        ):
            raise AssertionError(f"Invalid optimizer marker for rank {rank}: {optimizer}")

        stability = read_marker(args.audit_dir / f"realdata-stability-rank-{rank}.json")
        if (
            stability.get("kind") != "realdata_stability"
            or stability.get("stage") != "realdata"
            or stability.get("rank") != rank
            or stability.get("world_size") != args.world_size
            or stability.get("global_step") != args.max_steps
            or stability.get("max_steps") != args.max_steps
            or stability.get("finite_loss_log_count") != args.max_steps
            or stability.get("finite_grad_norm_log_count") != args.max_steps
            or stability.get("audio_batch_count") != args.max_steps
            or stability.get("audio_sample_count") != args.max_steps
            or int(stability.get("realized_audio_tokens", 0)) <= 0
            or int(stability.get("min_audio_tokens", -1)) < 0
            or int(stability.get("max_audio_tokens", 751)) > 750
            or not stability.get("optimizer_type")
        ):
            raise AssertionError(f"Invalid real-data stability marker for rank {rank}: {stability}")
        aggregate_audio_samples += int(stability["audio_sample_count"])
        aggregate_audio_tokens += int(stability["realized_audio_tokens"])
        print(
            f"[realdata-marker] rank={rank} cuda={fsdp['cuda_device']} "
            f"first_prefix_tokens={first_prefix_tokens} dtensor_parameters={fsdp['dtensor_parameter_count']} "
            f"finite_losses={stability['finite_loss_log_count']} "
            f"finite_grad_norms={stability['finite_grad_norm_log_count']} "
            f"audio_samples={stability['audio_sample_count']} "
            f"audio_tokens={stability['realized_audio_tokens']} "
            f"audio_token_range=[{stability['min_audio_tokens']},{stability['max_audio_tokens']}] "
            f"global_step={optimizer['global_step']}"
        )

    if observed_devices != {0, 1, 2, 3}:
        raise AssertionError(f"Expected CUDA devices 0-3, observed {sorted(observed_devices)}")
    if aggregate_audio_samples != global_samples:
        raise AssertionError(
            f"Global real-data sample count mismatch: expected={global_samples} actual={aggregate_audio_samples}"
        )
    print(f"[mixture-window] positions=0..{global_samples - 1} pool_counts={dict(pool_counts)} task_counts={task_counts}")
    print(
        f"[runtime-audio] global_samples={aggregate_audio_samples} "
        f"realized_audio_tokens={aggregate_audio_tokens}"
    )
    print("========== HUGINN WHISPER DYNAMIC90S REALDATA FSDP4 MARKERS PASSED ==========")


if __name__ == "__main__":
    main()
