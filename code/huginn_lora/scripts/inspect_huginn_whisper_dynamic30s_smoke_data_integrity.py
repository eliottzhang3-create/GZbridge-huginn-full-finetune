#!/usr/bin/env python3
"""Smoke-only data identity and deterministic-resume integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import load_pool_registry  # noqa: E402
from data_pipeline.indexed_atomic_mixture import (  # noqa: E402
    POOL_ORDER,
    SAMPLER_VERSION,
    DeterministicHierarchicalMixture,
)


FINGERPRINT_GATE = "huginn_whisper_dynamic30s_smoke_data_identity_v1"
STATISTICS_VERSION = "huginn_dynamic30s_training_statistics_v2"
HASH_ALGORITHM = "sha256"
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--registry", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--registry", type=Path, required=True)
    verify.add_argument("--fingerprint", type=Path, required=True)

    resume = subparsers.add_parser("verify-resume-state")
    resume.add_argument("--registry", type=Path, required=True)
    resume.add_argument("--state", type=Path, required=True)
    resume.add_argument("--seed", type=int, required=True)
    resume.add_argument("--start-position", type=int, required=True)

    subparsers.add_parser("self-test")
    return parser.parse_args()


def sha256_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Data identity artifact is missing: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    after = resolved.stat()
    if (
        size != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise RuntimeError(
            "Artifact changed while hashing: "
            f"path={resolved} read={size} before_size={before.st_size} "
            f"after_size={after.st_size} before_mtime_ns={before.st_mtime_ns} "
            f"after_mtime_ns={after.st_mtime_ns}"
        )
    return {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def fingerprint_registry(registry_path: Path) -> dict[str, Any]:
    resolved = registry_path.expanduser().resolve()
    registry_identity = sha256_file(resolved)
    registry = load_pool_registry(resolved)
    pools: dict[str, Any] = {}
    for name in POOL_ORDER:
        entry = registry["pools"][name]
        pools[name] = {
            "record_count": int(entry["record_count"]),
            "manifest": sha256_file(Path(entry["manifest_path"])),
            "index": sha256_file(Path(entry["index_path"])),
        }
    registry_identity_after = sha256_file(resolved)
    if registry_identity_after != registry_identity:
        raise RuntimeError(
            f"Pool registry changed while its artifacts were being fingerprinted: {resolved}"
        )
    return {
        "gate": FINGERPRINT_GATE,
        "hash_algorithm": HASH_ALGORITHM,
        "registry": registry_identity,
        "registry_contract_version": registry.get("contract_version"),
        "duration_policy": registry.get("duration_policy"),
        "sampling_weights": registry.get("sampling_weights"),
        "pools": pools,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f"{resolved.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, resolved)


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Required data-integrity JSON is missing or empty: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Data-integrity JSON is not an object: {resolved}")
    return payload


def verify_fingerprint(registry_path: Path, fingerprint_path: Path) -> dict[str, Any]:
    expected = load_json(fingerprint_path)
    actual = fingerprint_registry(registry_path)
    if expected != actual:
        changed: list[str] = []
        if expected.get("registry") != actual.get("registry"):
            changed.append("registry")
        expected_pools = expected.get("pools", {})
        actual_pools = actual.get("pools", {})
        for name in POOL_ORDER:
            for artifact in ("manifest", "index"):
                if expected_pools.get(name, {}).get(artifact) != actual_pools.get(name, {}).get(artifact):
                    changed.append(f"{name}:{artifact}")
        raise RuntimeError(
            "Dynamic-30s data identity changed between save and resume: "
            f"changed={changed} expected={expected} actual={actual}"
        )
    return actual


def validate_resume_state_payload(
    state: dict[str, Any],
    registry: dict[str, Any],
    *,
    seed: int,
    start_position: int,
) -> dict[str, int]:
    if state.get("statistics_version") != STATISTICS_VERSION:
        raise RuntimeError(
            f"Resume statistics version mismatch: {state.get('statistics_version')!r}"
        )
    if state.get("sampler_version") != SAMPLER_VERSION:
        raise RuntimeError(f"Resume sampler version mismatch: {state.get('sampler_version')!r}")
    if int(state.get("sampler_seed", -1)) != seed:
        raise RuntimeError(
            f"Resume sampler seed mismatch: state={state.get('sampler_seed')} expected={seed}"
        )
    if int(state.get("next_global_position", -1)) != start_position:
        raise RuntimeError(
            "Resume next position mismatch: "
            f"state={state.get('next_global_position')} expected={start_position}"
        )
    pools = state.get("pools")
    if not isinstance(pools, dict) or set(pools) != set(POOL_ORDER):
        raise RuntimeError(f"Resume state pool set mismatch: {pools}")
    pool_sizes = {
        name: int(registry["pools"][name]["record_count"])
        for name in POOL_ORDER
    }
    planner = DeterministicHierarchicalMixture(pool_sizes=pool_sizes, seed=seed)
    expected_counts = planner.pool_occurrence_counts_before(start_position)
    actual_counts = {name: int(pools[name]["sample_count"]) for name in POOL_ORDER}
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Resume per-pool occurrence counts do not match the deterministic sequence: "
            f"actual={actual_counts} expected={expected_counts} position={start_position}"
        )
    if int(state.get("total_samples", -1)) != start_position:
        raise RuntimeError(
            f"Resume total sample mismatch: state={state.get('total_samples')} expected={start_position}"
        )
    for name in POOL_ORDER:
        count = expected_counts[name]
        pool_size = pool_sizes[name]
        entry = pools[name]
        expected_epoch, expected_offset = divmod(count, pool_size)
        if (
            int(entry.get("pool_size", -1)) != pool_size
            or int(entry.get("completed_pool_epochs", -1)) != expected_epoch
            or int(entry.get("current_pool_epoch_offset", -1)) != expected_offset
        ):
            raise RuntimeError(
                f"Resume pool epoch/offset mismatch for {name}: "
                f"entry={entry} expected_epoch={expected_epoch} expected_offset={expected_offset}"
            )
    return expected_counts


def verify_resume_state(
    registry_path: Path,
    state_path: Path,
    *,
    seed: int,
    start_position: int,
) -> dict[str, int]:
    registry = load_pool_registry(registry_path.expanduser().resolve())
    state = load_json(state_path)
    return validate_resume_state_payload(
        state,
        registry,
        seed=seed,
        start_position=start_position,
    )


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="huginn-dynamic30s-integrity-") as temporary:
        root = Path(temporary)
        first = root / "first.jsonl"
        second = root / "second.jsonl"
        first.write_text('{"uid":"A"}\n{"uid":"B"}\n', encoding="utf-8")
        second.write_text('{"uid":"B"}\n{"uid":"A"}\n', encoding="utf-8")
        if sha256_file(first)["sha256"] == sha256_file(second)["sha256"]:
            raise AssertionError("Manifest reorder did not change the SHA256 identity")

    pool_sizes = {name: 11 + index for index, name in enumerate(POOL_ORDER)}
    seed = 20260730
    position = 64
    planner = DeterministicHierarchicalMixture(pool_sizes=pool_sizes, seed=seed)
    counts = planner.pool_occurrence_counts_before(position)
    registry = {
        "pools": {
            name: {"record_count": pool_sizes[name]}
            for name in POOL_ORDER
        }
    }
    state = {
        "statistics_version": STATISTICS_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "sampler_seed": seed,
        "next_global_position": position,
        "total_samples": position,
        "pools": {
            name: {
                "sample_count": counts[name],
                "pool_size": pool_sizes[name],
                "completed_pool_epochs": counts[name] // pool_sizes[name],
                "current_pool_epoch_offset": counts[name] % pool_sizes[name],
            }
            for name in POOL_ORDER
        },
    }
    validate_resume_state_payload(state, registry, seed=seed, start_position=position)
    corrupted = json.loads(json.dumps(state))
    first_name, second_name = POOL_ORDER[:2]
    corrupted["pools"][first_name]["sample_count"] += 1
    corrupted["pools"][second_name]["sample_count"] -= 1
    try:
        validate_resume_state_payload(
            corrupted,
            registry,
            seed=seed,
            start_position=position,
        )
    except RuntimeError as exc:
        if "per-pool occurrence counts" not in str(exc):
            raise
    else:
        raise AssertionError("Count-preserving resume corruption was not rejected")
    print("[self-test] manifest_reorder_detected=true count_preserving_corruption_detected=true")


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = fingerprint_registry(args.registry)
        write_json_atomic(args.output, payload)
        print(f"[data-identity] frozen={args.output.expanduser().resolve()}")
        return
    if args.command == "verify":
        verify_fingerprint(args.registry, args.fingerprint)
        print(f"[data-identity] verified={args.fingerprint.expanduser().resolve()}")
        return
    if args.command == "verify-resume-state":
        counts = verify_resume_state(
            args.registry,
            args.state,
            seed=args.seed,
            start_position=args.start_position,
        )
        print(
            f"[resume-sequence] position={args.start_position} "
            f"deterministic_pool_counts={counts}"
        )
        return
    if args.command == "self-test":
        run_self_test()
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
