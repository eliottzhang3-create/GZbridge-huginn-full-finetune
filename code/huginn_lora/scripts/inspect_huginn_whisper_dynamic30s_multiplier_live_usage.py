#!/usr/bin/env python3
"""Read-only audit of the frozen multiplier schedule and an active formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.finite_multiplier_pool import (  # noqa: E402
    COMPONENT_ORDER,
    EXPECTED_MULTIPLIERS,
    POOL_ORDER,
    SAMPLER_VERSION,
    STATISTICS_VERSION,
    FiniteMultiplierPool,
    load_multiplier_registry,
)


DEFAULT_REGISTRY = (
    REPO_ROOT
    / "data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m"
    / "multiplier_pool_registry.json"
)
DEFAULT_FORMAL_ROOT = (
    REPO_ROOT / "outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4"
)
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_GLOBAL_SIZE = 8
PER_RANK_BATCH_SIZE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--recent-duration-window", type=int, default=8192)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolve_run_root(args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        run_root = args.run_root.expanduser().resolve()
        if not run_root.is_dir():
            raise FileNotFoundError(f"Multiplier formal run root is missing: {run_root}")
        return run_root
    formal_root = args.formal_root.expanduser().resolve()
    search_roots = []
    for root in (formal_root, REPO_ROOT / "outputs"):
        resolved = root.resolve()
        if resolved.is_dir() and resolved not in search_roots:
            search_roots.append(resolved)
    candidates_by_path: dict[Path, float] = {}
    discovered_plans: list[Path] = []
    discovered_statistics: list[Path] = []
    skipped_names = {
        "swift_output",
        "save_phase",
        "resume_phase",
        "pytorch_model_fsdp_0",
        "training_statistics",
        "tensorboard",
    }

    def bounded_directories(root: Path, maximum_depth: int = 3):
        pending = [(root, 0)]
        while pending:
            directory, depth = pending.pop()
            yield directory
            if depth >= maximum_depth:
                continue
            try:
                children = list(directory.iterdir())
            except OSError:
                continue
            for child in children:
                if (
                    not child.is_dir()
                    or child.name in skipped_names
                    or child.name.startswith("checkpoint-")
                    or child.name.startswith("optimizer_")
                ):
                    continue
                pending.append((child, depth + 1))

    for search_root in search_roots:
        for candidate in bounded_directories(search_root):
            plan_path = candidate / "multiplier_formal_training_plan.json"
            latest = candidate / "training_statistics/latest.json"
            if plan_path.is_file():
                discovered_plans.append(plan_path)
            if latest.is_file():
                discovered_statistics.append(latest)
            if plan_path.is_file() and latest.is_file():
                candidates_by_path[candidate.resolve()] = latest.stat().st_mtime
    candidates = list(candidates_by_path)
    if not candidates:
        raise FileNotFoundError(
            "No multiplier formal run with a paired plan and live statistics was found. "
            f"searched={search_roots} "
            f"plans={[str(path) for path in sorted(set(discovered_plans))[-10:]]} "
            f"statistics={[str(path) for path in sorted(set(discovered_statistics))[-10:]]}. "
            "Set HUGINN_MULTIPLIER_LIVE_AUDIT_RUN_ROOT to the exact active run root if it is outside outputs."
        )
    return max(candidates, key=lambda path: candidates_by_path[path])


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training-statistics history is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Statistics row {line_number} is not an object: {path}")
            rows.append(payload)
    if not rows:
        raise RuntimeError(f"Training-statistics history is empty: {path}")
    return rows


def expected_pool_counts(registry: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(registry["pools"][name]["record_count"])
        for name in POOL_ORDER
    }


def validate_frozen_artifacts(
    registry_path: Path,
    registry: dict[str, Any],
) -> dict[str, Any]:
    preparation_path = registry_path.with_name("multiplier_pool_report.json")
    audit_path = registry_path.with_name("multiplier_pool_audit.json")
    preparation = load_json(preparation_path)
    audit = load_json(audit_path)
    if (
        not preparation.get("validation_passed")
        or preparation.get("sampler_version") != SAMPLER_VERSION
        or Path(preparation.get("registry_path", "")).resolve() != registry_path
        or not audit.get("validation_passed")
        or audit.get("sampler_version") != SAMPLER_VERSION
        or Path(audit.get("registry", "")).resolve() != registry_path
    ):
        raise AssertionError(
            f"Multiplier preparation/audit reports are invalid: preparation={preparation_path} audit={audit_path}"
        )
    schedule_artifact = preparation.get("schedule_artifact", {})
    if (
        Path(schedule_artifact.get("path", "")).resolve()
        != Path(registry["schedule_path"]).resolve()
        or schedule_artifact.get("sha256") != registry["schedule_sha256"]
        or audit.get("schedule_audit", {}).get("schedule_sha256")
        != registry["schedule_sha256"]
    ):
        raise AssertionError("Multiplier schedule identity differs across frozen reports")
    selection_artifacts = preparation.get("selection_artifacts")
    if not isinstance(selection_artifacts, dict):
        raise AssertionError(f"Preparation report has no selection artifacts: {preparation_path}")
    checked_selection_sha256 = {}
    for name, artifact in selection_artifacts.items():
        path = Path(artifact["path"]).resolve()
        actual_sha256 = sha256_file(path)
        if actual_sha256 != artifact["sha256"]:
            raise AssertionError(f"Selection artifact changed: component={name} path={path}")
        registry_selection = registry["components"][name].get("selection_index_path")
        if registry_selection is None or Path(registry_selection).resolve() != path:
            raise AssertionError(f"Selection path differs from registry: component={name}")
        checked_selection_sha256[name] = actual_sha256
    source_registry_path = Path(registry["source_registry_path"]).resolve()
    if sha256_file(source_registry_path) != registry["source_registry_sha256"]:
        raise AssertionError(f"Source registry changed: {source_registry_path}")
    return {
        "preparation_report": str(preparation_path),
        "pool_audit_report": str(audit_path),
        "source_filter_audit": audit.get("source_filter_audit"),
        "selection_sha256": checked_selection_sha256,
        "schedule_sha256": registry["schedule_sha256"],
        "source_registry_sha256": registry["source_registry_sha256"],
    }


def validate_statistics_shape(
    payload: dict[str, Any],
    *,
    registry: dict[str, Any],
) -> tuple[int, int]:
    if payload.get("statistics_version") != STATISTICS_VERSION:
        raise AssertionError(
            f"Statistics version mismatch: {payload.get('statistics_version')!r}"
        )
    if payload.get("sampler_version") != SAMPLER_VERSION:
        raise AssertionError(f"Sampler version mismatch: {payload.get('sampler_version')!r}")
    if int(payload.get("sampler_seed", -1)) != int(registry["seed"]):
        raise AssertionError(f"Statistics seed mismatch: {payload.get('sampler_seed')!r}")
    step = int(payload.get("global_step", -1))
    total_samples = int(payload.get("total_samples", -1))
    if step < 0 or total_samples != step * GLOBAL_BATCH_SIZE:
        raise AssertionError(
            f"Statistics step/sample mismatch: step={step} samples={total_samples}"
        )
    if int(payload.get("next_global_position", -1)) != total_samples:
        raise AssertionError(f"Statistics next position mismatch: {payload}")
    pools = payload.get("pools")
    if not isinstance(pools, dict) or set(pools) != set(POOL_ORDER):
        raise AssertionError(f"Statistics pool set mismatch: {pools}")
    if sum(int(pools[name]["sample_count"]) for name in POOL_ORDER) != total_samples:
        raise AssertionError(f"Statistics pool counts do not sum to {total_samples}: {pools}")
    for name, size in expected_pool_counts(registry).items():
        if int(pools[name].get("pool_size", -1)) != size:
            raise AssertionError(
                f"Statistics pool size mismatch for {name}: {pools[name].get('pool_size')} != {size}"
            )
    return step, total_samples


def capture_counts(
    component_counts: Counter[str],
    pool_counts: Counter[str],
    replica_counts: Counter[tuple[str, int]],
) -> dict[str, Any]:
    return {
        "component_counts": {
            name: int(component_counts[name])
            for name in COMPONENT_ORDER
        },
        "pool_counts": {
            name: int(pool_counts[name])
            for name in POOL_ORDER
        },
        "replica_counts": {
            f"{name}:replica-{replica}": int(replica_counts[(name, replica)])
            for name in COMPONENT_ORDER
            for replica in range(EXPECTED_MULTIPLIERS[name])
        },
    }


def scan_global_schedule(
    registry_path: Path,
    registry: dict[str, Any],
    snapshot_positions: set[int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    total = int(registry["total_records"])
    seen = bytearray(total)
    component_counts: Counter[str] = Counter()
    pool_counts: Counter[str] = Counter()
    replica_counts: Counter[tuple[str, int]] = Counter()
    snapshots: dict[int, dict[str, Any]] = {}
    digest = hashlib.sha256()
    transitions = 0
    previous_component: str | None = None
    current_run = 0
    maximum_run = 0
    mixed_global_batches = 0
    batch_components: set[str] = set()
    if 0 in snapshot_positions:
        snapshots[0] = capture_counts(component_counts, pool_counts, replica_counts)
    with FiniteMultiplierPool(registry_path) as pool:
        for position in range(total):
            selection = pool.selection(position)
            slot = selection.schedule_slot
            if seen[slot]:
                raise AssertionError(f"Global schedule repeats slot={slot} at position={position}")
            seen[slot] = 1
            digest.update(int(slot).to_bytes(8, "little"))
            component_counts[selection.component_name] += 1
            pool_counts[selection.pool_name] += 1
            replica_counts[(selection.component_name, selection.replica_id)] += 1
            batch_components.add(selection.component_name)
            if previous_component == selection.component_name:
                current_run += 1
            else:
                if previous_component is not None:
                    transitions += 1
                maximum_run = max(maximum_run, current_run)
                current_run = 1
                previous_component = selection.component_name
            consumed = position + 1
            if consumed % GLOBAL_BATCH_SIZE == 0:
                mixed_global_batches += int(len(batch_components) > 1)
                batch_components.clear()
            if consumed in snapshot_positions:
                snapshots[consumed] = capture_counts(
                    component_counts,
                    pool_counts,
                    replica_counts,
                )
    maximum_run = max(maximum_run, current_run)
    if not all(seen):
        missing = [index for index, value in enumerate(seen) if not value][:10]
        raise AssertionError(f"Global schedule omits virtual slots: {missing}")
    if digest.hexdigest() != registry["schedule_sha256"]:
        raise AssertionError(
            f"Schedule digest mismatch: {digest.hexdigest()} != {registry['schedule_sha256']}"
        )
    for name in COMPONENT_ORDER:
        entry = registry["components"][name]
        selected_count = int(entry["selected_record_count"])
        expected_count = selected_count * EXPECTED_MULTIPLIERS[name]
        if component_counts[name] != expected_count:
            raise AssertionError(
                f"Expanded component mismatch for {name}: {component_counts[name]} != {expected_count}"
            )
        for replica in range(EXPECTED_MULTIPLIERS[name]):
            if replica_counts[(name, replica)] != selected_count:
                raise AssertionError(
                    f"Replica coverage mismatch for {name}:replica-{replica}"
                )
    expected_pools = expected_pool_counts(registry)
    if dict(pool_counts) != expected_pools:
        raise AssertionError(f"Full pool counts mismatch: {pool_counts} != {expected_pools}")
    global_batches = total // GLOBAL_BATCH_SIZE
    report = {
        "total_records": total,
        "schedule_sha256": digest.hexdigest(),
        "component_counts": dict(component_counts),
        "pool_counts": dict(pool_counts),
        "replica_counts": {
            f"{name}:replica-{replica}": count
            for (name, replica), count in sorted(replica_counts.items())
        },
        "component_transitions": transitions,
        "maximum_same_component_run": maximum_run,
        "mixed_global_batches": mixed_global_batches,
        "global_batches": global_batches,
        "mixed_global_batch_ratio": mixed_global_batches / global_batches,
        "complete_permutation": True,
    }
    return report, snapshots


def compare_statistics_to_schedule(
    history: list[dict[str, Any]],
    latest: dict[str, Any],
    snapshots: dict[int, dict[str, Any]],
    registry: dict[str, Any],
) -> dict[str, Any]:
    rows = [*history]
    if rows[-1] != latest:
        rows.append(latest)
    previous_step = -1
    checked_positions: list[int] = []
    for payload in rows:
        step, total_samples = validate_statistics_shape(payload, registry=registry)
        if step < previous_step:
            raise AssertionError(
                f"Statistics history moved backwards: previous={previous_step} current={step}"
            )
        previous_step = step
        expected = snapshots.get(total_samples)
        if expected is None:
            raise AssertionError(f"No schedule snapshot was captured at position {total_samples}")
        actual_pools = {
            name: int(payload["pools"][name]["sample_count"])
            for name in POOL_ORDER
        }
        if actual_pools != expected["pool_counts"]:
            raise AssertionError(
                f"Consumed pool counts differ from schedule at position {total_samples}: "
                f"actual={actual_pools} expected={expected['pool_counts']}"
            )
        checked_positions.append(total_samples)
    latest_step, latest_samples = validate_statistics_shape(latest, registry=registry)
    expected_latest = snapshots[latest_samples]
    return {
        "latest_global_step": latest_step,
        "latest_total_samples": latest_samples,
        "checked_statistics_rows": len(rows),
        "checked_positions": sorted(set(checked_positions)),
        "expected_component_counts": expected_latest["component_counts"],
        "expected_pool_counts": expected_latest["pool_counts"],
        "expected_replica_counts": expected_latest["replica_counts"],
        "statistics_match_frozen_schedule": True,
    }


def effective_duration(record: dict[str, Any]) -> float:
    raw = record.get("raw_duration_sec")
    if raw is not None:
        duration = float(raw)
    else:
        audio = record.get("audio") if isinstance(record.get("audio"), dict) else {}
        start = audio.get("start_sec")
        end = audio.get("end_sec")
        if start is None or end is None:
            raise ValueError(f"Record has no usable duration metadata: {record.get('uid')}")
        duration = float(end) - float(start)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration for record {record.get('uid')}: {duration}")
    return min(duration, 30.0)


def audio_token_count(duration: float) -> int:
    feature_frames = min(3000, int(duration * 100.0))
    encoder_frames = feature_frames // 2
    if encoder_frames < 12:
        return 0
    return (encoder_frames - 12) // 12 + 1


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * probability))))
    return float(ordered[index])


def padding_report(values: list[float], group_size: int) -> dict[str, Any]:
    complete = len(values) // group_size * group_size
    if complete == 0:
        return {"groups": 0, "padding_amplification": None, "mean_group_max": None}
    values = values[:complete]
    true_total = sum(values)
    padded_total = 0.0
    maxima = []
    for start in range(0, complete, group_size):
        maximum = max(values[start : start + group_size])
        maxima.append(maximum)
        padded_total += maximum * group_size
    return {
        "groups": len(maxima),
        "padding_amplification": padded_total / true_total if true_total else None,
        "mean_group_max": sum(maxima) / len(maxima),
        "p95_group_max": quantile(maxima, 0.95),
    }


def inspect_recent_durations(
    registry_path: Path,
    total_samples: int,
    window: int,
) -> dict[str, Any]:
    if window <= 0:
        raise ValueError("recent-duration-window must be positive")
    start = max(0, total_samples - window)
    start -= start % MICRO_BATCH_GLOBAL_SIZE
    durations: list[float] = []
    tokens: list[float] = []
    by_component: dict[str, list[float]] = {name: [] for name in COMPONENT_ORDER}
    formats: Counter[str] = Counter()
    with FiniteMultiplierPool(registry_path) as pool:
        for position in range(start, total_samples):
            selection = pool.selection(position)
            record = pool.record(selection)
            duration = effective_duration(record)
            durations.append(duration)
            tokens.append(float(audio_token_count(duration)))
            by_component[selection.component_name].append(duration)
            audio = record.get("audio") if isinstance(record.get("audio"), dict) else {}
            formats[str(audio.get("format", "unknown")).lower() or "unknown"] += 1
    component_report = {}
    for name, values in by_component.items():
        if not values:
            continue
        component_report[name] = {
            "samples": len(values),
            "mean_seconds": sum(values) / len(values),
            "p50_seconds": quantile(values, 0.50),
            "p90_seconds": quantile(values, 0.90),
            "p99_seconds": quantile(values, 0.99),
            "at_30s_ratio": sum(value >= 30.0 for value in values) / len(values),
        }
    return {
        "position_start": start,
        "position_end_exclusive": total_samples,
        "samples": len(durations),
        "mean_seconds": sum(durations) / len(durations) if durations else 0.0,
        "p50_seconds": quantile(durations, 0.50),
        "p90_seconds": quantile(durations, 0.90),
        "p99_seconds": quantile(durations, 0.99),
        "at_30s_ratio": (
            sum(value >= 30.0 for value in durations) / len(durations) if durations else 0.0
        ),
        "mean_audio_content_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
        "p95_audio_content_tokens": quantile(tokens, 0.95),
        "format_counts": dict(formats),
        "components": component_report,
        "contiguous_per_rank_b2_padding": padding_report(durations, PER_RANK_BATCH_SIZE),
        "global_microbatch_b8_upper_bound": padding_report(durations, MICRO_BATCH_GLOBAL_SIZE),
        "audio_decode": False,
        "metadata_only": True,
    }


def exact_duration_statistics(latest: dict[str, Any]) -> dict[str, Any]:
    pools = latest["pools"]
    result = {}
    for name in POOL_ORDER:
        count = int(pools[name]["sample_count"])
        duration = float(pools[name]["effective_duration_seconds"])
        result[name] = {
            "sample_count": count,
            "sample_ratio": float(pools[name]["sample_ratio"]),
            "effective_duration_hours": duration / 3600.0,
            "duration_ratio": float(pools[name]["duration_ratio"]),
            "mean_effective_seconds": duration / count if count else 0.0,
        }
    return result


def main() -> None:
    args = parse_args()
    registry_path = args.registry.expanduser().resolve()
    registry = load_multiplier_registry(registry_path)
    schedule_path = Path(registry["schedule_path"])
    if sha256_file(schedule_path) != registry["schedule_sha256"]:
        raise AssertionError("Frozen multiplier schedule SHA256 changed")
    frozen_artifacts = validate_frozen_artifacts(registry_path, registry)
    run_root = resolve_run_root(args)
    plan_path = run_root / "multiplier_formal_training_plan.json"
    latest_path = run_root / "training_statistics/latest.json"
    history_path = run_root / "training_statistics/training_statistics.jsonl"
    plan = load_json(plan_path)
    latest = load_json(latest_path)
    history = load_history(history_path)
    if (
        plan.get("schedule_sha256") != registry["schedule_sha256"]
        or int(plan.get("seed", -1)) != int(registry["seed"])
        or int(plan.get("total_records", -1)) != int(registry["total_records"])
        or int(plan.get("max_steps", -1)) != int(registry["max_steps"])
        or int(plan.get("global_batch_size", -1)) != GLOBAL_BATCH_SIZE
    ):
        raise AssertionError(f"Formal plan differs from the frozen registry: {plan_path}")
    _latest_step, latest_samples = validate_statistics_shape(latest, registry=registry)
    snapshot_positions = {
        validate_statistics_shape(payload, registry=registry)[1]
        for payload in history
    }
    snapshot_positions.add(latest_samples)
    full_schedule, snapshots = scan_global_schedule(
        registry_path,
        registry,
        snapshot_positions,
    )
    consumption = compare_statistics_to_schedule(
        history,
        latest,
        snapshots,
        registry,
    )
    recent_durations = inspect_recent_durations(
        registry_path,
        latest_samples,
        args.recent_duration_window,
    )
    report = {
        "gate": "huginn_whisper_dynamic30s_multiplier_live_usage_audit_v1",
        "validation_passed": True,
        "audit_time": datetime.now().astimezone().isoformat(),
        "read_only": True,
        "model_load": False,
        "audio_decode": False,
        "audio_copy": False,
        "registry": str(registry_path),
        "run_root": str(run_root),
        "plan": str(plan_path),
        "latest_statistics": str(latest_path),
        "frozen_artifacts": frozen_artifacts,
        "full_schedule": full_schedule,
        "live_consumption": consumption,
        "exact_cumulative_duration_statistics": exact_duration_statistics(latest),
        "recent_duration_probe": recent_durations,
    }
    output = (
        args.output_report.expanduser().resolve()
        if args.output_report is not None
        else registry_path.parent
        / "audits"
        / f"multiplier_live_usage_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    write_json_atomic(output, report)
    print("========== HUGINN WHISPER DYNAMIC30S MULTIPLIER LIVE USAGE AUDIT PASSED ==========")
    print(
        f"[live] run={run_root} step={consumption['latest_global_step']} "
        f"samples={consumption['latest_total_samples']}"
    )
    print(
        f"[schedule] records={full_schedule['total_records']} complete_permutation=true "
        f"mixed_batch_ratio={full_schedule['mixed_global_batch_ratio']:.6f} "
        f"max_component_run={full_schedule['maximum_same_component_run']}"
    )
    print(f"[consumed-components] {consumption['expected_component_counts']}")
    print(f"[consumed-pools] {consumption['expected_pool_counts']}")
    print(f"[duration-exact] {report['exact_cumulative_duration_statistics']}")
    print(
        f"[duration-recent] samples={recent_durations['samples']} "
        f"mean={recent_durations['mean_seconds']:.3f}s "
        f"p50={recent_durations['p50_seconds']:.3f}s "
        f"p90={recent_durations['p90_seconds']:.3f}s "
        f"p99={recent_durations['p99_seconds']:.3f}s "
        f"at30={recent_durations['at_30s_ratio']:.6f}"
    )
    print(f"[duration-components] {recent_durations['components']}")
    print(f"[formats] {recent_durations['format_counts']}")
    print(f"[padding-b2] {recent_durations['contiguous_per_rank_b2_padding']}")
    print(f"[report] {output}")


if __name__ == "__main__":
    main()
