#!/usr/bin/env python3
"""Prepare and audit the combined HRM-audio tiny-overfit/resume gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_RECORDS = 89_658
EXPECTED_FIXTURE_RECORDS = 32
EXPECTED_UNIQUE_RECORDS = 4
EXPECTED_LORA_TENSORS = 512
EXPECTED_H_LORA_TENSORS = 256
EXPECTED_L_LORA_TENSORS = 256
EXPECTED_ALIGNER_TENSORS = 20
EXPECTED_OPTIMIZER_STATES = EXPECTED_LORA_TENSORS + EXPECTED_ALIGNER_TENSORS
EXPECTED_USER_PROMPT = "Listen to the audio and describe it."
EXPECTED_TOP_LEVEL_KEYS = {"messages", "audios", "metadata"}
ALIGNER_MARKERS = ("temporal_compressor.", "audio_projector.", "audio_boundary_embeddings.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build the deterministic real-audio tiny-overfit fixture.")
    prepare.add_argument("--source-manifest", type=Path, required=True)
    prepare.add_argument("--source-stats", type=Path)
    prepare.add_argument("--fixture-manifest", type=Path, required=True)
    prepare.add_argument("--output-report", type=Path, required=True)
    prepare.add_argument("--unique-records", type=int, default=EXPECTED_UNIQUE_RECORDS)
    prepare.add_argument("--fixture-records", type=int, default=EXPECTED_FIXTURE_RECORDS)

    audit = subparsers.add_parser("audit", help="Audit overfit behavior and exact fresh-process resume continuity.")
    audit.add_argument("--fixture-manifest", type=Path, required=True)
    audit.add_argument("--prepare-report", type=Path, required=True)
    audit.add_argument("--checkpoint-before-resume", type=Path, required=True)
    audit.add_argument("--checkpoint-after-resume", type=Path, required=True)
    audit.add_argument("--resume-boundary-report", type=Path, required=True)
    audit.add_argument("--resume-runtime-report", type=Path, required=True)
    audit.add_argument("--output-report", type=Path, required=True)
    audit.add_argument("--step-before-resume", type=int, default=12)
    audit.add_argument("--step-after-resume", type=int, default=24)
    audit.add_argument("--lora-rank", type=int, default=16)
    audit.add_argument("--lora-alpha", type=int, default=32)
    audit.add_argument("--lora-dropout", type=float, default=0.0)
    audit.add_argument("--learning-rate", type=float, default=1e-4)
    audit.add_argument("--minimum-relative-loss-reduction", type=float, default=0.10)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def canonical_adapter_key(key: str) -> str:
    for marker in ("H_module.", "L_module."):
        if marker in key:
            return key[key.index(marker) :].replace(".default.", ".")
    raise RuntimeError(f"Unexpected HRM adapter key: {key}")


def canonical_aligner_key(key: str) -> str | None:
    marker = next((candidate for candidate in ALIGNER_MARKERS if candidate in key), None)
    if marker is None:
        return None
    return key[key.index(marker) :].replace("original_module.", "").replace("modules_to_save.default.", "")


def verify_wav(path: Path, *, line_number: int) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"Fixture source line {line_number} has a missing/non-absolute WAV: {path}")
    with wave.open(str(path), "rb") as handle:
        report = {
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "sample_rate": handle.getframerate(),
            "compression": handle.getcomptype(),
            "frame_count": handle.getnframes(),
        }
    expected = {"channels": 1, "sample_width_bytes": 2, "sample_rate": 32_000, "compression": "NONE"}
    mismatches = {
        key: {"expected": value, "actual": report[key]}
        for key, value in expected.items()
        if report[key] != value
    }
    if mismatches or report["frame_count"] <= 0:
        raise RuntimeError(f"Fixture source line {line_number} WAV mismatch: {report} mismatches={mismatches}")
    return report


def validate_hrm_record(record: Any, *, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != EXPECTED_TOP_LEVEL_KEYS:
        raise RuntimeError(
            f"HRM manifest line {line_number} schema mismatch: "
            f"type={type(record)} keys={sorted(record) if isinstance(record, dict) else None}"
        )
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise RuntimeError(f"HRM manifest line {line_number} must have two messages: {messages}")
    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    if roles != ["user", "assistant"]:
        raise RuntimeError(f"HRM manifest line {line_number} role mismatch: {roles}")
    user_prompt = messages[0].get("content")
    caption = messages[1].get("content")
    if user_prompt != EXPECTED_USER_PROMPT:
        raise RuntimeError(f"HRM manifest line {line_number} user prompt mismatch: {user_prompt!r}")
    if not isinstance(caption, str) or not caption.strip():
        raise RuntimeError(f"HRM manifest line {line_number} caption is empty")
    audios = record.get("audios")
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
        raise RuntimeError(f"HRM manifest line {line_number} audio field mismatch: {audios}")
    audio_path = Path(audios[0]).expanduser()
    wav = verify_wav(audio_path, line_number=line_number)
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"HRM manifest line {line_number} metadata is not an object")
    if metadata.get("dataset") != "audiocaps_v2" or metadata.get("split") != "train":
        raise RuntimeError(f"HRM manifest line {line_number} metadata mismatch: {metadata}")
    sample_id = metadata.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise RuntimeError(f"HRM manifest line {line_number} sample_id is empty")
    return {
        "line_number": line_number,
        "audio_path": str(audio_path),
        "sample_id": sample_id,
        "caption": caption,
        "wav": wav,
    }


def prepare_fixture(args: argparse.Namespace) -> None:
    if args.unique_records <= 0 or args.fixture_records <= 0:
        raise ValueError("unique-records and fixture-records must be positive")
    if args.fixture_records % args.unique_records:
        raise ValueError(
            f"fixture-records must be divisible by unique-records: "
            f"fixture={args.fixture_records} unique={args.unique_records}"
        )
    source_manifest = args.source_manifest.expanduser().resolve()
    source_stats_path = (
        args.source_stats.expanduser().resolve()
        if args.source_stats is not None
        else source_manifest.with_suffix(f"{source_manifest.suffix}.stats.json")
    )
    fixture_manifest = args.fixture_manifest.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    for path, name in ((source_manifest, "source manifest"), (source_stats_path, "source stats")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if fixture_manifest.exists() or output_report.exists():
        raise FileExistsError(f"Refusing to overwrite fixture/report: fixture={fixture_manifest} report={output_report}")
    source_stats = json.loads(source_stats_path.read_text(encoding="utf-8"))
    expected_stats = {
        "dataset": "audiocaps_v2",
        "split": "train",
        "route": "hrm_text_audio_whisper",
        "template_contract": "hrm_text_audio_direct_user_assistant",
        "transformation": "remove_generic_system_message_only",
        "record_count": EXPECTED_SOURCE_RECORDS,
        "unique_audio_path_count": EXPECTED_SOURCE_RECORDS,
        "unique_sample_id_count": EXPECTED_SOURCE_RECORDS,
        "audio_path_verification": "passed",
        "wav_header_verification": "passed",
        "source_manifest_unchanged": True,
        "all_non_message_fields_preserved": True,
        "user_messages_preserved": True,
        "assistant_messages_preserved": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": source_stats.get(key)}
        for key, expected in expected_stats.items()
        if source_stats.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"HRM AudioCaps source stats mismatch: {mismatches}")
    source_sha_before = sha256_file(source_manifest)
    if source_sha_before != source_stats.get("output_manifest_sha256"):
        raise RuntimeError(
            f"HRM AudioCaps source hash mismatch: actual={source_sha_before} "
            f"stats={source_stats.get('output_manifest_sha256')}"
        )

    selected: list[dict[str, Any]] = []
    selected_reports: list[dict[str, Any]] = []
    audio_paths: set[str] = set()
    sample_ids: set[str] = set()
    with source_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"HRM AudioCaps source contains a blank line at {line_number}")
            record = json.loads(line)
            record_report = validate_hrm_record(record, line_number=line_number)
            if record_report["audio_path"] in audio_paths or record_report["sample_id"] in sample_ids:
                continue
            selected.append(record)
            selected_reports.append(record_report)
            audio_paths.add(record_report["audio_path"])
            sample_ids.add(record_report["sample_id"])
            if len(selected) == args.unique_records:
                break
    if len(selected) != args.unique_records:
        raise RuntimeError(f"Unable to select {args.unique_records} unique valid records; found={len(selected)}")

    fixture_records = [selected[index % len(selected)] for index in range(args.fixture_records)]
    fixture_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = fixture_manifest.with_name(f".{fixture_manifest.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in fixture_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(fixture_manifest)
    source_sha_after = sha256_file(source_manifest)
    if source_sha_after != source_sha_before:
        raise RuntimeError("HRM AudioCaps source changed while preparing the tiny-overfit fixture")
    fixture_sha = sha256_file(fixture_manifest)
    repetition_counts = Counter(record["metadata"]["sample_id"] for record in fixture_records)
    report = {
        "status": "OK",
        "source_manifest": str(source_manifest),
        "source_stats": str(source_stats_path),
        "source_manifest_sha256": source_sha_before,
        "source_manifest_unchanged": True,
        "fixture_manifest": str(fixture_manifest),
        "fixture_manifest_sha256": fixture_sha,
        "fixture_record_count": len(fixture_records),
        "unique_record_count": len(selected),
        "repetition_counts": dict(sorted(repetition_counts.items())),
        "role_sequence": ["user", "assistant"],
        "user_prompt": EXPECTED_USER_PROMPT,
        "selected_records": selected_reports,
        "first_fixture_record": fixture_records[0],
    }
    atomic_write_json(output_report, report)
    print("========== HRM AUDIO TINY-OVERFIT FIXTURE READY ==========")
    print(f"[fixture] status=OK source_sha256={source_sha_before}")
    print(f"[fixture] records={len(fixture_records)} unique={len(selected)} repetitions={dict(repetition_counts)}")
    print(f"[fixture] sha256={fixture_sha} path={fixture_manifest}")
    print(f"[fixture] report={output_report}")


def scalar_step(value: Any) -> int:
    import torch

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise RuntimeError(f"Optimizer step tensor is not scalar: shape={tuple(value.shape)}")
        value = value.item()
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeError(f"Optimizer step is invalid: {value!r}")
    return int(numeric)


def optimizer_state_report(payload: dict[str, Any], *, expected_step: int, expected_lr: float) -> dict[str, Any]:
    state = payload.get("state")
    param_groups = payload.get("param_groups")
    if not isinstance(state, dict) or not isinstance(param_groups, list):
        raise RuntimeError(f"Invalid optimizer state payload: keys={list(payload)}")
    steps = []
    missing_steps = []
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, dict) or "step" not in parameter_state:
            missing_steps.append(str(parameter_id))
        else:
            steps.append(scalar_step(parameter_state["step"]))
    if len(state) != EXPECTED_OPTIMIZER_STATES or missing_steps or len(steps) != EXPECTED_OPTIMIZER_STATES:
        raise RuntimeError(
            f"Optimizer state coverage mismatch: states={len(state)} steps={len(steps)} "
            f"expected={EXPECTED_OPTIMIZER_STATES} missing={missing_steps[:20]}"
        )
    step_counts = Counter(steps)
    if step_counts != {expected_step: EXPECTED_OPTIMIZER_STATES}:
        raise RuntimeError(f"Optimizer step continuity mismatch: expected={expected_step} actual={dict(step_counts)}")
    learning_rates = [float(group.get("lr")) for group in param_groups]
    if not learning_rates or any(
        not math.isclose(value, expected_lr, rel_tol=0.0, abs_tol=1e-12) for value in learning_rates
    ):
        raise RuntimeError(f"Optimizer learning-rate mismatch: expected={expected_lr} actual={learning_rates}")
    return {
        "state_count": len(state),
        "step_counts": dict(step_counts),
        "param_group_count": len(param_groups),
        "learning_rates": learning_rates,
    }


def scheduler_state_report(payload: dict[str, Any], *, expected_step: int) -> dict[str, Any]:
    last_epoch = payload.get("last_epoch")
    if last_epoch is None or int(last_epoch) != expected_step:
        raise RuntimeError(f"Scheduler last_epoch mismatch: expected={expected_step} actual={last_epoch}")
    return {
        "last_epoch": int(last_epoch),
        "step_count": int(payload.get("_step_count", -1)),
        "last_lr": [float(value) for value in payload.get("_last_lr", [])],
    }


def checkpoint_trainables(checkpoint: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    adapter_raw = read_safetensors(checkpoint / "adapter_model.safetensors")
    adapter = {canonical_adapter_key(key): tensor for key, tensor in adapter_raw.items()}
    if len(adapter) != len(adapter_raw):
        raise RuntimeError(f"Duplicate canonical LoRA tensors in {checkpoint}")
    aligner_raw = read_safetensors(checkpoint / "vit.safetensors")
    invalid_aligner = [key for key in aligner_raw if canonical_aligner_key(key) is None]
    aligner = {
        canonical_aligner_key(key): tensor
        for key, tensor in aligner_raw.items()
        if canonical_aligner_key(key) is not None
    }
    if invalid_aligner or len(aligner) != len(aligner_raw):
        raise RuntimeError(
            f"Invalid/duplicate canonical aligner tensors in {checkpoint}: invalid={invalid_aligner[:20]}"
        )
    return adapter, aligner


def inspect_checkpoint(
    checkpoint: Path,
    *,
    expected_step: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    checkpoint = checkpoint.expanduser().resolve()
    required_names = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "vit.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    missing = [name for name in required_names if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete HRM audio checkpoint {checkpoint}: missing={missing}")
    adapter_config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    expected_config = {"r": lora_rank, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout}
    config_mismatches = {}
    for key, expected in expected_config.items():
        actual = adapter_config.get(key)
        matches = (
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
            if isinstance(expected, float) and isinstance(actual, (int, float))
            else actual == expected
        )
        if not matches:
            config_mismatches[key] = {"expected": expected, "actual": actual}
    if config_mismatches:
        raise RuntimeError(f"Adapter config mismatch in {checkpoint}: {config_mismatches}")
    adapter, aligner = checkpoint_trainables(checkpoint)
    h_count = sum(key.startswith("H_module.") for key in adapter)
    l_count = sum(key.startswith("L_module.") for key in adapter)
    if (
        len(adapter) != EXPECTED_LORA_TENSORS
        or h_count != EXPECTED_H_LORA_TENSORS
        or l_count != EXPECTED_L_LORA_TENSORS
    ):
        raise RuntimeError(
            f"LoRA checkpoint coverage mismatch in {checkpoint}: total={len(adapter)} H={h_count} L={l_count}"
        )
    if len(aligner) != EXPECTED_ALIGNER_TENSORS:
        raise RuntimeError(f"Aligner checkpoint coverage mismatch in {checkpoint}: {len(aligner)}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != expected_step:
        raise RuntimeError(
            f"Trainer state global_step mismatch in {checkpoint}: "
            f"expected={expected_step} actual={trainer_state.get('global_step')}"
        )
    if int(trainer_state.get("max_steps", -1)) != expected_step:
        raise RuntimeError(
            f"Trainer state max_steps mismatch in {checkpoint}: "
            f"expected={expected_step} actual={trainer_state.get('max_steps')}"
        )
    optimizer_payload = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    scheduler_payload = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False)
    optimizer_report = optimizer_state_report(
        optimizer_payload,
        expected_step=expected_step,
        expected_lr=learning_rate,
    )
    scheduler_report = scheduler_state_report(scheduler_payload, expected_step=expected_step)
    files = {
        name: {
            "bytes": (checkpoint / name).stat().st_size,
            "sha256": sha256_file(checkpoint / name),
        }
        for name in required_names
    }
    if any(item["bytes"] <= 0 for item in files.values()):
        raise RuntimeError(f"Checkpoint contains an empty required file: {files}")
    return {
        "path": str(checkpoint),
        "global_step": expected_step,
        "adapter_tensor_count": len(adapter),
        "H_adapter_tensor_count": h_count,
        "L_adapter_tensor_count": l_count,
        "aligner_tensor_count": len(aligner),
        "adapter_dtype_counts": dict(Counter(str(tensor.dtype) for tensor in adapter.values())),
        "aligner_dtype_counts": dict(Counter(str(tensor.dtype) for tensor in aligner.values())),
        "optimizer": optimizer_report,
        "scheduler": scheduler_report,
        "files": files,
        "trainer_state": trainer_state,
    }


def tensor_update_report(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    *,
    name: str,
) -> dict[str, Any]:
    if set(before) != set(after):
        raise RuntimeError(
            f"{name} key mismatch across resume: "
            f"missing={sorted(set(before) - set(after))[:20]} "
            f"unexpected={sorted(set(after) - set(before))[:20]}"
        )
    changed = []
    max_abs_diff = 0.0
    squared_l2 = 0.0
    for key in sorted(before):
        left = before[key]
        right = after[key]
        if left.shape != right.shape:
            raise RuntimeError(f"{name} tensor shape changed for {key}: {tuple(left.shape)} -> {tuple(right.shape)}")
        difference = left.float() - right.float()
        current_max = float(difference.abs().max().item())
        max_abs_diff = max(max_abs_diff, current_max)
        squared_l2 += float(difference.square().sum().item())
        if current_max > 0.0:
            changed.append(key)
    if not changed:
        raise RuntimeError(f"No {name} tensors changed after resumed optimization")
    return {
        "tensor_count": len(before),
        "changed_tensor_count": len(changed),
        "unchanged_tensor_count": len(before) - len(changed),
        "max_abs_diff": max_abs_diff,
        "l2_diff": math.sqrt(squared_l2),
        "changed_preview": changed[:20],
    }


def extract_losses(trainer_state: dict[str, Any]) -> dict[int, float]:
    losses = {}
    for item in trainer_state.get("log_history", []):
        if not isinstance(item, dict) or "loss" not in item or "step" not in item:
            continue
        step = int(item["step"])
        loss = float(item["loss"])
        if step > 0 and math.isfinite(loss):
            losses[step] = loss
    return losses


def loss_reduction_report(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    *,
    step_before: int,
    step_after: int,
    minimum_relative_reduction: float,
) -> dict[str, Any]:
    before_losses = extract_losses(before_state)
    after_losses = extract_losses(after_state)
    resumed_steps = sorted(step for step in after_losses if step > step_before)
    if not resumed_steps:
        raise RuntimeError(f"Resumed trainer state contains no losses after step {step_before}: {after_losses}")
    combined = dict(before_losses)
    combined.update(after_losses)
    missing = [step for step in range(1, step_after + 1) if step not in combined]
    if missing:
        raise RuntimeError(f"Missing per-step losses for tiny-overfit audit: {missing}")
    ordered = [combined[step] for step in range(1, step_after + 1)]
    window_size = min(3, len(ordered) // 2)
    initial_mean = sum(ordered[:window_size]) / window_size
    final_mean = sum(ordered[-window_size:]) / window_size
    relative_reduction = (initial_mean - final_mean) / initial_mean
    if not math.isfinite(relative_reduction) or relative_reduction < minimum_relative_reduction:
        raise RuntimeError(
            "Tiny-overfit loss reduction is insufficient: "
            f"initial_mean={initial_mean} final_mean={final_mean} "
            f"relative_reduction={relative_reduction} required={minimum_relative_reduction}"
        )
    return {
        "step_losses": {str(step): combined[step] for step in range(1, step_after + 1)},
        "initial_window_steps": list(range(1, window_size + 1)),
        "final_window_steps": list(range(step_after - window_size + 1, step_after + 1)),
        "initial_window_mean": initial_mean,
        "final_window_mean": final_mean,
        "relative_reduction": relative_reduction,
        "required_relative_reduction": minimum_relative_reduction,
        "minimum_loss": min(ordered),
        "resume_history_preserved_in_second_checkpoint": all(
            step in after_losses for step in range(1, step_before + 1)
        ),
        "resumed_logged_steps": resumed_steps,
    }


def audit_combined_gate(args: argparse.Namespace) -> None:
    if args.step_before_resume <= 0 or args.step_after_resume <= args.step_before_resume:
        raise ValueError("Resume steps must satisfy 0 < before < after")
    if not 0.0 < args.minimum_relative_loss_reduction < 1.0:
        raise ValueError("minimum-relative-loss-reduction must be in (0, 1)")
    fixture_manifest = args.fixture_manifest.expanduser().resolve()
    prepare_report_path = args.prepare_report.expanduser().resolve()
    boundary_report_path = args.resume_boundary_report.expanduser().resolve()
    runtime_report_path = args.resume_runtime_report.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    for path, name in (
        (fixture_manifest, "fixture manifest"),
        (prepare_report_path, "prepare report"),
        (boundary_report_path, "resume boundary report"),
        (runtime_report_path, "resume runtime report"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    prepare_report = json.loads(prepare_report_path.read_text(encoding="utf-8"))
    if prepare_report.get("status") != "OK" or prepare_report.get("fixture_manifest_sha256") != sha256_file(fixture_manifest):
        raise RuntimeError(f"Fixture preparation report/hash mismatch: {prepare_report}")
    boundary_report = json.loads(boundary_report_path.read_text(encoding="utf-8"))
    runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
    if boundary_report.get("status") != "OK" or runtime_report.get("status") != "OK":
        raise RuntimeError(
            f"Resume runtime audit failed: boundary={boundary_report.get('status')} "
            f"runtime={runtime_report.get('status')}"
        )
    checkpoint_before = inspect_checkpoint(
        args.checkpoint_before_resume,
        expected_step=args.step_before_resume,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
    )
    checkpoint_after = inspect_checkpoint(
        args.checkpoint_after_resume,
        expected_step=args.step_after_resume,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
    )
    before_adapter, before_aligner = checkpoint_trainables(args.checkpoint_before_resume)
    after_adapter, after_aligner = checkpoint_trainables(args.checkpoint_after_resume)
    before_h = {key: value for key, value in before_adapter.items() if key.startswith("H_module.")}
    after_h = {key: value for key, value in after_adapter.items() if key.startswith("H_module.")}
    before_l = {key: value for key, value in before_adapter.items() if key.startswith("L_module.")}
    after_l = {key: value for key, value in after_adapter.items() if key.startswith("L_module.")}
    updates = {
        "H_lora": tensor_update_report(before_h, after_h, name="H-stack LoRA"),
        "L_lora": tensor_update_report(before_l, after_l, name="L-stack LoRA"),
        "aligner": tensor_update_report(before_aligner, after_aligner, name="aligner"),
    }
    losses = loss_reduction_report(
        checkpoint_before["trainer_state"],
        checkpoint_after["trainer_state"],
        step_before=args.step_before_resume,
        step_after=args.step_after_resume,
        minimum_relative_reduction=args.minimum_relative_loss_reduction,
    )
    if boundary_report.get("checkpoint") != str(args.checkpoint_before_resume.expanduser().resolve()):
        raise RuntimeError(f"Resume boundary audited the wrong checkpoint: {boundary_report}")
    if int(boundary_report.get("global_step", -1)) != args.step_before_resume:
        raise RuntimeError(f"Resume boundary global_step mismatch: {boundary_report}")
    if int(runtime_report.get("final_global_step", -1)) != args.step_after_resume:
        raise RuntimeError(f"Resume runtime final step mismatch: {runtime_report}")
    report = {
        "status": "OK",
        "policy": {
            "framework": "ms-swift SwiftSft/Seq2SeqTrainer",
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 32,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "aligner_learning_rate": args.learning_rate,
            "step_before_resume": args.step_before_resume,
            "step_after_resume": args.step_after_resume,
        },
        "fixture": prepare_report,
        "checkpoint_before_resume": checkpoint_before,
        "resume_boundary": boundary_report,
        "resume_runtime": runtime_report,
        "checkpoint_after_resume": checkpoint_after,
        "continued_updates": updates,
        "tiny_overfit": losses,
        "rng_state_changed": (
            checkpoint_before["files"]["rng_state.pth"]["sha256"]
            != checkpoint_after["files"]["rng_state.pth"]["sha256"]
        ),
    }
    if not report["rng_state_changed"]:
        raise RuntimeError("RNG state did not advance across resumed training")
    atomic_write_json(output_report, report)
    print("========== HRM AUDIO TINY-OVERFIT + RESUME AUDIT ==========")
    print(
        f"[resume] checkpoint-{args.step_before_resume} -> checkpoint-{args.step_after_resume} "
        f"optimizer_steps_exact=True scheduler_steps_exact=True boundary_weights_exact=True"
    )
    print(f"[updates] {json.dumps(updates, ensure_ascii=False)}")
    print(
        f"[tiny-overfit] initial_mean={losses['initial_window_mean']:.6f} "
        f"final_mean={losses['final_window_mean']:.6f} "
        f"relative_reduction={losses['relative_reduction']:.6f}"
    )
    print(f"[result] status=OK output_report={output_report}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_fixture(args)
    elif args.command == "audit":
        audit_combined_gate(args)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
