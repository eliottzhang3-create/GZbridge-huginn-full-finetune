#!/usr/bin/env python3
"""Preflight and checkpoint audit for formal two-epoch HRM AudioCaps-v2 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import audit_hrm_audio_tiny_overfit_resume as checkpoint_audit


EXPECTED_RECORDS = 89_658
EXPECTED_EPOCHS = 2
EXPECTED_MICRO_BATCH = 8
EXPECTED_GRADIENT_ACCUMULATION = 4
EXPECTED_EFFECTIVE_BATCH = 32
EXPECTED_STEPS_PER_EPOCH = 2_802
EXPECTED_TOTAL_STEPS = 5_604
EXPECTED_USER_PROMPT = "Listen to the audio and describe it."
EXPECTED_TOP_LEVEL_KEYS = {"messages", "audios", "metadata"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--stats", type=Path, required=True)
    preflight.add_argument("--output-report", type=Path, required=True)
    preflight.add_argument("--resume-checkpoint", type=Path)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--preflight-report", type=Path, required=True)
    audit.add_argument("--epoch1-checkpoint", type=Path, required=True)
    audit.add_argument("--epoch2-checkpoint", type=Path, required=True)
    audit.add_argument("--output-report", type=Path, required=True)
    audit.add_argument("--lora-rank", type=int, default=16)
    audit.add_argument("--lora-alpha", type=int, default=32)
    audit.add_argument("--lora-dropout", type=float, default=0.0)
    audit.add_argument("--learning-rate", type=float, default=1e-4)
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


def validate_stats(manifest: Path, stats_path: Path) -> tuple[dict[str, Any], str]:
    if not manifest.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Formal manifest/stats missing: manifest={manifest} stats={stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "audiocaps_v2",
        "split": "train",
        "route": "hrm_text_audio_whisper",
        "template_contract": "hrm_text_audio_direct_user_assistant",
        "transformation": "remove_generic_system_message_only",
        "record_count": EXPECTED_RECORDS,
        "unique_audio_path_count": EXPECTED_RECORDS,
        "unique_sample_id_count": EXPECTED_RECORDS,
        "duplicate_audio_path_count": 0,
        "duplicate_sample_id_count": 0,
        "audio_path_verification": "passed",
        "wav_header_verification": "passed",
        "source_manifest_unchanged": True,
        "all_non_message_fields_preserved": True,
        "user_messages_preserved": True,
        "assistant_messages_preserved": True,
    }
    mismatches = {
        key: {"expected": value, "actual": stats.get(key)}
        for key, value in expected.items()
        if stats.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Formal HRM AudioCaps stats mismatch: {mismatches}")
    manifest_sha = sha256_file(manifest)
    if manifest_sha != stats.get("output_manifest_sha256"):
        raise RuntimeError(
            f"Formal HRM AudioCaps hash mismatch: actual={manifest_sha} "
            f"stats={stats.get('output_manifest_sha256')}"
        )
    return stats, manifest_sha


def validate_manifest_schema(manifest: Path) -> dict[str, Any]:
    record_count = 0
    first_record = None
    last_record = None
    for line_number, line in enumerate(manifest.open("r", encoding="utf-8"), start=1):
        if not line.strip():
            raise RuntimeError(f"Formal manifest contains a blank line at {line_number}")
        record = json.loads(line)
        if not isinstance(record, dict) or set(record) != EXPECTED_TOP_LEVEL_KEYS:
            raise RuntimeError(f"Formal manifest schema mismatch at line {line_number}: {record}")
        messages = record.get("messages")
        roles = [item.get("role") if isinstance(item, dict) else None for item in messages or []]
        if roles != ["user", "assistant"]:
            raise RuntimeError(f"Formal manifest role mismatch at line {line_number}: {roles}")
        if messages[0].get("content") != EXPECTED_USER_PROMPT:
            raise RuntimeError(f"Formal manifest user prompt mismatch at line {line_number}")
        if not isinstance(messages[1].get("content"), str) or not messages[1]["content"].strip():
            raise RuntimeError(f"Formal manifest caption is empty at line {line_number}")
        audios = record.get("audios")
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
            raise RuntimeError(f"Formal manifest audio field mismatch at line {line_number}: {audios}")
        metadata = record.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("dataset") != "audiocaps_v2"
            or metadata.get("split") != "train"
            or not isinstance(metadata.get("sample_id"), str)
            or not metadata["sample_id"]
        ):
            raise RuntimeError(f"Formal manifest metadata mismatch at line {line_number}: {metadata}")
        record_count += 1
        if first_record is None:
            first_record = record
        last_record = record
    if record_count != EXPECTED_RECORDS:
        raise RuntimeError(f"Formal manifest record count mismatch: expected={EXPECTED_RECORDS} actual={record_count}")
    return {
        "record_count": record_count,
        "first_record": first_record,
        "last_record": last_record,
        "full_schema_scan": "passed",
    }


def validate_swift_interface() -> dict[str, Any]:
    from dataclasses import fields
    from swift.arguments.sft_args import SftArguments

    expected_versions = {
        "ms-swift": "4.4.2",
        "transformers": "5.9.0",
        "torch": "2.11.0+cu128",
        "peft": "0.18.1",
    }
    versions = {name: version(name) for name in expected_versions}
    mismatches = {
        name: {"expected": expected_versions[name], "actual": actual}
        for name, actual in versions.items()
        if actual != expected_versions[name]
    }
    if mismatches:
        raise RuntimeError(f"Formal HRM environment mismatch: {mismatches}")
    available = {field.name for field in fields(SftArguments)}
    required = {
        "num_train_epochs",
        "save_strategy",
        "save_total_limit",
        "resume_from_checkpoint",
        "lazy_tokenize",
        "dataset_shuffle",
        "train_dataloader_shuffle",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"Installed Swift lacks formal-training arguments: {missing}")
    return {"versions": versions, "required_argument_fields": sorted(required)}


def preflight(args: argparse.Namespace) -> None:
    manifest = args.manifest.expanduser().resolve()
    stats_path = args.stats.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    stats, manifest_sha = validate_stats(manifest, stats_path)
    schema = validate_manifest_schema(manifest)
    swift_interface = validate_swift_interface()
    micro_batches_per_epoch = math.ceil(EXPECTED_RECORDS / EXPECTED_MICRO_BATCH)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / EXPECTED_GRADIENT_ACCUMULATION)
    total_steps = steps_per_epoch * EXPECTED_EPOCHS
    if steps_per_epoch != EXPECTED_STEPS_PER_EPOCH or total_steps != EXPECTED_TOTAL_STEPS:
        raise RuntimeError(
            f"Formal step calculation mismatch: micro_batches={micro_batches_per_epoch} "
            f"steps_per_epoch={steps_per_epoch} total={total_steps}"
        )
    resume_report = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
        state_path = resume_checkpoint / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"Formal resume checkpoint lacks trainer_state.json: {resume_checkpoint}")
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(trainer_state.get("global_step", -1)) != EXPECTED_STEPS_PER_EPOCH:
            raise RuntimeError(
                "Formal resume is intentionally restricted to the epoch-1 checkpoint: "
                f"expected_step={EXPECTED_STEPS_PER_EPOCH} actual={trainer_state.get('global_step')}"
            )
        resume_report = {
            "checkpoint": str(resume_checkpoint),
            "global_step": int(trainer_state["global_step"]),
        }
    report = {
        "status": "OK",
        "manifest": str(manifest),
        "stats": str(stats_path),
        "manifest_sha256": manifest_sha,
        "source_manifest_sha256": stats.get("source_manifest_sha256"),
        "schema": schema,
        "swift_interface": swift_interface,
        "training_plan": {
            "epochs": EXPECTED_EPOCHS,
            "records": EXPECTED_RECORDS,
            "micro_batch_size": EXPECTED_MICRO_BATCH,
            "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
            "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
            "micro_batches_per_epoch": micro_batches_per_epoch,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "lazy_tokenize": True,
            "dataset_shuffle": True,
            "train_dataloader_shuffle": True,
            "save_strategy": "epoch",
            "expected_checkpoints": [EXPECTED_STEPS_PER_EPOCH, EXPECTED_TOTAL_STEPS],
        },
        "resume": resume_report,
    }
    atomic_write_json(output_report, report)
    print("========== HRM AUDIO FORMAL TRAINING PREFLIGHT ==========")
    print(f"[manifest] records={EXPECTED_RECORDS} sha256={manifest_sha} schema_scan=passed")
    print(
        f"[schedule] epochs=2 micro_batch=8 gradient_accumulation=4 "
        f"effective_batch=32 steps_per_epoch={steps_per_epoch} total_steps={total_steps}"
    )
    print("[data] lazy_tokenize=True dataset_shuffle=True train_dataloader_shuffle=True")
    print(f"[resume] {resume_report}")
    print(f"[result] status=OK output_report={output_report}")


def finite_loss_report(trainer_state: dict[str, Any]) -> dict[str, Any]:
    entries = []
    invalid = []
    for item in trainer_state.get("log_history", []):
        if not isinstance(item, dict) or "loss" not in item:
            continue
        try:
            step = int(item.get("step", -1))
            loss = float(item["loss"])
        except (TypeError, ValueError):
            invalid.append(item)
            continue
        if step <= 0 or not math.isfinite(loss):
            invalid.append(item)
        else:
            entries.append((step, loss))
    if invalid or len(entries) < 500:
        raise RuntimeError(f"Formal loss history is incomplete/non-finite: count={len(entries)} invalid={invalid[:20]}")
    entries.sort()
    window = min(20, len(entries) // 2)
    initial_mean = sum(loss for _, loss in entries[:window]) / window
    final_mean = sum(loss for _, loss in entries[-window:]) / window
    if final_mean >= initial_mean:
        raise RuntimeError(
            f"Formal loss did not improve: initial_mean={initial_mean} final_mean={final_mean}"
        )
    return {
        "logged_loss_count": len(entries),
        "first_logged_step": entries[0][0],
        "last_logged_step": entries[-1][0],
        "minimum_loss": min(loss for _, loss in entries),
        "maximum_loss": max(loss for _, loss in entries),
        "initial_window_size": window,
        "initial_window_mean": initial_mean,
        "final_window_mean": final_mean,
        "relative_reduction": (initial_mean - final_mean) / initial_mean,
    }


def audit_checkpoints(args: argparse.Namespace) -> None:
    manifest = args.manifest.expanduser().resolve()
    preflight_path = args.preflight_report.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    if not manifest.is_file() or not preflight_path.is_file():
        raise FileNotFoundError(f"Formal audit inputs missing: manifest={manifest} preflight={preflight_path}")
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_report.get("status") != "OK" or preflight_report.get("manifest_sha256") != sha256_file(manifest):
        raise RuntimeError(f"Formal preflight report/hash mismatch: {preflight_report}")
    checkpoint1 = checkpoint_audit.inspect_checkpoint(
        args.epoch1_checkpoint,
        expected_step=EXPECTED_STEPS_PER_EPOCH,
        expected_max_steps=EXPECTED_TOTAL_STEPS,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
    )
    checkpoint2 = checkpoint_audit.inspect_checkpoint(
        args.epoch2_checkpoint,
        expected_step=EXPECTED_TOTAL_STEPS,
        expected_max_steps=EXPECTED_TOTAL_STEPS,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
    )
    epoch1 = float(checkpoint1["trainer_state"].get("epoch", -1))
    epoch2 = float(checkpoint2["trainer_state"].get("epoch", -1))
    if not math.isclose(epoch1, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"Epoch-1 checkpoint epoch mismatch: {epoch1}")
    if not math.isclose(epoch2, 2.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"Epoch-2 checkpoint epoch mismatch: {epoch2}")
    adapter1, aligner1 = checkpoint_audit.checkpoint_trainables(args.epoch1_checkpoint)
    adapter2, aligner2 = checkpoint_audit.checkpoint_trainables(args.epoch2_checkpoint)
    h1 = {key: value for key, value in adapter1.items() if key.startswith("H_module.")}
    h2 = {key: value for key, value in adapter2.items() if key.startswith("H_module.")}
    l1 = {key: value for key, value in adapter1.items() if key.startswith("L_module.")}
    l2 = {key: value for key, value in adapter2.items() if key.startswith("L_module.")}
    updates = {
        "H_lora": checkpoint_audit.tensor_update_report(h1, h2, name="formal H-stack LoRA"),
        "L_lora": checkpoint_audit.tensor_update_report(l1, l2, name="formal L-stack LoRA"),
        "aligner": checkpoint_audit.tensor_update_report(aligner1, aligner2, name="formal aligner"),
    }
    losses = finite_loss_report(checkpoint2["trainer_state"])
    rng_changed = (
        checkpoint1["files"]["rng_state.pth"]["sha256"]
        != checkpoint2["files"]["rng_state.pth"]["sha256"]
    )
    if not rng_changed:
        raise RuntimeError("Formal RNG state did not advance between epoch checkpoints")
    report = {
        "status": "OK",
        "preflight": preflight_report,
        "epoch1_checkpoint": checkpoint1,
        "epoch2_checkpoint": checkpoint2,
        "epoch1_to_epoch2_updates": updates,
        "losses": losses,
        "rng_state_changed": rng_changed,
    }
    atomic_write_json(output_report, report)
    print("========== HRM AUDIO FORMAL TRAINING CHECKPOINT AUDIT ==========")
    print(f"[checkpoints] epoch1={args.epoch1_checkpoint} epoch2={args.epoch2_checkpoint}")
    print(f"[updates] {json.dumps(updates, ensure_ascii=False)}")
    print(f"[losses] {json.dumps(losses, ensure_ascii=False)}")
    print(f"[result] status=OK output_report={output_report}")


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "audit":
        audit_checkpoints(args)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
