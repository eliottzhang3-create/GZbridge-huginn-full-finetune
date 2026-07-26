#!/usr/bin/env python3
"""Audit the real ms-swift lora_llm trainability split for HRM audio."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import wave
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch


MODEL_TYPE = "hrm_text_audio_whisper"
TEMPLATE_TYPE = "hrm_text_audio"
EXPECTED_HRM_PARAMETERS = 1_182_795_264
EXPECTED_ALIGNER_PARAMETERS = 39_538_176
EXPECTED_ALIGNER_TENSORS = 20
RANK8_LORA_PARAMETERS = 8_257_536
EXPECTED_LORA_MODULES = 256
EXPECTED_LORA_TENSORS = EXPECTED_LORA_MODULES * 2
EXPECTED_PROJECTION_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.gate_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
TARGET_PATTERN = re.compile(
    r"(?:^|\.)(H_module|L_module)\.layers\.(\d+)\."
    r"(self_attn\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_test_wav(path: Path) -> None:
    sample_rate = 16_000
    frame_count = sample_rate // 4
    frames = bytearray()
    for index in range(frame_count):
        value = int(0.2 * 32767 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def create_test_dataset(path: Path, wav_path: Path) -> None:
    samples = [
        {
            "messages": [
                {"role": "user", "content": "What is 1 + 1?"},
                {"role": "assistant", "content": "2."},
            ],
            "audios": [str(wav_path)],
        },
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 3?"},
                {"role": "assistant", "content": "5."},
            ],
            "audios": [str(wav_path)],
        },
    ]
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")


def find_unique_module(model: torch.nn.Module, class_name: str) -> torch.nn.Module:
    matches = [module for module in model.modules() if module.__class__.__name__ == class_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {class_name}, found {len(matches)}")
    return matches[0]


def parameter_ids(module: torch.nn.Module | None) -> set[int]:
    if module is None:
        return set()
    return {id(parameter) for parameter in module.parameters()}


def canonical_aligner_name(name: str) -> str:
    markers = ("temporal_compressor.", "audio_projector.", "audio_boundary_embeddings.")
    marker = next((candidate for candidate in markers if candidate in name), None)
    if marker is None:
        raise RuntimeError(f"Unable to canonicalize aligner parameter name: {name}")
    suffix = name[name.index(marker) :]
    return suffix.replace("original_module.", "").replace("modules_to_save.default.", "")


def expected_lora_parameters(rank: int) -> int:
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    numerator = RANK8_LORA_PARAMETERS * rank
    if numerator % 8:
        raise RuntimeError(f"Unable to derive exact HRM audio LoRA parameter count for rank={rank}")
    return numerator // 8


def lora_module_report(
    model: torch.nn.Module,
    *,
    expected_rank: int = 8,
) -> tuple[dict[str, Any], set[int]]:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    canonical: list[str] = []
    invalid: list[str] = []
    lora_parameter_ids: set[int] = set()
    ranks: dict[int, int] = defaultdict(int)
    for name, module in modules:
        match = TARGET_PATTERN.search(name)
        if match is None:
            invalid.append(name)
            continue
        stack_module, layer, suffix = match.groups()
        canonical.append(f"{stack_module}.layers.{layer}.{suffix}")
        for adapter_module_name in ("lora_A", "lora_B"):
            adapter_modules = getattr(module, adapter_module_name)
            for adapter_module in adapter_modules.values():
                lora_parameter_ids.update(parameter_ids(adapter_module))
        for rank in getattr(module, "r", {}).values():
            ranks[int(rank)] += 1

    expected = {
        f"{stack}_module.layers.{layer}.{suffix}"
        for stack in ("H", "L")
        for layer in range(16)
        for suffix in EXPECTED_PROJECTION_SUFFIXES
    }
    canonical_set = set(canonical)
    missing = sorted(expected - canonical_set)
    unexpected = sorted(canonical_set - expected)
    duplicate_count = len(canonical) - len(canonical_set)
    if len(modules) != EXPECTED_LORA_MODULES or invalid or missing or unexpected or duplicate_count:
        raise RuntimeError(
            "HRM audio LoRA target coverage mismatch: "
            f"count={len(modules)} invalid={invalid[:20]} missing={missing[:20]} "
            f"unexpected={unexpected[:20]} duplicates={duplicate_count}"
        )
    if ranks != {expected_rank: EXPECTED_LORA_MODULES}:
        raise RuntimeError(f"Unexpected HRM audio LoRA ranks: {dict(ranks)}")

    report = {
        "count": len(modules),
        "H_count": sum(name.startswith("H_module.") for name in canonical),
        "L_count": sum(name.startswith("L_module.") for name in canonical),
        "rank_counts": dict(ranks),
        "canonical_preview": sorted(canonical)[:20],
    }
    if report["H_count"] != 128 or report["L_count"] != 128:
        raise RuntimeError(f"HRM audio LoRA H/L split mismatch: {report}")
    return report, lora_parameter_ids


def audit_parameters(
    model: torch.nn.Module,
    wrapper: torch.nn.Module,
    *,
    expected_lora_rank: int = 8,
) -> dict[str, Any]:
    expected_lora_count = expected_lora_parameters(expected_lora_rank)
    expected_total_trainable = EXPECTED_ALIGNER_PARAMETERS + expected_lora_count
    lora_report, lora_ids = lora_module_report(model, expected_rank=expected_lora_rank)
    audio_ids = parameter_ids(wrapper.audio_encoder)
    aligner_ids = set().union(
        parameter_ids(wrapper.temporal_compressor),
        parameter_ids(wrapper.audio_projector),
        parameter_ids(wrapper.audio_boundary_embeddings),
    )
    hrm_ids = parameter_ids(wrapper.model) | parameter_ids(wrapper.lm_head)

    groups: dict[str, dict[str, Any]] = {
        name: {"total": 0, "trainable": 0, "trainable_tensors": 0, "trainable_preview": []}
        for name in ("audio_encoder", "aligner", "hrm_base", "lora", "other")
    }
    aligner_copies: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "trainable": 0})
    seen_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        identity = id(parameter)
        if identity in seen_ids:
            continue
        seen_ids.add(identity)
        if identity in lora_ids:
            group = "lora"
        elif identity in audio_ids:
            group = "audio_encoder"
        elif identity in aligner_ids:
            group = "aligner"
        elif identity in hrm_ids:
            group = "hrm_base"
        else:
            group = "other"
        count = parameter.numel()
        groups[group]["total"] += count
        if parameter.requires_grad:
            groups[group]["trainable"] += count
            groups[group]["trainable_tensors"] += 1
            if len(groups[group]["trainable_preview"]) < 20:
                groups[group]["trainable_preview"].append(name)
        if group == "aligner":
            canonical_name = canonical_aligner_name(name)
            aligner_copies[canonical_name]["total"] += 1
            if parameter.requires_grad:
                aligner_copies[canonical_name]["trainable"] += 1

    failures: list[str] = []
    if groups["audio_encoder"]["total"] <= 0 or groups["audio_encoder"]["trainable"] != 0:
        failures.append(f"Whisper must be present and fully frozen: {groups['audio_encoder']}")
    if groups["aligner"]["trainable"] != EXPECTED_ALIGNER_PARAMETERS:
        failures.append(
            f"Aligner trainable count must be {EXPECTED_ALIGNER_PARAMETERS}: {groups['aligner']}"
        )
    allowed_aligner_totals = {EXPECTED_ALIGNER_PARAMETERS, 2 * EXPECTED_ALIGNER_PARAMETERS}
    if groups["aligner"]["total"] not in allowed_aligner_totals:
        failures.append(
            "Aligner must have either one direct copy or one original plus one modules_to_save copy: "
            f"{groups['aligner']}"
        )
    invalid_aligner_copies = {
        name: copies
        for name, copies in aligner_copies.items()
        if copies["trainable"] != 1 or copies["total"] not in (1, 2)
    }
    if invalid_aligner_copies:
        failures.append(
            "Every aligner tensor must have exactly one trainable effective copy: "
            f"{dict(list(invalid_aligner_copies.items())[:20])}"
        )
    if len(aligner_copies) != EXPECTED_ALIGNER_TENSORS:
        failures.append(
            f"Aligner canonical tensor count must be {EXPECTED_ALIGNER_TENSORS}, got {len(aligner_copies)}"
        )
    if groups["hrm_base"] != {
        "total": EXPECTED_HRM_PARAMETERS,
        "trainable": 0,
        "trainable_tensors": 0,
        "trainable_preview": [],
    }:
        failures.append(f"Original HRM base must be exact and frozen: {groups['hrm_base']}")
    if groups["lora"]["total"] != expected_lora_count:
        failures.append(f"LoRA parameter count must be {expected_lora_count}: {groups['lora']}")
    if groups["lora"]["trainable"] != expected_lora_count:
        failures.append(f"Every LoRA parameter must be trainable: {groups['lora']}")
    if groups["lora"]["trainable_tensors"] != EXPECTED_LORA_TENSORS:
        failures.append(f"LoRA trainable tensor count must be {EXPECTED_LORA_TENSORS}: {groups['lora']}")
    if groups["other"]["total"] != 0:
        failures.append(f"Unclassified parameters are forbidden: {groups['other']}")
    total_trainable = sum(group["trainable"] for group in groups.values())
    if total_trainable != expected_total_trainable:
        failures.append(f"Total trainable count must be {expected_total_trainable}, got {total_trainable}")
    if failures:
        raise RuntimeError("HRM audio Swift trainability mismatch: " + " | ".join(failures))
    return {
        "groups": groups,
        "total": sum(group["total"] for group in groups.values()),
        "trainable": total_trainable,
        "frozen": sum(group["total"] for group in groups.values()) - total_trainable,
        "lora": lora_report,
        "aligner_canonical_tensor_count": len(aligner_copies),
    }


def cuda_report() -> dict[str, Any]:
    index = torch.cuda.current_device()
    return {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "allocated_gib": torch.cuda.memory_allocated(index) / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved(index) / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 1024**3,
    }


def main() -> None:
    args = parse_args()
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
        raise RuntimeError(f"Unexpected HRM audio Swift environment: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM audio Swift trainability audit requires CUDA")

    wrapper_model_path = args.wrapper_model_path.resolve()
    plugin_path = args.plugin_path.resolve()
    run_dir = args.run_dir.resolve()
    output_report = args.output_report.resolve()
    if not wrapper_model_path.is_dir():
        raise FileNotFoundError(f"Wrapper model directory is missing: {wrapper_model_path}")
    if not plugin_path.is_file():
        raise FileNotFoundError(f"Swift plugin is missing: {plugin_path}")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    wav_path = run_dir / "synthetic_16k_mono.wav"
    dataset_path = run_dir / "trainability_fixture.jsonl"
    swift_output_dir = run_dir / "swift_output"
    create_test_wav(wav_path)
    create_test_dataset(dataset_path, wav_path)

    from swift.pipelines.train.sft import SwiftSft

    argv = [
        "--model", str(wrapper_model_path),
        "--model_type", MODEL_TYPE,
        "--template", TEMPLATE_TYPE,
        "--external_plugins", str(plugin_path),
        "--dataset", str(dataset_path),
        "--split_dataset_ratio", "0",
        "--max_length", "128",
        "--output_dir", str(swift_output_dir),
        "--tuner_type", "lora_llm",
        "--tuner_backend", "peft",
        "--target_modules", "all-linear",
        "--freeze_llm", "true",
        "--freeze_vit", "true",
        "--freeze_aligner", "false",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0",
        "--learning_rate", "1e-4",
        "--aligner_lr", "1e-4",
        "--max_steps", "1",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing", "false",
        "--save_strategy", "no",
        "--dataloader_num_workers", "0",
        "--dataloader_pin_memory", "false",
        "--dataset_num_proc", "1",
        "--lazy_tokenize", "false",
        "--attn_impl", "sdpa",
        "--bf16", "true",
        "--report_to", "none",
    ]

    class InspectSwiftSft(SwiftSft):
        def train(self, trainer):
            model = trainer.model
            wrapper = find_unique_module(model, "HrmTextAudioForConditionalGeneration")
            config = wrapper.config
            recurrence = {
                "H_cycles": int(config.H_cycles),
                "L_cycles": int(config.L_cycles),
                "L_bp_cycles": [int(value) for value in config.L_bp_cycles],
                "prefix_lm": bool(config.prefix_lm),
            }
            expected_recurrence = {
                "H_cycles": 2,
                "L_cycles": 3,
                "L_bp_cycles": [0, 3],
                "prefix_lm": True,
            }
            if recurrence != expected_recurrence:
                raise RuntimeError(f"Swift changed HRM recurrence semantics: {recurrence}")
            if not bool(config.freeze_audio_encoder) or not bool(config.freeze_text_backbone):
                raise RuntimeError(
                    "Wrapper freeze flags changed: "
                    f"audio={config.freeze_audio_encoder} text={config.freeze_text_backbone}"
                )
            if int(trainer.state.global_step) != 0:
                raise RuntimeError(f"Audit unexpectedly observed a training update: step={trainer.state.global_step}")

            model.train()
            if wrapper.audio_encoder.training:
                raise RuntimeError("Frozen Whisper encoder entered training mode")
            parameter_report = audit_parameters(model, wrapper)
            memory = cuda_report()
            report = {
                "status": "OK",
                "python": sys.version.split()[0],
                "packages": versions,
                "model_type": MODEL_TYPE,
                "template": TEMPLATE_TYPE,
                "tuner_type": "lora_llm",
                "trainer_type": f"{type(trainer).__module__}.{type(trainer).__name__}",
                "model_type_runtime": f"{type(model).__module__}.{type(model).__name__}",
                "wrapper_type": f"{type(wrapper).__module__}.{type(wrapper).__name__}",
                "global_step": int(trainer.state.global_step),
                "optimizer_created": trainer.optimizer is not None,
                "recurrence": recurrence,
                "parameters": parameter_report,
                "cuda": memory,
                "fixture": {"dataset": str(dataset_path), "wav": str(wav_path)},
                "swift_argv": argv,
            }
            atomic_write_json(output_report, report)
            print("========== HRM AUDIO SWIFT TRAINABILITY AUDIT ==========", flush=True)
            print(f"[trainer] type={report['trainer_type']} global_step={report['global_step']}", flush=True)
            print(f"[runtime] model={report['model_type_runtime']} wrapper={report['wrapper_type']}", flush=True)
            print(f"[recurrence] {recurrence}", flush=True)
            for group, values in parameter_report["groups"].items():
                print(
                    f"[parameters] {group} total={values['total']} trainable={values['trainable']} "
                    f"trainable_tensors={values['trainable_tensors']}",
                    flush=True,
                )
            print(f"[lora] {parameter_report['lora']}", flush=True)
            print(
                f"[parameters] total={parameter_report['total']} "
                f"trainable={parameter_report['trainable']} frozen={parameter_report['frozen']}",
                flush=True,
            )
            print(f"[memory] {memory}", flush=True)
            print(f"[result] status=OK output_report={output_report}", flush=True)
            return {"status": "inspected", "output_report": str(output_report)}

    print("========== INSPECT HRM AUDIO SWIFT LORA_LLM TRAINABILITY ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[packages] {versions}", flush=True)
    print(f"[wrapper-model-path] {wrapper_model_path}", flush=True)
    print(f"[plugin-path] {plugin_path}", flush=True)
    print(f"[dataset] {dataset_path}", flush=True)
    print(f"[output-report] {output_report}", flush=True)
    print(
        "[expected] Whisper=0 HRM-base=0 aligner=39538176 LoRA=8257536 total-trainable=47795712",
        flush=True,
    )
    print("[argv] " + " ".join(argv), flush=True)
    InspectSwiftSft(argv).main()


if __name__ == "__main__":
    main()
