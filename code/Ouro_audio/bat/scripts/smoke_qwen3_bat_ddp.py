#!/usr/bin/env python3
"""Strict 8-rank DDP smoke for the Qwen3 BAT Stage-III route.

This is deliberately independent of the BAT three-stage curriculum.  The
manifest must contain the Stage-III A/B/C/D/E route, while the smoke itself
uses a short configurable number of optimizer steps.  It audits
the real ms-swift/PEFT/Accelerate DDP path and supports a resumable second
phase through ``--resume-from-checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from bat.configs.training import BAT_TRAINING
from bat.cache_contract import assert_local_arrow_cache
from bat.qwen3_compile import (
    compile_qwen3_transformer_core,
    dynamo_counter_summary,
    prepare_compile_runtime,
)
from smoke_qwen3_bat_lora import (
    EXPECTED_AUDIO_TOKENS,
    EXPECTED_QWEN3_LAYERS,
    TARGET_MODULES,
    adapter_config_report,
    find_module,
    find_trainable_module,
    gradient_report,
    lora_report,
    normalized_name,
    optimizer_report,
    package_version,
    parameter_group_name,
    parameter_report,
    require_environment,
    shape_tuple,
)


MODEL_TYPE = "qwen3_bat_spatial_ast"
TEMPLATE_TYPE = "qwen3_bat_audio_prefix"
EXPECTED_WORLD_SIZE = 8
EXPECTED_LOCAL_BATCH = 8
EXPECTED_GLOBAL_BATCH = EXPECTED_WORLD_SIZE * EXPECTED_LOCAL_BATCH
EXPECTED_RECORDS = 128
EXPECTED_SMOKE_STEPS = 2
MAX_LENGTH_CEILING = 512
STAGE3_TYPES = {"A", "B", "C", "D", "E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=EXPECTED_RECORDS)
    parser.add_argument("--max-steps", type=int, default=EXPECTED_SMOKE_STEPS)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=EXPECTED_LOCAL_BATCH)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Manifest line {line_number} is not an object")
            rows.append(value)
    return rows


def validate_stage3_manifest(path: Path, expected_records: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected_records:
        raise RuntimeError(f"Expected {expected_records} records, got {len(rows)}")
    invalid: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        bat_type = str(row.get("bat_type", "")).upper()
        bat_stage = str(row.get("bat_stage", "")).upper()
        if bat_type not in STAGE3_TYPES or bat_stage not in {"III", "STAGE3", "STAGE3-MIXUP"}:
            invalid.append({"index": index, "bat_type": bat_type, "bat_stage": bat_stage})
    if invalid:
        raise RuntimeError(f"Manifest is not the Stage-III A/B/C/D/E route: {invalid[:8]}")
    return rows


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gather_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DDP is not initialized")
    gathered: list[dict[str, Any] | None] = [None] * world_size()
    dist.all_gather_object(gathered, report)
    return [item for item in gathered if item is not None]


def audit_audio_batch(model: torch.nn.Module, inputs: dict[str, Any]) -> dict[str, Any]:
    waveforms = inputs.get("audio_waveforms")
    records = inputs.get("bat_audio_records")
    if not torch.is_tensor(waveforms) or tuple(waveforms.shape[1:]) != (2, 320000):
        raise RuntimeError(f"Unexpected Qwen3 DDP waveform batch: {getattr(waveforms, 'shape', None)}")
    if not bool(torch.isfinite(waveforms.float()).all().item()):
        raise RuntimeError("Qwen3 DDP waveform batch contains NaN or Inf")
    if not isinstance(records, list) or len(records) != waveforms.shape[0]:
        raise RuntimeError("Qwen3 DDP audio provenance is missing or misaligned")
    causal = find_module(model, "Qwen3ForCausalLM")
    renderer = getattr(causal, "audio_renderer", None)
    if renderer is None:
        raise RuntimeError("Qwen3 model has no BATAudioRenderer")
    actual = waveforms.detach().float().cpu()
    details: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"Audio record {index} is not an object")
        audio_id = str(record.get("audio_id", ""))
        reverb_id = str(record.get("reverb_id", ""))
        if not audio_id or not reverb_id:
            raise RuntimeError(f"Audio record {index} lacks primary source references")
        audio_path = renderer._resolve_audio(renderer.audio_root, audio_id)
        reverb_path = renderer._resolve_reverb(renderer.reverb_root, reverb_id)
        second_paths = None
        second_audio = record.get("audio_id2")
        second_reverb = record.get("reverb_id2")
        if second_audio not in (None, "", "null") or second_reverb not in (None, "", "null"):
            if second_audio in (None, "", "null") or second_reverb in (None, "", "null"):
                raise RuntimeError(f"Partial second source in record {index}")
            second_paths = {
                "audio": str(renderer._resolve_audio(renderer.audio_root, str(second_audio))),
                "reverb": str(renderer._resolve_reverb(renderer.reverb_root, str(second_reverb))),
            }
        expected = renderer.render_record(record).float().cpu()
        error = float((actual[index] - expected).abs().max().item())
        if error > 1e-5:
            raise RuntimeError(f"AudioSet/RIR render mismatch at index={index}: {error}")
        rms = float(torch.sqrt(torch.mean(actual[index] ** 2)).item())
        if rms <= 0:
            raise RuntimeError(f"Rendered waveform is silent at index={index}")
        details.append({
            "index": index,
            "audio_id": audio_id,
            "audio_path": str(audio_path),
            "reverb_id": reverb_id,
            "reverb_path": str(reverb_path),
            "second_source": second_paths,
            "waveform_shape": list(actual[index].shape),
            "waveform_rms": rms,
            "independent_render_max_abs_error": error,
        })
    return {
        "status": "ok",
        "source_metadata_present": True,
        "audio_root": str(renderer.audio_root),
        "reverb_root": str(renderer.reverb_root),
        "batch_shape": list(actual.shape),
        "records": details,
    }


def audit_dynamic_padding(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    if input_ids.ndim != 2 or labels.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise RuntimeError(
            "Dynamic-padding tensors are not aligned: "
            f"input={tuple(input_ids.shape)} labels={tuple(labels.shape)} attention={tuple(attention_mask.shape)}"
        )
    batch_size, batch_width = input_ids.shape
    if batch_width < EXPECTED_AUDIO_TOKENS + 1 or batch_width > MAX_LENGTH_CEILING:
        raise RuntimeError(
            f"Dynamic batch width={batch_width} outside [{EXPECTED_AUDIO_TOKENS + 1}, {MAX_LENGTH_CEILING}]"
        )
    active_lengths = attention_mask.to(torch.long).sum(dim=1)
    if int(active_lengths.max().item()) != batch_width:
        raise RuntimeError(
            f"Collator width={batch_width} does not equal longest active length={int(active_lengths.max().item())}"
        )
    if int(active_lengths.min().item()) < EXPECTED_AUDIO_TOKENS + 1:
        raise RuntimeError(f"Invalid active length(s): {active_lengths.tolist()}")
    observed_within_batch = len(set(int(value) for value in active_lengths.tolist())) >= 2
    if not bool((attention_mask[:, :EXPECTED_AUDIO_TOKENS] == 1).all().item()):
        raise RuntimeError("Audio prefix positions are not all active in attention_mask")
    padding_mask = attention_mask == 0
    if padding_mask.any() and not bool((labels[padding_mask] == -100).all().item()):
        raise RuntimeError("Batch padding positions are not fully ignored by labels")
    for row_index, row in enumerate(attention_mask.tolist()):
        seen_padding = False
        for value in row:
            if value == 0:
                seen_padding = True
            elif seen_padding:
                raise RuntimeError(f"Non-right padding detected in row {row_index}: {row}")
    valid_label_counts = (labels != -100).sum(dim=1)
    if not bool((labels[:, :EXPECTED_AUDIO_TOKENS] == -100).all().item()):
        raise RuntimeError("Audio prefix labels are not fully ignored")
    if bool((valid_label_counts <= 0).any().item()):
        raise RuntimeError("At least one sample has no assistant-response target labels")
    target_spans: list[dict[str, int]] = []
    for row_index in range(batch_size):
        target_indices = torch.nonzero(labels[row_index] != -100, as_tuple=False).flatten()
        if target_indices.numel() == 0:
            raise RuntimeError(f"No assistant target labels in row {row_index}")
        first = int(target_indices[0].item())
        last = int(target_indices[-1].item())
        expected = torch.arange(first, last + 1, device=target_indices.device)
        if not torch.equal(target_indices, expected):
            raise RuntimeError(f"Assistant target labels are not contiguous in row {row_index}")
        if first <= EXPECTED_AUDIO_TOKENS:
            raise RuntimeError(
                "Assistant target starts at the audio boundary; no user/system prompt labels were masked "
                f"in row {row_index}: first_target_index={first}"
            )
        target_spans.append({"first_target_index": first, "last_target_index": last})
    return {
        "mode": "dynamic_batch_padding",
        "max_length_ceiling": MAX_LENGTH_CEILING,
        "batch_sequence_length": batch_width,
        "per_example_active_lengths": [int(value) for value in active_lengths.tolist()],
        "observed_within_batch": observed_within_batch,
        "per_example_assistant_target_counts": [int(value) for value in valid_label_counts.tolist()],
        "assistant_target_spans": target_spans,
        "padding_token_count": int(padding_mask.sum().item()),
        "padding_labels_all_ignore": True,
        "audio_prefix_labels_all_ignore": True,
        "assistant_targets_present": True,
    }


def checkpoint_path(output_dir: Path, expected_step: int) -> Path:
    candidates = sorted(path for path in output_dir.rglob(f"checkpoint-{expected_step}") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one checkpoint-{expected_step} below {output_dir}, found {candidates}")
    return candidates[0]


def checkpoint_report_for_step(path: Path, expected_step: int) -> dict[str, Any]:
    from safetensors import safe_open

    adapter = path / "adapter_model.safetensors"
    if not adapter.is_file():
        raise RuntimeError(f"Missing adapter checkpoint: {adapter}")
    with safe_open(str(adapter), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
    lora_keys = [key for key in keys if "lora_" in key]
    qformer_keys = [key for key in keys if "audio_qformer" in key]
    unexpected = [key for key in keys if key not in lora_keys and key not in qformer_keys]
    expected_lora_tensors = EXPECTED_QWEN3_LAYERS * len(TARGET_MODULES) * 2
    if len(lora_keys) != expected_lora_tensors or not qformer_keys or unexpected:
        raise RuntimeError(
            f"Unexpected Qwen3 DDP adapter keys: lora={len(lora_keys)} expected={expected_lora_tensors} "
            f"qformer={len(qformer_keys)} unexpected={unexpected[:8]}"
        )
    lora_modules = {key.split(".lora_", 1)[0] for key in lora_keys if ".lora_" in key}
    if len(lora_modules) != EXPECTED_QWEN3_LAYERS * len(TARGET_MODULES):
        raise RuntimeError(f"Unexpected saved Qwen3 LoRA module count: {len(lora_modules)}")
    if any(path.rsplit(".", 1)[-1] not in TARGET_MODULES for path in lora_modules):
        raise RuntimeError("Saved Qwen3 LoRA contains a module outside q_proj/v_proj")
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing trainer_state.json: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    actual_step = int(state.get("global_step", -1))
    if actual_step != expected_step:
        raise RuntimeError(f"Expected checkpoint global_step={expected_step}, got {actual_step}")
    state_files: dict[str, Any] = {}
    missing: list[str] = []
    for group, names in {
        "optimizer": ("optimizer.pt", "optimizer.bin", "optimizer.safetensors"),
        "scheduler": ("scheduler.pt", "scheduler.bin"),
    }.items():
        found = next((name for name in names if (path / name).is_file()), None)
        state_files[group] = found
        if found is None:
            missing.append(group)
    rng_files = sorted({
        item.name
        for pattern in ("rng_state_*.pth", "rng_state_*.pt")
        for item in path.glob(pattern)
        if item.is_file()
    })
    rng_indices = sorted({
        int(match.group(1))
        for name in rng_files
        if (match := re.fullmatch(r"rng_state_(\d+)\.(?:pth|pt)", name)) is not None
    })
    missing_rng = [index for index in range(EXPECTED_WORLD_SIZE) if index not in rng_indices]
    if missing_rng:
        missing.append(f"rng_state_{missing_rng[0]}..rng_state_{missing_rng[-1]}")
    state_files["rng"] = rng_files
    if missing:
        raise RuntimeError(f"Incomplete resumable checkpoint {path}: missing={missing} present_rng={rng_files}")
    return {
        "path": str(path),
        "tensor_count": len(keys),
        "lora_tensor_count": len(lora_keys),
        "expected_lora_tensor_count": expected_lora_tensors,
        "qformer_tensor_count": len(qformer_keys),
        "unexpected_tensor_count": len(unexpected),
        "global_step": actual_step,
        "state_files": state_files,
        "rng_rank_indices": rng_indices,
        "adapter_config": adapter_config_report(path / "adapter_config.json"),
    }


def read_checkpoint_step(path: Path) -> int:
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing trainer_state.json: {state_path}")
    return int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", -1))


def make_argv(args: argparse.Namespace, target_steps: int, save_steps: int, warmup_steps: int, resume: Path | None) -> list[str]:
    argv = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", str(MAX_LENGTH_CEILING),
        "--remove_unused_columns", "false", "--output_dir", str(args.output_dir),
        "--tuner_type", "lora", "--tuner_backend", "peft", "--target_modules", *TARGET_MODULES,
        "--modules_to_save", "audio_qformer", "--freeze_llm", "true", "--freeze_vit", "true",
        "--freeze_aligner", "false", "--lora_rank", str(BAT_TRAINING.lora_rank),
        "--lora_alpha", str(BAT_TRAINING.lora_alpha), "--lora_dropout", str(BAT_TRAINING.lora_dropout),
        "--learning_rate", str(BAT_TRAINING.learning_rate), "--lr_scheduler_type", "cosine",
        "--warmup_steps", str(warmup_steps), "--max_steps", str(target_steps), "--num_train_epochs", "2",
        "--per_device_train_batch_size", str(args.per_device_batch_size), "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing", "false", "--logging_steps", "1", "--save_strategy", "steps",
        "--save_steps", str(save_steps), "--save_total_limit", "2", "--save_only_model", "false",
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--dataloader_pin_memory", "true" if args.dataloader_num_workers > 0 else "false",
        "--dataset_num_proc", "1",
        "--lazy_tokenize", "true", "--load_from_cache_file", "false", "--loss_scale", "default",
        "--is_binary_loss_scale", "true",
        "--seed", "42", "--data_seed", "42", "--optim", "adamw_torch", "--adam_beta1", str(BAT_TRAINING.beta1),
        "--adam_beta2", str(BAT_TRAINING.beta2), "--weight_decay", str(BAT_TRAINING.weight_decay),
        "--attn_impl", "sdpa", "--bf16", "true", "--report_to", "none",
        "--ddp_find_unused_parameters", "false",
    ]
    if resume is not None:
        argv.extend(["--resume_from_checkpoint", str(resume)])
    return argv


def main() -> None:
    os.environ["BAT_FIXED_SEQUENCE_LENGTH"] = "false"
    os.environ["BAT_MAX_SEQUENCE_LENGTH"] = str(MAX_LENGTH_CEILING)
    args = parse_args()
    os.environ["BAT_AUDIO_AUDIT"] = "1"
    BAT_TRAINING.validate()
    require_environment()
    current_rank = rank()
    current_world = world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(current_rank)))
    if current_world != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"Qwen3 DDP smoke requires WORLD_SIZE=8, got {current_world}")
    if args.per_device_batch_size <= 0:
        raise ValueError("per-device-batch-size must be positive")
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader-num-workers must be non-negative")
    if local_rank < 0 or local_rank >= current_world:
        raise RuntimeError(f"Invalid LOCAL_RANK={local_rank}")
    torch.cuda.set_device(local_rank)
    for path in (args.model_path, args.plugin_path, args.dataset):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    rows = validate_stage3_manifest(args.dataset.resolve(), int(args.expected_records))
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if str(args.output_report).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_report}")
    target_steps = int(args.max_steps)
    if target_steps <= 0:
        raise ValueError(f"--max-steps must be positive, got {target_steps}")
    resume_checkpoint = args.resume_from_checkpoint.expanduser().resolve() if args.resume_from_checkpoint else None
    initial_step = 0
    if resume_checkpoint is not None:
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(resume_checkpoint)
        initial_step = read_checkpoint_step(resume_checkpoint)
        if initial_step < 0 or initial_step >= target_steps:
            raise RuntimeError(f"Invalid resume step={initial_step} target={target_steps}")
    expected_steps = target_steps - initial_step
    if expected_steps <= 0:
        raise RuntimeError(f"No remaining optimizer steps: target={target_steps} initial={initial_step}")
    save_steps = int(args.save_steps) if args.save_steps is not None else target_steps
    warmup_steps = min(1, target_steps)
    from swift.pipelines.train.sft import SwiftSft

    argv = make_argv(args, target_steps, save_steps, warmup_steps, resume_checkpoint)

    class AuditedDistributedSwiftSft(SwiftSft):
        def train(self, trainer):
            cache_root = os.environ.get("BAT_LOCAL_ARROW_CACHE")
            modelscope_cache = os.environ.get("MODELSCOPE_CACHE")
            if not cache_root or not modelscope_cache:
                raise RuntimeError("BAT_LOCAL_ARROW_CACHE and MODELSCOPE_CACHE are required for the resume smoke")
            cache_audit = assert_local_arrow_cache(trainer.train_dataset, [cache_root, modelscope_cache])
            if cache_audit.get("status") != "ok":
                raise RuntimeError(f"Qwen3 resume smoke Arrow cache audit failed: {cache_audit}")
            model = trainer.model
            causal = find_module(model, "Qwen3ForCausalLM")
            qformer = find_trainable_module(model, "BATQFormer")
            encoder = find_module(model, "SpatialASTAudioEncoder")
            qformer_instances = [module for module in model.modules() if module.__class__.__name__ == "BATQFormer"]
            if len(qformer_instances) != 2:
                raise RuntimeError(f"Expected PEFT original+trainable Q-Former copies, found {len(qformer_instances)}")
            if sum(any(parameter.requires_grad for parameter in module.parameters()) for module in qformer_instances) != 1:
                raise RuntimeError("Expected exactly one trainable Q-Former copy")
            if bool(getattr(causal.config, "use_cache", True)):
                raise RuntimeError("Qwen3 KV cache must be disabled for training")
            if any(parameter.requires_grad for parameter in encoder.parameters()):
                raise RuntimeError("Spatial-AST is unexpectedly trainable")
            if not any(parameter.requires_grad for parameter in qformer.parameters()):
                raise RuntimeError("Q-Former is unexpectedly frozen")
            model.train()
            encoder.eval()
            compile_report: dict[str, Any] = {
                "requested": bool(args.torch_compile),
                "dynamic": bool(args.compile_dynamic),
                "mode": args.compile_mode,
                "target": None,
                "runtime": None,
                "step_counters": [],
                "reuse_verified": False,
                "reuse_observation": "not_requested" if not args.torch_compile else "pending",
            }
            if args.torch_compile:
                compile_report["runtime"] = prepare_compile_runtime()
                _, target_report = compile_qwen3_transformer_core(
                    causal,
                    mode=args.compile_mode,
                    dynamic=args.compile_dynamic,
                )
                compile_report.update(target_report)
            parameters = parameter_report(model)
            lora = lora_report(model)
            trace: dict[str, Any] = {
                "forward": 0, "backward": 0, "layer_forward": 0, "layer_backward": 0,
                "loss": None, "batch": None, "audio_batch": None, "gradient_audit": None,
                "past_key_values_present": None, "audio_forward_audit": None,
                "arrow_cache_audit": cache_audit,
            }
            handles: list[Any] = []
            if not args.torch_compile:
                layer0 = causal.model.layers[0]
                handles.append(layer0.register_forward_hook(lambda *_: trace.__setitem__("layer_forward", trace["layer_forward"] + 1)))
                handles.append(layer0.register_full_backward_hook(lambda *_: trace.__setitem__("layer_backward", trace["layer_backward"] + 1)))
            original_compute_loss = trainer.compute_loss
            original_backward = trainer.accelerator.backward

            def compute_loss(actual_model, inputs, return_outputs=False, num_items_in_batch=None):
                trace["forward"] += 1
                # Swift may consume/transform these auxiliary tensors while
                # computing its loss.  Preserve the real batch contract
                # before delegating to the trainer.
                labels_before = inputs["labels"].detach().clone()
                loss_scale_before = inputs.get("loss_scale")
                if torch.is_tensor(loss_scale_before):
                    loss_scale_before = loss_scale_before.detach().clone()
                result = original_compute_loss(actual_model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                loss, outputs = result
                if trace["loss"] is None:
                    logits = outputs.logits
                    labels = labels_before
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    waveform = inputs.get("audio_waveforms")
                    dynamic_padding = audit_dynamic_padding(input_ids, labels, attention_mask)
                    if input_ids.shape[0] != args.per_device_batch_size:
                        raise RuntimeError(f"Unexpected Qwen3 local batch size: {shape_tuple(input_ids)}")
                    if shape_tuple(waveform) != (args.per_device_batch_size, 2, 320000):
                        raise RuntimeError(f"Unexpected Qwen3 local waveform shape: {shape_tuple(waveform)}")
                    if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(labels.shape):
                        raise RuntimeError(f"Unexpected Qwen3 logits shape: {shape_tuple(logits)}")
                    if not bool(torch.isfinite(logits.float()).all().item()) or not bool(torch.isfinite(loss.float()).all().item()):
                        raise RuntimeError("Qwen3 logits/loss contain NaN or Inf")
                    trace["past_key_values_present"] = getattr(outputs, "past_key_values", None) is not None
                    if trace["past_key_values_present"]:
                        raise RuntimeError("Qwen3 KV cache is unexpectedly enabled")
                    if not bool((labels[:, :EXPECTED_AUDIO_TOKENS] == -100).all().item()):
                        raise RuntimeError("Qwen3 audio prefix labels are not fully masked")
                    logits_float = logits.float()
                    shifted_logits = logits_float[:, :-1].contiguous()
                    shifted_labels = labels[:, 1:].contiguous()
                    shifted_valid = shifted_labels != -100
                    shifted_count = int(shifted_valid.sum().item())
                    if shifted_count <= 0:
                        raise RuntimeError("No valid shifted targets in Qwen3 DDP batch")
                    shifted_token_losses = F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                        shifted_labels.reshape(-1), ignore_index=-100, reduction="none",
                    )
                    manual_sum = shifted_token_losses[shifted_valid.reshape(-1)].sum()
                    manual_value = float((manual_sum / shifted_count).detach().cpu())

                    # Reproduce ms-swift 4.4.2's loss path exactly.  In the
                    # distributed trainer the denominator is supplied by
                    # ``num_items_in_batch`` and may differ from the local
                    # shifted-token count.  Swift also implements the shift
                    # by rolling labels/loss_scale over the full sequence.
                    swift_labels = torch.roll(labels, shifts=-1, dims=-1).reshape(-1)
                    swift_label_matrix = swift_labels.reshape_as(labels)
                    # Explicitly prove that Swift's full-sequence roll is
                    # exactly the standard causal-LM next-token alignment:
                    # prediction at position t is trained against label t+1.
                    if not torch.equal(swift_label_matrix[:, :-1], shifted_labels):
                        raise RuntimeError(
                            "Swift label roll is not equivalent to next-token labels[:, 1:]"
                        )
                    if not bool((swift_label_matrix[:, -1] == -100).all().item()):
                        raise RuntimeError(
                            "Swift roll exposes a non-ignored wrapped final target; "
                            "next-token loss would be misaligned"
                        )
                    swift_token_losses = F.cross_entropy(
                        logits_float.reshape(-1, logits_float.shape[-1]),
                        swift_labels,
                        ignore_index=-100,
                        reduction="none",
                    )
                    swift_valid = swift_labels != -100
                    if loss_scale_before is not None:
                        if not torch.is_tensor(loss_scale_before):
                            raise RuntimeError(f"Unexpected loss_scale type: {type(loss_scale_before).__name__}")
                        swift_scale = torch.roll(loss_scale_before, shifts=-1, dims=-1).reshape(-1).to(swift_token_losses.dtype)
                        swift_token_losses = swift_token_losses * swift_scale
                        loss_scale_binary_equivalent = bool(
                            torch.equal(swift_scale, swift_valid.to(swift_scale.dtype))
                        )
                    else:
                        loss_scale_binary_equivalent = True
                    swift_sum = swift_token_losses.sum()
                    if not math.isclose(
                        float(swift_sum.detach().cpu()),
                        float(manual_sum.detach().cpu()),
                        rel_tol=2e-3,
                        abs_tol=2e-3,
                    ):
                        raise RuntimeError(
                            "Swift next-token loss sum differs from explicit "
                            "logits[:, :-1] vs labels[:, 1:] loss sum"
                        )
                    if num_items_in_batch is None:
                        denominator = shifted_count
                    elif torch.is_tensor(num_items_in_batch):
                        denominator = int(num_items_in_batch.detach().cpu().item())
                    else:
                        denominator = int(num_items_in_batch)
                    if denominator <= 0:
                        raise RuntimeError(f"Invalid Swift loss denominator: {denominator}")
                    # With token-averaged DDP, ``num_items_in_batch`` is the
                    # global valid-item count while this rank owns only its
                    # local loss sum.  Transformers rescales by world_size
                    # before DDP's gradient averaging so the resulting
                    # gradient equals global_loss_sum / global_item_count.
                    ddp_world_size_rescale = current_world if num_items_in_batch is not None else 1
                    swift_formula_value = float(
                        (swift_sum / denominator * ddp_world_size_rescale).detach().cpu()
                    )
                    trainer_value = float(loss.detach().float().cpu())
                    if not math.isclose(trainer_value, swift_formula_value, rel_tol=2e-3, abs_tol=2e-3):
                        raise RuntimeError(
                            "Qwen3 DDP Swift loss mismatch: "
                            f"trainer={trainer_value} reproduced={swift_formula_value} "
                            f"manual_shifted_local_mean={manual_value} denominator={denominator}"
                        )
                    if not loss_scale_binary_equivalent:
                        raise RuntimeError(
                            "Qwen3 DDP loss_scale is not equivalent to the labels -100 mask"
                        )
                    trace["audio_batch"] = audit_audio_batch(model, inputs)
                    audio_audit = getattr(causal, "_qwen3_bat_last_audio_forward_audit", None)
                    if not isinstance(audio_audit, dict) or not audio_audit.get("audio_prefix_replaced"):
                        raise RuntimeError("Qwen3 audio prefix replacement audit was not captured")
                    trace["audio_forward_audit"] = audio_audit
                    trace["batch"] = {
                        "input_ids_shape": list(input_ids.shape), "labels_shape": list(labels.shape),
                        "attention_mask_shape": list(attention_mask.shape), "audio_waveforms_shape": list(waveform.shape),
                        "audio_prefix_label_ignore_count": int((labels[:, :EXPECTED_AUDIO_TOKENS] == -100).sum().item()),
                        "valid_shifted_target_count": shifted_count,
                        "manual_shifted_loss_sum": float(manual_sum.detach().cpu()),
                        "manual_shifted_ce": manual_value,
                        "swift_loss_denominator": denominator,
                        "ddp_world_size_rescale": ddp_world_size_rescale,
                        "swift_reproduced_ce": swift_formula_value,
                        "loss_scale_binary_equivalent": loss_scale_binary_equivalent,
                        "loss_scope": "assistant_response_only_via_labels_-100",
                        "loss_scale_argument": "default",
                        "dynamic_padding": dynamic_padding,
                        "trainer_ce": trainer_value,
                        "next_token_alignment": {
                            "logits_slice": "logits[:, :-1]",
                            "label_slice": "labels[:, 1:]",
                            "swift_equivalent": "torch.roll(labels, -1) with final wrapped target ignored",
                            "verified": True,
                        },
                        "shift_verified": True,
                    }
                    trace["loss"] = {"value": trainer_value, "logits_shape": list(logits.shape), "labels_shape": list(labels.shape)}
                if args.torch_compile:
                    trace.setdefault("compile_step_counters", []).append(dynamo_counter_summary())
                return result if return_outputs else loss

            def backward(loss, **kwargs):
                trace["backward"] += 1
                result = original_backward(loss, **kwargs)
                if trace["gradient_audit"] is None:
                    trace["gradient_audit"] = gradient_report(model)
                return result

            trainer.compute_loss = compute_loss
            trainer.accelerator.backward = backward
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                result = super().train(trainer)
            finally:
                trainer.compute_loss = original_compute_loss
                trainer.accelerator.backward = original_backward
                for handle in handles:
                    handle.remove()
            if int(trainer.state.global_step) != target_steps:
                raise RuntimeError(f"Rank {current_rank} expected global_step={target_steps}, got {trainer.state.global_step}")
            if trace["forward"] != expected_steps or trace["backward"] != expected_steps:
                raise RuntimeError(f"Unexpected Qwen3 DDP forward/backward counts: {trace}")
            if not args.torch_compile and (trace["layer_forward"] != expected_steps or trace["layer_backward"] != expected_steps):
                raise RuntimeError(f"Unexpected Qwen3 DDP layer counts: {trace}")
            if args.torch_compile:
                counters = trace.get("compile_step_counters", [])
                if not counters or counters[-1]["unique_graphs"] <= 0:
                    raise RuntimeError(f"Qwen3 DDP compile produced no graph: {counters}")
                unique_graphs = [item["unique_graphs"] for item in counters]
                if len(unique_graphs) > 1 and any(value != unique_graphs[0] for value in unique_graphs[1:]):
                    raise RuntimeError(f"Qwen3 DDP compile graph was not reused: {unique_graphs}")
                compile_report["step_counters"] = counters
                compile_report["unique_graphs"] = unique_graphs[-1]
                if len(unique_graphs) >= 2:
                    compile_report["reuse_verified"] = True
                    compile_report["reuse_observation"] = "verified_across_steps"
                else:
                    # A one-step resume run can validate that the compiled
                    # core executes, but cannot observe reuse within the same
                    # process. The preceding fresh compile smoke already
                    # owns the two-step graph-reuse assertion.
                    compile_report["reuse_verified"] = False
                    compile_report["reuse_observation"] = "insufficient_steps_for_in_process_reuse"
            if trace["audio_batch"] is None or trace["gradient_audit"] is None:
                raise RuntimeError("Missing Qwen3 DDP audio or gradient audit")
            local_optimizer = optimizer_report(trainer, model)
            local_report = {
                "rank": current_rank, "local_rank": local_rank, "world_size": current_world,
                "parameters": parameters, "lora": lora, "optimizer": local_optimizer,
                "compile": compile_report,
                "forward_audit": trace, "global_step": int(trainer.state.global_step),
                "elapsed_seconds": time.perf_counter() - started,
                "memory": {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                },
            }
            reports = gather_reports(local_report)
            barrier()
            checkpoint: dict[str, Any] | None = None
            audit_error: str | None = None
            if current_rank == 0:
                try:
                    if sorted(int(item["rank"]) for item in reports) != list(range(EXPECTED_WORLD_SIZE)):
                        raise RuntimeError("Missing rank report")
                    all_active_lengths = [
                        int(value)
                        for report_item in reports
                        for value in report_item["forward_audit"]["batch"]["dynamic_padding"]["per_example_active_lengths"]
                    ]
                    if len(set(all_active_lengths)) < 2:
                        raise RuntimeError("Dynamic padding was not observed across the global batch")
                    signatures = [
                        (item["parameters"]["trainable_parameter_counts"], item["lora"]["module_count"],
                         item["lora"]["rank"], tuple(item["lora"]["target_modules"]),
                         item["optimizer"]["betas"], item["optimizer"]["weight_decay_groups"])
                        for item in reports
                    ]
                    if any(signature != signatures[0] for signature in signatures[1:]):
                        raise RuntimeError(f"Rank contracts differ: {signatures}")
                    if args.torch_compile:
                        compile_signatures = [
                            (
                                bool(item["compile"]["requested"]),
                                bool(item["compile"]["dynamic"]),
                                item["compile"].get("target"),
                                int(item["compile"].get("unique_graphs", 0)),
                                bool(item["compile"].get("reuse_verified")),
                                item["compile"].get("reuse_observation"),
                            )
                            for item in reports
                        ]
                        if any(signature != compile_signatures[0] for signature in compile_signatures[1:]):
                            raise RuntimeError(f"Compile contracts differ across ranks: {compile_signatures}")
                        if not compile_signatures[0][0] or compile_signatures[0][1] or compile_signatures[0][3] <= 0:
                            raise RuntimeError(f"Invalid static compile contract: {compile_signatures[0]}")
                        if (
                            compile_signatures[0][5] == "verified_across_steps"
                            and not compile_signatures[0][4]
                        ):
                            raise RuntimeError(f"Compile reuse contract is inconsistent: {compile_signatures[0]}")
                    local_batches = [item["forward_audit"]["batch"]["input_ids_shape"][0] for item in reports]
                    if local_batches != [args.per_device_batch_size] * EXPECTED_WORLD_SIZE:
                        raise RuntimeError(f"Local batch audit failed: {local_batches}")
                    global_count = sum(int(item["forward_audit"]["batch"]["valid_shifted_target_count"]) for item in reports)
                    global_sum = sum(float(item["forward_audit"]["batch"]["manual_shifted_loss_sum"]) for item in reports)
                    if global_count <= 0:
                        raise RuntimeError("Global valid shifted target count is zero")
                    checkpoint = checkpoint_report_for_step(checkpoint_path(args.output_dir, target_steps), target_steps)
                    global_ce = global_sum / global_count
                except Exception as exc:
                    audit_error = f"{type(exc).__name__}: {exc}"
                    global_count = 0
                    global_ce = 0.0
            else:
                global_count = 0
                global_ce = 0.0
            errors: list[str | None] = [None] * EXPECTED_WORLD_SIZE
            dist.all_gather_object(errors, audit_error)
            if any(error is not None for error in errors):
                raise RuntimeError(f"Qwen3 DDP post-training audit failed: {errors}")
            if current_rank == 0:
                report = {
                    "status": "ok",
                    "route": "stage3_ab_cde",
                    "curriculum": False,
                    "stage3_types": sorted(STAGE3_TYPES),
                    "distributed": {
                        "backend": dist.get_backend(), "world_size": EXPECTED_WORLD_SIZE,
                        "per_device_batch_size": args.per_device_batch_size,
                        "global_batch_size": EXPECTED_WORLD_SIZE * args.per_device_batch_size,
                        "gradient_accumulation_steps": 1, "dataset_records": len(rows),
                        "target_global_step": target_steps, "initial_global_step": initial_step,
                        "optimizer_steps": expected_steps,
                        "resumed_from_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
                        "global_valid_shifted_target_count": global_count, "global_manual_shifted_ce": global_ce,
                        "rank_peak_allocated_bytes": [int(item["memory"]["peak_allocated_bytes"]) for item in reports],
                        "rank_peak_reserved_bytes": [int(item["memory"]["peak_reserved_bytes"]) for item in reports],
                        "rank_reports": reports,
                    },
                    "checkpoint": checkpoint,
                    "packages": {name: package_version(name) for name in ("ms-swift", "transformers", "peft", "accelerate")},
                    "argv": argv,
                }
                write_json(args.output_report, report)
                print(
                    f"[ddp] backend={dist.get_backend()} world_size={current_world} "
                    f"per_device_batch={args.per_device_batch_size} "
                    f"global_batch={EXPECTED_WORLD_SIZE * args.per_device_batch_size}",
                    flush=True,
                )
                print(f"[route] stage3_ab_cde curriculum=false types={sorted(STAGE3_TYPES)}", flush=True)
                print(f"[checkpoint] {json.dumps(checkpoint, ensure_ascii=False)}", flush=True)
                print(f"[report] {args.output_report}", flush=True)
                print("[status] ok", flush=True)
            barrier()
            return result

    print("========== QWEN3 BAT STAGE-III 8-RANK DDP SMOKE ==========")
    print(f"[rank] rank={current_rank} local_rank={local_rank} world_size={current_world}")
    print(f"[packages] ms-swift={package_version('ms-swift')} transformers={package_version('transformers')} torch={package_version('torch')}")
    print(f"[route] stage3_ab_cde curriculum=false records={len(rows)}")
    print(f"[schedule] target_steps={target_steps} initial_step={initial_step} expected_optimizer_steps={expected_steps}")
    if current_rank == 0:
        print(f"[argv] {' '.join(argv)}")
    AuditedDistributedSwiftSft(argv).main()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
