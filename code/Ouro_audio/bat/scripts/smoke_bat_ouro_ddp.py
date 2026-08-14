#!/usr/bin/env python3
"""Strict 8-rank DDP smoke for the BAT Spatial-AST -> Q-Former -> Ouro path.

This intentionally runs only two optimizer steps over a private 16-record
Stage-I manifest.  With eight ranks and per-device batch size two, every rank
must receive two examples and the effective global batch is 16.  The script
audits the real ms-swift/Accelerate training path rather than simulating an
all-reduce.
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
from smoke_bat_ouro_lora import (
    TARGET_MODULES,
    find_module,
    lora_report,
    optimizer_report,
    package_version,
    parameter_report,
    require_environment,
)


MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
EXPECTED_WORLD_SIZE = 8
EXPECTED_LOCAL_BATCH = BAT_TRAINING.per_device_batch_size
EXPECTED_DATASET_RECORDS = 16
EXPECTED_RECURRENT_STEPS = 4
EXPECTED_OPTIMIZER_STEPS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=EXPECTED_DATASET_RECORDS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
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


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def audit_audio_batch(model: torch.nn.Module, inputs: dict[str, Any]) -> dict[str, Any]:
    """Prove that the lazy batch contains the expected AudioSet/RIR renderings."""
    waveforms = inputs.get("audio_waveforms")
    records = inputs.get("bat_audio_records")
    if not torch.is_tensor(waveforms):
        raise RuntimeError("Lazy BAT batch has no tensor audio_waveforms")
    if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (2, 320000):
        raise RuntimeError(f"Unexpected lazy waveform batch shape: {tuple(waveforms.shape)}")
    if not bool(torch.isfinite(waveforms).all().item()):
        raise RuntimeError("Lazy BAT waveform batch contains NaN or Inf")
    if not isinstance(records, list) or len(records) != waveforms.shape[0]:
        raise RuntimeError(
            "Lazy BAT source metadata is missing or misaligned: "
            f"records={type(records).__name__}/{len(records) if isinstance(records, list) else None} "
            f"batch={waveforms.shape[0]}"
        )

    causal = find_module(model, "OuroForCausalLM")
    renderer = getattr(causal, "audio_renderer", None)
    if renderer is None:
        raise RuntimeError("Ouro BAT model has no attached BATAudioRenderer")
    # The renderer is intentionally CPU-side and owns the read-only input
    # roots. Re-rendering only this first smoke batch gives an independent
    # equality check without adding work to the real training loop.
    actual = waveforms.detach().float().cpu()
    details: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"Lazy BAT record {index} is not a dictionary: {type(record).__name__}")
        audio_id = str(record.get("audio_id", ""))
        reverb_id = str(record.get("reverb_id", ""))
        if not audio_id or not reverb_id:
            raise RuntimeError(f"Lazy BAT record {index} lacks audio_id/reverb_id: {record}")
        audio_path = renderer._resolve_audio(renderer.audio_root, audio_id)
        reverb_path = renderer._resolve_reverb(renderer.reverb_root, reverb_id)
        second_audio_id = record.get("audio_id2")
        second_reverb_id = record.get("reverb_id2")
        second_paths = None
        if second_audio_id not in (None, "", "null") or second_reverb_id not in (None, "", "null"):
            if second_audio_id in (None, "", "null") or second_reverb_id in (None, "", "null"):
                raise RuntimeError(f"Lazy BAT record {index} has a partial second source: {record}")
            second_paths = {
                "audio": str(renderer._resolve_audio(renderer.audio_root, str(second_audio_id))),
                "reverb": str(renderer._resolve_reverb(renderer.reverb_root, str(second_reverb_id))),
            }
        expected = renderer.render_record(record).float().cpu()
        difference = (actual[index] - expected).abs()
        max_abs_error = float(difference.max().item())
        if max_abs_error > 1e-5:
            raise RuntimeError(
                f"Lazy BAT waveform does not match an independent AudioSet/RIR render: "
                f"index={index} max_abs_error={max_abs_error}"
            )
        waveform_rms = float(torch.sqrt(torch.mean(actual[index] ** 2)).item())
        if waveform_rms <= 0.0:
            raise RuntimeError(f"Lazy BAT rendered waveform is silent: index={index}")
        details.append({
            "index": index,
            "audio_id": audio_id,
            "audio_path": str(audio_path),
            "reverb_id": reverb_id,
            "reverb_path": str(reverb_path),
            "second_source": second_paths,
            "waveform_shape": list(actual[index].shape),
            "waveform_rms": waveform_rms,
            "independent_render_max_abs_error": max_abs_error,
        })
    return {
        "status": "ok",
        "source_metadata_present": True,
        "audio_root": str(renderer.audio_root),
        "reverb_root": str(renderer.reverb_root),
        "batch_shape": list(actual.shape),
        "records": details,
    }


def gradient_audit(model: torch.nn.Module) -> dict[str, Any]:
    """Audit the first real backward: only Q-Former and LoRA may receive grads."""
    groups = {"qformer": 0, "lora": 0, "spatial_ast": 0, "gate": 0, "ouro_native": 0, "other": 0}
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if "lora_A" in name or "lora_B" in name:
            group = "lora"
        elif "audio_qformer" in name:
            group = "qformer"
        elif "spatial_ast_encoder" in name:
            group = "spatial_ast"
        elif "early_exit_gate" in name:
            group = "gate"
        elif ".model." in name or name.endswith("lm_head.weight"):
            group = "ouro_native"
        else:
            group = "other"
        groups[group] += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite.append(name)
    if groups["qformer"] <= 0 or groups["lora"] <= 0:
        raise RuntimeError(f"Expected finite Q-Former and LoRA gradients, got {groups}")
    if any(groups[key] for key in ("spatial_ast", "gate", "ouro_native", "other")):
        raise RuntimeError(f"Frozen BAT components unexpectedly received gradients: {groups}")
    if nonfinite:
        raise RuntimeError(f"Non-finite trainable gradients: {nonfinite[:10]}")
    return {"finite_gradient_parameter_counts": groups, "nonfinite_names": nonfinite}


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gather_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DDP smoke did not initialize torch.distributed")
    gathered: list[dict[str, Any] | None] = [None] * world_size()
    dist.all_gather_object(gathered, report)
    return [item for item in gathered if item is not None]


def checkpoint_path(output_dir: Path, expected_step: int) -> Path:
    candidates = sorted(path for path in output_dir.rglob(f"checkpoint-{expected_step}") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one checkpoint-{expected_step} below {output_dir}, found {candidates}")
    return candidates[0]


def checkpoint_report_for_step(
    path: Path,
    expected_step: int,
    expected_world_size: int = EXPECTED_WORLD_SIZE,
) -> dict[str, Any]:
    from safetensors import safe_open

    adapter = path / "adapter_model.safetensors"
    if not adapter.is_file():
        raise RuntimeError(f"Missing adapter checkpoint: {adapter}")
    with safe_open(str(adapter), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
    lora_keys = [key for key in keys if "lora_" in key]
    qformer_keys = [key for key in keys if "audio_qformer" in key]
    unexpected = [key for key in keys if key not in lora_keys and key not in qformer_keys]
    if not lora_keys or not qformer_keys or unexpected:
        raise RuntimeError(
            f"Unexpected BAT checkpoint keys: lora={len(lora_keys)} "
            f"qformer={len(qformer_keys)} unexpected={unexpected[:10]}"
        )

    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing trainer_state.json: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    actual_step = int(state.get("global_step", -1))
    if actual_step != expected_step:
        raise RuntimeError(f"Expected checkpoint global_step={expected_step}, got {actual_step}")

    alternatives = {
        "optimizer": ("optimizer.pt", "optimizer.bin", "optimizer.safetensors"),
        "scheduler": ("scheduler.pt", "scheduler.bin"),
    }
    state_files: dict[str, str | list[str] | None] = {}
    missing_state: list[str] = []
    for group, names in alternatives.items():
        found = next((name for name in names if (path / name).is_file()), None)
        state_files[group] = found
        if found is None:
            missing_state.append(f"{group}={names}")

    # Transformers saves one RNG file per process for distributed training:
    # rng_state_0.pth, rng_state_1.pth, ... .  The single-process fallback is
    # rng_state.pth.  The old audit only checked the fallback name and thus
    # falsely rejected an otherwise valid DDP checkpoint.
    rng_rank_files: list[str] = []
    for pattern in ("rng_state_*.pth", "rng_state_*.pt"):
        rng_rank_files.extend(item.name for item in path.glob(pattern) if item.is_file())
    rng_rank_files = sorted(set(rng_rank_files))
    rng_indices = sorted(
        {
            int(match.group(1))
            for name in rng_rank_files
            if (match := re.fullmatch(r"rng_state_(\d+)\.(?:pth|pt)", name)) is not None
        }
    )
    legacy_rng = next(
        (name for name in ("rng_state.pth", "rng_state.pt") if (path / name).is_file()),
        None,
    )
    missing_rng_ranks: list[int] = []
    if expected_world_size > 1:
        missing_rng_ranks = [index for index in range(expected_world_size) if index not in rng_indices]
        if missing_rng_ranks:
            missing_state.append(
                f"rng_state_{missing_rng_ranks[0]}..rng_state_{missing_rng_ranks[-1]}"
            )
        state_files["rng"] = rng_rank_files or None
        rng_mode = "per_rank"
    else:
        if legacy_rng is None:
            missing_state.append("rng=('rng_state.pth', 'rng_state.pt')")
        state_files["rng"] = legacy_rng
        rng_mode = "single_process"

    adapter_config = path / "adapter_config.json"
    if not adapter_config.is_file():
        missing_state.append("adapter_config.json")
    if missing_state:
        raise RuntimeError(
            f"Incomplete resumable checkpoint {path}: missing {missing_state}; "
            f"present_rng_files={rng_rank_files or legacy_rng or []}"
        )

    return {
        "path": str(path),
        "tensor_count": len(keys),
        "lora_tensor_count": len(lora_keys),
        "qformer_tensor_count": len(qformer_keys),
        "unexpected_tensor_count": len(unexpected),
        "global_step": actual_step,
        "state_files": state_files,
        "rng_state_mode": rng_mode,
        "rng_rank_indices": rng_indices,
        "rng_missing_ranks": missing_rng_ranks,
        "adapter_config": str(adapter_config),
    }


def read_checkpoint_step(path: Path) -> int:
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Resume checkpoint is missing trainer_state.json: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state.get("global_step", -1))


def main() -> None:
    args = parse_args()
    # Enable provenance metadata and runtime prefix audits only for this
    # smoke. Formal training keeps the batch free of audit-only dictionaries.
    os.environ["BAT_AUDIO_AUDIT"] = "1"
    BAT_TRAINING.validate()
    require_environment()

    current_rank = rank()
    current_world = world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(current_rank)))
    if current_world != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"DDP smoke requires WORLD_SIZE=8, got {current_world}")
    if local_rank < 0 or local_rank >= current_world:
        raise RuntimeError(f"Invalid LOCAL_RANK={local_rank} for WORLD_SIZE={current_world}")
    torch.cuda.set_device(local_rank)

    for path in (args.model_path, args.plugin_path, args.dataset):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    expected_records = int(args.expected_records)
    if expected_records <= 0:
        raise ValueError(f"--expected-records must be positive, got {expected_records}")
    dataset_records = count_jsonl(args.dataset)
    if dataset_records != expected_records:
        raise RuntimeError(
            f"DDP smoke requires exactly {expected_records} JSONL records; "
            f"got {dataset_records} from {args.dataset}"
        )
    if args.output_dir.exists() and current_rank == 0:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if str(args.output_report).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_report}")

    from swift.pipelines.train.sft import SwiftSft

    schedule = BAT_TRAINING.schedule(
        dataset_size=expected_records,
        world_size=EXPECTED_WORLD_SIZE,
        gradient_accumulation_steps=1,
        stage_name="I",
    )
    target_steps = int(args.max_steps) if args.max_steps is not None else int(schedule["total_steps"])
    if target_steps <= 0:
        raise ValueError(f"--max-steps must be positive, got {target_steps}")
    initial_step = 0
    resume_checkpoint = None
    if args.resume_from_checkpoint is not None:
        resume_checkpoint = args.resume_from_checkpoint.expanduser().resolve()
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(resume_checkpoint)
        initial_step = read_checkpoint_step(resume_checkpoint)
        if initial_step < 0 or initial_step >= target_steps:
            raise RuntimeError(
                f"Invalid resume step={initial_step} for target_steps={target_steps}: {resume_checkpoint}"
            )
    expected_optimizer_steps = target_steps - initial_step
    if int(schedule["effective_batch_size"]) != 16 or expected_optimizer_steps <= 0:
        raise RuntimeError(f"Unexpected DDP smoke schedule: {schedule}")
    warmup_steps = min(int(schedule["warmup_steps"]), target_steps)
    save_steps = int(args.save_steps) if args.save_steps is not None else target_steps
    if save_steps <= 0:
        raise ValueError(f"--save-steps must be positive, got {save_steps}")

    argv = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", "512",
        "--remove_unused_columns", "false", "--output_dir", str(args.output_dir),
        "--tuner_type", "lora", "--tuner_backend", "peft", "--target_modules", *TARGET_MODULES,
        "--modules_to_save", "audio_qformer", "--freeze_llm", "true", "--freeze_vit", "true",
        "--freeze_aligner", "false", "--lora_rank", str(BAT_TRAINING.lora_rank),
        "--lora_alpha", str(BAT_TRAINING.lora_alpha), "--lora_dropout", str(BAT_TRAINING.lora_dropout),
        "--learning_rate", str(BAT_TRAINING.learning_rate), "--lr_scheduler_type", "cosine",
        "--warmup_steps", str(warmup_steps), "--max_steps", str(target_steps),
        "--num_train_epochs", str(schedule["epochs"]), "--per_device_train_batch_size", str(EXPECTED_LOCAL_BATCH),
        "--gradient_accumulation_steps", "1", "--gradient_checkpointing", "false", "--logging_steps", "1",
        "--save_strategy", "steps", "--save_steps", str(save_steps), "--save_total_limit", "2", "--save_only_model", "false",
        "--dataloader_num_workers", "4", "--dataloader_pin_memory", "true", "--dataset_num_proc", "1",
        # Keep smoke/preflight on the same lazy multimodal path as formal
        # training; eager mode renders the whole dataset inside Dataset.map.
        "--lazy_tokenize", "true", "--load_from_cache_file", "false", "--loss_scale", "all", "--seed", "42",
        "--data_seed", "42", "--optim", "adamw_torch", "--adam_beta1", str(BAT_TRAINING.beta1),
        "--adam_beta2", str(BAT_TRAINING.beta2), "--weight_decay", str(BAT_TRAINING.weight_decay),
        "--attn_impl", "sdpa", "--bf16", "true", "--ddp_find_unused_parameters", "false",
        "--average_tokens_across_devices", "false",
        "--report_to", "none",
    ]
    if resume_checkpoint is not None:
        argv.extend(["--resume_from_checkpoint", str(resume_checkpoint)])

    class AuditedDistributedSwiftSft(SwiftSft):
        def train(self, trainer):
            if hasattr(trainer.accelerator, "unwrap_model"):
                model = trainer.accelerator.unwrap_model(trainer.model)
            else:
                model = trainer.model
            causal = find_module(model, "OuroForCausalLM")
            ouro = find_module(model, "OuroModel")
            if int(getattr(causal.config, "total_ut_steps", -1)) != EXPECTED_RECURRENT_STEPS:
                raise RuntimeError("Ouro total_ut_steps is not 4")
            if float(getattr(causal, "early_exit_threshold", -1)) != 1.0:
                raise RuntimeError("Ouro early_exit_threshold is not 1.0")
            causal.config.use_cache = False
            ouro.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.train()

            parameters = parameter_report(model)
            lora = lora_report(model)
            trace: dict[str, Any] = {
                "rank": current_rank, "local_rank": local_rank, "world_size": current_world,
                "forward": 0, "backward": 0, "layer": 0, "gate": 0,
                "layer_backward": 0, "gate_backward": 0, "loss": None, "batch": None,
                "audio_encoder_forward": 0, "audio_encoder_input_shape": None,
                "audio_encoder_output_shape": None, "qformer_forward": 0,
                "qformer_input_shape": None, "qformer_output_shape": None,
                "audio_batch": None, "prefix_audit": None, "gradient_audit": None,
            }
            handles: list[Any] = []
            first_layer = ouro.layers[0]
            audio_encoder = find_module(model, "SpatialASTAudioEncoder")
            qformer = find_module(model, "BATQFormer")

            def audio_encoder_hook(_module, hook_inputs, hook_output):
                trace["audio_encoder_forward"] += 1
                trace["audio_encoder_input_shape"] = list(hook_inputs[0].shape)
                trace["audio_encoder_output_shape"] = list(hook_output.shape)

            def qformer_hook(_module, hook_inputs, hook_output):
                trace["qformer_forward"] += 1
                trace["qformer_input_shape"] = list(hook_inputs[0].shape)
                trace["qformer_output_shape"] = list(hook_output.shape)

            handles.append(first_layer.register_forward_hook(lambda *_: trace.__setitem__("layer", trace["layer"] + 1)))
            handles.append(ouro.early_exit_gate.register_forward_hook(lambda *_: trace.__setitem__("gate", trace["gate"] + 1)))
            handles.append(first_layer.register_full_backward_hook(lambda *_: trace.__setitem__("layer_backward", trace["layer_backward"] + 1)))
            handles.append(ouro.early_exit_gate.register_full_backward_hook(lambda *_: trace.__setitem__("gate_backward", trace["gate_backward"] + 1)))
            handles.append(audio_encoder.register_forward_hook(audio_encoder_hook))
            handles.append(qformer.register_forward_hook(qformer_hook))
            original_compute_loss = trainer.compute_loss
            original_backward = trainer.accelerator.backward

            def compute_loss(actual_model, inputs, return_outputs=False, num_items_in_batch=None):
                trace["forward"] += 1
                labels_before = inputs["labels"].detach().clone()
                audio_batch_before = inputs.get("audio_waveforms")
                audio_records_before = inputs.get("bat_audio_records")
                loss_scale_before = inputs.get("loss_scale")
                if torch.is_tensor(loss_scale_before):
                    loss_scale_before = loss_scale_before.detach().clone()
                result = original_compute_loss(actual_model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                loss, outputs = result
                if trace["loss"] is None:
                    logits = outputs.logits
                    labels = labels_before
                    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
                        raise RuntimeError(f"Cannot audit shifted CE: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
                    logits_float = logits.float()
                    shifted_logits = logits_float[:, :-1].contiguous()
                    shifted_labels = labels[:, 1:].contiguous()
                    manual_token_loss = F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                        shifted_labels.reshape(-1), ignore_index=-100, reduction="none",
                    )
                    manual_valid = shifted_labels.reshape(-1) != -100
                    manual_sum = manual_token_loss[manual_valid].sum()
                    manual_count = int(manual_valid.sum().item())
                    manual_value = float((manual_sum / manual_count).detach().cpu())

                    # Reproduce ms-swift v4.4.2's per-token loss path:
                    # roll labels left, optionally apply loss_scale, then
                    # divide the token-loss sum by num_items_in_batch.
                    swift_labels = torch.roll(labels, shifts=-1, dims=-1).reshape(-1)
                    swift_token_loss = F.cross_entropy(
                        logits_float.reshape(-1, logits_float.shape[-1]),
                        swift_labels,
                        ignore_index=-100,
                        reduction="none",
                    )
                    swift_valid = swift_labels != -100
                    if loss_scale_before is not None:
                        if not torch.is_tensor(loss_scale_before):
                            raise RuntimeError(f"Unexpected loss_scale type: {type(loss_scale_before).__name__}")
                        swift_scale = torch.roll(loss_scale_before, shifts=-1, dims=-1).reshape(-1).to(swift_token_loss.dtype)
                        swift_token_loss = swift_token_loss * swift_scale
                        expected_scale = swift_valid.to(swift_scale.dtype)
                        loss_scale_binary_equivalent = bool(torch.equal(swift_scale, expected_scale))
                    else:
                        swift_scale = None
                        loss_scale_binary_equivalent = True
                    swift_sum = swift_token_loss.sum()
                    if num_items_in_batch is None:
                        denominator = manual_count
                    elif torch.is_tensor(num_items_in_batch):
                        denominator = int(num_items_in_batch.detach().cpu().item())
                    else:
                        denominator = int(num_items_in_batch)
                    if denominator <= 0:
                        raise RuntimeError(f"Invalid Swift loss denominator: {denominator}")
                    swift_formula_value = float((swift_sum / denominator).detach().cpu())
                    trainer_value = float(loss.detach().float().cpu())
                    if not math.isclose(swift_formula_value, trainer_value, rel_tol=2e-3, abs_tol=2e-3):
                        raise RuntimeError(
                            "Swift loss formula mismatch: "
                            f"trainer={trainer_value} reproduced={swift_formula_value} "
                            f"manual_local_mean={manual_value} denominator={denominator}"
                        )
                    if not loss_scale_binary_equivalent:
                        raise RuntimeError(
                            "loss_scale is not equivalent to the labels -100 mask; "
                            "ordinary CE contract is not verified"
                        )
                    if int(inputs["input_ids"].shape[0]) != EXPECTED_LOCAL_BATCH:
                        raise RuntimeError(f"Rank {current_rank} local batch is not 2: {tuple(inputs['input_ids'].shape)}")
                    if not bool((labels[:, :BAT_TRAINING.audio_token_count] == -100).all().item()):
                        raise RuntimeError("Audio prefix labels are not fully masked")
                    trace["audio_batch"] = audit_audio_batch(model, {
                        "audio_waveforms": audio_batch_before,
                        "bat_audio_records": audio_records_before,
                    })
                    if trace["audio_encoder_forward"] <= 0 or trace["qformer_forward"] <= 0:
                        raise RuntimeError("Lazy audio encoder/Q-Former did not run in the real training forward")
                    if trace["audio_encoder_input_shape"] != [EXPECTED_LOCAL_BATCH, 2, 320000]:
                        raise RuntimeError(f"Unexpected Spatial-AST input shape: {trace['audio_encoder_input_shape']}")
                    if trace["audio_encoder_output_shape"] != [EXPECTED_LOCAL_BATCH, 515, 768]:
                        raise RuntimeError(f"Unexpected Spatial-AST output shape: {trace['audio_encoder_output_shape']}")
                    if trace["qformer_input_shape"] != [EXPECTED_LOCAL_BATCH, 515, 768]:
                        raise RuntimeError(f"Unexpected Q-Former input shape: {trace['qformer_input_shape']}")
                    if trace["qformer_output_shape"] != [EXPECTED_LOCAL_BATCH, 64, 2048]:
                        raise RuntimeError(f"Unexpected Q-Former output shape: {trace['qformer_output_shape']}")
                    prefix_audit = getattr(causal, "_ouro_bat_last_audio_forward_audit", None)
                    if not isinstance(prefix_audit, dict) or not prefix_audit.get("audio_prefix_replaced"):
                        raise RuntimeError("Ouro audio prefix replacement audit was not captured")
                    if prefix_audit.get("inputs_embeds_shape") != list(inputs["input_ids"].shape):
                        raise RuntimeError(f"Audio/text embedding width mismatch: {prefix_audit}")
                    trace["prefix_audit"] = prefix_audit
                    trace["batch"] = {
                        "input_ids_shape": list(inputs["input_ids"].shape),
                        "labels_shape": list(labels.shape),
                        "attention_mask_shape": list(inputs["attention_mask"].shape),
                        "audio_waveforms_shape": list(inputs["audio_waveforms"].shape),
                        "audio_prefix_label_ignore_count": int((labels[:, :BAT_TRAINING.audio_token_count] == -100).sum().item()),
                        "valid_shifted_target_count": manual_count,
                        "manual_shifted_loss_sum": float(manual_sum.detach().cpu()),
                        "manual_shifted_ce": manual_value,
                        "swift_loss_denominator": denominator,
                        "swift_reproduced_ce": swift_formula_value,
                        "loss_scale_binary_equivalent": loss_scale_binary_equivalent,
                        "trainer_ce": trainer_value,
                        "shift_verified": True,
                    }
                    trace["loss"] = {"value": trainer_value, "logits_shape": list(logits.shape), "labels_shape": list(labels.shape)}
                return result if return_outputs else loss

            def backward(loss, **kwargs):
                trace["backward"] += 1
                result = original_backward(loss, **kwargs)
                if trace["gradient_audit"] is None:
                    trace["gradient_audit"] = gradient_audit(model)
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

            local_optimizer = optimizer_report(trainer, model)
            if int(trainer.state.global_step) != target_steps:
                raise RuntimeError(f"Rank {current_rank} expected global_step={target_steps}, got {trainer.state.global_step}")
            if trace["forward"] != expected_optimizer_steps or trace["backward"] != expected_optimizer_steps:
                raise RuntimeError(f"Rank {current_rank} unexpected trainer counts: {trace}")
            if trace["layer"] != EXPECTED_RECURRENT_STEPS * expected_optimizer_steps or trace["gate"] != EXPECTED_RECURRENT_STEPS * expected_optimizer_steps:
                raise RuntimeError(f"Rank {current_rank} unexpected recurrent forward counts: {trace}")
            if trace["layer_backward"] != EXPECTED_RECURRENT_STEPS * expected_optimizer_steps:
                raise RuntimeError(f"Rank {current_rank} unexpected recurrent backward count: {trace}")
            if trace["audio_encoder_forward"] != expected_optimizer_steps:
                raise RuntimeError(f"Rank {current_rank} unexpected Spatial-AST calls: {trace}")
            if trace["qformer_forward"] != expected_optimizer_steps:
                raise RuntimeError(f"Rank {current_rank} unexpected Q-Former calls: {trace}")
            if (
                trace["audio_batch"] is None
                or trace["prefix_audit"] is None
                or trace["gradient_audit"] is None
            ):
                raise RuntimeError(f"Rank {current_rank} missing lazy audio provenance audit: {trace}")
            local_report = {
                "rank": current_rank, "local_rank": local_rank, "world_size": current_world,
                "parameters": parameters, "lora": lora, "optimizer": local_optimizer,
                "forward_audit": {**trace, "expected_recurrent_steps": EXPECTED_RECURRENT_STEPS, "use_cache": False},
                "elapsed_seconds": time.perf_counter() - started,
                "global_step": int(trainer.state.global_step),
                "memory": {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "current_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "current_reserved_bytes": int(torch.cuda.memory_reserved()),
                },
            }
            reports = gather_reports(local_report)
            if len(reports) != EXPECTED_WORLD_SIZE:
                raise RuntimeError(f"Expected {EXPECTED_WORLD_SIZE} rank reports, got {len(reports)}")
            checkpoint: dict[str, Any] | None = None
            rank0_audit_error: str | None = None
            global_token_count = 0
            global_ce = 0.0
            if current_rank == 0:
                try:
                    rank_ids = sorted(int(item["rank"]) for item in reports)
                    if rank_ids != list(range(EXPECTED_WORLD_SIZE)):
                        raise RuntimeError(f"Unexpected rank reports: {rank_ids}")
                    local_batches = [item["forward_audit"]["batch"]["input_ids_shape"][0] for item in reports]
                    if local_batches != [EXPECTED_LOCAL_BATCH] * EXPECTED_WORLD_SIZE:
                        raise RuntimeError(f"Local batch audit failed: {local_batches}")
                    global_loss_sum = sum(
                        float(item["forward_audit"]["batch"]["manual_shifted_loss_sum"])
                        for item in reports
                    )
                    global_token_count = sum(
                        int(item["forward_audit"]["batch"]["valid_shifted_target_count"])
                        for item in reports
                    )
                    if global_token_count <= 0:
                        raise RuntimeError("Global shifted-token count is zero")
                    global_ce = global_loss_sum / global_token_count
                    checkpoint = checkpoint_report_for_step(
                        checkpoint_path(args.output_dir, target_steps),
                        target_steps,
                        expected_world_size=EXPECTED_WORLD_SIZE,
                    )
                except Exception as exc:
                    rank0_audit_error = f"{type(exc).__name__}: {exc}"

            audit_errors: list[str | None] = [None] * EXPECTED_WORLD_SIZE
            dist.all_gather_object(audit_errors, rank0_audit_error)
            if any(error is not None for error in audit_errors):
                raise RuntimeError(f"Distributed post-training audit failed: {audit_errors}")

            if current_rank == 0:
                report = {
                    "status": "ok",
                    "distributed": {
                        "backend": dist.get_backend(), "world_size": EXPECTED_WORLD_SIZE,
                        "per_device_batch_size": EXPECTED_LOCAL_BATCH,
                        "gradient_accumulation_steps": 1, "global_batch_size": 16,
                        "dataset_records": expected_records, "optimizer_steps": expected_optimizer_steps,
                        "target_global_step": target_steps, "initial_global_step": initial_step,
                        "resumed_from_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
                        "global_valid_shifted_target_count": global_token_count,
                        "global_manual_shifted_ce": global_ce,
                        "rank_peak_allocated_bytes": [
                            int(item["memory"]["peak_allocated_bytes"]) for item in reports
                        ],
                        "rank_peak_reserved_bytes": [
                            int(item["memory"]["peak_reserved_bytes"]) for item in reports
                        ],
                        "rank_reports": reports,
                    },
                    "paper_contract": {
                        "sound_source": BAT_TRAINING.sound_source,
                        "optimizer": BAT_TRAINING.optimizer,
                        "betas": [BAT_TRAINING.beta1, BAT_TRAINING.beta2],
                        "weight_decay": BAT_TRAINING.weight_decay,
                        "learning_rate": BAT_TRAINING.learning_rate,
                        "scheduler": BAT_TRAINING.scheduler,
                        "stage": "I", "stage_epochs": 2,
                    },
                    "checkpoint": checkpoint,
                    "argv": argv,
                    "packages": {name: package_version(name) for name in ("ms-swift", "transformers", "peft", "accelerate")},
                    "result": {"trainer": f"{trainer.__class__.__module__}.{trainer.__class__.__name__}", "result_type": type(result).__name__},
                }
                write_json(args.output_report, report)
                print(f"[ddp] backend={dist.get_backend()} world_size={EXPECTED_WORLD_SIZE} global_batch=16", flush=True)
                print(
                    f"[memory] peak_allocated_max={max(report['distributed']['rank_peak_allocated_bytes'])} "
                    f"peak_reserved_max={max(report['distributed']['rank_peak_reserved_bytes'])}",
                    flush=True,
                )
                print(f"[checkpoint] {json.dumps(checkpoint, ensure_ascii=False)}", flush=True)
                print(f"[report] {args.output_report}", flush=True)
                print("[status] ok", flush=True)
            barrier()
            return result

    print("========== BAT OURO DDP 8-RANK SMOKE ==========")
    print(f"[rank] rank={current_rank} local_rank={local_rank} world_size={current_world}")
    print(f"[packages] ms-swift={package_version('ms-swift')} transformers={package_version('transformers')} torch={package_version('torch')}")
    print(f"[schedule] {json.dumps(dict(schedule), ensure_ascii=False)}")
    if current_rank == 0:
        print(f"[argv] {' '.join(argv)}")
    AuditedDistributedSwiftSft(argv).main()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
