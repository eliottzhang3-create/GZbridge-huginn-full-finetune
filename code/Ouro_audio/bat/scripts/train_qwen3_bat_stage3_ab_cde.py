#!/usr/bin/env python3
"""Train Qwen3-4B on the BAT Stage-III A/B/C/D/E route.

The manifest already contains two deterministic curriculum epochs.  Each
epoch is ordered as A+B followed by C+D+E, so runtime dataset and dataloader
shuffle remain disabled.  Spatial-AST is frozen FP32, the Q-Former is
randomly initialized and trainable, Qwen3 native weights are frozen, and LoRA
is applied only to Qwen3 q_proj/v_proj modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from bat.configs.training import BAT_TRAINING
from bat.curriculum import count_jsonl
from bat.qwen3_compile import compile_qwen3_transformer_core, prepare_compile_runtime


MODEL_TYPE = "qwen3_bat_spatial_ast"
TEMPLATE_TYPE = "qwen3_bat_audio_prefix"
PER_DEVICE_BATCH_SIZE = 2
WORLD_SIZE_REQUIRED = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 0.001
MAX_SEQUENCE_LENGTH = 176
EXPECTED_QWEN3_LAYERS = 36
EXPECTED_LORA_TARGETS = ("q_proj", "v_proj")
PERIODIC_SAVE_STEPS = 6_000
MAX_PERIODIC_CHECKPOINTS = 9
WARMUP_RATIO = 0.13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE_REQUIRED)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def validate_base_contract() -> None:
    if BAT_TRAINING.sound_source != "AudioSet-20K" or not BAT_TRAINING.audio_normalization:
        raise ValueError("BAT AudioSet/normalization contract is invalid")
    if BAT_TRAINING.augmentation or BAT_TRAINING.weighted_sampling:
        raise ValueError("BAT augmentation and weighted sampling must remain disabled")
    if BAT_TRAINING.optimizer != "AdamW" or (BAT_TRAINING.beta1, BAT_TRAINING.beta2) != (0.9, 0.95):
        raise ValueError("BAT optimizer contract must be AdamW with betas (0.9, 0.95)")
    if BAT_TRAINING.weight_decay != 0.05:
        raise ValueError("BAT weight decay must remain 0.05")
    if (BAT_TRAINING.lora_rank, BAT_TRAINING.lora_alpha, BAT_TRAINING.lora_dropout) != (8, 32, 0.05):
        raise ValueError("BAT LoRA contract must remain rank=8 alpha=32 dropout=0.05")
    if BAT_TRAINING.lora_target_modules != EXPECTED_LORA_TARGETS:
        raise ValueError("BAT LoRA target modules must remain q_proj and v_proj")
    if BAT_TRAINING.audio_token_count != 64:
        raise ValueError("BAT audio token contract must remain 64")


def load_and_validate_report(path: Path, global_batch_size: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "ok":
        raise ValueError(f"Stage-III route report is not ok: {path}")
    if report.get("route") != "stage3_ab_cde_2epoch":
        raise ValueError(f"Unexpected route in report: {report.get('route')!r}")
    # The manifest/report was composed with global batch 64.  The same ordered
    # manifest is intentionally trained here with global batch 16, so compute
    # the actual optimizer schedule from written records below.
    if int(report.get("global_batch_size", -1)) != 64:
        raise ValueError(f"Unexpected manifest composition batch: {report.get('global_batch_size')}")
    if int(report.get("per_device_batch_size", -1)) != 8 or int(report.get("world_size", -1)) != 8:
        raise ValueError("Unexpected source route report distributed metadata")
    if int(report.get("gradient_accumulation_steps", -1)) != GRADIENT_ACCUMULATION_STEPS:
        raise ValueError("Stage-III route requires gradient accumulation 1")
    if report.get("runtime_shuffle") is not False:
        raise ValueError("Stage-III route must disable runtime shuffle")
    blocks = report.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 4:
        raise ValueError("Stage-III route must contain four ordered blocks")
    expected = [(1, "A+B"), (1, "C+D+E"), (2, "A+B"), (2, "C+D+E")]
    observed = [(int(item.get("epoch", -1)), str(item.get("group"))) for item in blocks]
    if observed != expected:
        raise ValueError(f"Unexpected block order: {observed}")
    previous_record = 0
    for item in blocks:
        start_record = int(item.get("start_record", -1))
        end_record = int(item.get("end_record", -1))
        written = int(item.get("written_records", -1))
        if start_record != previous_record:
            raise ValueError(f"Non-contiguous Stage-III block: {item}")
        if written <= 0 or written % global_batch_size:
            raise ValueError(f"Invalid Stage-III block step range: {item}")
        if end_record != start_record + written:
            raise ValueError(f"Invalid Stage-III block record range: {item}")
        previous_record = end_record
    if int(report.get("total_records", -1)) != previous_record:
        raise ValueError("Stage-III report total records do not match block ranges")
    if previous_record % global_batch_size:
        raise ValueError(f"Manifest records are not divisible by actual global batch: {previous_record}/{global_batch_size}")

    actual_step = 0
    actual_boundaries: dict[str, int] = {}
    for item in blocks:
        actual_step += int(item["written_records"]) // global_batch_size
        if str(item["group"]) == "C+D+E":
            actual_boundaries[str(int(item["epoch"]))] = actual_step
    if set(actual_boundaries) != {"1", "2"}:
        raise ValueError(f"Unable to compute actual epoch boundaries: {actual_boundaries}")
    report["actual_training_schedule"] = {
        "global_batch_size": global_batch_size,
        "total_steps": actual_step,
        "warmup_steps": int(math.ceil(actual_step * WARMUP_RATIO)),
        "epoch_boundary_steps": actual_boundaries,
        "route_report_global_batch_size": int(report["global_batch_size"]),
        "route_report_learning_rate": report.get("learning_rate"),
    }
    return report


def checkpoint_state_report(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    )
    missing = [name for name in required if not (path / name).is_file()]
    rng_files = sorted(list(path.glob("rng_state_*.pth")) + list(path.glob("rng_state_*.pt")))
    if len(rng_files) < WORLD_SIZE_REQUIRED:
        missing.append(f"rng_state_*.pth>={WORLD_SIZE_REQUIRED}")

    keys: list[str] = []
    adapter = path / "adapter_model.safetensors"
    if adapter.is_file():
        with safe_open(str(adapter), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
    lora_keys = [key for key in keys if "lora_" in key]
    qformer_keys = [key for key in keys if "audio_qformer" in key]
    unexpected = [key for key in keys if key not in lora_keys and key not in qformer_keys]
    expected_lora_tensors = EXPECTED_QWEN3_LAYERS * len(EXPECTED_LORA_TARGETS) * 2
    if len(lora_keys) != expected_lora_tensors:
        missing.append(f"lora_tensors={expected_lora_tensors},got={len(lora_keys)}")
    if not qformer_keys:
        missing.append("audio_qformer tensors")
    if unexpected:
        missing.append(f"unexpected adapter tensors: {unexpected[:4]}")

    state = {}
    trainer_state = path / "trainer_state.json"
    if trainer_state.is_file():
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
    marker = {}
    marker_path = path / "stage3_epoch.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "missing": missing,
        "status": "ok" if not missing else "incomplete",
        "tensor_count": len(keys),
        "lora_tensor_count": len(lora_keys),
        "expected_lora_tensor_count": expected_lora_tensors,
        "qformer_tensor_count": len(qformer_keys),
        "unexpected_tensor_count": len(unexpected),
        "global_step": int(state.get("global_step", -1)),
        "stage_marker": marker,
        "rng_files": [item.name for item in rng_files],
    }


def retained_checkpoint_paths(output_dir: Path) -> list[Path]:
    """Return native Trainer checkpoint directories in global-step order."""
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if step > 0:
            checkpoints.append((step, path))
    return [path for _, path in sorted(checkpoints, key=lambda item: item[0])]


def main() -> None:
    args = parse_args()
    validate_base_contract()
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.world_size != WORLD_SIZE_REQUIRED or actual_world_size != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"Qwen3 Stage-III training requires WORLD_SIZE=8, got launcher={args.world_size} actual={actual_world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0 or local_rank >= actual_world_size:
        raise RuntimeError(f"Invalid LOCAL_RANK={local_rank}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 Stage-III DDP training requires CUDA")
    torch.cuda.set_device(local_rank)
    for path in (args.model_path, args.plugin_path, args.dataset, args.report):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    if str(args.output_dir).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_dir}")
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(args.resume_from_checkpoint)
    elif rank() == 0 and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory for a fresh run: {args.output_dir}")

    global_batch_size = PER_DEVICE_BATCH_SIZE * actual_world_size * GRADIENT_ACCUMULATION_STEPS
    report = load_and_validate_report(args.report, global_batch_size)
    dataset_records = count_jsonl(args.dataset)
    if dataset_records != int(report["total_records"]):
        raise RuntimeError(f"Manifest count mismatch: actual={dataset_records} report={report['total_records']}")

    from swift.pipelines.train.sft import SwiftSft
    from smoke_qwen3_bat_lora import find_module, find_trainable_module

    class Qwen3Stage3SwiftSft(SwiftSft):
        def train(self, trainer):
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
            if args.torch_compile:
                runtime = prepare_compile_runtime()
                _, target = compile_qwen3_transformer_core(
                    causal,
                    mode=args.compile_mode,
                    dynamic=args.compile_dynamic,
                )
                print(f"[compile] runtime={json.dumps(runtime, ensure_ascii=False)} target={json.dumps(target, ensure_ascii=False)}", flush=True)

            result = super().train(trainer)
            if rank() == 0:
                effective_output_dir = Path(trainer.args.output_dir).resolve()
                checkpoints = retained_checkpoint_paths(effective_output_dir)
                if not checkpoints:
                    raise RuntimeError(
                        "No periodic checkpoint was retained; expected native Trainer saves "
                        f"every {PERIODIC_SAVE_STEPS} steps"
                    )
                if len(checkpoints) > MAX_PERIODIC_CHECKPOINTS:
                    raise RuntimeError(
                        "Native checkpoint retention exceeded limit: "
                        f"found={len(checkpoints)} limit={MAX_PERIODIC_CHECKPOINTS}"
                    )
                print(
                    f"[checkpoint] retained={len(checkpoints)} "
                    f"save_steps={PERIODIC_SAVE_STEPS} save_total_limit={MAX_PERIODIC_CHECKPOINTS}",
                    flush=True,
                )
                for checkpoint in checkpoints:
                    audit = checkpoint_state_report(checkpoint)
                    print(f"[checkpoint] audit={json.dumps(audit, ensure_ascii=False)}", flush=True)
                    if audit["status"] != "ok":
                        raise RuntimeError(f"Incomplete Qwen3 Stage-III checkpoint: {audit}")
            return result

    actual_schedule = report["actual_training_schedule"]
    total_steps = int(actual_schedule["total_steps"])
    warmup_steps = int(actual_schedule["warmup_steps"])
    argv: list[str] = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", str(MAX_SEQUENCE_LENGTH),
        "--output_dir", str(args.output_dir), "--tuner_type", "lora", "--tuner_backend", "peft",
        "--target_modules", *EXPECTED_LORA_TARGETS, "--modules_to_save", "audio_qformer",
        "--freeze_llm", "true", "--freeze_vit", "true", "--freeze_aligner", "false",
        "--lora_rank", str(BAT_TRAINING.lora_rank), "--lora_alpha", str(BAT_TRAINING.lora_alpha),
        "--lora_dropout", str(BAT_TRAINING.lora_dropout), "--learning_rate", str(LEARNING_RATE),
        "--lr_scheduler_type", "cosine", "--warmup_steps", str(warmup_steps),
        "--max_steps", str(total_steps), "--num_train_epochs", "1",
        "--per_device_train_batch_size", str(PER_DEVICE_BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--gradient_checkpointing", "false", "--logging_steps", "100",
        "--save_strategy", "steps", "--save_steps", str(PERIODIC_SAVE_STEPS),
        "--save_only_model", "false", "--save_total_limit", str(MAX_PERIODIC_CHECKPOINTS),
        "--remove_unused_columns", "false", "--dataloader_num_workers", "0",
        "--dataloader_pin_memory", "false", "--dataloader_drop_last", "false",
        "--dataset_num_proc", "1", "--lazy_tokenize", "true", "--load_from_cache_file", "false",
        "--loss_scale", "all", "--seed", "42", "--data_seed", "42", "--optim", "adamw_torch",
        "--adam_beta1", str(BAT_TRAINING.beta1), "--adam_beta2", str(BAT_TRAINING.beta2),
        "--weight_decay", str(BAT_TRAINING.weight_decay), "--attn_impl", "sdpa", "--bf16", "true",
        "--ddp_find_unused_parameters", "false", "--average_tokens_across_devices", "false", "--report_to", "none",
    ]
    if args.resume_from_checkpoint is not None:
        argv.extend(["--resume_from_checkpoint", str(args.resume_from_checkpoint)])

    print("========== QWEN3-4B BAT STAGE-III A+B -> C+D+E TRAINING ==========")
    print(f"[ddp] world_size={actual_world_size} per_device_batch_size={PER_DEVICE_BATCH_SIZE} global_batch_size={global_batch_size}")
    print(f"[route] manifest={args.dataset} records={dataset_records} total_steps={total_steps}")
    print(f"[model] Qwen3-4B base; Spatial-AST frozen FP32; Q-Former trainable random init; Qwen3 native frozen")
    print(f"[lora] targets={EXPECTED_LORA_TARGETS} rank=8 alpha=32 dropout=0.05")
    print(f"[audio] tokens=64 sequence_length={MAX_SEQUENCE_LENGTH} RIR=crop_or_zero_pad_to_2s")
    print(f"[schedule] learning_rate={LEARNING_RATE} warmup_steps={warmup_steps} scheduler=half-cycle cosine decay")
    print(f"[compile] requested={args.torch_compile} target=Qwen3ForCausalLM.model dynamic={args.compile_dynamic} mode={args.compile_mode}")
    print(
        f"[checkpoint] native_save_strategy=steps save_steps={PERIODIC_SAVE_STEPS} "
        f"save_total_limit={MAX_PERIODIC_CHECKPOINTS} full_resumable=true"
    )
    print(f"[data] dataloader_num_workers=0 pin_memory=false")
    if rank() == 0:
        print(f"[argv] {' '.join(argv)}")
    Qwen3Stage3SwiftSft(argv).main()


if __name__ == "__main__":
    main()
