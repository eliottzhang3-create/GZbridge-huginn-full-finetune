#!/usr/bin/env python3
"""Launch one full BAT curriculum stage through ms-swift.

The launcher intentionally receives a prepared private JSONL manifest.  This
keeps the official QA records and lazy AudioSet/RIR renderer separate from the
trainer, while making the exact sample count available for the warm-up and
cosine step calculation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bat.configs.training import BAT_TRAINING

MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage", choices=("I", "II", "III"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    args = parse_args()
    BAT_TRAINING.validate()
    process_rank = int(os.environ.get("RANK", "0"))
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world_size != args.world_size:
        raise RuntimeError(
            f"Distributed world-size mismatch: launcher={actual_world_size} argument={args.world_size}"
        )
    if actual_world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0 or local_rank >= actual_world_size:
            raise RuntimeError(f"Invalid LOCAL_RANK={local_rank} for WORLD_SIZE={actual_world_size}")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("BAT DDP training requires CUDA")
        torch.cuda.set_device(local_rank)
    if str(args.output_dir).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_dir}")
    for path in (args.model_path, args.plugin_path, args.dataset):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    if process_rank == 0 and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output_dir}")
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(args.resume_from_checkpoint)

    dataset_size = count_jsonl(args.dataset)
    schedule = BAT_TRAINING.schedule(
        dataset_size=dataset_size,
        world_size=args.world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        stage_name=args.stage,
    )
    from swift.pipelines.train.sft import SwiftSft

    argv: list[str] = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "true", "--train_dataloader_shuffle", "true",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", "512",
        "--output_dir", str(args.output_dir), "--tuner_type", "lora", "--tuner_backend", "peft",
        "--target_modules", *BAT_TRAINING.lora_target_modules, "--modules_to_save", "audio_qformer",
        "--freeze_llm", "true", "--freeze_vit", "true", "--freeze_aligner", "false",
        "--lora_rank", str(BAT_TRAINING.lora_rank), "--lora_alpha", str(BAT_TRAINING.lora_alpha),
        "--lora_dropout", str(BAT_TRAINING.lora_dropout), "--learning_rate", str(BAT_TRAINING.learning_rate),
        "--lr_scheduler_type", "cosine", "--warmup_steps", str(schedule["warmup_steps"]),
        "--max_steps", str(schedule["total_steps"]), "--num_train_epochs", str(schedule["epochs"]),
        "--per_device_train_batch_size", str(BAT_TRAINING.per_device_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        # The validated 8x5090 BAT path uses ordinary autograd.  The memory
        # preflight peaked at only ~6.9 GiB/GPU, so enabling checkpointing
        # would add an untested recomputation path without a memory benefit.
        "--gradient_checkpointing", "false", "--logging_steps", "100", "--save_strategy", "steps",
        "--save_steps", str(max(1, int(schedule["steps_per_epoch"]))), "--save_total_limit", "2",
        "--save_only_model", "false", "--remove_unused_columns", "false", "--dataloader_num_workers", "4",
        "--dataloader_pin_memory", "true", "--dataset_num_proc", "1", "--lazy_tokenize", "false",
        "--load_from_cache_file", "false", "--loss_scale", "all", "--seed", "42", "--data_seed", "42",
        "--optim", "adamw_torch", "--adam_beta1", str(BAT_TRAINING.beta1), "--adam_beta2", str(BAT_TRAINING.beta2),
        "--weight_decay", str(BAT_TRAINING.weight_decay), "--attn_impl", "sdpa", "--bf16", "true",
        "--ddp_find_unused_parameters", "false", "--average_tokens_across_devices", "false",
        "--report_to", "none",
    ]
    if args.resume_from_checkpoint is not None:
        argv.extend(["--resume_from_checkpoint", str(args.resume_from_checkpoint)])

    print("========== BAT OURO STAGE TRAINING ==========")
    print(f"[rank] rank={process_rank} world_size={actual_world_size}")
    print(f"[stage] {args.stage} dataset={args.dataset} records={dataset_size}")
    paper_contract = {
        "sound_source": BAT_TRAINING.sound_source,
        "audio_normalization": BAT_TRAINING.audio_normalization,
        "augmentation": BAT_TRAINING.augmentation,
        "weighted_sampling": BAT_TRAINING.weighted_sampling,
        "optimizer": BAT_TRAINING.optimizer,
        "betas": [BAT_TRAINING.beta1, BAT_TRAINING.beta2],
        "weight_decay": BAT_TRAINING.weight_decay,
        "learning_rate": BAT_TRAINING.learning_rate,
        "scheduler": BAT_TRAINING.scheduler,
        "warmup_epochs": BAT_TRAINING.warmup_epochs,
        "epoch_partitioning_factor": BAT_TRAINING.epoch_partitioning_factor,
        "batch_size": BAT_TRAINING.per_device_batch_size,
    }
    print(f"[paper] {json.dumps(paper_contract, ensure_ascii=False)}")
    print(f"[schedule] {json.dumps(dict(schedule), ensure_ascii=False)}")
    if process_rank == 0:
        print(f"[argv] {' '.join(argv)}")
    SwiftSft(argv).main()


if __name__ == "__main__":
    main()
