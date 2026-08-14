#!/usr/bin/env python3
"""Run BAT Stage-I/II/III as one ordered, continuous Swift training job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bat.configs.training import BAT_TRAINING
from bat.curriculum import count_jsonl, load_report, validate_curriculum_report

MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--curriculum-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def main() -> None:
    args = parse_args()
    BAT_TRAINING.validate()
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world_size != args.world_size:
        raise RuntimeError(f"World-size mismatch: launcher={actual_world_size} argument={args.world_size}")
    if args.world_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("world-size and gradient-accumulation-steps must be positive")
    if actual_world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0 or local_rank >= actual_world_size:
            raise RuntimeError(f"Invalid LOCAL_RANK={local_rank}")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Continuous BAT DDP training requires CUDA")
        torch.cuda.set_device(local_rank)

    for path in (args.model_path, args.plugin_path, args.dataset, args.curriculum_report):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    if str(args.output_dir).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_dir}")
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(args.resume_from_checkpoint)
    if args.resume_from_checkpoint is None:
        if rank() == 0 and args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Refusing non-empty output directory for a fresh run: {args.output_dir}")
    elif rank() == 0:
        checkpoint = args.resume_from_checkpoint.resolve()
        output_root = args.output_dir.resolve()
        if checkpoint.parent != output_root:
            raise ValueError(
                "Resume checkpoint must be directly inside --output-dir so subsequent curriculum "
                f"boundary checkpoints remain in one run directory: checkpoint={checkpoint} output={output_root}"
            )

    global_batch_size = BAT_TRAINING.per_device_batch_size * args.world_size * args.gradient_accumulation_steps
    curriculum_report = load_report(args.curriculum_report)
    validate_curriculum_report(curriculum_report, global_batch_size)
    dataset_records = count_jsonl(args.dataset)
    if dataset_records != int(curriculum_report["total_records"]):
        raise RuntimeError(
            f"Curriculum manifest count mismatch: actual={dataset_records} "
            f"report={curriculum_report['total_records']}"
        )

    from swift.pipelines.train.sft import SwiftSft
    from curriculum_checkpoint import CurriculumBoundaryCheckpointCallback

    class ContinuousCurriculumSwiftSft(SwiftSft):
        def train(self, trainer):
            callback = CurriculumBoundaryCheckpointCallback(
                args.curriculum_report,
                global_batch_size,
                checkpoint_root=args.output_dir,
            )
            trainer.add_callback(callback)
            result = super().train(trainer)
            missing = callback.missing_boundary_steps()
            if missing:
                raise RuntimeError(f"Missing curriculum boundary checkpoints: {missing}")
            if rank() == 0:
                for step, stage in sorted(callback.step_to_stage.items()):
                    checkpoint_dir = args.output_dir / f"checkpoint-{step}"
                    marker = checkpoint_dir / "curriculum_stage.json"
                    if not marker.is_file():
                        raise RuntimeError(f"Missing curriculum marker for Stage-{stage}: {marker}")
                    print(f"[checkpoint] stage={stage} global_step={step} path={checkpoint_dir}", flush=True)
            return result

    warmup_steps = int(curriculum_report["warmup_steps"])
    total_steps = int(curriculum_report["total_steps"])
    argv: list[str] = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", "512",
        "--output_dir", str(args.output_dir), "--tuner_type", "lora", "--tuner_backend", "peft",
        "--target_modules", *BAT_TRAINING.lora_target_modules, "--modules_to_save", "audio_qformer",
        "--freeze_llm", "true", "--freeze_vit", "true", "--freeze_aligner", "false",
        "--lora_rank", str(BAT_TRAINING.lora_rank), "--lora_alpha", str(BAT_TRAINING.lora_alpha),
        "--lora_dropout", str(BAT_TRAINING.lora_dropout), "--learning_rate", str(BAT_TRAINING.learning_rate),
        "--lr_scheduler_type", "cosine", "--warmup_steps", str(warmup_steps),
        "--max_steps", str(total_steps), "--num_train_epochs", "1",
        "--per_device_train_batch_size", str(BAT_TRAINING.per_device_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--gradient_checkpointing", "false", "--logging_steps", "100",
        "--save_strategy", "no", "--save_only_model", "false", "--save_total_limit", "3",
        "--remove_unused_columns", "false", "--dataloader_num_workers", "4",
        "--dataloader_pin_memory", "true", "--dataloader_drop_last", "false",
        "--dataset_num_proc", "1", "--lazy_tokenize", "true", "--load_from_cache_file", "false",
        "--loss_scale", "all", "--seed", "42", "--data_seed", "42", "--optim", "adamw_torch",
        "--adam_beta1", str(BAT_TRAINING.beta1), "--adam_beta2", str(BAT_TRAINING.beta2),
        "--weight_decay", str(BAT_TRAINING.weight_decay), "--attn_impl", "sdpa", "--bf16", "true",
        "--ddp_find_unused_parameters", "false", "--average_tokens_across_devices", "false",
        "--report_to", "none",
    ]
    if args.resume_from_checkpoint is not None:
        argv.extend(["--resume_from_checkpoint", str(args.resume_from_checkpoint)])

    print("========== BAT OURO CONTINUOUS CURRICULUM TRAINING ==========")
    print(f"[rank] rank={rank()} world_size={actual_world_size}")
    print(f"[curriculum] manifest={args.dataset} records={dataset_records}")
    print(f"[curriculum] boundaries={curriculum_report['boundary_steps']}")
    print(
        f"[curriculum] shuffle={curriculum_report['shuffle_policy']} "
        f"runtime_shuffle={curriculum_report['runtime_shuffle']}"
    )
    print(f"[schedule] {json.dumps({'total_steps': total_steps, 'warmup_steps': warmup_steps, 'scheduler': 'cosine', 'global_batch_size': global_batch_size}, ensure_ascii=False)}")
    if rank() == 0:
        print(f"[argv] {' '.join(argv)}")
    ContinuousCurriculumSwiftSft(argv).main()


if __name__ == "__main__":
    main()
