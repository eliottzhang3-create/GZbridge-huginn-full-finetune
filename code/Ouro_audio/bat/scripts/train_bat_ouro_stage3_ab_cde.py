#!/usr/bin/env python3
"""Train Ouro on the custom two-epoch Stage-III A+B -> C+D+E route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bat.configs.training import BAT_TRAINING
from bat.curriculum import count_jsonl


MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
PER_DEVICE_BATCH_SIZE = 8
WORLD_SIZE_REQUIRED = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE_REQUIRED)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
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
    if BAT_TRAINING.lora_target_modules != ("q_proj", "v_proj"):
        raise ValueError("BAT LoRA target modules must remain q_proj and v_proj")


def load_and_validate_report(path: Path, global_batch_size: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "ok":
        raise ValueError(f"Stage-III route report is not ok: {path}")
    if report.get("route") != "stage3_ab_cde_2epoch":
        raise ValueError(f"Unexpected route in report: {report.get('route')!r}")
    if int(report.get("global_batch_size", -1)) != global_batch_size:
        raise ValueError(
            f"Global batch mismatch: report={report.get('global_batch_size')} expected={global_batch_size}"
        )
    if int(report.get("per_device_batch_size", -1)) != PER_DEVICE_BATCH_SIZE:
        raise ValueError("Stage-III route report has an unexpected per-device batch size")
    if int(report.get("world_size", -1)) != WORLD_SIZE_REQUIRED:
        raise ValueError("Stage-III route report has an unexpected world size")
    if int(report.get("gradient_accumulation_steps", -1)) != GRADIENT_ACCUMULATION_STEPS:
        raise ValueError("Stage-III route requires gradient accumulation 1")
    if float(report.get("learning_rate", -1.0)) != LEARNING_RATE:
        raise ValueError("Stage-III route requires learning rate 0.002")
    if report.get("runtime_shuffle") is not False:
        raise ValueError("Stage-III route must disable runtime shuffle")
    blocks = report.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 4:
        raise ValueError("Stage-III route must contain four ordered blocks")
    expected = [(1, "A+B"), (1, "C+D+E"), (2, "A+B"), (2, "C+D+E")]
    observed = [(int(item.get("epoch", -1)), str(item.get("group"))) for item in blocks]
    if observed != expected:
        raise ValueError(f"Unexpected block order: {observed}")
    previous_step = 0
    previous_record = 0
    for item in blocks:
        start_step = int(item.get("start_step", -1))
        end_step = int(item.get("end_step", -1))
        start_record = int(item.get("start_record", -1))
        end_record = int(item.get("end_record", -1))
        written = int(item.get("written_records", -1))
        if start_step != previous_step or start_record != previous_record:
            raise ValueError(f"Non-contiguous Stage-III block: {item}")
        if written <= 0 or written % global_batch_size or end_step != start_step + written // global_batch_size:
            raise ValueError(f"Invalid Stage-III block step range: {item}")
        if end_record != start_record + written:
            raise ValueError(f"Invalid Stage-III block record range: {item}")
        previous_step = end_step
        previous_record = end_record
    if int(report.get("total_steps", -1)) != previous_step or int(report.get("total_records", -1)) != previous_record:
        raise ValueError("Stage-III report totals do not match block ranges")
    boundaries = report.get("epoch_boundary_steps")
    if boundaries != {"1": 13_630, "2": 27_260}:
        raise ValueError(f"Unexpected epoch boundary steps: {boundaries}")
    if int(report.get("warmup_steps", -1)) != 3_544:
        raise ValueError(f"Unexpected warmup_steps: {report.get('warmup_steps')}")
    return report


def checkpoint_state_report(path: Path) -> dict[str, Any]:
    required = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "stage3_epoch.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    rng_files = sorted(list(path.glob("rng_state_*.pth")) + list(path.glob("rng_state_*.pt")))
    if len(rng_files) < WORLD_SIZE_REQUIRED:
        missing.append(f"rng_state_*.pth>={WORLD_SIZE_REQUIRED}")
    return {"path": str(path), "missing": missing, "rng_files": [item.name for item in rng_files], "status": "ok" if not missing else "incomplete"}


def main() -> None:
    args = parse_args()
    validate_base_contract()
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.world_size != WORLD_SIZE_REQUIRED or actual_world_size != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"This route requires WORLD_SIZE=8, got launcher={args.world_size} actual={actual_world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0 or local_rank >= actual_world_size:
        raise RuntimeError(f"Invalid LOCAL_RANK={local_rank}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-III DDP training requires CUDA")
    torch.cuda.set_device(local_rank)

    for path in (args.model_path, args.plugin_path, args.dataset, args.report):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    if str(args.output_dir).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_dir}")
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(args.resume_from_checkpoint)
    if args.resume_from_checkpoint is None and rank() == 0 and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory for a fresh run: {args.output_dir}")
    if args.resume_from_checkpoint is not None and rank() == 0:
        try:
            args.resume_from_checkpoint.resolve().relative_to(args.output_dir.resolve())
        except ValueError as exc:
            raise ValueError("Resume checkpoint must be inside --output-dir") from exc

    global_batch_size = PER_DEVICE_BATCH_SIZE * actual_world_size * GRADIENT_ACCUMULATION_STEPS
    report = load_and_validate_report(args.report, global_batch_size)
    dataset_records = count_jsonl(args.dataset)
    if dataset_records != int(report["total_records"]):
        raise RuntimeError(f"Manifest count mismatch: actual={dataset_records} report={report['total_records']}")

    from swift.pipelines.train.sft import SwiftSft
    from stage3_ab_cde_checkpoint import Stage3EpochCheckpointCallback

    class Stage3AbCdeSwiftSft(SwiftSft):
        def train(self, trainer):
            effective_output_dir = Path(trainer.args.output_dir).resolve()
            callback = Stage3EpochCheckpointCallback(
                args.report,
                checkpoint_root=effective_output_dir,
                resume_checkpoint=args.resume_from_checkpoint,
            )
            trainer.add_callback(callback)
            result = super().train(trainer)
            missing = callback.missing_boundary_steps()
            if missing:
                raise RuntimeError(f"Missing Stage-III epoch checkpoints: {missing}")
            if rank() == 0:
                for step, epoch in sorted(callback.step_to_epoch.items()):
                    checkpoint = effective_output_dir / f"checkpoint-{step}"
                    audit = checkpoint_state_report(checkpoint)
                    print(f"[checkpoint] epoch={epoch} global_step={step} audit={json.dumps(audit, ensure_ascii=False)}", flush=True)
                    if audit["status"] != "ok":
                        raise RuntimeError(f"Incomplete Stage-III checkpoint: {audit}")
            return result

    total_steps = int(report["total_steps"])
    warmup_steps = int(report["warmup_steps"])
    argv: list[str] = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", "512",
        "--output_dir", str(args.output_dir), "--tuner_type", "lora", "--tuner_backend", "peft",
        "--target_modules", *BAT_TRAINING.lora_target_modules, "--modules_to_save", "audio_qformer",
        "--freeze_llm", "true", "--freeze_vit", "true", "--freeze_aligner", "false",
        "--lora_rank", str(BAT_TRAINING.lora_rank), "--lora_alpha", str(BAT_TRAINING.lora_alpha),
        "--lora_dropout", str(BAT_TRAINING.lora_dropout), "--learning_rate", str(LEARNING_RATE),
        "--lr_scheduler_type", "cosine", "--warmup_steps", str(warmup_steps),
        "--max_steps", str(total_steps), "--num_train_epochs", "1",
        "--per_device_train_batch_size", str(PER_DEVICE_BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--gradient_checkpointing", "false", "--logging_steps", "100",
        "--save_strategy", "no", "--save_only_model", "false", "--save_total_limit", "2",
        "--remove_unused_columns", "false", "--dataloader_num_workers", "4",
        "--dataloader_pin_memory", "true", "--dataloader_drop_last", "false",
        "--dataset_num_proc", "1", "--lazy_tokenize", "true", "--load_from_cache_file", "false",
        "--loss_scale", "all", "--seed", "42", "--data_seed", "42", "--optim", "adamw_torch",
        "--adam_beta1", str(BAT_TRAINING.beta1), "--adam_beta2", str(BAT_TRAINING.beta2),
        "--weight_decay", str(BAT_TRAINING.weight_decay), "--attn_impl", "sdpa", "--bf16", "true",
        "--ddp_find_unused_parameters", "false", "--average_tokens_across_devices", "false", "--report_to", "none",
    ]
    if args.resume_from_checkpoint is not None:
        argv.extend(["--resume_from_checkpoint", str(args.resume_from_checkpoint)])

    print("========== BAT OURO STAGE-III A+B -> C+D+E TRAINING ==========")
    print(f"[ddp] world_size={actual_world_size} per_device_batch_size={PER_DEVICE_BATCH_SIZE} global_batch_size={global_batch_size}")
    print(f"[route] manifest={args.dataset} records={dataset_records} total_steps={total_steps}")
    print(f"[schedule] learning_rate={LEARNING_RATE} warmup_steps={warmup_steps} scheduler=half-cycle cosine decay")
    print(f"[checkpoint] epoch_boundaries={report['epoch_boundary_steps']}")
    if rank() == 0:
        print(f"[argv] {' '.join(argv)}")
    Stage3AbCdeSwiftSft(argv).main()


if __name__ == "__main__":
    main()
