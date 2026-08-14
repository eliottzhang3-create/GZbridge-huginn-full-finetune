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
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from bat.configs.training import BAT_TRAINING
from smoke_bat_ouro_lora import (
    TARGET_MODULES,
    checkpoint_report,
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


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gather_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DDP smoke did not initialize torch.distributed")
    gathered: list[dict[str, Any] | None] = [None] * world_size()
    dist.all_gather_object(gathered, report)
    return [item for item in gathered if item is not None]


def checkpoint_path(output_dir: Path) -> Path:
    candidates = sorted(path for path in output_dir.rglob("checkpoint-2") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one checkpoint-2 below {output_dir}, found {candidates}")
    return candidates[0]


def main() -> None:
    args = parse_args()
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
    dataset_records = count_jsonl(args.dataset)
    if dataset_records != EXPECTED_DATASET_RECORDS:
        raise RuntimeError(
            f"DDP smoke requires exactly {EXPECTED_DATASET_RECORDS} JSONL records; "
            f"got {dataset_records} from {args.dataset}"
        )
    if args.output_dir.exists() and current_rank == 0:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if str(args.output_report).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output path: {args.output_report}")

    from swift.pipelines.train.sft import SwiftSft

    schedule = BAT_TRAINING.schedule(
        dataset_size=EXPECTED_DATASET_RECORDS,
        world_size=EXPECTED_WORLD_SIZE,
        gradient_accumulation_steps=1,
        stage_name="I",
    )
    if int(schedule["effective_batch_size"]) != 16 or int(schedule["total_steps"]) != EXPECTED_OPTIMIZER_STEPS:
        raise RuntimeError(f"Unexpected DDP smoke schedule: {schedule}")

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
        "--warmup_steps", str(schedule["warmup_steps"]), "--max_steps", str(schedule["total_steps"]),
        "--num_train_epochs", str(schedule["epochs"]), "--per_device_train_batch_size", str(EXPECTED_LOCAL_BATCH),
        "--gradient_accumulation_steps", "1", "--gradient_checkpointing", "false", "--logging_steps", "1",
        "--save_strategy", "steps", "--save_steps", "2", "--save_total_limit", "1", "--save_only_model", "false",
        "--dataloader_num_workers", "0", "--dataloader_pin_memory", "false", "--dataset_num_proc", "1",
        "--lazy_tokenize", "false", "--load_from_cache_file", "false", "--loss_scale", "all", "--seed", "42",
        "--data_seed", "42", "--optim", "adamw_torch", "--adam_beta1", str(BAT_TRAINING.beta1),
        "--adam_beta2", str(BAT_TRAINING.beta2), "--weight_decay", str(BAT_TRAINING.weight_decay),
        "--attn_impl", "sdpa", "--bf16", "true", "--report_to", "none",
    ]

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
            }
            handles: list[Any] = []
            first_layer = ouro.layers[0]
            handles.append(first_layer.register_forward_hook(lambda *_: trace.__setitem__("layer", trace["layer"] + 1)))
            handles.append(ouro.early_exit_gate.register_forward_hook(lambda *_: trace.__setitem__("gate", trace["gate"] + 1)))
            handles.append(first_layer.register_full_backward_hook(lambda *_: trace.__setitem__("layer_backward", trace["layer_backward"] + 1)))
            handles.append(ouro.early_exit_gate.register_full_backward_hook(lambda *_: trace.__setitem__("gate_backward", trace["gate_backward"] + 1)))
            original_compute_loss = trainer.compute_loss
            original_backward = trainer.accelerator.backward

            def compute_loss(actual_model, inputs, return_outputs=False, num_items_in_batch=None):
                trace["forward"] += 1
                result = original_compute_loss(actual_model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                loss, outputs = result
                if trace["loss"] is None:
                    logits = outputs.logits
                    labels = inputs["labels"]
                    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
                        raise RuntimeError(f"Cannot audit shifted CE: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
                    shifted_logits = logits[:, :-1].contiguous()
                    shifted_labels = labels[:, 1:].contiguous()
                    manual_loss = F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                        shifted_labels.reshape(-1), ignore_index=-100,
                    )
                    trainer_value = float(loss.detach().float().cpu())
                    manual_value = float(manual_loss.detach().float().cpu())
                    if not math.isclose(manual_value, trainer_value, rel_tol=2e-3, abs_tol=2e-3):
                        raise RuntimeError(f"Trainer CE mismatch: trainer={trainer_value} manual={manual_value}")
                    if int(inputs["input_ids"].shape[0]) != EXPECTED_LOCAL_BATCH:
                        raise RuntimeError(f"Rank {current_rank} local batch is not 2: {tuple(inputs['input_ids'].shape)}")
                    if not bool((labels[:, :BAT_TRAINING.audio_token_count] == -100).all().item()):
                        raise RuntimeError("Audio prefix labels are not fully masked")
                    trace["batch"] = {
                        "input_ids_shape": list(inputs["input_ids"].shape),
                        "labels_shape": list(labels.shape),
                        "attention_mask_shape": list(inputs["attention_mask"].shape),
                        "audio_waveforms_shape": list(inputs["audio_waveforms"].shape),
                        "audio_prefix_label_ignore_count": int((labels[:, :BAT_TRAINING.audio_token_count] == -100).sum().item()),
                        "valid_shifted_target_count": int((shifted_labels != -100).sum().item()),
                        "manual_shifted_ce": manual_value,
                        "trainer_ce": trainer_value,
                        "shift_verified": True,
                    }
                    trace["loss"] = {"value": trainer_value, "logits_shape": list(logits.shape), "labels_shape": list(labels.shape)}
                return result if return_outputs else loss

            def backward(loss, **kwargs):
                trace["backward"] += 1
                return original_backward(loss, **kwargs)

            trainer.compute_loss = compute_loss
            trainer.accelerator.backward = backward
            started = time.perf_counter()
            try:
                result = super().train(trainer)
            finally:
                trainer.compute_loss = original_compute_loss
                trainer.accelerator.backward = original_backward
                for handle in handles:
                    handle.remove()

            local_optimizer = optimizer_report(trainer, model)
            if int(trainer.state.global_step) != EXPECTED_OPTIMIZER_STEPS:
                raise RuntimeError(f"Rank {current_rank} expected global_step=2, got {trainer.state.global_step}")
            if trace["forward"] != EXPECTED_OPTIMIZER_STEPS or trace["backward"] != EXPECTED_OPTIMIZER_STEPS:
                raise RuntimeError(f"Rank {current_rank} unexpected trainer counts: {trace}")
            if trace["layer"] != EXPECTED_RECURRENT_STEPS * EXPECTED_OPTIMIZER_STEPS or trace["gate"] != EXPECTED_RECURRENT_STEPS * EXPECTED_OPTIMIZER_STEPS:
                raise RuntimeError(f"Rank {current_rank} unexpected recurrent forward counts: {trace}")
            if trace["layer_backward"] != EXPECTED_RECURRENT_STEPS * EXPECTED_OPTIMIZER_STEPS:
                raise RuntimeError(f"Rank {current_rank} unexpected recurrent backward count: {trace}")
            local_report = {
                "rank": current_rank, "local_rank": local_rank, "world_size": current_world,
                "parameters": parameters, "lora": lora, "optimizer": local_optimizer,
                "forward_audit": {**trace, "expected_recurrent_steps": EXPECTED_RECURRENT_STEPS, "use_cache": False},
                "elapsed_seconds": time.perf_counter() - started,
                "global_step": int(trainer.state.global_step),
            }
            reports = gather_reports(local_report)
            if len(reports) != EXPECTED_WORLD_SIZE:
                raise RuntimeError(f"Expected {EXPECTED_WORLD_SIZE} rank reports, got {len(reports)}")
            if current_rank == 0:
                rank_ids = sorted(int(item["rank"]) for item in reports)
                if rank_ids != list(range(EXPECTED_WORLD_SIZE)):
                    raise RuntimeError(f"Unexpected rank reports: {rank_ids}")
                local_batches = [item["forward_audit"]["batch"]["input_ids_shape"][0] for item in reports]
                if local_batches != [EXPECTED_LOCAL_BATCH] * EXPECTED_WORLD_SIZE:
                    raise RuntimeError(f"Local batch audit failed: {local_batches}")
                checkpoint = checkpoint_report(checkpoint_path(args.output_dir))
                report = {
                    "status": "ok",
                    "distributed": {
                        "backend": dist.get_backend(), "world_size": EXPECTED_WORLD_SIZE,
                        "per_device_batch_size": EXPECTED_LOCAL_BATCH,
                        "gradient_accumulation_steps": 1, "global_batch_size": 16,
                        "dataset_records": EXPECTED_DATASET_RECORDS, "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
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
    main()
