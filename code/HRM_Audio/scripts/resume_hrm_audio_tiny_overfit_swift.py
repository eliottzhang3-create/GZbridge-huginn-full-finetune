#!/usr/bin/env python3
"""Resume the HRM-audio tiny-overfit run through ms-swift with a boundary audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

import audit_hrm_audio_tiny_overfit_resume as gate_audit
import inspect_hrm_audio_swift_trainability as trainability_audit


MODEL_TYPE = "hrm_text_audio_whisper"
TEMPLATE_TYPE = "hrm_text_audio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-report", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--step-before-resume", type=int, default=12)
    parser.add_argument("--step-after-resume", type=int, default=24)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--aligner-learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_trainables(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    from peft import get_peft_model_state_dict

    adapter_raw = get_peft_model_state_dict(model)
    adapter = {
        gate_audit.canonical_adapter_key(key): tensor.detach().cpu()
        for key, tensor in adapter_raw.items()
    }
    wrapper = trainability_audit.find_unique_module(model, "HrmTextAudioForConditionalGeneration")
    aligner = {
        gate_audit.canonical_aligner_key(key): tensor.detach().cpu()
        for key, tensor in wrapper.state_dict().items()
        if gate_audit.canonical_aligner_key(key) is not None
    }
    return adapter, aligner


def exact_runtime_reload_report(
    checkpoint: Path,
    model: torch.nn.Module,
) -> dict[str, Any]:
    saved_adapter, saved_aligner = gate_audit.checkpoint_trainables(checkpoint)
    runtime_adapter, runtime_aligner = runtime_trainables(model)

    def compare(
        saved: dict[str, torch.Tensor],
        runtime: dict[str, torch.Tensor],
        *,
        name: str,
        expected_count: int,
    ) -> dict[str, Any]:
        if len(saved) != expected_count or set(saved) != set(runtime):
            raise RuntimeError(
                f"Resume-boundary {name} key mismatch: saved={len(saved)} runtime={len(runtime)} "
                f"missing={sorted(set(saved) - set(runtime))[:20]} "
                f"unexpected={sorted(set(runtime) - set(saved))[:20]}"
            )
        max_abs_diff = 0.0
        dtype_pairs = {}
        for key in saved:
            left = saved[key]
            right = runtime[key]
            if left.shape != right.shape:
                raise RuntimeError(
                    f"Resume-boundary {name} shape mismatch for {key}: "
                    f"saved={tuple(left.shape)} runtime={tuple(right.shape)}"
                )
            max_abs_diff = max(max_abs_diff, float((left.float() - right.float()).abs().max().item()))
            pair = f"{left.dtype}->{right.dtype}"
            dtype_pairs[pair] = dtype_pairs.get(pair, 0) + 1
        if max_abs_diff != 0.0:
            raise RuntimeError(f"Resume-boundary {name} tensors are not exact: max_abs_diff={max_abs_diff}")
        return {
            "tensor_count": len(saved),
            "max_abs_diff": max_abs_diff,
            "dtype_pairs": dtype_pairs,
        }

    return {
        "adapter": compare(
            saved_adapter,
            runtime_adapter,
            name="LoRA",
            expected_count=gate_audit.EXPECTED_LORA_TENSORS,
        ),
        "aligner": compare(
            saved_aligner,
            runtime_aligner,
            name="aligner",
            expected_count=gate_audit.EXPECTED_ALIGNER_TENSORS,
        ),
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
        raise RuntimeError(f"Unexpected HRM resume environment: {mismatches}")
    if args.step_before_resume <= 0 or args.step_after_resume <= args.step_before_resume:
        raise ValueError("Resume steps must satisfy 0 < before < after")
    if args.micro_batch_size * args.gradient_accumulation_steps != 32:
        raise ValueError("The formal HRM resume gate requires effective batch size 32")
    if not torch.cuda.is_available():
        raise RuntimeError("HRM audio resume gate requires CUDA")
    wrapper_model_path = args.wrapper_model_path.expanduser().resolve()
    plugin_path = args.plugin_path.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    checkpoint = args.resume_from_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    boundary_report_path = args.boundary_report.expanduser().resolve()
    runtime_report_path = args.runtime_report.expanduser().resolve()
    for path, name in (
        (wrapper_model_path, "wrapper model"),
        (plugin_path, "plugin"),
        (dataset, "dataset"),
        (checkpoint, "resume checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    from swift.pipelines.train.sft import SwiftSft
    from transformers import TrainerCallback

    boundary_capture: dict[str, Any] = {}

    class ResumeBoundaryCallback(TrainerCallback):
        def on_train_begin(
            self,
            training_args,
            state,
            control,
            model=None,
            optimizer=None,
            lr_scheduler=None,
            **kwargs,
        ):
            del training_args, control, kwargs
            if boundary_capture:
                raise RuntimeError("Resume boundary callback was invoked more than once")
            if model is None or optimizer is None or lr_scheduler is None:
                raise RuntimeError("Resume boundary callback did not receive model/optimizer/scheduler")
            if int(state.global_step) != args.step_before_resume:
                raise RuntimeError(
                    f"Resume boundary global_step mismatch: expected={args.step_before_resume} actual={state.global_step}"
                )
            persistent_state = exact_runtime_reload_report(checkpoint, model)
            optimizer_payload = optimizer.state_dict()
            optimizer_report = gate_audit.optimizer_state_report(
                optimizer_payload,
                expected_step=args.step_before_resume,
                expected_lr=args.learning_rate,
            )
            scheduler_payload = lr_scheduler.state_dict()
            scheduler_report = gate_audit.scheduler_state_report(
                scheduler_payload,
                expected_step=args.step_before_resume,
            )
            wrapper = trainability_audit.find_unique_module(model, "HrmTextAudioForConditionalGeneration")
            parameter_report = trainability_audit.audit_parameters(
                model,
                wrapper,
                expected_lora_rank=args.lora_rank,
            )
            boundary_capture.update(
                {
                    "status": "OK",
                    "checkpoint": str(checkpoint),
                    "global_step": int(state.global_step),
                    "persistent_state": persistent_state,
                    "optimizer": optimizer_report,
                    "scheduler": scheduler_report,
                    "parameters": parameter_report,
                }
            )
            atomic_write_json(boundary_report_path, boundary_capture)
            print("========== HRM AUDIO RESUME BOUNDARY AUDIT ==========")
            print(f"[resume-boundary] checkpoint={checkpoint} global_step={state.global_step}")
            print(f"[resume-boundary] persistent_state={json.dumps(persistent_state, ensure_ascii=False)}")
            print(f"[resume-boundary] optimizer={json.dumps(optimizer_report, ensure_ascii=False)}")
            print(f"[resume-boundary] scheduler={json.dumps(scheduler_report, ensure_ascii=False)}")
            print("[resume-boundary] status=OK")

    save_interval = args.step_after_resume
    argv = [
        "--model", str(wrapper_model_path),
        "--model_type", MODEL_TYPE,
        "--template", TEMPLATE_TYPE,
        "--external_plugins", str(plugin_path),
        "--dataset", str(dataset),
        "--split_dataset_ratio", "0",
        "--dataset_shuffle", "false",
        "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false",
        "--group_by_length", "false",
        "--max_length", "192",
        "--output_dir", str(output_dir),
        "--tuner_type", "lora_llm",
        "--tuner_backend", "peft",
        "--target_modules", "all-linear",
        "--freeze_llm", "true",
        "--freeze_vit", "true",
        "--freeze_aligner", "false",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--learning_rate", str(args.learning_rate),
        "--aligner_lr", str(args.aligner_learning_rate),
        "--lr_scheduler_type", "constant",
        "--warmup_ratio", "0",
        "--max_steps", str(args.step_after_resume),
        "--per_device_train_batch_size", str(args.micro_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--gradient_checkpointing", "false",
        "--logging_steps", "1",
        "--save_strategy", "steps",
        "--save_steps", str(save_interval),
        "--save_total_limit", "1",
        "--save_only_model", "false",
        "--dataloader_num_workers", "0",
        "--dataloader_pin_memory", "false",
        "--dataset_num_proc", "1",
        "--lazy_tokenize", "false",
        "--seed", "42",
        "--data_seed", "42",
        "--optim", "adamw_torch",
        "--attn_impl", "sdpa",
        "--bf16", "true",
        "--report_to", "none",
        "--resume_from_checkpoint", str(checkpoint),
    ]

    class AuditedResumeSwiftSft(SwiftSft):
        def train(self, trainer):
            wrapper = trainability_audit.find_unique_module(
                trainer.model,
                "HrmTextAudioForConditionalGeneration",
            )
            wrapper.config.use_cache = False
            if hasattr(trainer.model, "gradient_checkpointing_disable"):
                trainer.model.gradient_checkpointing_disable()
            trainer.add_callback(ResumeBoundaryCallback())
            started = time.perf_counter()
            train_result = super().train(trainer)
            elapsed = time.perf_counter() - started
            if not boundary_capture:
                raise RuntimeError("Resume boundary callback was never invoked")
            if int(trainer.state.global_step) != args.step_after_resume:
                raise RuntimeError(
                    f"Resumed Trainer final global_step mismatch: "
                    f"expected={args.step_after_resume} actual={trainer.state.global_step}"
                )
            resumed_losses = {
                int(item["step"]): float(item["loss"])
                for item in trainer.state.log_history
                if isinstance(item, dict)
                and "step" in item
                and "loss" in item
                and int(item["step"]) > args.step_before_resume
                and math.isfinite(float(item["loss"]))
            }
            expected_resumed_steps = set(range(args.step_before_resume + 1, args.step_after_resume + 1))
            if set(resumed_losses) != expected_resumed_steps:
                raise RuntimeError(
                    f"Resumed loss-step coverage mismatch: expected={sorted(expected_resumed_steps)} "
                    f"actual={sorted(resumed_losses)}"
                )
            runtime_report = {
                "status": "OK",
                "packages": versions,
                "checkpoint": str(checkpoint),
                "output_dir": str(output_dir),
                "start_global_step": args.step_before_resume,
                "final_global_step": int(trainer.state.global_step),
                "elapsed_seconds": elapsed,
                "resumed_losses": {str(key): value for key, value in sorted(resumed_losses.items())},
                "log_history": trainer.state.log_history,
                "boundary_report": str(boundary_report_path),
                "argv": argv,
                "train_result_type": f"{type(train_result).__module__}.{type(train_result).__name__}",
            }
            atomic_write_json(runtime_report_path, runtime_report)
            print("========== HRM AUDIO RESUMED TRAINING AUDIT ==========")
            print(
                f"[resume-runtime] start_step={args.step_before_resume} "
                f"final_step={trainer.state.global_step} elapsed_seconds={elapsed:.3f}"
            )
            print(f"[resume-runtime] losses={json.dumps(runtime_report['resumed_losses'])}")
            print(f"[resume-runtime] status=OK report={runtime_report_path}")
            return train_result

    print("========== HRM AUDIO FRESH-PROCESS SWIFT RESUME ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[packages] {versions}")
    print(f"[resume] checkpoint={checkpoint} step={args.step_before_resume}->{args.step_after_resume}")
    print("[framework] swift.pipelines.train.sft.SwiftSft + swift Seq2SeqTrainer")
    print("[argv] " + " ".join(argv))
    AuditedResumeSwiftSft(argv).main()


if __name__ == "__main__":
    main()
