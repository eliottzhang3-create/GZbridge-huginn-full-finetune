#!/usr/bin/env python3
"""Run BAT Stage-I/II/III as one ordered, continuous Swift training job."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

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
    parser.add_argument(
        "--max-sequence-length", type=int, default=176,
        help="Fixed full Ouro sequence width during training, including the 64 audio prefix tokens.",
    )
    parser.add_argument(
        "--torch-compile", action=argparse.BooleanOptionalAction, default=False,
        help="Compile only OuroForCausalLM.model with the validated static-shape DDP configuration.",
    )
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--compile-dynamic", action=argparse.BooleanOptionalAction, default=False,
        help="Must remain false for the fixed-width production graph; enabled only for explicit experiments.",
    )
    parser.add_argument(
        "--compile-audit-output", type=Path, default=None,
        help="Optional private JSON report for compile counters, step speed, and first-batch contracts.",
    )
    parser.add_argument("--logging-steps", type=int, default=100)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


class CompileTrainingAuditCallback(TrainerCallback):
    """Small opt-in callback used by the compile fresh/resume smoke."""

    def __init__(self, output: Path, compile_report: dict[str, Any], audit_state: dict[str, Any]):
        self.output = output.resolve()
        self.compile_report = compile_report
        self.audit_state = audit_state
        self.step_started: float | None = None
        self.step_wall_seconds: list[dict[str, Any]] = []

    def on_step_begin(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        self.step_started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        if self.step_started is not None:
            self.step_wall_seconds.append({
                "global_step": int(state.global_step),
                "wall_seconds": time.perf_counter() - self.step_started,
            })
        self.step_started = None

    def on_train_end(self, args, state, control, **kwargs):
        del control, kwargs
        from bat.ouro_compile import dynamo_counter_summary

        report = {
            "status": "ok" if not self.audit_state.get("issues") else "incomplete",
            "compile": dict(self.compile_report),
            "dynamo_counters": dynamo_counter_summary(),
            "global_step": int(state.global_step),
            "step_wall_seconds": self.step_wall_seconds,
            "first_step_wall_seconds": (
                self.step_wall_seconds[0]["wall_seconds"] if self.step_wall_seconds else None
            ),
            "steady_state_step_wall_seconds": self.step_wall_seconds[1:],
            "batch_contract": self.audit_state,
            "effective_output_dir": str(Path(args.output_dir).resolve()),
        }
        if rank() == 0:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output.with_name(self.output.name + ".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.output)
            print(f"[compile-audit] report={self.output}", flush=True)


def main() -> None:
    args = parse_args()
    BAT_TRAINING.validate()
    if args.max_sequence_length != 176:
        raise ValueError(
            "The current BAT production contract is fixed at 176 full tokens; "
            f"got --max-sequence-length={args.max_sequence_length}"
        )
    if args.compile_dynamic:
        raise ValueError(
            "Production curriculum compile requires --no-compile-dynamic; "
            "dynamic=True is not allowed after the validated static-shape smoke."
        )
    if args.logging_steps <= 0:
        raise ValueError("--logging-steps must be positive")
    os.environ["BAT_MAX_SEQUENCE_LENGTH"] = str(args.max_sequence_length)
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
        try:
            checkpoint.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(
                "Resume checkpoint must be inside --output-dir so it belongs to this curriculum run: "
                f"checkpoint={checkpoint} output={output_root}"
            ) from exc

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
            # Swift may materialize a run-specific directory such as
            # ``output_dir/v0-<timestamp>``.  TrainerArguments contains that
            # effective directory; the outer CLI path is only the requested
            # parent directory and must not be used for checkpoint inspection.
            effective_output_dir = Path(trainer.args.output_dir).resolve()
            compile_report: dict[str, Any] = {
                "requested": bool(args.torch_compile),
                "enabled": False,
                "mode": args.compile_mode,
                "dynamic": bool(args.compile_dynamic),
                "static_sequence_length": args.max_sequence_length,
            }
            if args.torch_compile:
                from bat.ouro_compile import (
                    compile_ouro_transformer_core,
                    find_ouro_causal_model,
                    prepare_compile_runtime,
                )

                runtime_report = prepare_compile_runtime()
                compile_started = time.perf_counter()
                causal = find_ouro_causal_model(trainer.model)
                compiled_core, target_report = compile_ouro_transformer_core(
                    causal, mode=args.compile_mode, dynamic=False
                )
                compile_report.update({
                    "enabled": True,
                    **runtime_report,
                    **target_report,
                    "wrapper_class": type(compiled_core).__name__,
                    "setup_seconds": time.perf_counter() - compile_started,
                })
                trainer._ouro_compile_report = compile_report
                if rank() == 0:
                    print(f"[compile] {json.dumps(compile_report, ensure_ascii=False)}", flush=True)

            audit_state: dict[str, Any] = {
                "forward_calls_observed": 0,
                "input_ids_shapes": [],
                "labels_shapes": [],
                "attention_mask_shapes": [],
                "audio_waveforms_shapes": [],
                "audio_prefix_label_ignore_count": 0,
                "padding_label_violation_count": 0,
                "attention_padding_positions": 0,
                "issues": [],
            }
            if args.compile_audit_output is not None:
                original_compute_loss = trainer.compute_loss

                def audited_compute_loss(model, inputs, *compute_args, **compute_kwargs):
                    input_ids = inputs.get("input_ids")
                    labels = inputs.get("labels")
                    attention_mask = inputs.get("attention_mask")
                    waveforms = inputs.get("audio_waveforms")
                    if hasattr(input_ids, "shape"):
                        audit_state["forward_calls_observed"] += 1
                        audit_state["input_ids_shapes"].append(list(input_ids.shape))
                        if tuple(input_ids.shape[-1:]) != (args.max_sequence_length,):
                            audit_state["issues"].append(
                                f"input_width={tuple(input_ids.shape)} expected={args.max_sequence_length}"
                            )
                    if hasattr(labels, "shape"):
                        audit_state["labels_shapes"].append(list(labels.shape))
                        if input_ids is not None and tuple(labels.shape) != tuple(input_ids.shape):
                            audit_state["issues"].append(
                                f"labels_shape={tuple(labels.shape)} input_ids_shape={tuple(input_ids.shape)}"
                            )
                        if labels.shape[-1] >= 64:
                            prefix_ignored = int((labels[:, :64] == -100).sum().item())
                            audit_state["audio_prefix_label_ignore_count"] += prefix_ignored
                            expected_prefix = int(labels.shape[0] * 64)
                            if prefix_ignored != expected_prefix:
                                audit_state["issues"].append(
                                    f"audio_prefix_labels={prefix_ignored} expected={expected_prefix}"
                                )
                    if hasattr(attention_mask, "shape"):
                        audit_state["attention_mask_shapes"].append(list(attention_mask.shape))
                        if input_ids is not None and tuple(attention_mask.shape) != tuple(input_ids.shape):
                            audit_state["issues"].append(
                                f"attention_mask_shape={tuple(attention_mask.shape)} input_ids_shape={tuple(input_ids.shape)}"
                            )
                        if labels is not None:
                            padding = attention_mask == 0
                            audit_state["attention_padding_positions"] += int(padding.sum().item())
                            violations = int(((labels != -100) & padding).sum().item())
                            audit_state["padding_label_violation_count"] += violations
                            if violations:
                                audit_state["issues"].append(
                                    f"padding_labels_not_ignored={violations}"
                                )
                    if hasattr(waveforms, "shape"):
                        audit_state["audio_waveforms_shapes"].append(list(waveforms.shape))
                        if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (2, 320000):
                            audit_state["issues"].append(
                                f"audio_waveforms_shape={tuple(waveforms.shape)} expected=[B,2,320000]"
                            )
                    return original_compute_loss(model, inputs, *compute_args, **compute_kwargs)

                # An instance attribute is intentionally used here: Trainer
                # calls it with (model, inputs, ...), and no descriptor rebinding
                # is needed for this audit wrapper.
                trainer.compute_loss = audited_compute_loss
                trainer.add_callback(CompileTrainingAuditCallback(
                    args.compile_audit_output, compile_report, audit_state
                ))

            callback = CurriculumBoundaryCheckpointCallback(
                args.curriculum_report,
                global_batch_size,
                checkpoint_root=effective_output_dir,
                resume_checkpoint=args.resume_from_checkpoint,
            )
            trainer.add_callback(callback)
            result = super().train(trainer)
            # Re-read after Swift returns as an additional guard against a
            # pipeline that finalizes or rewrites its versioned run directory
            # during setup/training.
            effective_output_dir = Path(trainer.args.output_dir).resolve()
            missing = callback.missing_boundary_steps()
            if missing:
                raise RuntimeError(f"Missing curriculum boundary checkpoints: {missing}")
            if args.compile_audit_output is not None and audit_state["issues"]:
                raise RuntimeError(f"Compile training audit failed: {audit_state['issues']}")
            if rank() == 0:
                for step, stage in sorted(callback.step_to_stage.items()):
                    marker = callback.marker_paths.get(step)
                    if marker is None or not marker.is_file():
                        expected = effective_output_dir / f"checkpoint-{step}" / "curriculum_stage.json"
                        raise RuntimeError(f"Missing curriculum marker for Stage-{stage}: {expected}")
                    current_marker = effective_output_dir / f"checkpoint-{step}" / "curriculum_stage.json"
                    location = "current" if marker.resolve() == current_marker.resolve() else "resume-source"
                    print(
                        f"[checkpoint] stage={stage} global_step={step} path={marker.parent} location={location}",
                        flush=True,
                    )
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
        "--gradient_checkpointing", "false", "--logging_steps", str(args.logging_steps),
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
