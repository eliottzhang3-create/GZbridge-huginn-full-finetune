#!/usr/bin/env python3
"""Short DDP smoke for the current Stage-III A+B -> C+D+E implementation.

The smoke uses the real lazy AudioSet/RIR -> Spatial-AST -> Q-Former -> Ouro
path with Stage-III's per-device batch size of eight.  It audits the actual
Spatial-AST output dtype, recurrent execution, shifted CE, frozen/trainable
parameters and a resumable distributed checkpoint.  The remote wrapper runs
this once to global step 1 and once more from that checkpoint to step 2.
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
from smoke_bat_ouro_ddp import (
    EXPECTED_OURO_HIDDEN_SIZE,
    EXPECTED_RECURRENT_STEPS,
    audit_audio_batch,
    barrier,
    checkpoint_path,
    checkpoint_report_for_step,
    find_active_qformer,
    find_module,
    gather_reports,
    gradient_audit,
    read_checkpoint_step,
    world_size,
)
from smoke_bat_ouro_lora import (
    TARGET_MODULES,
    lora_report,
    optimizer_report,
    package_version,
    parameter_report,
    parameter_group_name,
    require_environment,
    shape_tuple,
)


MODEL_TYPE = "ouro_bat_spatial_ast"
TEMPLATE_TYPE = "ouro_bat_audio_prefix"
EXPECTED_WORLD_SIZE = 8
EXPECTED_LOCAL_BATCH = 8
EXPECTED_GLOBAL_BATCH = 64
EXPECTED_DATASET_RECORDS = 128
EXPECTED_SPATIAL_AST_DTYPE = "torch.float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=EXPECTED_DATASET_RECORDS)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def add_issue(trace: dict[str, Any], message: str) -> None:
    issues = trace.setdefault("issues", [])
    if message not in issues:
        issues.append(message)


def module_parameter_audit(module: torch.nn.Module) -> dict[str, Any]:
    parameters = list(module.named_parameters())
    trainable = [name for name, parameter in parameters if parameter.requires_grad]
    return {
        "parameter_count": sum(int(parameter.numel()) for _, parameter in parameters),
        "trainable_parameter_count": sum(int(parameter.numel()) for _, parameter in parameters if parameter.requires_grad),
        "parameter_tensor_count": len(parameters),
        "trainable_tensor_count": len(trainable),
        "trainable_name_preview": trainable[:10],
        "dtype_set": sorted({str(parameter.dtype) for _, parameter in parameters}),
    }


def main() -> None:
    os.environ["BAT_AUDIO_AUDIT"] = "1"
    args = parse_args()
    require_environment()
    BAT_TRAINING.validate()

    current_rank = rank()
    current_world = world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(current_rank)))
    if current_world != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"Stage-III resume smoke requires WORLD_SIZE=8, got {current_world}")
    if not 0 <= local_rank < current_world:
        raise RuntimeError(f"Invalid LOCAL_RANK={local_rank}")
    torch.cuda.set_device(local_rank)

    for path in (args.model_path, args.plugin_path, args.dataset):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    if args.max_steps <= 0 or args.save_steps <= 0:
        raise ValueError("--max-steps and --save-steps must be positive")
    expected_records = int(args.expected_records)
    actual_records = count_jsonl(args.dataset)
    if actual_records != expected_records:
        raise RuntimeError(f"Expected {expected_records} records, found {actual_records}: {args.dataset}")
    if str(args.output_report).replace("\\", "/").startswith("/hpc_stor03/public"):
        raise ValueError(f"Refusing public output report: {args.output_report}")
    resume_checkpoint = args.resume_from_checkpoint.expanduser().resolve() if args.resume_from_checkpoint else None
    initial_step = 0
    if resume_checkpoint is not None:
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(resume_checkpoint)
        initial_step = read_checkpoint_step(resume_checkpoint)
        if initial_step < 0 or initial_step >= args.max_steps:
            raise RuntimeError(f"Invalid resume step={initial_step} for target={args.max_steps}")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty fresh smoke output: {args.output_dir}")
    expected_optimizer_steps = args.max_steps - initial_step
    if expected_optimizer_steps <= 0:
        raise RuntimeError("Resume phase has no remaining optimizer steps")

    from swift.pipelines.train.sft import SwiftSft

    # One warmup step is enough to exercise scheduler restore in the tiny
    # smoke; the formal route's 13% warmup is validated by its own report.
    warmup_steps = min(1, args.max_steps)
    argv: list[str] = [
        "--model", str(args.model_path), "--model_type", MODEL_TYPE, "--template", TEMPLATE_TYPE,
        "--external_plugins", str(args.plugin_path), "--dataset", str(args.dataset),
        "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false",
        "--sortish_sampler", "false", "--group_by_length", "false", "--max_length", "512",
        "--remove_unused_columns", "false", "--output_dir", str(args.output_dir),
        "--tuner_type", "lora", "--tuner_backend", "peft", "--target_modules", *TARGET_MODULES,
        "--modules_to_save", "audio_qformer", "--freeze_llm", "true", "--freeze_vit", "true",
        "--freeze_aligner", "false", "--lora_rank", str(BAT_TRAINING.lora_rank),
        "--lora_alpha", str(BAT_TRAINING.lora_alpha), "--lora_dropout", str(BAT_TRAINING.lora_dropout),
        "--learning_rate", "0.002", "--lr_scheduler_type", "cosine",
        "--warmup_steps", str(warmup_steps), "--max_steps", str(args.max_steps), "--num_train_epochs", "1",
        "--per_device_train_batch_size", str(EXPECTED_LOCAL_BATCH), "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing", "false", "--logging_steps", "1", "--save_strategy", "steps",
        "--save_steps", str(args.save_steps), "--save_total_limit", "2", "--save_only_model", "false",
        "--dataloader_num_workers", "4", "--dataloader_pin_memory", "true", "--dataloader_drop_last", "false",
        "--dataset_num_proc", "1", "--lazy_tokenize", "true", "--load_from_cache_file", "false",
        "--loss_scale", "all", "--seed", "42", "--data_seed", "42", "--optim", "adamw_torch",
        "--adam_beta1", str(BAT_TRAINING.beta1), "--adam_beta2", str(BAT_TRAINING.beta2),
        "--weight_decay", str(BAT_TRAINING.weight_decay), "--attn_impl", "sdpa", "--bf16", "true",
        "--ddp_find_unused_parameters", "false", "--average_tokens_across_devices", "false",
        "--report_to", "none",
    ]
    if resume_checkpoint is not None:
        argv.extend(["--resume_from_checkpoint", str(resume_checkpoint)])

    class AuditedStage3ResumeSwiftSft(SwiftSft):
        def train(self, trainer):
            model = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer.accelerator, "unwrap_model") else trainer.model
            causal = find_module(model, "OuroForCausalLM")
            ouro = find_module(model, "OuroModel")
            audio_encoder = find_module(model, "SpatialASTAudioEncoder")
            qformer = find_active_qformer(model)
            trace: dict[str, Any] = {
                "rank": current_rank,
                "local_rank": local_rank,
                "world_size": current_world,
                "forward": 0,
                "backward": 0,
                "layer_forward": 0,
                "gate_forward": 0,
                "layer_backward": 0,
                "gate_backward": 0,
                "audio_encoder_forward": 0,
                "audio_encoder_input_shape": None,
                "audio_encoder_output_shape": None,
                "audio_encoder_output_dtype": None,
                "qformer_forward": 0,
                "qformer_input_shape": None,
                "qformer_output_shape": None,
                "qformer_output_dtype": None,
                "batch": None,
                "audio_batch": None,
                "prefix_audit": None,
                "gradient_audit": None,
                "spatial_ast_parameter_audit": module_parameter_audit(audio_encoder),
                "gate_parameter_audit": module_parameter_audit(ouro.early_exit_gate),
                "past_key_values_present": None,
                "issues": [],
            }
            if trace["spatial_ast_parameter_audit"]["trainable_tensor_count"] != 0:
                add_issue(trace, "Spatial-AST_has_trainable_parameters")
            if trace["gate_parameter_audit"]["trainable_tensor_count"] != 0:
                add_issue(trace, "early_exit_gate_has_trainable_parameters")
            if int(getattr(causal.config, "total_ut_steps", -1)) != EXPECTED_RECURRENT_STEPS:
                add_issue(trace, "total_ut_steps_is_not_4")
            if float(getattr(causal, "early_exit_threshold", -1.0)) != 1.0:
                add_issue(trace, "early_exit_threshold_is_not_1")
            causal.config.use_cache = False
            ouro.config.use_cache = False
            if hasattr(getattr(causal, "generation_config", None), "use_cache"):
                causal.generation_config.use_cache = False
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.train()

            handles: list[Any] = []
            first_layer = ouro.layers[0]
            handles.append(first_layer.register_forward_hook(lambda *_: trace.__setitem__("layer_forward", trace["layer_forward"] + 1)))
            handles.append(ouro.early_exit_gate.register_forward_hook(lambda *_: trace.__setitem__("gate_forward", trace["gate_forward"] + 1)))
            handles.append(first_layer.register_full_backward_hook(lambda *_: trace.__setitem__("layer_backward", trace["layer_backward"] + 1)))
            handles.append(ouro.early_exit_gate.register_full_backward_hook(lambda *_: trace.__setitem__("gate_backward", trace["gate_backward"] + 1)))

            def audio_hook(_module, hook_inputs, hook_output):
                trace["audio_encoder_forward"] += 1
                if torch.is_tensor(hook_inputs[0]):
                    trace["audio_encoder_input_shape"] = list(hook_inputs[0].shape)
                if torch.is_tensor(hook_output):
                    trace["audio_encoder_output_shape"] = list(hook_output.shape)
                    trace["audio_encoder_output_dtype"] = str(hook_output.dtype)
                else:
                    add_issue(trace, f"spatial_ast_hook_output_not_tensor:{type(hook_output).__name__}")

            def qformer_hook(_module, hook_inputs, hook_output):
                trace["qformer_forward"] += 1
                if hook_inputs and torch.is_tensor(hook_inputs[0]):
                    trace["qformer_input_shape"] = list(hook_inputs[0].shape)
                if torch.is_tensor(hook_output):
                    trace["qformer_output_shape"] = list(hook_output.shape)
                    trace["qformer_output_dtype"] = str(hook_output.dtype)

            handles.append(audio_encoder.register_forward_hook(audio_hook))
            handles.append(qformer.register_forward_hook(qformer_hook))
            original_compute_loss = trainer.compute_loss
            original_backward = trainer.accelerator.backward

            def compute_loss(actual_model, inputs, return_outputs=False, num_items_in_batch=None):
                trace["forward"] += 1
                labels_before = inputs["labels"].detach().clone()
                audio_before = inputs.get("audio_waveforms")
                records_before = inputs.get("bat_audio_records")
                loss_scale_before = inputs.get("loss_scale")
                if torch.is_tensor(loss_scale_before):
                    loss_scale_before = loss_scale_before.detach().clone()
                result = original_compute_loss(actual_model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                loss, outputs = result
                if trace["batch"] is None:
                    logits = outputs.logits
                    labels = labels_before
                    input_ids = inputs.get("input_ids")
                    attention_mask = inputs.get("attention_mask")
                    try:
                        if not (torch.is_tensor(input_ids) and torch.is_tensor(attention_mask)):
                            raise RuntimeError("missing input_ids/attention_mask")
                        if shape_tuple(input_ids) != shape_tuple(labels) or shape_tuple(attention_mask) != shape_tuple(input_ids):
                            raise RuntimeError(f"input/label/attention shapes: {shape_tuple(input_ids)}/{shape_tuple(labels)}/{shape_tuple(attention_mask)}")
                        if logits.ndim != 3 or logits.shape[:2] != labels.shape:
                            raise RuntimeError(f"logits/labels shape mismatch: {tuple(logits.shape)}/{tuple(labels.shape)}")
                        if getattr(outputs, "past_key_values", None) is not None:
                            trace["past_key_values_present"] = True
                            raise RuntimeError("KV cache is enabled during training")
                        trace["past_key_values_present"] = False
                        shifted_logits = logits.float()[:, :-1].contiguous()
                        shifted_labels = labels[:, 1:].contiguous()
                        valid = shifted_labels.reshape(-1) != -100
                        manual_losses = F.cross_entropy(
                            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                            shifted_labels.reshape(-1), ignore_index=-100, reduction="none",
                        )
                        manual_count = int(valid.sum().item())
                        if manual_count <= 0:
                            raise RuntimeError("no valid shifted targets")
                        manual_sum = manual_losses[valid].sum()
                        manual_ce = float((manual_sum / manual_count).detach().cpu())
                        swift_labels = torch.roll(labels, shifts=-1, dims=-1).reshape(-1)
                        swift_losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), swift_labels, ignore_index=-100, reduction="none")
                        swift_valid = swift_labels != -100
                        if torch.is_tensor(loss_scale_before):
                            swift_scale = torch.roll(loss_scale_before, shifts=-1, dims=-1).reshape(-1).to(swift_losses.dtype)
                            swift_losses = swift_losses * swift_scale
                            binary_equivalent = bool(torch.equal(swift_scale, swift_valid.to(swift_scale.dtype)))
                        else:
                            binary_equivalent = True
                        denominator = int(num_items_in_batch.detach().cpu().item()) if torch.is_tensor(num_items_in_batch) else int(num_items_in_batch or manual_count)
                        swift_ce = float((swift_losses.sum() / denominator).detach().cpu())
                        trainer_ce = float(loss.detach().float().cpu())
                        if not math.isclose(swift_ce, trainer_ce, rel_tol=2e-3, abs_tol=2e-3):
                            raise RuntimeError(f"Swift CE mismatch trainer={trainer_ce} reproduced={swift_ce}")
                        if not binary_equivalent:
                            raise RuntimeError("loss_scale is not equivalent to shifted -100 masking")
                        if input_ids.shape[0] != EXPECTED_LOCAL_BATCH:
                            raise RuntimeError(f"local batch is {input_ids.shape[0]}, expected {EXPECTED_LOCAL_BATCH}")
                        if not bool((labels[:, :64] == -100).all().item()):
                            raise RuntimeError("audio prefix labels are not fully masked")
                        if not bool(torch.isfinite(logits.float()).all().item()) or not bool(torch.isfinite(loss.float()).all().item()):
                            raise RuntimeError("non-finite logits or loss")
                        trace["audio_batch"] = audit_audio_batch(model, {"audio_waveforms": audio_before, "bat_audio_records": records_before})
                        prefix = getattr(causal, "_ouro_bat_last_audio_forward_audit", None)
                        trace["prefix_audit"] = prefix
                        if not isinstance(prefix, dict) or not prefix.get("audio_prefix_replaced"):
                            raise RuntimeError("audio prefix replacement audit missing")
                        embedding_shape = tuple(int(value) for value in (prefix.get("inputs_embeds_shape") or ()))
                        if embedding_shape[:2] != tuple(input_ids.shape) or embedding_shape != (*tuple(input_ids.shape), EXPECTED_OURO_HIDDEN_SIZE):
                            raise RuntimeError(f"audio/text embedding shape mismatch: input={tuple(input_ids.shape)} embedding={embedding_shape}")
                        if trace["audio_encoder_input_shape"] != [EXPECTED_LOCAL_BATCH, 2, 320000]:
                            raise RuntimeError(f"Spatial-AST input shape={trace['audio_encoder_input_shape']}")
                        if trace["audio_encoder_output_shape"] != [EXPECTED_LOCAL_BATCH, 515, 768]:
                            raise RuntimeError(f"Spatial-AST output shape={trace['audio_encoder_output_shape']}")
                        if trace["audio_encoder_output_dtype"] != EXPECTED_SPATIAL_AST_DTYPE:
                            raise RuntimeError(
                                f"Spatial-AST output dtype={trace['audio_encoder_output_dtype']} expected={EXPECTED_SPATIAL_AST_DTYPE}"
                            )
                        if trace["qformer_input_shape"] != [EXPECTED_LOCAL_BATCH, 515, 768] or trace["qformer_output_shape"] != [EXPECTED_LOCAL_BATCH, 64, 2048]:
                            raise RuntimeError(f"Q-Former shapes input={trace['qformer_input_shape']} output={trace['qformer_output_shape']}")
                        trace["batch"] = {
                            "input_ids_shape": list(input_ids.shape),
                            "labels_shape": list(labels.shape),
                            "attention_mask_shape": list(attention_mask.shape),
                            "audio_waveforms_shape": list(audio_before.shape) if torch.is_tensor(audio_before) else None,
                            "valid_shifted_target_count": manual_count,
                            "manual_shifted_ce": manual_ce,
                            "swift_reproduced_ce": swift_ce,
                            "trainer_ce": trainer_ce,
                            "loss_scale_binary_equivalent": binary_equivalent,
                            "shift_verified": True,
                        }
                    except Exception as exc:
                        add_issue(trace, f"forward_audit:{type(exc).__name__}: {exc}")
                return result if return_outputs else loss

            def backward(loss, **kwargs):
                trace["backward"] += 1
                result = original_backward(loss, **kwargs)
                if trace["gradient_audit"] is None:
                    try:
                        trace["gradient_audit"] = gradient_audit(model)
                    except Exception as exc:
                        add_issue(trace, f"gradient_audit:{type(exc).__name__}: {exc}")
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

            if trace["audio_encoder_output_dtype"] != EXPECTED_SPATIAL_AST_DTYPE:
                add_issue(trace, "spatial_ast_not_fp32")
            if trace["forward"] != expected_optimizer_steps or trace["backward"] != expected_optimizer_steps:
                add_issue(trace, f"optimizer_step_count={trace['forward']}/{trace['backward']} expected={expected_optimizer_steps}")
            if trace["layer_forward"] != EXPECTED_RECURRENT_STEPS * expected_optimizer_steps:
                add_issue(trace, f"recurrent_layer_forward={trace['layer_forward']}")
            if trace["gate_forward"] != EXPECTED_RECURRENT_STEPS * expected_optimizer_steps:
                add_issue(trace, f"recurrent_gate_forward={trace['gate_forward']}")
            if trace["layer_backward"] != EXPECTED_RECURRENT_STEPS * expected_optimizer_steps:
                add_issue(trace, f"recurrent_layer_backward={trace['layer_backward']}")
            if trace["gate_backward"] <= 0:
                add_issue(trace, "gate_backward_not_observed")

            local_report = {
                "rank": current_rank,
                "parameters": parameter_report(model),
                "lora": lora_report(model),
                "optimizer": optimizer_report(trainer, model),
                "forward_audit": trace,
                "global_step": int(trainer.state.global_step),
                "elapsed_seconds": time.perf_counter() - started,
                "memory": {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                },
            }
            reports = gather_reports(local_report)
            barrier()

            checkpoint: dict[str, Any] | None = None
            rank0_error: str | None = None
            if current_rank == 0:
                combined_issues = [item for report_item in reports for item in report_item["forward_audit"].get("issues", [])]
                try:
                    if sorted(int(item["rank"]) for item in reports) != list(range(EXPECTED_WORLD_SIZE)):
                        combined_issues.append("rank_reports_incomplete")
                    if any(int(item["global_step"]) != args.max_steps for item in reports):
                        combined_issues.append("global_step_mismatch_across_ranks")
                    checkpoint = checkpoint_report_for_step(
                        checkpoint_path(args.output_dir, args.max_steps), args.max_steps, expected_world_size=EXPECTED_WORLD_SIZE
                    )
                except Exception as exc:
                    rank0_error = f"checkpoint_audit:{type(exc).__name__}: {exc}"
                    combined_issues.append(rank0_error)
                status = "ok" if not combined_issues else "incomplete"
                output = {
                    "status": status,
                    "phase": "resume" if resume_checkpoint is not None else "fresh",
                    "dataset": str(args.dataset),
                    "dataset_records": actual_records,
                    "distributed": {
                        "backend": dist.get_backend(),
                        "world_size": EXPECTED_WORLD_SIZE,
                        "per_device_batch_size": EXPECTED_LOCAL_BATCH,
                        "global_batch_size": EXPECTED_GLOBAL_BATCH,
                        "initial_global_step": initial_step,
                        "target_global_step": args.max_steps,
                        "optimizer_steps": expected_optimizer_steps,
                        "resumed_from_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
                        "rank_peak_allocated_bytes": [int(item["memory"]["peak_allocated_bytes"]) for item in reports],
                        "rank_peak_reserved_bytes": [int(item["memory"]["peak_reserved_bytes"]) for item in reports],
                        "rank_reports": reports,
                    },
                    "contract": {
                        "total_ut_steps": EXPECTED_RECURRENT_STEPS,
                        "early_exit_threshold": 1.0,
                        "use_cache": False,
                        "spatial_ast_expected_output_dtype": EXPECTED_SPATIAL_AST_DTYPE,
                        "spatial_ast_policy": "strict audit; current code is not modified by this smoke",
                        "learning_rate": 0.002,
                        "lora_target_modules": list(TARGET_MODULES),
                        "lora_rank": BAT_TRAINING.lora_rank,
                        "qformer_random_initialization": True,
                    },
                    "checkpoint": checkpoint,
                    "issues": sorted(set(combined_issues)),
                    "argv": argv,
                    "packages": {name: package_version(name) for name in ("ms-swift", "transformers", "peft", "accelerate")},
                }
                if status != "ok" and rank0_error is None:
                    rank0_error = f"audit_status_incomplete: {sorted(set(combined_issues))}"
                write_json(args.output_report, output)
                print(f"[phase] {output['phase']} initial_step={initial_step} target_step={args.max_steps}", flush=True)
                print(f"[spatial-ast] dtype={trace['audio_encoder_output_dtype']} expected={EXPECTED_SPATIAL_AST_DTYPE}", flush=True)
                print(f"[checkpoint] {json.dumps(checkpoint, ensure_ascii=False)}", flush=True)
                print(f"[report] {args.output_report}", flush=True)
                print(f"[status] {status} issues={sorted(set(combined_issues))}", flush=True)
            errors: list[str | None] = [None] * EXPECTED_WORLD_SIZE
            dist.all_gather_object(errors, rank0_error)
            if any(error is not None for error in errors):
                raise RuntimeError(f"Stage-III resume smoke failed: {errors}")
            barrier()
            if current_rank == 0 and Path(args.output_report).is_file():
                saved = json.loads(Path(args.output_report).read_text(encoding="utf-8"))
                if saved.get("status") != "ok":
                    raise RuntimeError(f"Stage-III resume smoke audit failed: {saved.get('issues')}")
            barrier()
            return result

    print("========== BAT OURO STAGE-III SPATIAL-AST/CHECKPOINT/RESUME SMOKE ==========")
    print(f"[ddp] world_size={current_world} local_batch={EXPECTED_LOCAL_BATCH} global_batch={EXPECTED_GLOBAL_BATCH}")
    print(f"[phase] {'resume' if resume_checkpoint else 'fresh'} initial_step={initial_step} target_step={args.max_steps}")
    if current_rank == 0:
        print(f"[argv] {' '.join(argv)}")
    AuditedStage3ResumeSwiftSft(argv).main()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
