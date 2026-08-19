#!/usr/bin/env python3
"""Full BAT multimodal DDP test with synthetic fixed waveforms.

This intentionally bypasses JSONL, HuggingFace Datasets/Arrow, AudioSet,
RIR resolution, soundfile, and SciPy convolution.  It still loads the real
registered Ouro/Qwen3 BAT model, frozen Spatial-AST, random trainable
Q-Former, PEFT LoRA, and executes real DDP forward/backward/all-reduce.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model

from bat_diagnostics import require_private_absolute, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("ouro", "qwen3"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=176)
    return parser.parse_args()


def import_plugin(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"bat_fixed_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def group_name(name: str) -> str:
    if "lora_A" in name or "lora_B" in name:
        return "lora"
    if "audio_qformer" in name:
        return "qformer"
    if "spatial_ast_encoder" in name:
        return "spatial_ast"
    if any(value in name for value in ("model.", "lm_head.", "embed_tokens.")):
        return "native"
    return "other"


def trainable_audit(model: torch.nn.Module) -> dict[str, Any]:
    counts = {key: 0 for key in ("lora", "qformer", "spatial_ast", "native", "other")}
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            counts[group_name(name)] += parameter.numel()
            names.append(name)
    if counts["lora"] <= 0 or counts["qformer"] <= 0:
        raise RuntimeError(f"Fixed waveform test did not enable LoRA and Q-Former: {counts}")
    if any(counts[key] for key in ("spatial_ast", "native", "other")):
        raise RuntimeError(f"Unexpected trainable group in fixed waveform test: {counts}")
    return {"trainable_parameter_counts": counts, "trainable_name_count": len(names), "trainable_name_preview": names[:20]}


def main() -> None:
    args = parse_args()
    output = require_private_absolute(args.output_report)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("Launch fixed waveform test with torchrun")
    if args.local_batch_size <= 0 or args.steps <= 0 or args.sequence_length != 176:
        raise ValueError("Expected positive batch/steps and fixed sequence length 176")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    plugin = import_plugin(args.plugin_path.resolve())
    model_type = "ouro_bat_spatial_ast" if args.model_type == "ouro" else "qwen3_bat_spatial_ast"
    if plugin.MODEL_TYPE != model_type:
        raise RuntimeError(f"Plugin/model type mismatch: expected={model_type} actual={plugin.MODEL_TYPE}")
    from swift import get_model_processor

    model, processor = get_model_processor(
        str(args.model_path.resolve()),
        model_type=model_type,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    model.train()
    lora_model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["audio_qformer"],
        ),
    )
    lora_model.to(device)
    audit = trainable_audit(lora_model)
    ddp = torch.nn.parallel.DistributedDataParallel(lora_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has no pad/eos token")
    input_ids = torch.full((args.local_batch_size, args.sequence_length), int(pad_id), device=device, dtype=torch.long)
    text_start = 64
    text_ids = torch.full((args.local_batch_size, args.sequence_length - text_start), int(pad_id), device=device, dtype=torch.long)
    input_ids[:, text_start:] = text_ids
    labels = torch.full_like(input_ids, -100)
    labels[:, text_start + 1 :] = text_ids[:, 1:]
    attention_mask = torch.ones_like(input_ids)
    # The input is deliberately fixed across all steps.  No renderer or
    # filesystem data is touched; this is a pure model/DDP isolation test.
    torch.cuda.manual_seed_all(1234 + rank)
    waveforms = torch.randn(args.local_batch_size, 2, 320000, device=device, dtype=torch.float32) * 0.01
    optimizer = torch.optim.AdamW((parameter for parameter in ddp.parameters() if parameter.requires_grad), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
    times: list[float] = []
    losses: list[float] = []
    try:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            outputs = ddp(input_ids=input_ids, attention_mask=attention_mask, labels=labels, audio_waveforms=waveforms, use_cache=False)
            loss = outputs.loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Non-finite fixed waveform loss at step={step}")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - started)
            losses.append(float(loss.detach().item()))
            print(f"[rank{rank}] step={step + 1}/{args.steps} loss={losses[-1]:.6f} seconds={times[-1]:.3f}", flush=True)
    finally:
        if rank == 0:
            gathered: list[dict[str, Any] | None] = [None] * world_size
        else:
            gathered = []
        local = {"rank": rank, "times": times, "losses": losses, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        dist.gather_object(local, gathered if rank == 0 else None, dst=0)
        dist.barrier()
        if rank == 0:
            report = {
                "status": "ok",
                "model_type": args.model_type,
                "model_path": str(args.model_path.resolve()),
                "world_size": world_size,
                "local_batch_size": args.local_batch_size,
                "global_batch_size": world_size * args.local_batch_size,
                "sequence_length": args.sequence_length,
                "audio_prefix_tokens": 64,
                "renderer_bypassed": True,
                "spatial_ast_executed": True,
                "ddp_backend": "nccl",
                "packages": {name: version(name) for name in ("ms-swift", "transformers", "peft")},
                "trainable_audit": audit,
                "ranks": gathered,
                "rank0_step_p50_seconds": statistics.median(times) if times else None,
            }
            write_json(output, report)
            print(f"[summary] model={args.model_type} world={world_size} p50={report['rank0_step_p50_seconds']}", flush=True)
            print(f"[report] {output}", flush=True)
            print("[status] ok", flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
