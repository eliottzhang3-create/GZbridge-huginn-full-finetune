"""Simple(ish), self-contained finetuning script. Training on GSM8k like in this example will not improve flexible extract,
but the model will quickly learn the format and strict match will rise.

built around minimal train.py variant

Almost all of the credit for this file goes to SeanMcLeish.
"""


####################################################################################################
# Imports.
####################################################################################################

import time

global_start_time = time.time()
import os
import socket
import json

from typing import TYPE_CHECKING, Any, Optional
import sys
import datetime
import shutil

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset, Dataset, load_from_disk
from contextlib import nullcontext
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

USE_LOCAL_CODE = False

# Check device health immediately after loading torch and standard libraries without loading cuda/hip/dist:
nvml_count = torch.cuda._device_count_amdsmi() if torch.version.hip else torch.cuda._device_count_nvml()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}")


if TYPE_CHECKING:
    import torch.distributed
    import torch.version
    import torch._dynamo.config

from dataclasses import dataclass, field


end_time = time.time()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")


@dataclass
class CLISettings:
    run_name: str = "huginn-gsm8k-fsdp-pilot"
    out_path: str = "outputs"
    dataset_location: str = "/hpc_stor03/sjtu_home/jinwei.zhang/code/recurrent-pretraining-main/data/gsm8k_train.jsonl"
    val_dataset_location: str = "/hpc_stor03/sjtu_home/jinwei.zhang/code/recurrent-pretraining-main/data/gsm8k_test.jsonl"
    model_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
    dataset_args: dict[str, Any] = field(default_factory=lambda: dict(q_col="question", a_col="answer"))
    dataset_config: Optional[str] = None
    max_seq_length: int = 128
    max_samples: Optional[int] = 200
    micro_batch_size: int = 1
    compile: bool = False
    max_steps: Optional[int] = 3
    epochs: int = 1
    batch_size: int = 5
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(lr=1e-7, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    )
    scheduler_args: dict[float, Any] = field(default_factory=lambda: dict(warmup=0.1, cooldown=0.1, min_lr_ratio=0.001))
    eval_interval: int = 1_000_000_000
    seed: int = 74
    take_loss_over_all_tokens: bool = False
    max_grad_norm: float = 0.25
    precision: str = "fp16-mixed"
    gradient_checkpointing: bool = False
    save_interval: int = 0
    save_final_checkpoint: bool = False
    use_fsdp: bool = True
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_cpu_offload: bool = False

    def __post_init__(self):
        pass


@dataclass
class Message:
    role: str
    content: str


def is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


def seed_everything(seed):
    import random  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def unwrap_model(model):
    while hasattr(model, "_fsdp_wrapped_module"):
        model = model._fsdp_wrapped_module
    if hasattr(model, "module"):
        model = model.module
    return model


def get_unwrapped_model(state):
    return unwrap_model(state["model"])

def sample_shared_num_steps(state, device):
    base_model = get_unwrapped_model(state)

    if state["distributed"]:
        step_pair = torch.zeros(2, device=device, dtype=torch.long)
        if state["rank"] == 0:
            sampled_no_grad, sampled_with_grad = base_model.randomized_iteration_sampler()
            step_pair[0] = int(sampled_no_grad)
            step_pair[1] = int(sampled_with_grad)
        torch.distributed.broadcast(step_pair, src=0)
    else:
        sampled_no_grad, sampled_with_grad = base_model.randomized_iteration_sampler()
        step_pair = torch.tensor(
            [int(sampled_no_grad), int(sampled_with_grad)],
            device=device,
            dtype=torch.long,
        )

    return step_pair

def save_fsdp_checkpoint(state, cfg, ckpt_name):
    model = state["model"]
    save_dir = f"{cfg.out_path}/{cfg.run_name}/{ckpt_name}"

    if state["distributed"] and cfg.use_fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        if is_main_process():
            unwrapped_model = get_unwrapped_model(state)
            unwrapped_model.save_pretrained(save_dir, state_dict=cpu_state)
            state["tokenizer"].save_pretrained(save_dir)
    else:
        if is_main_process():
            unwrapped_model = get_unwrapped_model(state)
            unwrapped_model.save_pretrained(save_dir)
            state["tokenizer"].save_pretrained(save_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

from collections import deque

def init_debug_history(state, maxlen=6):
    if "debug_history" not in state:
        state["debug_history"] = deque(maxlen=maxlen)

def capture_param_snapshot(model, names):
    out = {}
    for name, p in model.named_parameters():
        if name in names:
            item = {
                "dtype": str(p.dtype),
                "shape": tuple(p.shape),
                "numel": int(p.numel()),
            }

            pdata = p.detach()
            if pdata.numel() == 0:
                item["param_abs_max"] = "empty_shard"
            else:
                item["param_abs_max"] = float(pdata.abs().max().item())

            if p.grad is not None:
                g = p.grad.detach()
                item["grad_numel"] = int(g.numel())
                if g.numel() == 0:
                    item["grad_abs_max"] = "empty_shard"
                    item["grad_finite"] = True
                else:
                    finite = torch.isfinite(g).all()
                    item["grad_finite"] = bool(finite.item())
                    if finite:
                        item["grad_abs_max"] = float(g.abs().max().item())
                    else:
                        item["grad_abs_max"] = "nonfinite"
            else:
                item["grad_numel"] = None
                item["grad_abs_max"] = None
                item["grad_finite"] = None

            out[name] = item
    return out

"""
def append_debug_history(state, model, data_step, optimizer_step, loss, log_ppl, lr, grad_norm=None):
    init_debug_history(state)
    watched = [
        "transformer.wte.weight",
        "transformer.prelude.0.attn.proj.weight",
        "transformer.prelude.0.mlp.proj.weight",
    ]
    state["debug_history"].append({
        "rank": state["rank"],
        "data_step": data_step,
        "optimizer_step": optimizer_step,
        "loss": None if loss is None else float(loss.detach().float().item()),
        "log_ppl": None if log_ppl is None else float(log_ppl.detach().float().item()),
        "lr": float(lr),
        "grad_norm": None if grad_norm is None else float(grad_norm.detach().float().item() if torch.is_tensor(grad_norm) else grad_norm),
        "params": capture_param_snapshot(get_unwrapped_model(state), watched),
    })

def dump_debug_history(state, prefix):
    if "debug_history" not in state:
        print(f"{prefix} no debug history")
        return
    print(f"{prefix} recent_history_begin")
    for item in state["debug_history"]:
        print(f"{prefix} {item}")
    print(f"{prefix} recent_history_end")
"""

def tensor_summary(name, x):
    if x is None:
        return f"{name}=None"
    x = x.detach()
    msg = f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if x.numel() == 0:
        return msg + " empty"
    if x.is_floating_point():
        x32 = x.float()
        finite = torch.isfinite(x32)
        msg += (
            f" min={x32.min().item():.4e}"
            f" max={x32.max().item():.4e}"
            f" mean={x32.mean().item():.4e}"
            f" finite={bool(finite.all().item())}"
        )
        if not finite.all():
            msg += f" nonfinite={(~finite).sum().item()}"
    else:
        msg += f" min={x.min().item()} max={x.max().item()}"
    return msg

def clean_param_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


def _safe_abs_max(x: torch.Tensor):
    if x.numel() == 0:
        return None
    x = x.detach()
    finite = torch.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(x[finite].abs().max().item())


def _safe_norm(x: torch.Tensor):
    if x.numel() == 0:
        return None
    x = x.detach()
    finite = torch.isfinite(x)
    if not finite.all():
        return float("nan")
    return float(x.float().norm().item())


def collect_param_debug_stats(model):
    stats = []

    for name, p in model.named_parameters():
        item = {
            "name": clean_param_name(name),
            "shape": tuple(p.shape),
            "param_dtype": str(p.dtype),
            "param_numel": int(p.numel()),
            "param_nonfinite": 0,
            "param_abs_max": None,
            "param_norm": None,
            "grad_present": p.grad is not None,
            "grad_dtype": None,
            "grad_numel": 0,
            "grad_nonfinite": 0,
            "grad_abs_max": None,
            "grad_norm": None,
        }

        pdata = p.detach()
        if pdata.numel() > 0:
            item["param_nonfinite"] = int((~torch.isfinite(pdata)).sum().item())
            item["param_abs_max"] = _safe_abs_max(pdata.float())
            item["param_norm"] = _safe_norm(pdata.float())

        if p.grad is not None:
            g = p.grad.detach()
            item["grad_dtype"] = str(g.dtype)
            item["grad_numel"] = int(g.numel())
            if g.numel() > 0:
                item["grad_nonfinite"] = int((~torch.isfinite(g)).sum().item())
                item["grad_abs_max"] = _safe_abs_max(g.float())
                item["grad_norm"] = _safe_norm(g.float())

        stats.append(item)

    return stats

def print_param_debug_report(model, rank, tag, topk=12):
    stats = collect_param_debug_stats(model)

    bad_param = [x for x in stats if x["param_nonfinite"] > 0]
    bad_grad = [x for x in stats if x["grad_nonfinite"] > 0]

    grad_candidates = [x for x in stats if x["grad_abs_max"] is not None]
    grad_candidates = sorted(
        grad_candidates,
        key=lambda x: (float("-inf") if x["grad_abs_max"] != x["grad_abs_max"] else x["grad_abs_max"]),
        reverse=True,
    )

    param_candidates = [x for x in stats if x["param_abs_max"] is not None]
    param_candidates = sorted(
        param_candidates,
        key=lambda x: (float("-inf") if x["param_abs_max"] != x["param_abs_max"] else x["param_abs_max"]),
        reverse=True,
    )

    print(
        f"[param-debug][rank={rank}] tag={tag} "
        f"total_params={len(stats)} bad_param_tensors={len(bad_param)} bad_grad_tensors={len(bad_grad)}"
    )

    if bad_param:
        print(f"[param-debug][rank={rank}] tag={tag} bad_param_begin")
        for x in bad_param[:topk]:
            print(
                f"[param-debug][rank={rank}] bad_param "
                f"name={x['name']} shape={x['shape']} dtype={x['param_dtype']} "
                f"nonfinite={x['param_nonfinite']} abs_max={x['param_abs_max']} norm={x['param_norm']}"
            )
        print(f"[param-debug][rank={rank}] tag={tag} bad_param_end")

    if bad_grad:
        print(f"[param-debug][rank={rank}] tag={tag} bad_grad_begin")
        for x in bad_grad[:topk]:
            print(
                f"[param-debug][rank={rank}] bad_grad "
                f"name={x['name']} shape={x['shape']} grad_dtype={x['grad_dtype']} "
                f"nonfinite={x['grad_nonfinite']} grad_abs_max={x['grad_abs_max']} grad_norm={x['grad_norm']}"
            )
        print(f"[param-debug][rank={rank}] tag={tag} bad_grad_end")

    print(f"[param-debug][rank={rank}] tag={tag} top_grad_begin")
    for x in grad_candidates[:topk]:
        print(
            f"[param-debug][rank={rank}] top_grad "
            f"name={x['name']} shape={x['shape']} grad_dtype={x['grad_dtype']} "
            f"grad_abs_max={x['grad_abs_max']} grad_norm={x['grad_norm']} grad_nonfinite={x['grad_nonfinite']}"
        )
    print(f"[param-debug][rank={rank}] tag={tag} top_grad_end")

    print(f"[param-debug][rank={rank}] tag={tag} top_param_begin")
    for x in param_candidates[:topk]:
        print(
            f"[param-debug][rank={rank}] top_param "
            f"name={x['name']} shape={x['shape']} param_dtype={x['param_dtype']} "
            f"param_abs_max={x['param_abs_max']} param_norm={x['param_norm']} param_nonfinite={x['param_nonfinite']}"
        )
    print(f"[param-debug][rank={rank}] tag={tag} top_param_end")

def clean_param_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


def _safe_abs_max(x: torch.Tensor):
    if x.numel() == 0:
        return None
    x = x.detach()
    finite = torch.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(x[finite].abs().max().item())


def _safe_norm(x: torch.Tensor):
    if x.numel() == 0:
        return None
    x = x.detach()
    finite = torch.isfinite(x)
    if not finite.all():
        return float("nan")
    return float(x.float().norm().item())


def collect_param_debug_stats(model):
    stats = []

    for name, p in model.named_parameters():
        item = {
            "name": clean_param_name(name),
            "shape": tuple(p.shape),
            "param_dtype": str(p.dtype),
            "param_numel": int(p.numel()),
            "param_nonfinite": 0,
            "param_abs_max": None,
            "param_norm": None,
            "grad_present": p.grad is not None,
            "grad_dtype": None,
            "grad_numel": 0,
            "grad_nonfinite": 0,
            "grad_abs_max": None,
            "grad_norm": None,
        }

        pdata = p.detach()
        if pdata.numel() > 0:
            item["param_nonfinite"] = int((~torch.isfinite(pdata)).sum().item())
            item["param_abs_max"] = _safe_abs_max(pdata.float())
            item["param_norm"] = _safe_norm(pdata.float())

        if p.grad is not None:
            g = p.grad.detach()
            item["grad_dtype"] = str(g.dtype)
            item["grad_numel"] = int(g.numel())
            if g.numel() > 0:
                item["grad_nonfinite"] = int((~torch.isfinite(g)).sum().item())
                item["grad_abs_max"] = _safe_abs_max(g.float())
                item["grad_norm"] = _safe_norm(g.float())

        stats.append(item)

    return stats


def print_param_debug_report(model, rank, tag, topk=12):
    stats = collect_param_debug_stats(model)

    bad_param = [x for x in stats if x["param_nonfinite"] > 0]
    bad_grad = [x for x in stats if x["grad_nonfinite"] > 0]

    grad_candidates = [x for x in stats if x["grad_abs_max"] is not None]
    grad_candidates = sorted(
        grad_candidates,
        key=lambda x: (float("-inf") if x["grad_abs_max"] != x["grad_abs_max"] else x["grad_abs_max"]),
        reverse=True,
    )

    param_candidates = [x for x in stats if x["param_abs_max"] is not None]
    param_candidates = sorted(
        param_candidates,
        key=lambda x: (float("-inf") if x["param_abs_max"] != x["param_abs_max"] else x["param_abs_max"]),
        reverse=True,
    )

    print(
        f"[param-debug][rank={rank}] tag={tag} "
        f"total_params={len(stats)} bad_param_tensors={len(bad_param)} bad_grad_tensors={len(bad_grad)}"
    )

    if bad_param:
        print(f"[param-debug][rank={rank}] tag={tag} bad_param_begin")
        for x in bad_param[:topk]:
            print(
                f"[param-debug][rank={rank}] bad_param "
                f"name={x['name']} shape={x['shape']} dtype={x['param_dtype']} "
                f"nonfinite={x['param_nonfinite']} abs_max={x['param_abs_max']} norm={x['param_norm']}"
            )
        print(f"[param-debug][rank={rank}] tag={tag} bad_param_end")

    if bad_grad:
        print(f"[param-debug][rank={rank}] tag={tag} bad_grad_begin")
        for x in bad_grad[:topk]:
            print(
                f"[param-debug][rank={rank}] bad_grad "
                f"name={x['name']} shape={x['shape']} grad_dtype={x['grad_dtype']} "
                f"nonfinite={x['grad_nonfinite']} grad_abs_max={x['grad_abs_max']} grad_norm={x['grad_norm']}"
            )
        print(f"[param-debug][rank={rank}] tag={tag} bad_grad_end")

    print(f"[param-debug][rank={rank}] tag={tag} top_grad_begin")
    for x in grad_candidates[:topk]:
        print(
            f"[param-debug][rank={rank}] top_grad "
            f"name={x['name']} shape={x['shape']} grad_dtype={x['grad_dtype']} "
            f"grad_abs_max={x['grad_abs_max']} grad_norm={x['grad_norm']} grad_nonfinite={x['grad_nonfinite']}"
        )
    print(f"[param-debug][rank={rank}] tag={tag} top_grad_end")

    print(f"[param-debug][rank={rank}] tag={tag} top_param_begin")
    for x in param_candidates[:topk]:
        print(
            f"[param-debug][rank={rank}] top_param "
            f"name={x['name']} shape={x['shape']} param_dtype={x['param_dtype']} "
            f"param_abs_max={x['param_abs_max']} param_norm={x['param_norm']} param_nonfinite={x['param_nonfinite']}"
        )
    print(f"[param-debug][rank={rank}] tag={tag} top_param_end")

def debug_model_summary(model, rank):
    if rank != 0:
        return
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[debug-model] total_params={total_params:,} trainable_params={trainable_params:,}")
    dtype_buckets = {}
    for p in model.parameters():
        key = str(p.dtype)
        dtype_buckets[key] = dtype_buckets.get(key, 0) + p.numel()
    print(f"[debug-model] dtype_buckets={dtype_buckets}")

    for idx, (name, p) in enumerate(model.named_parameters()):
        print(
            f"[debug-model] param[{idx}] name={name} "
            f"shape={tuple(p.shape)} dtype={p.dtype} requires_grad={p.requires_grad}"
        )
        if idx >= 4:
            break


def debug_first_batch(state, input_ids, labels, data_step):
    if data_step != 1:
        return
    print(f"[debug-batch][rank={state['rank']}]", tensor_summary("input_ids", input_ids))
    print(f"[debug-batch][rank={state['rank']}]", tensor_summary("labels", labels))
    valid = (labels != -100)
    print(
        f"[debug-batch][rank={state['rank']}] supervised_tokens={valid.sum().item()} "
        f"total_tokens={labels.numel()} ratio={valid.float().mean().item():.4f}"
    )
    print(f"[debug-batch][rank={state['rank']}] sample_begin")
    print(state["tokenizer"].decode(input_ids[0].tolist(), skip_special_tokens=False)[:1000])
    print(f"[debug-batch][rank={state['rank']}] sample_end")

def debug_first_sample_alignment(state, raw_batch_input_ids, input_ids, labels, data_step, max_positions=160):
    if data_step != 1 or state["rank"] != 0:
        return

    tok = state["tokenizer"]

    raw_ids = raw_batch_input_ids[0].detach().cpu().tolist()
    in_ids = input_ids[0].detach().cpu().tolist()
    lab_ids = labels[0].detach().cpu().tolist()

    print("[debug-align] begin")
    print(f"[debug-align] raw_len={len(raw_ids)} input_len={len(in_ids)} label_len={len(lab_ids)}")

    print("[debug-align] raw_decoded_begin")
    print(tok.decode(raw_ids, skip_special_tokens=False))
    print("[debug-align] raw_decoded_end")

    limit = min(len(in_ids), len(lab_ids), max_positions)
    for i in range(limit):
        in_id = int(in_ids[i])
        lab_id = int(lab_ids[i])

        in_tok = tok.decode([in_id], skip_special_tokens=False).replace("\n", "\\n")
        if lab_id == -100:
            lab_tok = "<IGN>"
        else:
            lab_tok = tok.decode([lab_id], skip_special_tokens=False).replace("\n", "\\n")

        print(
            f"[debug-align] pos={i:03d} "
            f"input_id={in_id:<6d} input_tok={repr(in_tok):<18} "
            f"label_id={lab_id:<6d} label_tok={repr(lab_tok)}"
        )

    supervised_positions = [i for i, x in enumerate(lab_ids[:limit]) if x != -100]
    print(f"[debug-align] supervised_positions_first_{limit}={supervised_positions}")
    print("[debug-align] end")

    supervised_label_ids = [x for x in lab_ids if x != -100]
    print("[debug-align] supervised_label_text_begin")
    print(tok.decode(supervised_label_ids, skip_special_tokens=False))
    print("[debug-align] supervised_label_text_end")

    if supervised_positions:
        first_sup = supervised_positions[0]
        last_sup = supervised_positions[-1]
        print(f"[debug-align] first_supervised_pos={first_sup} last_supervised_pos={last_sup}")

def find_first_nonfinite_grad(model):
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            bad_count = (~torch.isfinite(g)).sum().item()
            return name, tuple(g.shape), str(g.dtype), bad_count
    return None

"""
def debug_grad_summary(model, rank, tag, topk=5):
    bad = []
    largest = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            bad.append((name, (~torch.isfinite(g)).sum().item(), str(g.dtype)))
            if len(bad) >= topk:
                break
        else:
            try:
                largest.append((g.abs().max().item(), name, str(g.dtype)))
            except Exception:
                pass
    print(f"[debug-grad][rank={rank}] tag={tag}")
    if bad:
        print(f"[debug-grad][rank={rank}] nonfinite={bad}")
    if largest:
        largest = sorted(largest, reverse=True)[:topk]
        print(f"[debug-grad][rank={rank}] largest={largest}")
"""

####################################################################################################
# Main driver functions.
####################################################################################################
DEFAULT_SYS_PROMPT = "You are a helpful assistant that can assist users with mathematical reasoning."


def startup(cfg: CLISettings):
    """The main setup function for the training script."""
    seed_everything(cfg.seed)

    ##########    Comms              ##############
    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    local_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    if torch.cuda.device_count() > 1:
        distributed = True
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", -1))),
            device_id=local_device,
            timeout=datetime.timedelta(hours=2),
        )
        world_size = torch.distributed.get_world_size()
        print(f"Comms formed on rank {rank} with device {local_device} out of world size {world_size}.")
    else:
        world_size = 1
        distributed = False

    torch.cuda.set_device(local_device)

    if cfg.precision == "bf16-true":
        torch.set_default_dtype(torch.bfloat16)
        weight_dtype = torch.bfloat16
        autocast_args = {"device_type": "cuda", "enabled": False, "dtype": torch.bfloat16}
    elif cfg.precision == "bf16-mixed":
        torch.set_default_dtype(torch.float32)
        weight_dtype = torch.float32
        autocast_args = {"device_type": "cuda", "enabled": True, "dtype": torch.bfloat16}
    elif cfg.precision == "fp16-true":
        torch.set_default_dtype(torch.float16)
        weight_dtype = torch.float16
        autocast_args = {"device_type": "cuda", "enabled": False, "dtype": torch.float16}
    elif cfg.precision == "fp16-mixed":
        torch.set_default_dtype(torch.float32)
        weight_dtype = torch.float32
        autocast_args = {"device_type": "cuda", "enabled": True, "dtype": torch.float16}
    else:
        raise ValueError(f"Unsupported precision: {cfg.precision}")

    ########## Model and tokenizer ##############
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=not USE_LOCAL_CODE,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(local_device)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=not USE_LOCAL_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0 and hasattr(model, "transformer"):
        print(
            "prelude0_type =",
            type(model.transformer.prelude[0]).__name__
            if hasattr(model.transformer, "prelude") and len(model.transformer.prelude) > 0
            else None,
        )
        print(
            "coda0_type =",
            type(model.transformer.coda[0]).__name__
            if hasattr(model.transformer, "coda") and len(model.transformer.coda) > 0
            else None,
        )
        print(
            "core_block_container_type =",
            type(model.transformer.core_block).__name__
            if hasattr(model.transformer, "core_block")
            else None,
        )
        print(
            "core_block0_type =",
            type(model.transformer.core_block[0]).__name__
            if hasattr(model.transformer, "core_block") and len(model.transformer.core_block) > 0
            else None,
        )

    debug_model_summary(model, rank)
    if rank == 0:
       print_param_debug_report(model, rank, "after_load", topk=8)

    if rank == 0:
        print_param_debug_report(model, rank, "after_load", topk=8)

    ##########  Distribute model   ##############
    if distributed and cfg.use_fsdp:
        mp_policy = None
        if cfg.precision == "fp16-true":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif cfg.precision == "fp16-mixed":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        elif cfg.precision == "bf16-true":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        sharding_strategy = ShardingStrategy.FULL_SHARD
        if cfg.fsdp_sharding_strategy == "shard_grad_op":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        def wrap_module_list_in_place(module_list, name):
            for i in range(len(module_list)):
                module_list[i] = FSDP(
                    module_list[i],
                    mixed_precision=mp_policy,
                    sharding_strategy=sharding_strategy,
                    device_id=local_device,
                    use_orig_params=True,
                )
            if rank == 0:
                print(f"[fsdp-wrap] wrapped {name}: {len(module_list)} blocks")

        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "prelude") and len(model.transformer.prelude) > 0:
                wrap_module_list_in_place(model.transformer.prelude, "prelude")
            if hasattr(model.transformer, "core_block") and len(model.transformer.core_block) > 0:
                wrap_module_list_in_place(model.transformer.core_block, "core_block")
            if hasattr(model.transformer, "coda") and len(model.transformer.coda) > 0:
                wrap_module_list_in_place(model.transformer.coda, "coda")

        model = FSDP(
            model,
            mixed_precision=mp_policy,
            sharding_strategy=sharding_strategy,
            device_id=local_device,
            use_orig_params=True,
            limit_all_gathers=True,
        )

    elif distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_device],
            find_unused_parameters=not cfg.compile,
            gradient_as_bucket_view=True,
        )

    if cfg.compile:
        model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")

    ##########     Optimizer       ##############
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim_config)

    ##########     Data            ##############
    def format_and_tokenize_examples(examples):
        conversations = []
        for idx in range(len(examples[cfg.dataset_args["q_col"]])):
            if cfg.dataset_args["q_col"] != "text":
                messages = [
                    Message(role="system", content=DEFAULT_SYS_PROMPT),
                    Message(role="user", content=examples[cfg.dataset_args["q_col"]][idx].strip()),
                    Message(role="Huginn", content=examples[cfg.dataset_args["a_col"]][idx].strip()),
                ]
            else:
                messages = tokenizer.bos_token + examples[cfg.dataset_args["q_col"]][idx].strip()
            conversations.append(messages)

        if cfg.dataset_args["q_col"] != "text":
            chat_encoding = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                padding="max_length",
                max_length=cfg.max_seq_length + 1,
                return_tensors="pt",
                return_dict=True,
                truncation=True,
            )
            if cfg.take_loss_over_all_tokens:
                chat_encoding["assistant_masks"] = chat_encoding["attention_mask"]
        else:
            chat_encoding = tokenizer(
                conversations,
                padding="max_length",
                max_length=cfg.max_seq_length + 1,
                return_tensors="pt",
                truncation=True,
            )
            chat_encoding["assistant_masks"] = chat_encoding["attention_mask"].clone()

        valid_supervised = (chat_encoding["assistant_masks"].bool() & chat_encoding["attention_mask"].bool()).sum(dim=1)
        return {
            "input_ids": chat_encoding["input_ids"],
            "mask": chat_encoding["assistant_masks"],
            "attention_mask": chat_encoding["attention_mask"],
            "valid_supervised": valid_supervised,
        }

    cfg.token_id_col_name = "input_ids"  # type: ignore
    dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"

    if is_main_process():
        if str(cfg.dataset_location).endswith(".jsonl"):
            dataset: Dataset = load_dataset("json", data_files=str(cfg.dataset_location), split="train")  # type: ignore
        elif cfg.dataset_config is None:
            dataset: Dataset = load_dataset(cfg.dataset_location, split="train")  # type: ignore
        else:
            dataset: Dataset = load_dataset(cfg.dataset_location, cfg.dataset_config, split="train")  # type: ignore

        if cfg.max_samples is not None:
            dataset = dataset.select(range(cfg.max_samples))

        if os.path.exists(dataset_save_dir):
            shutil.rmtree(dataset_save_dir)

        tokenized_dataset = dataset.map(
            format_and_tokenize_examples,
            num_proc=16,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=1024,
        )
        before_len = len(tokenized_dataset)
        tokenized_dataset = tokenized_dataset.filter(lambda x: x["valid_supervised"] > 0)
        after_len = len(tokenized_dataset)
        print(f"[debug-filter] before={before_len} after={after_len} removed={before_len - after_len}")

    if distributed:
        if is_main_process():
            tokenized_dataset.save_to_disk(dataset_save_dir)
        torch.distributed.barrier()
        tokenized_dataset = load_from_disk(dataset_save_dir)
        torch.distributed.barrier()

    if rank == 0:
        idx = int(torch.randint(len(tokenized_dataset), (1,)))
        print(f"-----------------------------------Processed Data example idx {idx}:----------------------------")
        print(tokenized_dataset[idx])
        print(tokenizer.decode(tokenized_dataset[idx]["input_ids"], skip_special_tokens=False))
        print("--------------------------------------------------------------------------------------------")

    tokenized_dataset.set_format("pt")

    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            tokenized_dataset,  # type: ignore
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            seed=cfg.seed,
        )
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            sampler=sampler,
            pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    ##########     Scheduler       ##############
    accumulation_steps = max(1, cfg.batch_size // cfg.micro_batch_size)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    max_training_steps = cfg.epochs * num_update_steps_per_epoch
    num_warmup_steps = math.ceil(cfg.scheduler_args["warmup"] * max_training_steps)  # type: ignore
    num_decay_steps = math.ceil(cfg.scheduler_args["cooldown"] * max_training_steps)  # type: ignore

    scheduler = get_scheduler(
        name="warmup_stable_decay",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_training_steps,
        scheduler_specific_kwargs={
            "num_decay_steps": num_decay_steps,
            "min_lr_ratio": cfg.scheduler_args["min_lr_ratio"],  # type: ignore
        },
    )

    use_grad_scaler = cfg.precision in {"fp16-mixed", "fp16-true"}
    scaler = ShardedGradScaler(enabled=use_grad_scaler)

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "dataloader": dataloader,
        "distributed": distributed,
        "rank": rank,
        "scheduler": scheduler,
        "autocast_args": autocast_args,
        "scaler": scaler,
    }

    cfg.world_size = world_size  # type: ignore
    return state, local_device


def train(state, device, cfg):
    model, optimizer = state["model"], state["optimizer"]
    model.train()
    init_debug_history(state)

    accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    optimizer_step = 0
    step_time = time.time()
    total_tokens = 0
    total_tokens_with_loss = 0
    tokens_in_step = 0
    stop_training = False

    metrics_to_agg_data_step = {
        "loss": [],
        "log_ppl": [],
    }

    for epoch in range(cfg.epochs):
        for data_step, inputs in enumerate(state["dataloader"], start=1):
            input_ids = inputs[cfg.token_id_col_name][:, :-1].to(dtype=torch.long, device=device, non_blocking=True)
            mask = ~(inputs["mask"].bool() & inputs["attention_mask"].bool())
            labels = torch.where(mask[:, 1:], -100, inputs[cfg.token_id_col_name][:, 1:]).to(
                dtype=torch.long, device=device, non_blocking=True
            )
            debug_first_sample_alignment(
                state,
                inputs[cfg.token_id_col_name],
                input_ids,
                labels,
                data_step,
            )

            debug_first_batch(state, input_ids, labels, data_step)
            total_tokens_with_loss += (labels != -100).sum().item()
            tokens_in_step += input_ids.numel()
            is_accumulating = data_step % accumulation_steps != 0

            shared_num_steps = sample_shared_num_steps(state, device)

            if data_step == 1:
                print(
                    f"[shared-steps][rank={state['rank']}] "
                    f"no_grad={int(shared_num_steps[0].item())} "
                    f"grad={int(shared_num_steps[1].item())}"
                )

            def tightly_scoped_fwd_bwd(model, input_ids, labels, num_steps):
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    with torch.autocast(**state["autocast_args"]):
                        outputs = model(input_ids, labels=labels, num_steps=num_steps)

                    scaled_loss = outputs["loss"] / accumulation_steps
                    if state["scaler"].is_enabled():
                        state["scaler"].scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    return (outputs["loss"].detach(), outputs["log_ppl"].detach())

            loss, log_ppl = tightly_scoped_fwd_bwd(model, input_ids, labels, shared_num_steps)
            current_lr = state["scheduler"].get_last_lr()[0] if hasattr(state["scheduler"], "get_last_lr") else cfg.optim_config["lr"]
            #append_debug_history(state, model, data_step, optimizer_step, loss, log_ppl, current_lr)

            local_bad_loss = (not torch.isfinite(loss)) or (not torch.isfinite(log_ppl))
            bad_loss_flag = torch.tensor(
                1 if local_bad_loss else 0,
                device=device,
                dtype=torch.int32,
            )

            if state["distributed"]:
                torch.distributed.all_reduce(bad_loss_flag, op=torch.distributed.ReduceOp.MAX)

            if bad_loss_flag.item() > 0:
                if local_bad_loss:
                    print(f"[nonfinite-loss][rank={state['rank']}] data_step={data_step} optimizer_step={optimizer_step}")
                    print("[nonfinite-loss]", tensor_summary("input_ids", input_ids))
                    print("[nonfinite-loss]", tensor_summary("labels", labels))
                    valid = (labels != -100)
                    print(
                        f"[nonfinite-loss][rank={state['rank']}] supervised_tokens={valid.sum().item()} "
                        f"total_tokens={labels.numel()} ratio={valid.float().mean().item():.4f}"
                    )
                    print(f"[nonfinite-loss][rank={state['rank']}] decoded_sample_begin")
                    print(state["tokenizer"].decode(input_ids[0].tolist(), skip_special_tokens=False)[:2000])
                    print(f"[nonfinite-loss][rank={state['rank']}] decoded_sample_end")
                raise RuntimeError(f"Global non-finite loss detected at data_step={data_step}")
            

            metrics_to_agg_data_step["loss"].append(loss.item())
            metrics_to_agg_data_step["log_ppl"].append(log_ppl.item())

            if not is_accumulating:
            
                if optimizer_step == 0:
                    print_param_debug_report(model, state["rank"], f"before_unscale_data_step_{data_step}", topk=10)

                if state["scaler"].is_enabled():
                    state["scaler"].unscale_(optimizer)

                if optimizer_step == 0:
                    print_param_debug_report(model, state["rank"], f"after_unscale_data_step_{data_step}", topk=10)

                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.max_grad_norm,
                    norm_type=2.0,
                )

                current_lr = state["scheduler"].get_last_lr()[0] if hasattr(state["scheduler"], "get_last_lr") else cfg.optim_config["lr"]
                #append_debug_history(state, model, data_step, optimizer_step, loss, log_ppl, current_lr, total_norm)
                #debug_grad_summary(model, state["rank"], f"before_step data_step={data_step}")

                
                local_bad_grad = not torch.isfinite(total_norm)
                bad_grad_flag = torch.tensor(
                    1 if local_bad_grad else 0,
                    device=device,
                    dtype=torch.int32,
                )

                if state["distributed"]:
                    torch.distributed.all_reduce(bad_grad_flag, op=torch.distributed.ReduceOp.MAX)

                if bad_grad_flag.item() > 0:
                    if local_bad_grad:
                        print(f"[nonfinite-gradnorm][rank={state['rank']}] data_step={data_step} optimizer_step={optimizer_step}")
                        print(f"[nonfinite-gradnorm][rank={state['rank']}] total_norm={total_norm}")

                        first_bad = find_first_nonfinite_grad(model)
                        if first_bad is None:
                            print(f"[nonfinite-gradnorm][rank={state['rank']}] no parameter-level nonfinite grad found, total norm still nonfinite")
                        else:
                            bad_name, bad_shape, bad_dtype, bad_count = first_bad
                            print(
                                f"[nonfinite-gradnorm][rank={state['rank']}] "
                                f"first_bad_grad name={bad_name} shape={bad_shape} dtype={bad_dtype} nonfinite_count={bad_count}"
                            )

                        print_param_debug_report(model, state["rank"], f"nonfinite_grad_data_step_{data_step}", topk=20)

                    raise RuntimeError(f"Global non-finite grad norm detected at data_step={data_step}")

                if state["scaler"].is_enabled():
                    state["scaler"].step(optimizer)
                    state["scaler"].update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                state["scheduler"].step()
                optimizer_step += 1


                if cfg.save_interval > 0 and optimizer_step % cfg.save_interval == 0:
                    if is_main_process():
                        print(f"[checkpoint] saving checkpoint-{optimizer_step}")
                    save_fsdp_checkpoint(state, cfg, f"checkpoint-{optimizer_step}")

                if state["rank"] == 0:
                    time_interval = (time.time() - step_time) / accumulation_steps
                    tok_sec = tokens_in_step * cfg.world_size / (time.time() - step_time)
                    print(
                        f"GPU: {model.device} | Step: {data_step:4d} | Updates: {optimizer_step:4d} | Time/step: {time_interval:2.4f}"
                        f" | Tok/sec={tok_sec:9.2f} | Loss: {loss:2.4f} / log-ppl: {log_ppl:2.4f} | Grad-Norm {total_norm.item():2.4f}"
                    )
                    total_tokens += tokens_in_step * cfg.world_size
                    step_time = time.time()
                    tokens_in_step = 0

            if optimizer_step and (optimizer_step % cfg.eval_interval == 0):
                validate(state, optimizer_step, cfg)

            if cfg.max_steps and optimizer_step >= cfg.max_steps:
                stop_training = True
                break

        if stop_training:
            break

    model.eval()
    return state


####################################################################################################
# Main control loop
####################################################################################################


def validate(state, step, cfg, task="gsm8k"):
    unwrapped_model = get_unwrapped_model(state)
    unwrapped_model.eval()
    if is_main_process():
        print(f"[validate-placeholder] step={step}, task={task}, validation is temporarily disabled for profiling.")
    unwrapped_model.train()


def main():
    """Encapsulates main scope away from import calls."""

    # Configuration loader
    cfg = CLISettings()

    # Print system setup
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    # set flags
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway
    torch._dynamo.config.optimize_ddp = "python_reducer"
    torch._dynamo.config.compiled_autograd = False

    train_time = time.time()

    state, device = startup(cfg)
    state = train(state, device, cfg)
    # validate(state, "final", cfg)

    if cfg.save_final_checkpoint:
        if is_main_process():
            print("[checkpoint] saving final_checkpoint")
        save_fsdp_checkpoint(state, cfg, "final_checkpoint")

    # Now exit
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"Training time: {str(datetime.timedelta(seconds=time.time() - train_time))} ")
        max_alloc = f"{torch.cuda.max_memory_allocated(device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(device) / float(1024**3):,.3f} GB"
        print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
        print("--------------------------------------------------------------------")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
    if is_main_process() and os.path.exists(dataset_save_dir):
        try:
            shutil.rmtree(dataset_save_dir)
        except OSError as e:
            print(f"[cleanup-warning] failed to remove {dataset_save_dir}: {e}")
    return cfg.run_name


def shutdown():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
    print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        run_name = main()
        print("--------------------------------------------------------------------")
        print(f"Run {run_name} finished without error.")
    except BaseException:
        print("--------------------------------------------------------------------")
        print("Run finished with errors.")
        raise
    finally:
        shutdown()  # guarantee NCCL deconstruction


if __name__ == "__main__":
    guarded_main()