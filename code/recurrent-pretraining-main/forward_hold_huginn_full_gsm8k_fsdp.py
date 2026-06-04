"""Forward-only FSDP hold script for keeping 8x V100 occupied on a single sample."""

import datetime
import json
import math
import os
import socket
import sys
import time

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer


global_start_time = time.time()
USE_LOCAL_CODE = False

# Check device health before heavy imports/runtime setup.
nvml_count = torch.cuda._device_count_amdsmi() if torch.version.hip else torch.cuda._device_count_nvml()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}")


if TYPE_CHECKING:
    import torch.distributed


@dataclass
class CLISettings:
    run_name: str = "huginn-gsm8k-forward-hold-v100"
    dataset_location: str = "/hpc_stor03/sjtu_home/jinwei.zhang/code/recurrent-pretraining-main/data/gsm8k_train.jsonl"
    model_name: str = os.getenv(
        "HUGINN_MODEL_DIR",
        "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125",
    )
    dataset_args: dict[str, Any] = field(default_factory=lambda: dict(q_col="question", a_col="answer"))
    max_seq_length: int = 128
    hold_steps: int = 100000
    log_interval: int = 50
    precision: str = "fp16-mixed"
    use_fsdp: bool = True
    fsdp_sharding_strategy: str = "full_shard"
    seed: int = 74


@dataclass
class Message:
    role: str
    content: str


DEFAULT_SYS_PROMPT = "You are a helpful assistant that can assist users with mathematical reasoning."


def is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
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


def sample_shared_num_steps(model, distributed, rank, device):
    base_model = unwrap_model(model)

    if distributed:
        step_pair = torch.zeros(2, device=device, dtype=torch.long)
        if rank == 0:
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


def _read_single_example(cfg: CLISettings) -> dict[str, str]:
    if str(cfg.dataset_location).endswith(".jsonl"):
        with open(cfg.dataset_location, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    return {
                        "question": item[cfg.dataset_args["q_col"]].strip(),
                        "answer": item[cfg.dataset_args["a_col"]].strip(),
                    }
        raise ValueError(f"No non-empty lines found in {cfg.dataset_location}")

    dataset = load_dataset(cfg.dataset_location, split="train")
    item = dataset[0]
    return {
        "question": item[cfg.dataset_args["q_col"]].strip(),
        "answer": item[cfg.dataset_args["a_col"]].strip(),
    }


def _build_single_batch(tokenizer, cfg: CLISettings, device):
    example = _read_single_example(cfg)
    messages = [
        Message(role="system", content=DEFAULT_SYS_PROMPT),
        Message(role="user", content=example["question"]),
        Message(role="Huginn", content=example["answer"]),
    ]

    chat_encoding = tokenizer.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        padding="max_length",
        max_length=cfg.max_seq_length + 1,
        return_tensors="pt",
        return_dict=True,
        truncation=True,
    )

    input_ids = chat_encoding["input_ids"][:, :-1].to(dtype=torch.long, device=device, non_blocking=True)
    mask = ~(chat_encoding["assistant_masks"].bool() & chat_encoding["attention_mask"].bool())
    labels = torch.where(mask[:, 1:], -100, chat_encoding["input_ids"][:, 1:]).to(
        dtype=torch.long,
        device=device,
        non_blocking=True,
    )

    if is_main_process():
        print("[forward-hold] sample_ready")
        print(f"[forward-hold] question_chars={len(example['question'])} answer_chars={len(example['answer'])}")
        print(f"[forward-hold] input_shape={tuple(input_ids.shape)} labels_shape={tuple(labels.shape)}")
        valid = (labels != -100)
        print(
            f"[forward-hold] supervised_tokens={valid.sum().item()} total_tokens={labels.numel()} "
            f"ratio={valid.float().mean().item():.4f}"
        )
        print("[forward-hold] sample_text_begin")
        print(tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)[:1200])
        print("[forward-hold] sample_text_end")

    return input_ids, labels


def startup(cfg: CLISettings):
    seed_everything(cfg.seed)

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
        distributed = False
        world_size = 1

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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=not USE_LOCAL_CODE,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(local_device)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=not USE_LOCAL_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
                print(f"[forward-hold] wrapped {name}: {len(module_list)} blocks")

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

    input_ids, labels = _build_single_batch(tokenizer, cfg, local_device)

    state = {
        "model": model,
        "tokenizer": tokenizer,
        "input_ids": input_ids,
        "labels": labels,
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "autocast_args": autocast_args,
    }
    return state, local_device


def hold_forward(state, device, cfg: CLISettings):
    model = state["model"]
    input_ids = state["input_ids"]
    labels = state["labels"]

    tokens_per_step = input_ids.numel() * state["world_size"]
    start_time = time.time()
    last_log_time = start_time

    if state["rank"] == 0:
        print(
            f"[forward-hold] starting run_name={cfg.run_name} hold_steps={cfg.hold_steps} "
            f"log_interval={cfg.log_interval} model_name={cfg.model_name}"
        )

    for step in range(1, cfg.hold_steps + 1):
        shared_num_steps = sample_shared_num_steps(
            model=model,
            distributed=state["distributed"],
            rank=state["rank"],
            device=device,
        )

        with nullcontext():
            with torch.no_grad():
                with torch.autocast(**state["autocast_args"]):
                    outputs = model(input_ids, labels=labels, num_steps=shared_num_steps)

        loss = outputs["loss"].detach()
        log_ppl = outputs["log_ppl"].detach()

        if (not torch.isfinite(loss)) or (not torch.isfinite(log_ppl)):
            raise RuntimeError(
                f"Non-finite forward output detected at step={step} "
                f"loss={loss.item()} log_ppl={log_ppl.item()}"
            )

        if step == 1 or step % cfg.log_interval == 0:
            torch.cuda.synchronize(device)
            now = time.time()
            interval = now - last_log_time
            total_elapsed = now - start_time
            interval_steps = cfg.log_interval if step != 1 else 1
            tok_per_sec = (tokens_per_step * interval_steps) / max(interval, 1e-6)

            if state["rank"] == 0:
                print(
                    f"[forward-hold] step={step:6d}/{cfg.hold_steps} "
                    f"no_grad={int(shared_num_steps[0].item())} grad={int(shared_num_steps[1].item())} "
                    f"loss={loss.item():.4f} log_ppl={log_ppl.item():.4f} "
                    f"tok_per_sec={tok_per_sec:9.2f} elapsed={str(datetime.timedelta(seconds=int(total_elapsed)))}"
                )
            last_log_time = now


def main():
    cfg = CLISettings()

    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"Launching forward hold run {cfg.run_name}")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU: {torch.cuda.get_device_name()}. {driver}.")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    state, device = startup(cfg)
    hold_forward(state, device, cfg)

    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"Forward hold finished: {cfg.run_name}")
        max_alloc = f"{torch.cuda.max_memory_allocated(device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(device) / float(1024**3):,.3f} GB"
        print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
        print("--------------------------------------------------------------------")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

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
        shutdown()


if __name__ == "__main__":
    guarded_main()
