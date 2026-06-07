"""Single-GPU smoke test for the Huginn audio experiment branch."""

import math
import random
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

@dataclass
class CLISettings:
    run_name: str = "huginn-audio-whisper-smoke"
    out_path: str = "outputs"
    base_model_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
    audio_model_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-whisper-v1"
    audio_encoder_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-small"
    max_seq_length: int = 128
    sample_rate: int = 16000
    duration_seconds: float = 1.0
    micro_batch_size: int = 1
    max_steps: int = 6
    lr: float = 1e-3
    precision: str = "bf16-mixed"
    fixed_num_steps_no_grad: int = 0
    fixed_num_steps_with_grad: int = 4
    seed: int = 74
    system_prompt: str = "You are a helpful assistant that answers questions about audio."


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SyntheticAudioQADataset(Dataset):
    def __init__(self, sample_rate: int, duration_seconds: float):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.examples = [
            self._make_example(220.0, "Is the tone low or high?", "The tone is low."),
            self._make_example(880.0, "Is the tone low or high?", "The tone is high."),
            self._make_example(330.0, "Is the tone low or high?", "The tone is low."),
            self._make_example(990.0, "Is the tone low or high?", "The tone is high."),
        ]

    def _make_example(self, frequency: float, question: str, answer: str) -> dict[str, Any]:
        steps = int(self.sample_rate * self.duration_seconds)
        t = torch.arange(steps, dtype=torch.float32) / self.sample_rate
        waveform = 0.2 * torch.sin(2 * math.pi * frequency * t)
        return {"audio": waveform.numpy(), "question": question, "answer": answer}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_collate_fn(tokenizer, processor, cfg: CLISettings):
    def collate_fn(batch):
        conversations = []
        for sample in batch:
            conversations.append(
                [
                    {"role": "system", "content": cfg.system_prompt},
                    {
                        "role": "user",
                        "content": f"Listen to the audio and answer the question.\nQuestion: {sample['question']}",
                    },
                    {"role": "Huginn", "content": sample["answer"]},
                ]
            )

        chat_encoding = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            padding="longest",
            max_length=cfg.max_seq_length + 1,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
        )

        input_ids = chat_encoding["input_ids"][:, :-1]
        assistant_masks = chat_encoding["assistant_masks"].bool()
        attention_mask = chat_encoding["attention_mask"]
        labels = torch.where(
            assistant_masks[:, 1:] & attention_mask[:, 1:].bool(),
            chat_encoding["input_ids"][:, 1:],
            torch.full_like(chat_encoding["input_ids"][:, 1:], -100),
        )

        audio = [sample["audio"] for sample in batch]
        audio_inputs = processor.feature_extractor(
            audio,
            sampling_rate=cfg.sample_rate,
            return_tensors="pt",
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask[:, :-1],
            "audio_input_features": audio_inputs["input_features"],
        }

    return collate_fn


def print_trainable_summary(model):
    trainable = [(name, tuple(param.shape)) for name, param in model.named_parameters() if param.requires_grad]
    frozen_count = sum(1 for _, param in model.named_parameters() if not param.requires_grad)
    print("[audio-smoke] trainable parameter tensors:")
    for name, shape in trainable:
        print(f"  - {name}: {shape}")
    print(f"[audio-smoke] frozen_param_tensors={frozen_count} trainable_param_tensors={len(trainable)}")


def resolve_model_dtype(precision: str) -> torch.dtype:
    if precision == "bf16-mixed":
        return torch.bfloat16
    if precision == "fp16-mixed":
        return torch.float16
    return torch.float32


def print_grad_summary(model):
    grad_stats = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            grad_stats.append((name, "missing", None))
        else:
            grad_stats.append((name, "ok", float(param.grad.detach().abs().max().item())))

    print("[audio-smoke] grad summary:")
    for name, status, abs_max in grad_stats:
        if status == "missing":
            print(f"  - {name}: grad=missing")
        else:
            print(f"  - {name}: grad_abs_max={abs_max:.4e}")

    missing = [name for name, status, _ in grad_stats if status == "missing"]
    if missing:
        raise RuntimeError(f"Trainable parameters missing gradients: {missing}")


def main():
    cfg = CLISettings()
    seed_everything(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("--------------------------------------------------------------------")
    print(f"---------------- Launching run {cfg.run_name} ----------------")
    print("--------------------------------------------------------------------")
    print(f"Host={socket.gethostname()} Python={sys.version.split(' (')[0]} Torch={torch.__version__} Device={device}")

    model_dtype = resolve_model_dtype(cfg.precision)
    config = AutoConfig.from_pretrained(cfg.audio_model_dir, trust_remote_code=True)
    config.audio_encoder_name = cfg.audio_encoder_name
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    load_result = model.load_huginn_backbone_from_pretrained(cfg.base_model_name, torch_dtype=torch.float32)
    print(f"[audio-smoke] backbone load missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")
    if load_result.unexpected_keys:
        print("[audio-smoke] unexpected_keys:", load_result.unexpected_keys[:10])
    if load_result.missing_keys:
        print("[audio-smoke] first_missing_keys:", load_result.missing_keys[:10])

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(cfg.audio_encoder_name)

    model.to(device=device, dtype=model_dtype)
    model.train()
    print_trainable_summary(model)
    print(f"[audio-smoke] model_dtype={model_dtype}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    dataset = SyntheticAudioQADataset(cfg.sample_rate, cfg.duration_seconds)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(tokenizer, processor, cfg),
    )

    autocast_dtype = torch.bfloat16 if cfg.precision == "bf16-mixed" else torch.float16
    num_steps = torch.tensor(
        [cfg.fixed_num_steps_no_grad, cfg.fixed_num_steps_with_grad],
        dtype=torch.long,
        device=device,
    )

    start_time = time.time()
    step = 0
    while step < cfg.max_steps:
        for batch in dataloader:
            step += 1
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_input_features = batch["audio_input_features"].to(device)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    num_steps=num_steps,
                    audio_input_features=audio_input_features,
                )

            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in smoke test at step={step}: {loss}")
            loss.backward()

            print_grad_summary(model)

            optimizer.step()
            print(f"[audio-smoke] step={step} loss={loss.item():.4f}")
            if step >= cfg.max_steps:
                break

    elapsed = time.time() - start_time
    print("--------------------------------------------------------------------")
    print(f"[audio-smoke] completed steps={cfg.max_steps} elapsed={elapsed:.2f}s")
    if device.type == "cuda":
        print(
            f"[audio-smoke] max_mem_allocated={torch.cuda.max_memory_allocated(device) / float(1024**3):.3f} GB "
            f"max_mem_reserved={torch.cuda.max_memory_reserved(device) / float(1024**3):.3f} GB"
        )
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    main()
