"""Single-GPU tiny-overfit training for the Huginn audio experiment branch."""

from __future__ import annotations

import json
import random
import socket
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


@dataclass
class CLISettings:
    run_name: str = "huginn-audio-whisper-tiny-overfit"
    out_path: str = "outputs"
    base_model_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
    audio_model_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-whisper-v1"
    audio_encoder_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-small"
    dataset_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn_tiny_train32"
    train_jsonl: str = "train.jsonl"
    max_seq_length: int = 192
    sample_rate: int = 16000
    micro_batch_size: int = 1
    epochs: int = 20
    max_steps: int = 200
    lr: float = 5e-4
    precision: str = "bf16-mixed"
    fixed_num_steps_no_grad: int = 0
    fixed_num_steps_with_grad: int = 4
    seed: int = 74
    log_interval: int = 10
    save_interval: int = 50
    system_prompt: str = "You are a helpful assistant that answers questions about audio."


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
        frames = wf.readframes(num_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav is supported in tiny overfit script: {path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)
    return audio


class TinyAudioQADataset(Dataset):
    def __init__(self, dataset_dir: str, train_jsonl: str):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        manifest_path = self.dataset_dir / train_jsonl
        with manifest_path.open("r", encoding="utf-8") as f:
            self.examples = [json.loads(line) for line in f]
        if not self.examples:
            raise ValueError(f"No training examples found in {manifest_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        record = self.examples[idx]
        audio_path = self.dataset_dir / record["audio_path"]
        waveform = load_wav_mono(audio_path)
        return {
            "audio": waveform,
            "question": record["question"],
            "answer": record["answer"],
            "audio_path": str(audio_path),
        }


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
            "audio_paths": [sample["audio_path"] for sample in batch],
        }

    return collate_fn


def resolve_model_dtype(precision: str) -> torch.dtype:
    if precision == "bf16-mixed":
        return torch.bfloat16
    if precision == "fp16-mixed":
        return torch.float16
    return torch.float32


def print_trainable_summary(model):
    trainable = [(name, tuple(param.shape)) for name, param in model.named_parameters() if param.requires_grad]
    frozen_count = sum(1 for _, param in model.named_parameters() if not param.requires_grad)
    print("[audio-tiny] trainable parameter tensors:")
    for name, shape in trainable:
        print(f"  - {name}: {shape}")
    print(f"[audio-tiny] frozen_param_tensors={frozen_count} trainable_param_tensors={len(trainable)}")


def print_grad_summary(model):
    grad_stats = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            grad_stats.append((name, "missing", None))
        else:
            grad_stats.append((name, "ok", float(param.grad.detach().abs().max().item())))

    print("[audio-tiny] grad summary:")
    for name, status, abs_max in grad_stats:
        if status == "missing":
            print(f"  - {name}: grad=missing")
        else:
            print(f"  - {name}: grad_abs_max={abs_max:.4e}")

    missing = [name for name, status, _ in grad_stats if status == "missing"]
    if missing:
        raise RuntimeError(f"Trainable parameters missing gradients: {missing}")


def save_trainable_state(model, save_dir: Path, step: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if dict(model.named_parameters()).get(name, None) is not None
        and dict(model.named_parameters())[name].requires_grad
    }
    torch.save(trainable_state, checkpoint_dir / "trainable_state.pt")
    print(f"[audio-tiny] saved trainable checkpoint to {checkpoint_dir}")


def main():
    cfg = CLISettings()
    seed_everything(cfg.seed)

    output_dir = Path(cfg.out_path) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("--------------------------------------------------------------------")
    print(f"---------------- Launching run {cfg.run_name} ----------------")
    print("--------------------------------------------------------------------")
    print(f"Host={socket.gethostname()} Python={sys.version.split(' (')[0]} Torch={torch.__version__} Device={device}")
    print(f"[audio-tiny] dataset_dir={cfg.dataset_dir}")

    model_dtype = resolve_model_dtype(cfg.precision)
    config = AutoConfig.from_pretrained(cfg.audio_model_dir, trust_remote_code=True)
    config.audio_encoder_name = cfg.audio_encoder_name
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    load_result = model.load_huginn_backbone_from_pretrained(cfg.base_model_name, torch_dtype=torch.float32)
    print(f"[audio-tiny] backbone load missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")
    if load_result.unexpected_keys:
        print("[audio-tiny] unexpected_keys:", load_result.unexpected_keys[:10])
    if load_result.missing_keys:
        print("[audio-tiny] first_missing_keys:", load_result.missing_keys[:10])

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(cfg.audio_encoder_name)

    model.to(device=device, dtype=model_dtype)
    model.train()
    print_trainable_summary(model)
    print(f"[audio-tiny] model_dtype={model_dtype}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    dataset = TinyAudioQADataset(cfg.dataset_dir, cfg.train_jsonl)
    print(f"[audio-tiny] dataset_size={len(dataset)}")
    print(f"[audio-tiny] first_audio_path={dataset[0]['audio_path']}")
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
    for epoch in range(1, cfg.epochs + 1):
        print(f"[audio-tiny] epoch={epoch} begin")
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
                raise RuntimeError(f"Non-finite loss in tiny overfit at step={step}: {loss}")
            loss.backward()

            if step == 1 or step % cfg.log_interval == 0:
                print_grad_summary(model)

            optimizer.step()
            print(f"[audio-tiny] epoch={epoch} step={step} loss={loss.item():.4f} audio={batch['audio_paths'][0]}")

            if step % cfg.save_interval == 0:
                save_trainable_state(model, output_dir, step)

            if step >= cfg.max_steps:
                break

        if step >= cfg.max_steps:
            break

    save_trainable_state(model, output_dir, step)

    elapsed = time.time() - start_time
    print("--------------------------------------------------------------------")
    print(f"[audio-tiny] completed steps={step} elapsed={elapsed:.2f}s")
    if device.type == "cuda":
        print(
            f"[audio-tiny] max_mem_allocated={torch.cuda.max_memory_allocated(device) / float(1024**3):.3f} GB "
            f"max_mem_reserved={torch.cuda.max_memory_reserved(device) / float(1024**3):.3f} GB"
        )
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    main()
