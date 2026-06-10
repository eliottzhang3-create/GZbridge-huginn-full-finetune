"""Post-training analysis for Huginn audio ClothoAQA runs.

Compares three evaluation conditions on a sampled subset of the test split:
1. normal audio
2. shuffled audio
3. zeroed audio
"""

from __future__ import annotations

import json
import random
import socket
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


@dataclass
class CLISettings:
    run_name: str = "huginn-audio-whisper-clotho-aqa-v2-analysis"
    out_path: str = "outputs"
    base_model_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
    audio_model_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-whisper-v1"
    audio_encoder_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-small"
    dataset_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn"
    test_jsonl: str = "test.jsonl"
    checkpoint_dir: str = (
        "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
        "code/recurrent-pretraining-main/outputs/huginn-audio-whisper-clotho-aqa-v2/checkpoint-7029"
    )
    sample_count: int = 100
    max_seq_length: int = 192
    target_sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    micro_batch_size: int = 1
    precision: str = "bf16-mixed"
    eval_num_steps_no_grad: int = 4
    eval_num_steps_with_grad: int = 4
    seed: int = 74
    system_prompt: str = "You are a helpful assistant that answers questions about audio."


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_waveform(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.shape[0] / float(source_sr)
    target_length = max(1, int(round(duration * target_sr)))
    src_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    tgt_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(tgt_positions, src_positions, audio).astype(np.float32)


def load_wav_mono(path: Path, target_sr: int, max_audio_seconds: float) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        source_sr = wf.getframerate()
        num_frames = wf.getnframes()
        frames = wf.readframes(num_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav is supported: {path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    audio = resample_waveform(audio, source_sr, target_sr)
    max_samples = int(round(max_audio_seconds * target_sr))
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    return audio


class ClothoAQASubset(Dataset):
    def __init__(self, cfg: CLISettings):
        self.dataset_dir = Path(cfg.dataset_dir)
        manifest_path = self.dataset_dir / cfg.test_jsonl
        with manifest_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        if not records:
            raise ValueError(f"No test examples found in {manifest_path}")

        rng = random.Random(cfg.seed)
        selected = records if cfg.sample_count >= len(records) else rng.sample(records, cfg.sample_count)

        self.examples = []
        for record in selected:
            audio_path = self.dataset_dir / record["audio_path"]
            waveform = load_wav_mono(audio_path, cfg.target_sample_rate, cfg.max_audio_seconds)
            self.examples.append(
                {
                    "audio": waveform,
                    "question": record["question"],
                    "answer": record["answer"],
                    "audio_path": str(audio_path),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ConditionDataset(Dataset):
    def __init__(self, base_examples: list[dict], condition: str, seed: int):
        self.condition = condition
        self.examples = base_examples
        rng = random.Random(seed)
        self.shuffle_indices = list(range(len(base_examples)))
        rng.shuffle(self.shuffle_indices)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = dict(self.examples[idx])
        if self.condition == "normal":
            pass
        elif self.condition == "shuffled":
            sample["audio"] = self.examples[self.shuffle_indices[idx]]["audio"]
            sample["audio_path"] = self.examples[self.shuffle_indices[idx]]["audio_path"]
        elif self.condition == "zero":
            sample["audio"] = np.zeros_like(sample["audio"])
            sample["audio_path"] = "<zero-audio>"
        else:
            raise ValueError(f"Unsupported condition: {self.condition}")
        return sample


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
            sampling_rate=cfg.target_sample_rate,
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


def load_model(cfg: CLISettings, device: torch.device):
    model_dtype = resolve_model_dtype(cfg.precision)
    config = AutoConfig.from_pretrained(cfg.audio_model_dir, trust_remote_code=True)
    config.audio_encoder_name = cfg.audio_encoder_name
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    base_load = model.load_huginn_backbone_from_pretrained(cfg.base_model_name, torch_dtype=torch.float32)
    print(
        f"[audio-analysis] backbone load missing={len(base_load.missing_keys)} "
        f"unexpected={len(base_load.unexpected_keys)}"
    )
    if base_load.missing_keys:
        print("[audio-analysis] first_missing_keys:", base_load.missing_keys[:10])

    trainable_state_path = Path(cfg.checkpoint_dir) / "trainable_state.pt"
    state = torch.load(trainable_state_path, map_location="cpu")
    delta_load = model.load_state_dict(state, strict=False)
    print(
        f"[audio-analysis] delta load missing={len(delta_load.missing_keys)} "
        f"unexpected={len(delta_load.unexpected_keys)}"
    )
    if delta_load.unexpected_keys:
        print("[audio-analysis] unexpected delta keys:", delta_load.unexpected_keys[:10])

    model.to(device=device, dtype=model_dtype)
    model.eval()
    return model


def evaluate_condition(model, dataloader, cfg: CLISettings, device: torch.device, condition: str):
    autocast_dtype = torch.bfloat16 if cfg.precision == "bf16-mixed" else torch.float16
    num_steps = torch.tensor(
        [cfg.eval_num_steps_no_grad, cfg.eval_num_steps_with_grad],
        dtype=torch.long,
        device=device,
    )

    total_loss = 0.0
    total_batches = 0
    first_audio = None
    with torch.no_grad():
        for batch in dataloader:
            if first_audio is None:
                first_audio = batch["audio_paths"][0]

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
            total_loss += float(loss.item())
            total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    print(
        f"[audio-analysis] condition={condition} batches={total_batches} "
        f"mean_loss={mean_loss:.6f} first_audio={first_audio}"
    )
    return {
        "condition": condition,
        "mean_loss": mean_loss,
        "num_batches": total_batches,
        "first_audio": first_audio,
    }


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
    print(
        f"[audio-analysis] dataset_dir={cfg.dataset_dir} checkpoint_dir={cfg.checkpoint_dir} "
        f"sample_count={cfg.sample_count}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(cfg.audio_encoder_name)

    subset = ClothoAQASubset(cfg)
    print(f"[audio-analysis] subset_size={len(subset)}")
    base_examples = subset.examples
    collate_fn = build_collate_fn(tokenizer, processor, cfg)
    model = load_model(cfg, device)

    start_time = time.time()
    results = []
    for condition in ("normal", "shuffled", "zero"):
        condition_dataset = ConditionDataset(base_examples, condition, seed=cfg.seed)
        dataloader = DataLoader(
            condition_dataset,
            batch_size=cfg.micro_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        results.append(evaluate_condition(model, dataloader, cfg, device, condition))

    elapsed = time.time() - start_time
    summary = {
        "config": asdict(cfg),
        "results": results,
        "elapsed_seconds": elapsed,
    }
    summary_path = output_dir / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("--------------------------------------------------------------------")
    print(f"[audio-analysis] saved summary to {summary_path}")
    print(f"[audio-analysis] elapsed={elapsed:.2f}s")
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    main()
