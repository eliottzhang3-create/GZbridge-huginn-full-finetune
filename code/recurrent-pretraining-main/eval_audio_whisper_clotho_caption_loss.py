"""Evaluate mean token loss on the Clotho caption test set."""

from __future__ import annotations

import json
import random
import socket
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


@dataclass
class CLISettings:
    run_name: str = "huginn-audio-whisper-clotho-caption-loss-eval"
    base_model_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
    audio_model_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-whisper-v1"
    audio_encoder_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-small"
    checkpoint_dir: str = (
        "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
        "code/recurrent-pretraining-main/outputs/huginn-audio-whisper-clotho-caption-v1/checkpoint-2560"
    )
    dataset_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn"
    eval_manifest: str = "test.jsonl"
    max_seq_length: int = 192
    target_sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    batch_size: int = 5
    precision: str = "bf16-mixed"
    fixed_num_steps_no_grad: int = 2
    fixed_num_steps_with_grad: int = 4
    seed: int = 74
    system_prompt: str = "You are a helpful assistant that describes audio."
    user_prompt: str = "Listen to the audio and describe it."


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_model_dtype(precision: str) -> torch.dtype:
    if precision == "bf16-mixed":
        return torch.bfloat16
    if precision == "fp16-mixed":
        return torch.float16
    return torch.float32


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


class ClothoCaptionEvalDataset(Dataset):
    def __init__(self, dataset_dir: str, manifest_name: str, target_sr: int, max_audio_seconds: float):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.target_sr = target_sr
        self.max_audio_seconds = max_audio_seconds
        manifest_path = self.dataset_dir / manifest_name
        with manifest_path.open("r", encoding="utf-8") as f:
            if manifest_path.suffix.lower() == ".json":
                payload = json.load(f)
                if not isinstance(payload, list):
                    raise ValueError(f"Expected a JSON array in {manifest_path}")
                self.examples = payload
            else:
                self.examples = [json.loads(line) for line in f if line.strip()]
        if not self.examples:
            raise ValueError(f"No evaluation examples found in {manifest_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        last_error = None
        for offset in range(len(self.examples)):
            record = self.examples[(idx + offset) % len(self.examples)]
            audio_path = self.dataset_dir / record["audio_path"]
            try:
                waveform = load_wav_mono(audio_path, self.target_sr, self.max_audio_seconds)
                return {
                    "audio": waveform,
                    "caption": record["caption"],
                    "audio_path": str(audio_path),
                }
            except (EOFError, wave.Error, ValueError) as exc:
                last_error = exc
                print(f"[audio-caption-eval][bad-audio] skip path={audio_path} error={type(exc).__name__}: {exc}")
                continue

        raise RuntimeError(f"Failed to load any audio example from eval dataset; last_error={last_error}")


def build_collate_fn(tokenizer, processor, cfg: CLISettings):
    def collate_fn(batch):
        conversations = []
        for sample in batch:
            conversations.append(
                [
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": cfg.user_prompt},
                    {"role": "Huginn", "content": sample["caption"]},
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


def main():
    cfg = CLISettings()
    seed_everything(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_model_dtype(cfg.precision)
    print("--------------------------------------------------------------------")
    print(f"---------------- Launching run {cfg.run_name} ----------------")
    print("--------------------------------------------------------------------")
    print(f"Host={socket.gethostname()} Python={sys.version.split(' (')[0]} Torch={torch.__version__} Device={device}")
    print(f"[audio-caption-eval] dataset_dir={cfg.dataset_dir}")
    print(f"[audio-caption-eval] eval_manifest={cfg.eval_manifest}")
    print(f"[audio-caption-eval] checkpoint_dir={cfg.checkpoint_dir}")
    print(f"[audio-caption-eval] batch_size={cfg.batch_size}")

    config = AutoConfig.from_pretrained(cfg.audio_model_dir, trust_remote_code=True)
    config.audio_encoder_name = cfg.audio_encoder_name
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    base_load = model.load_huginn_backbone_from_pretrained(cfg.base_model_name, torch_dtype=torch.float32)
    print(
        f"[audio-caption-eval] backbone load missing={len(base_load.missing_keys)} "
        f"unexpected={len(base_load.unexpected_keys)}"
    )

    init_state_path = Path(cfg.checkpoint_dir) / "trainable_state.pt"
    init_state = torch.load(init_state_path, map_location="cpu")
    delta_load = model.load_state_dict(init_state, strict=False)
    print(
        f"[audio-caption-eval] delta load missing={len(delta_load.missing_keys)} "
        f"unexpected={len(delta_load.unexpected_keys)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(cfg.audio_encoder_name)

    model.to(device=device, dtype=model_dtype)
    model.eval()

    dataset = ClothoCaptionEvalDataset(cfg.dataset_dir, cfg.eval_manifest, cfg.target_sample_rate, cfg.max_audio_seconds)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer, processor, cfg),
    )
    print(f"[audio-caption-eval] dataset_size={len(dataset)} num_batches={len(dataloader)}")
    print(f"[audio-caption-eval] first_audio_path={dataset[0]['audio_path']}")

    fixed_num_steps = torch.tensor(
        [cfg.fixed_num_steps_no_grad, cfg.fixed_num_steps_with_grad],
        dtype=torch.long,
        device=device,
    )
    autocast_dtype = torch.bfloat16 if cfg.precision == "bf16-mixed" else torch.float16

    start_time = time.time()
    total_weighted_loss = 0.0
    total_target_tokens = 0
    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio_input_features = batch["audio_input_features"].to(device)

        valid_targets = int((labels != -100).sum().item())
        if valid_targets == 0:
            continue

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    num_steps=fixed_num_steps,
                    audio_input_features=audio_input_features,
                )

        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss during eval at step={step}: {loss}")

        total_weighted_loss += float(loss.item()) * valid_targets
        total_target_tokens += valid_targets

        if step == 1 or step % 50 == 0:
            print(
                f"[audio-caption-eval] step={step} batch_loss={loss.item():.4f} "
                f"valid_targets={valid_targets} audio={batch['audio_paths'][0]}"
            )

    if total_target_tokens == 0:
        raise RuntimeError("No valid target tokens found during evaluation.")

    mean_loss = total_weighted_loss / total_target_tokens
    elapsed = time.time() - start_time
    print("--------------------------------------------------------------------")
    print(
        f"[audio-caption-eval] checkpoint_dir={cfg.checkpoint_dir} "
        f"dataset_size={len(dataset)} num_batches={len(dataloader)} "
        f"target_tokens={total_target_tokens} mean_loss={mean_loss:.6f}"
    )
    print(f"[audio-caption-eval] elapsed={elapsed:.2f}s")
    if device.type == "cuda":
        print(
            f"[audio-caption-eval] max_mem_allocated={torch.cuda.max_memory_allocated(device) / float(1024**3):.3f} GB "
            f"max_mem_reserved={torch.cuda.max_memory_reserved(device) / float(1024**3):.3f} GB"
        )
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    main()
