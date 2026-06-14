"""Shared helpers for Huginn audio-text alignment evaluation."""

from __future__ import annotations

import argparse
import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


DEFAULT_BASE_MODEL_NAME = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
DEFAULT_AUDIO_MODEL_DIR = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-whisper-v1"
)
DEFAULT_AUDIO_ENCODER_NAME = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-small"
DEFAULT_CHECKPOINT_DIR = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "code/recurrent-pretraining-main/outputs/huginn-audio-whisper-clotho-caption-v1/checkpoint-2560"
)
DEFAULT_DATASET_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn"
DEFAULT_EVAL_MANIFEST = "test_expand.jsonl"


@dataclass
class GroupedEvalRecord:
    audio_path: str
    references: list[str]


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


def add_common_eval_args(parser: argparse.ArgumentParser, default_output_dir: str = "results"):
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--eval_manifest", default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL_NAME)
    parser.add_argument("--audio_model_dir", default=DEFAULT_AUDIO_MODEL_DIR)
    parser.add_argument("--audio_encoder_name", default=DEFAULT_AUDIO_ENCODER_NAME)
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--precision", default="bf16-mixed", choices=["bf16-mixed", "fp16-mixed", "fp32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=74)


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


def maybe_text_list(value) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        output = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    output.append(text)
        return output
    return []


def extract_references(record: dict) -> list[str]:
    refs: list[str] = []
    for key in ["references", "captions", "caption_list", "ref_captions"]:
        refs.extend(maybe_text_list(record.get(key)))
    if "caption" in record:
        refs.extend(maybe_text_list(record.get("caption")))
    if "text" in record:
        refs.extend(maybe_text_list(record.get("text")))

    deduped = []
    seen = set()
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def load_grouped_eval_records(dataset_dir: Path, manifest_name: str) -> list[GroupedEvalRecord]:
    manifest_path = dataset_dir / manifest_name
    with manifest_path.open("r", encoding="utf-8") as f:
        if manifest_path.suffix.lower() == ".json":
            payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {manifest_path}")
            records = payload
        else:
            records = [json.loads(line) for line in f if line.strip()]

    grouped: dict[str, list[str]] = {}
    for record in records:
        audio_path = record["audio_path"]
        refs = extract_references(record)
        grouped.setdefault(audio_path, []).extend(refs)

    output = []
    for audio_path, refs in grouped.items():
        deduped = []
        seen = set()
        for ref in refs:
            if ref not in seen:
                seen.add(ref)
                deduped.append(ref)
        output.append(GroupedEvalRecord(audio_path=audio_path, references=deduped))

    if not output:
        raise ValueError(f"No evaluation records found in {manifest_path}")
    return sorted(output, key=lambda item: item.audio_path)


class GroupedClothoEvalDataset(Dataset):
    def __init__(self, dataset_dir: str, manifest_name: str, target_sr: int, max_audio_seconds: float):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.target_sr = target_sr
        self.max_audio_seconds = max_audio_seconds
        self.records = load_grouped_eval_records(self.dataset_dir, manifest_name)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        last_error = None
        for offset in range(len(self.records)):
            record = self.records[(idx + offset) % len(self.records)]
            audio_path = self.dataset_dir / record.audio_path
            try:
                waveform = load_wav_mono(audio_path, self.target_sr, self.max_audio_seconds)
                return {
                    "audio": waveform,
                    "audio_path": str(audio_path),
                    "references": record.references,
                }
            except (EOFError, wave.Error, ValueError) as exc:
                last_error = exc
                print(f"[audio-align][bad-audio] skip path={audio_path} error={type(exc).__name__}: {exc}")
                continue

        raise RuntimeError(f"Failed to load any audio example from eval dataset; last_error={last_error}")


def build_audio_collate_fn(processor, target_sample_rate: int):
    def collate_fn(batch):
        audios = [sample["audio"] for sample in batch]
        audio_inputs = processor.feature_extractor(
            audios,
            sampling_rate=target_sample_rate,
            return_tensors="pt",
        )
        return {
            "audio_input_features": audio_inputs["input_features"],
            "audio_paths": [sample["audio_path"] for sample in batch],
            "references": [sample["references"] for sample in batch],
        }

    return collate_fn


def create_eval_dataloader(
    dataset_dir: str,
    eval_manifest: str,
    processor,
    target_sample_rate: int,
    max_audio_seconds: float,
    batch_size: int,
    selected_indices: list[int] | None = None,
) -> tuple[GroupedClothoEvalDataset, DataLoader, list[int]]:
    dataset = GroupedClothoEvalDataset(dataset_dir, eval_manifest, target_sample_rate, max_audio_seconds)
    if selected_indices is None:
        selected_indices = list(range(len(dataset)))
    subset = torch.utils.data.Subset(dataset, selected_indices)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_audio_collate_fn(processor, target_sample_rate),
    )
    return dataset, dataloader, selected_indices


def load_eval_components(
    checkpoint_dir: str,
    base_model_name: str,
    audio_model_dir: str,
    audio_encoder_name: str,
    precision: str,
    device: str,
):
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_model_dtype(precision)

    config = AutoConfig.from_pretrained(audio_model_dir, trust_remote_code=True)
    config.audio_encoder_name = audio_encoder_name
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    base_load = model.load_huginn_backbone_from_pretrained(base_model_name, torch_dtype=torch.float32)
    init_state_path = Path(checkpoint_dir) / "trainable_state.pt"
    init_state = torch.load(init_state_path, map_location="cpu")
    delta_load = model.load_state_dict(init_state, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(audio_encoder_name)

    model.to(device=torch_device, dtype=model_dtype)
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "device": torch_device,
        "model_dtype": model_dtype,
        "base_load": base_load,
        "delta_load": delta_load,
    }


def mean_pool_embeddings(embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return embeddings.mean(dim=1)
    mask = mask.to(embeddings.device).unsqueeze(-1).to(dtype=embeddings.dtype)
    summed = (embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def extract_audio_projected_tokens(model, audio_input_features: torch.Tensor) -> torch.Tensor:
    encoder_outputs = model.audio_encoder(input_features=audio_input_features, return_dict=True)
    audio_hidden = model.temporal_compressor(encoder_outputs.last_hidden_state)
    return model.audio_projector(audio_hidden)


def compute_audio_embeddings(
    model,
    audio_input_features: torch.Tensor,
) -> torch.Tensor:
    projected_tokens = extract_audio_projected_tokens(model, audio_input_features)
    return mean_pool_embeddings(projected_tokens)


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def summarize_references(references: Iterable[str], limit: int = 2) -> str:
    refs = list(references)
    if not refs:
        return "<empty>"
    shown = refs[:limit]
    suffix = "" if len(refs) <= limit else f" ... (+{len(refs) - limit} more)"
    return " | ".join(shown) + suffix


def sample_indices(total_size: int, sample_count: int, seed: int) -> list[int]:
    if sample_count >= total_size:
        return list(range(total_size))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_size), sample_count))


def ensure_non_empty_text(text: str, tokenizer) -> str:
    if text.strip():
        return text
    return tokenizer.eos_token or "."


def compute_reference_caption_embeddings(
    model,
    tokenizer,
    reference_groups: list[list[str]],
    device: torch.device,
    max_length: int = 192,
    batch_size: int = 64,
) -> torch.Tensor:
    flat_texts: list[str] = []
    owners: list[int] = []
    for group_idx, refs in enumerate(reference_groups):
        for ref in refs:
            flat_texts.append(ensure_non_empty_text(ref, tokenizer))
            owners.append(group_idx)

    if not flat_texts:
        raise ValueError("No references available to encode.")

    per_ref_embeddings = []
    with torch.inference_mode():
        for start in range(0, len(flat_texts), batch_size):
            chunk = flat_texts[start : start + batch_size]
            batch = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text_embeds = model.get_input_embeddings()(input_ids)
            pooled = mean_pool_embeddings(text_embeds, attention_mask)
            per_ref_embeddings.append(pooled)

    all_ref_embeddings = torch.cat(per_ref_embeddings, dim=0)
    hidden_dim = all_ref_embeddings.shape[1]
    grouped = torch.zeros((len(reference_groups), hidden_dim), device=device, dtype=all_ref_embeddings.dtype)
    counts = torch.zeros((len(reference_groups), 1), device=device, dtype=all_ref_embeddings.dtype)
    owner_tensor = torch.tensor(owners, device=device, dtype=torch.long)
    grouped.index_add_(0, owner_tensor, all_ref_embeddings)
    counts.index_add_(0, owner_tensor, torch.ones((len(owners), 1), device=device, dtype=all_ref_embeddings.dtype))
    return grouped / counts.clamp_min(1.0)


def collect_audio_embeddings_from_dataloader(
    model,
    dataloader: DataLoader,
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, list[str], list[list[str]]]:
    autocast_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    all_embeddings = []
    all_paths = []
    all_references = []
    with torch.inference_mode():
        for batch in dataloader:
            audio_input_features = batch["audio_input_features"].to(device)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                pooled = compute_audio_embeddings(model, audio_input_features)
            all_embeddings.append(pooled.float())
            all_paths.extend(batch["audio_paths"])
            all_references.extend(batch["references"])

    return torch.cat(all_embeddings, dim=0), all_paths, all_references
