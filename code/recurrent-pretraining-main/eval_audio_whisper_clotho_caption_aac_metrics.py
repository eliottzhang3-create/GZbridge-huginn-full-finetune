"""Generate Clotho captions and evaluate them with aac_metrics."""

from __future__ import annotations

import importlib.machinery
import json
import random
import socket
import sys
import time
import types
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def install_torchvision_stub():
    """Avoid optional torchvision imports inside metric dependencies."""
    if "torchvision" in sys.modules:
        return

    class _MissingTorchvisionAttr:
        def __init__(self, name: str):
            self._name = name

        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                f"torchvision stub attribute '{self._name}' was called. "
                "This evaluation script expects caption metrics only and should not need real torchvision."
            )

        def __getattr__(self, item: str):
            return _MissingTorchvisionAttr(f"{self._name}.{item}")

    root = types.ModuleType("torchvision")
    root.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    sys.modules["torchvision"] = root

    submodules = {}
    for name in ["datasets", "io", "models", "ops", "transforms", "utils", "_meta_registrations"]:
        module = types.ModuleType(f"torchvision.{name}")
        module.__spec__ = importlib.machinery.ModuleSpec(f"torchvision.{name}", loader=None)
        setattr(root, name, module)
        sys.modules[f"torchvision.{name}"] = module
        submodules[name] = module

    class _InterpolationMode:
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 1
        HAMMING = 4
        BOX = 5

    submodules["transforms"].InterpolationMode = _InterpolationMode
    submodules["models"].resnet50 = _MissingTorchvisionAttr("torchvision.models.resnet50")
    submodules["models"].vgg16 = _MissingTorchvisionAttr("torchvision.models.vgg16")
    submodules["models"].VGG16_Weights = _MissingTorchvisionAttr("torchvision.models.VGG16_Weights")
    submodules["models"].ResNet50_Weights = _MissingTorchvisionAttr("torchvision.models.ResNet50_Weights")
    submodules["models"].__getattr__ = lambda name: _MissingTorchvisionAttr(f"torchvision.models.{name}")
    root.__getattr__ = lambda name: _MissingTorchvisionAttr(f"torchvision.{name}")


install_torchvision_stub()

from aac_metrics import evaluate as aac_evaluate


@dataclass
class CLISettings:
    run_name: str = "huginn-audio-whisper-clotho-aac-metrics"
    out_path: str = "outputs"
    base_model_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
    audio_model_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-whisper-v1"
    audio_encoder_name: str = "/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-small"
    checkpoint_dir: str = (
        "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
        "code/recurrent-pretraining-main/outputs/huginn-audio-whisper-clotho-caption-v1/checkpoint-2560"
    )
    dataset_dir: str = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn"
    eval_manifest: str = "test_expand.jsonl"
    max_seq_length: int = 192
    target_sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    batch_size: int = 5
    precision: str = "bf16-mixed"
    seed: int = 74
    max_new_tokens: int = 64
    num_beams: int = 1
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


def maybe_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    result.append(text)
        return result
    return []


def extract_references(record: dict) -> list[str]:
    refs: list[str] = []
    for key in ["references", "captions", "caption_list", "ref_captions"]:
        refs.extend(maybe_text_list(record.get(key)))
    if "caption" in record:
        refs.extend(maybe_text_list(record.get("caption")))
    if "text" in record:
        refs.extend(maybe_text_list(record.get("text")))

    seen = set()
    deduped = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def load_grouped_eval_records(dataset_dir: Path, manifest_name: str) -> list[dict]:
    manifest_path = dataset_dir / manifest_name
    with manifest_path.open("r", encoding="utf-8") as f:
        if manifest_path.suffix.lower() == ".json":
            payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {manifest_path}")
            records = payload
        else:
            records = [json.loads(line) for line in f if line.strip()]

    grouped: dict[str, dict] = {}
    for record_idx, record in enumerate(records):
        audio_path = record["audio_path"]
        refs = extract_references(record)
        if audio_path not in grouped:
            grouped[audio_path] = {
                "audio_path": audio_path,
                "references": [],
                "source_record_indices": [],
            }
        grouped[audio_path]["source_record_indices"].append(record_idx)
        grouped[audio_path]["references"].extend(refs)

    final_records = []
    for group in grouped.values():
        seen = set()
        deduped_refs = []
        for ref in group["references"]:
            if ref not in seen:
                seen.add(ref)
                deduped_refs.append(ref)
        group["references"] = deduped_refs
        final_records.append(group)

    if not final_records:
        raise ValueError(f"No evaluation records found in {manifest_path}")
    return sorted(final_records, key=lambda item: item["audio_path"])


class ClothoCaptionEvalDataset(Dataset):
    def __init__(self, dataset_dir: str, manifest_name: str, target_sr: int, max_audio_seconds: float):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.target_sr = target_sr
        self.max_audio_seconds = max_audio_seconds
        self.examples = load_grouped_eval_records(self.dataset_dir, manifest_name)

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
                    "audio_path": str(audio_path),
                    "references": record["references"],
                }
            except (EOFError, wave.Error, ValueError) as exc:
                last_error = exc
                print(f"[audio-caption-aac][bad-audio] skip path={audio_path} error={type(exc).__name__}: {exc}")
                continue

        raise RuntimeError(f"Failed to load any audio example from eval dataset; last_error={last_error}")


def build_generation_collate_fn(tokenizer, processor, cfg: CLISettings):
    def collate_fn(batch):
        conversations = []
        for _sample in batch:
            conversations.append(
                [
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": cfg.user_prompt},
                ]
            )

        chat_encoding = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding="longest",
            max_length=cfg.max_seq_length,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
        )

        audio = [sample["audio"] for sample in batch]
        audio_inputs = processor.feature_extractor(
            audio,
            sampling_rate=cfg.target_sample_rate,
            return_tensors="pt",
        )

        return {
            "input_ids": chat_encoding["input_ids"],
            "attention_mask": chat_encoding["attention_mask"],
            "audio_input_features": audio_inputs["input_features"],
            "audio_paths": [sample["audio_path"] for sample in batch],
            "references": [sample["references"] for sample in batch],
        }

    return collate_fn


def clean_prediction(text: str) -> str:
    text = text.replace("<|end_turn|>", " ").replace("<|end_text|>", " ")
    text = " ".join(text.split())
    return text.strip()


def tensor_dict_to_python(data: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            converted[key] = float(value.item()) if value.numel() == 1 else value.tolist()
        else:
            converted[key] = value
    return converted


def get_metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


def main():
    cfg = CLISettings()
    seed_everything(cfg.seed)

    output_dir = Path(cfg.out_path) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_model_dtype(cfg.precision)
    print("--------------------------------------------------------------------")
    print(f"---------------- Launching run {cfg.run_name} ----------------")
    print("--------------------------------------------------------------------")
    print(f"Host={socket.gethostname()} Python={sys.version.split(' (')[0]} Torch={torch.__version__} Device={device}")
    print(f"[audio-caption-aac] dataset_dir={cfg.dataset_dir}")
    print(f"[audio-caption-aac] eval_manifest={cfg.eval_manifest}")
    print(f"[audio-caption-aac] checkpoint_dir={cfg.checkpoint_dir}")
    print(f"[audio-caption-aac] batch_size={cfg.batch_size} max_new_tokens={cfg.max_new_tokens}")

    config = AutoConfig.from_pretrained(cfg.audio_model_dir, trust_remote_code=True)
    config.audio_encoder_name = cfg.audio_encoder_name
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    base_load = model.load_huginn_backbone_from_pretrained(cfg.base_model_name, torch_dtype=torch.float32)
    print(
        f"[audio-caption-aac] backbone load missing={len(base_load.missing_keys)} "
        f"unexpected={len(base_load.unexpected_keys)}"
    )

    init_state_path = Path(cfg.checkpoint_dir) / "trainable_state.pt"
    init_state = torch.load(init_state_path, map_location="cpu")
    delta_load = model.load_state_dict(init_state, strict=False)
    print(
        f"[audio-caption-aac] delta load missing={len(delta_load.missing_keys)} "
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
        collate_fn=build_generation_collate_fn(tokenizer, processor, cfg),
    )
    multi_ref_count = [len(item["references"]) for item in dataset.examples]
    print(
        f"[audio-caption-aac] dataset_size={len(dataset)} num_batches={len(dataloader)} "
        f"min_refs={min(multi_ref_count)} max_refs={max(multi_ref_count)}"
    )
    print(f"[audio-caption-aac] first_audio_path={dataset[0]['audio_path']}")

    autocast_dtype = torch.bfloat16 if cfg.precision == "bf16-mixed" else torch.float16
    predictions = []
    references = []
    prediction_records = []
    start_time = time.time()

    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio_input_features = batch["audio_input_features"].to(device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_input_features=audio_input_features,
                    do_sample=False,
                    num_beams=cfg.num_beams,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

        prompt_len = input_ids.shape[1]
        generated_ids = outputs[:, prompt_len:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned = [clean_prediction(text) for text in decoded]

        for audio_path, pred, refs in zip(batch["audio_paths"], cleaned, batch["references"]):
            predictions.append(pred)
            references.append(refs)
            prediction_records.append(
                {
                    "audio_path": audio_path,
                    "prediction": pred,
                    "references": refs,
                }
            )

        if step == 1 or step % 20 == 0:
            print(
                f"[audio-caption-aac] step={step} generated={len(cleaned)} "
                f"sample_audio={batch['audio_paths'][0]}"
            )
            print(f"[audio-caption-aac] sample_prediction={cleaned[0]}")

    corpus_scores, sentence_scores = aac_evaluate(predictions, references)
    corpus_scores = tensor_dict_to_python(corpus_scores)
    sentence_scores = tensor_dict_to_python(sentence_scores)
    elapsed = time.time() - start_time
    cider_value = get_metric_value(corpus_scores, "cider_d", "cider")
    spice_value = get_metric_value(corpus_scores, "spice")
    spider_value = get_metric_value(corpus_scores, "spider")

    prediction_path = output_dir / "clotho_test_predictions.json"
    metrics_path = output_dir / "clotho_test_metrics.json"
    summary = {
        "checkpoint_dir": cfg.checkpoint_dir,
        "dataset_size": len(dataset),
        "num_batches": len(dataloader),
        "elapsed_seconds": elapsed,
        "metrics": {
            "cider_d": cider_value,
            "spice": spice_value,
            "spider": spider_value,
        },
        "all_corpus_scores": corpus_scores,
        "sentence_scores_keys": list(sentence_scores.keys()),
    }

    with prediction_path.open("w", encoding="utf-8") as f:
        json.dump(prediction_records, f, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("--------------------------------------------------------------------")
    print(f"[audio-caption-aac] predictions_path={prediction_path}")
    print(f"[audio-caption-aac] metrics_path={metrics_path}")
    print(
        f"[audio-caption-aac] checkpoint_dir={cfg.checkpoint_dir} "
        f"dataset_size={len(dataset)} num_batches={len(dataloader)} "
        f"CIDEr={summary['metrics']['cider_d']} "
        f"SPICE={summary['metrics']['spice']} "
        f"SPIDEr={summary['metrics']['spider']}"
    )
    print(f"[audio-caption-aac] elapsed={elapsed:.2f}s")
    if device.type == "cuda":
        print(
            f"[audio-caption-aac] max_mem_allocated={torch.cuda.max_memory_allocated(device) / float(1024**3):.3f} GB "
            f"max_mem_reserved={torch.cuda.max_memory_reserved(device) / float(1024**3):.3f} GB"
        )
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    main()
