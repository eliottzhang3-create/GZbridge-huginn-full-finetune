from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code" / "huginn_lora"))

from plugins.huginn_audio_swift import (  # noqa: E402
    AUDIO_MODEL_DIR,
    DEFAULT_MAX_AUDIO_SECONDS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SYSTEM_PROMPT,
    create_huginn_audio_model_and_processor,
    get_huginn_backbone_lora_target_modules,
    load_jsonl_records,
    normalize_audio_record,
    summarize_trainable_parameter_groups,
)

DATASET_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn_tiny_train32")
MANIFEST_NAME = "train.jsonl"
BATCH_SIZE = 2
PRECISION = "bf16"
FIXED_NUM_STEPS = (0, 4)


def resolve_dtype() -> torch.dtype:
    if PRECISION == "bf16":
        return torch.bfloat16
    if PRECISION == "fp16":
        return torch.float16
    return torch.float32


def print_group_summary(groups: dict[str, list[str]]):
    print("[audio-swift-smoke] trainable group summary:")
    for group_name, names in groups.items():
        print(f"  - {group_name}: {len(names)}")
        for name in names[:8]:
            print(f"    * {name}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype()

    print("--------------------------------------------------------------------")
    print("[audio-swift-smoke] launch")
    print("--------------------------------------------------------------------")
    print(f"[audio-swift-smoke] host={socket.gethostname()} python={sys.version.split(' (')[0]}")
    print(f"[audio-swift-smoke] torch={torch.__version__} device={device} dtype={dtype}")
    print(f"[audio-swift-smoke] audio_model_dir={AUDIO_MODEL_DIR}")
    print(f"[audio-swift-smoke] dataset_dir={DATASET_DIR}")

    model, processor = create_huginn_audio_model_and_processor(
        model_dir=str(AUDIO_MODEL_DIR),
        model_kwargs={},
    )
    model.to(device=device, dtype=dtype)
    model.train()

    groups = summarize_trainable_parameter_groups(model)
    print_group_summary(groups)
    if groups["audio_encoder"]:
        raise RuntimeError("audio_encoder should be frozen in smoke test")
    if groups["llm"]:
        raise RuntimeError("llm backbone base params should be frozen before LoRA injection")
    if not groups["aligner"]:
        raise RuntimeError("aligner parameters are unexpectedly empty")

    target_modules = get_huginn_backbone_lora_target_modules(model)
    print(f"[audio-swift-smoke] huginn_only_target_modules_count={len(target_modules)}")
    print("[audio-swift-smoke] first_target_modules:")
    for name in target_modules[:20]:
        print(f"  - {name}")

    records = load_jsonl_records(DATASET_DIR / MANIFEST_NAME, max_records=BATCH_SIZE)
    samples = [normalize_audio_record(record, dataset_dir=DATASET_DIR) for record in records]
    batch = processor.collate_audio_sft_batch(
        samples,
        max_length=DEFAULT_MAX_LENGTH,
        sample_rate=DEFAULT_SAMPLE_RATE,
        max_audio_seconds=DEFAULT_MAX_AUDIO_SECONDS,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-4,
    )
    optimizer.zero_grad(set_to_none=True)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audio_input_features = batch["audio_input_features"].to(device)
    num_steps = torch.tensor(FIXED_NUM_STEPS, dtype=torch.long, device=device)

    autocast_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    with torch.autocast(
        device_type="cuda",
        dtype=autocast_dtype,
        enabled=device.type == "cuda" and dtype in {torch.bfloat16, torch.float16},
    ):
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            num_steps=num_steps,
            audio_input_features=audio_input_features,
        )

    loss = outputs.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss in audio swift smoke: {loss}")
    loss.backward()

    missing_grads = []
    print("[audio-swift-smoke] aligner grad summary:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("audio_encoder.") or name.startswith("transformer.") or name.startswith("lm_head."):
            continue
        if param.grad is None:
            missing_grads.append(name)
            print(f"  - {name}: grad=missing")
        else:
            print(f"  - {name}: grad_abs_max={float(param.grad.detach().abs().max().item()):.4e}")

    if missing_grads:
        raise RuntimeError(f"Trainable adapter parameters missing gradients: {missing_grads}")

    optimizer.step()

    print(f"[audio-swift-smoke] loss={float(loss.item()):.4f}")
    print("[audio-swift-smoke] sample_queries:")
    for query in batch["queries"]:
        print(f"  - {query[:120]}")
    print("[audio-swift-smoke] sample_audio_paths:")
    for audio_path in batch["audio_paths"]:
        print(f"  - {audio_path}")
    print("--------------------------------------------------------------------")
    print("[audio-swift-smoke] finished without error")
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    main()
