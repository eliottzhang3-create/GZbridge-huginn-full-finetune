"""Inspect the WebDataset-to-Swift IterableDataset boundary without loading a model."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_sampled.json"
)


def load_plugin(path: Path):
    spec = importlib.util.spec_from_file_location("huginn_losatok_acavcaps_wds_swift", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load external plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    manifest = Path(os.environ.get("ACAVCAPS_WDS_MANIFEST", str(DEFAULT_MANIFEST))).expanduser().resolve()
    plugin_path = repo_root / "code" / "huginn_lora" / "plugins" / "huginn_losatok_acavcaps_wds_swift.py"
    os.environ["ACAVCAPS_WDS_MANIFEST"] = str(manifest)
    os.environ.setdefault("ACAVCAPS_WDS_MAX_TARS_PER_STAGE", "2")

    print("========== ACAVCAPS SWIFT ITERABLE DATASET INSPECT ==========")
    print(f"[env] python={sys.version.split()[0]}")
    print(f"[env] platform={platform.platform()}")
    print(f"[manifest] path={manifest}")
    print(f"[plugin] path={plugin_path}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    max_tars = int(os.environ["ACAVCAPS_WDS_MAX_TARS_PER_STAGE"])
    expected_stage_counts: dict[str, int] = {}
    for stage in manifest_payload.get("stages", []):
        name = str(stage.get("name"))
        selected_tars = stage.get("tars", [])[:max_tars]
        counts = [entry.get("json_count") for entry in selected_tars]
        if not counts or any(not isinstance(count, int) for count in counts):
            raise ValueError(
                "Swift IterableDataset inspect requires sampled manifest json_count values for "
                f"stage={name}; run the sampled preflight first"
            )
        expected_stage_counts[name] = sum(counts)

    plugin = load_plugin(plugin_path)
    from datasets import IterableDataset as HFIterableDataset
    from swift.dataset import load_dataset

    print(f"[swift] registered_loader={plugin.DATASET_NAME}")
    try:
        train_dataset, val_dataset = load_dataset(
            str(manifest),
            split_dataset_ratio=0.0,
            shuffle=False,
            num_proc=1,
            streaming=True,
        )
    except TypeError as first_error:
        print(f"[swift] load_dataset_compat_retry={type(first_error).__name__}: {first_error}")
        train_dataset, val_dataset = load_dataset(
            str(manifest),
            split_dataset_ratio=0.0,
            shuffle=False,
            num_proc=1,
        )

    if not isinstance(train_dataset, HFIterableDataset):
        raise TypeError(f"Swift returned {type(train_dataset)}, expected datasets.IterableDataset")
    if val_dataset is not None:
        raise RuntimeError(f"split_dataset_ratio=0 must not produce a validation dataset: {type(val_dataset)}")

    print(f"[dataset] type={type(train_dataset)}")
    print(f"[dataset] iterable={isinstance(train_dataset, HFIterableDataset)}")
    rows = []
    stage_counts: dict[str, int] = {}
    for index, row in enumerate(train_dataset):
        if not isinstance(row, dict):
            raise TypeError(f"row {index} has type {type(row).__name__}")
        messages = row.get("messages")
        audios = row.get("audios")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"row {index} has unexpected messages={messages!r}")
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], dict):
            raise ValueError(f"row {index} has unexpected audios={audios!r}")
        audio_bytes = audios[0].get("audio_bytes")
        if not isinstance(audio_bytes, (bytes, bytearray, memoryview)) or len(audio_bytes) == 0:
            raise TypeError(f"row {index} audio_bytes is invalid: {type(audio_bytes).__name__}")
        stage_name = str(audios[0].get("stage"))
        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
        if index < 8:
            rows.append(row)
            print(
                f"[row] index={index} stage={stage_name} sample_id={audios[0].get('sample_id')} "
                f"audio_bytes={len(audio_bytes)} caption_chars={len(messages[-1].get('content', ''))}",
                flush=True,
            )
        if (index + 1) % 5000 == 0:
            print(f"[progress] rows={index + 1} stage_counts={stage_counts}", flush=True)

    if len(rows) != 8:
        raise RuntimeError(f"IterableDataset ended before the inspect quota: rows={len(rows)}")
    if stage_counts != expected_stage_counts:
        raise RuntimeError(f"Swift IterableDataset stage counts mismatch: {stage_counts} != {expected_stage_counts}")
    waveform = plugin._BASE_PLUGIN.decode_audio_bytes_16k(
        bytes(rows[0]["audios"][0]["audio_bytes"]),
        f"{rows[0]['audios'][0].get('stage')}:{rows[0]['audios'][0].get('sample_id')}",
    )
    if waveform.numel() <= 0:
        raise RuntimeError("LoSATok template byte decoder returned an empty waveform")
    print(f"[audio] bytes_decoder=pass waveform_samples={waveform.numel()} sample_rate=16000")
    print(f"[summary] rows_checked={sum(stage_counts.values())} stage_counts={stage_counts}")
    print("========== ACAVCAPS SWIFT ITERABLE DATASET INSPECT DONE ==========")
    print("[result] status=PASS swift_load_dataset=pass hf_iterable=pass bytes_schema=pass bytes_decode=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
