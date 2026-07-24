"""ACAVCAPS WebDataset registration for the LoSATok Swift routes.

The public ACAVCAPS tree is read-only. This plugin consumes the private
tar-level schedule manifest and exposes a finite Hugging Face IterableDataset
whose rows contain the JSON caption and FLAC bytes from each WebDataset sample.
The base model plugin selects legacy fixed-32 or dynamic 90-second prefixes from
``HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS``; this dataset layer is shared by both.
The model/template registration is imported from the existing LoSATok plugin;
only this ACAVCAPS route adds the dataset registration.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_PLUGIN_PATH = Path(__file__).with_name("huginn_losatok_swift.py")
DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_sampled.json"
)
MANIFEST_ENV = "ACAVCAPS_WDS_MANIFEST"
BUFFER_ENV = "ACAVCAPS_WDS_BUFFER_SIZE"
MAX_TARS_ENV = "ACAVCAPS_WDS_MAX_TARS_PER_STAGE"
DATASET_NAME = "acavcaps_wds"
EXPECTED_STAGE_NAMES = ("stage1", "stage2", "stage3")
DEFAULT_BUFFER_SIZE = 512


def _load_base_plugin() -> Any:
    spec = importlib.util.spec_from_file_location("huginn_losatok_swift_base", BASE_PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load LoSATok base plugin: {BASE_PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_PLUGIN = _load_base_plugin()


def _manifest_path(dataset_meta: Any | None = None) -> Path:
    configured = os.environ.get(MANIFEST_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    metadata_path = getattr(dataset_meta, "dataset_path", None)
    if metadata_path:
        return Path(str(metadata_path)).expanduser().resolve()
    return DEFAULT_MANIFEST


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ACAVCAPS WebDataset manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), list):
        raise ValueError(f"Invalid ACAVCAPS WebDataset manifest: {path}")
    stage_names = tuple(str(stage.get("name", "unknown")) for stage in payload["stages"])
    if stage_names != EXPECTED_STAGE_NAMES:
        raise ValueError(f"Unexpected ACAVCAPS stage order: {stage_names}")
    dataset_root = Path(str(payload.get("dataset_root", ""))).resolve()
    expected_root = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
    if dataset_root != expected_root:
        raise ValueError(f"Unexpected ACAVCAPS dataset_root: {dataset_root}")
    return payload


def _caption(payload: Any, label: str) -> str:
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: JSON payload is not an object")
    value = payload.get("long")
    if isinstance(value, str):
        captions = [value.strip()] if value.strip() else []
    elif isinstance(value, list):
        captions = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    else:
        captions = []
    if not captions:
        raise ValueError(f"{label}: JSON long has no non-empty caption")
    return captions[0]


def _buffer_size(manifest: dict[str, Any]) -> int:
    configured = os.environ.get(BUFFER_ENV, "").strip()
    value = int(configured) if configured else int(manifest.get("sample_shuffle_buffer", DEFAULT_BUFFER_SIZE))
    if value <= 0:
        raise ValueError(f"{BUFFER_ENV} must be positive, got {value}")
    return value


def _max_tars_per_stage() -> int | None:
    configured = os.environ.get(MAX_TARS_ENV, "").strip()
    if not configured:
        return None
    value = int(configured)
    if value <= 0:
        raise ValueError(f"{MAX_TARS_ENV} must be positive when set, got {value}")
    return value


def iter_acavcaps_rows(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    try:
        import webdataset as wds
    except Exception as exc:  # noqa: BLE001 - expose remote environment failure
        raise RuntimeError(f"webdataset import failed: {type(exc).__name__}: {exc}") from exc

    dataset_root = Path(str(manifest["dataset_root"])).resolve()
    buffer_size = _buffer_size(manifest)
    max_tars_per_stage = _max_tars_per_stage()
    for stage in manifest["stages"]:
        stage_name = str(stage["name"])
        stage_tars = stage.get("tars", [])
        if max_tars_per_stage is not None:
            stage_tars = stage_tars[:max_tars_per_stage]
        for tar_index, tar_entry in enumerate(stage_tars):
            if not isinstance(tar_entry, dict) or not isinstance(tar_entry.get("path"), str):
                raise ValueError(f"Invalid tar entry in {stage_name}: {tar_entry!r}")
            tar_path = Path(str(tar_entry["path"])).resolve()
            if dataset_root not in tar_path.parents:
                raise ValueError(f"ACAVCAPS tar is outside the public root: {tar_path}")
            if not tar_path.is_file():
                raise FileNotFoundError(f"ACAVCAPS tar does not exist: {tar_path}")

            dataset = wds.WebDataset(str(tar_path), shardshuffle=False)
            try:
                dataset = dataset.shuffle(buffer_size)
            except TypeError:
                dataset = dataset.shuffle(size=buffer_size)
            for sample in dataset:
                key = str(sample.get("__key__", ""))
                if not key:
                    raise ValueError(f"{stage_name}:{tar_path}: sample has no __key__")
                json_bytes = sample.get("json")
                flac_bytes = sample.get("flac")
                if not isinstance(json_bytes, (bytes, bytearray, memoryview)):
                    raise TypeError(f"{stage_name}:{tar_path}:{key}: json is not bytes")
                if not isinstance(flac_bytes, (bytes, bytearray, memoryview)):
                    raise TypeError(f"{stage_name}:{tar_path}:{key}: flac is not bytes")
                label = f"{stage_name}:{tar_path.name}:{key}"
                payload = json.loads(bytes(json_bytes).decode("utf-8"))
                caption = _caption(payload, label)
                yield {
                    "messages": [
                        {
                            "role": "system",
                            "content": _BASE_PLUGIN.DEFAULT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": "Listen to the audio and describe it.",
                        },
                        {"role": "assistant", "content": caption},
                    ],
                    "audios": [
                        {
                            "audio_bytes": bytes(flac_bytes),
                            "audio_format": "flac",
                            "sample_id": key,
                            "stage": stage_name,
                            "tar_path": str(tar_path),
                            "tar_index": str(tar_index),
                        }
                    ],
                    "metadata": {
                        "stage": stage_name,
                        "sample_id": key,
                        "tar_path": str(tar_path),
                    },
                }


def build_dataset(manifest_path: Path):
    from datasets import IterableDataset

    manifest = _load_manifest(manifest_path)
    return IterableDataset.from_generator(lambda: iter_acavcaps_rows(manifest))


def _register_dataset() -> None:
    try:
        from swift.dataset import register_dataset
    except ImportError:
        from swift.dataset.register import register_dataset  # type: ignore
    try:
        from swift.dataset.register import DatasetMeta
    except ImportError:
        from swift.llm import DatasetMeta  # type: ignore
    from swift.dataset.loader import DatasetLoader

    manifest = _manifest_path()

    class ACAVCAPSWDSLoader(DatasetLoader):
        def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
            del dataset_syntax, use_hf
            path = _manifest_path(dataset_meta)
            dataset = build_dataset(path)
            print(
                f"[HuginnLoSATokACAVCAPS] loaded IterableDataset manifest={path} "
                f"buffer_size={_buffer_size(_load_manifest(path))} "
                f"max_tars_per_stage={_max_tars_per_stage() or 'all'}"
            )
            return dataset

    metadata = DatasetMeta(
        dataset_path=str(manifest),
        dataset_name=DATASET_NAME,
        loader=ACAVCAPSWDSLoader,
    )
    try:
        register_dataset(metadata, exist_ok=True)
    except TypeError as exc:
        if "exist_ok" not in str(exc):
            raise
        register_dataset(metadata)
    print(
        f"[HuginnLoSATokACAVCAPS] registered dataset path={manifest} "
        f"name={DATASET_NAME} loader={ACAVCAPSWDSLoader.__name__}"
    )


_register_dataset()
