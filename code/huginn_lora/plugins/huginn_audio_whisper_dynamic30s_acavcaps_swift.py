"""Current Whisper dynamic-30s ACAVCAPS flat-tar dataset route.

The model/template registration comes from the current Whisper plugin.  This
module adds only a private-manifest WebDataset loader whose top-level tar list
has already been globally shuffled across all ACAVCAPS source stages.  FLAC
bytes are not materialized into dataset rows: the current Whisper template
reopens the public tar and decodes the referenced member at training time.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_PLUGIN_PATH = Path(__file__).with_name("huginn_audio_whisper_dynamic90s_swift.py")
DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps/acavcaps_flat_global_tar_shuffle_seed20260723.json"
)
MANIFEST_ENV = "ACAVCAPS_FLAT_MANIFEST"
BUFFER_ENV = "ACAVCAPS_FLAT_SAMPLE_SHUFFLE_BUFFER"
MAX_TARS_ENV = "ACAVCAPS_FLAT_MAX_TARS"
DATASET_NAME = "huginn_audio_whisper_dynamic30s_acavcaps"
EXPECTED_POLICY = "global_tar_order_shuffle_all_stages_v1_per_tar_buffer_shuffle"
EXPECTED_DATASET_ROOT = Path("/hpc_stor03/public/shared/data/raa/ACAVCAPS")
EXPECTED_SOURCE_STAGE_ORDER = ("stage1", "stage2", "stage3")
EXPECTED_TAR_COUNT = 1071
EXPECTED_SAMPLE_COUNT = 4_664_169
DEFAULT_BUFFER_SIZE = 512


def _load_base_plugin() -> Any:
    module_name = "huginn_audio_whisper_dynamic90s_base_acavcaps"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, BASE_PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load current Whisper base plugin: {BASE_PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
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
        raise FileNotFoundError(f"ACAVCAPS flat manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ACAVCAPS flat manifest root is not an object: {path}")
    if payload.get("schedule_policy") != EXPECTED_POLICY:
        raise ValueError(
            "ACAVCAPS flat manifest schedule policy mismatch: "
            f"{payload.get('schedule_policy')!r}"
        )
    if Path(str(payload.get("dataset_root", ""))).resolve() != EXPECTED_DATASET_ROOT.resolve():
        raise ValueError(f"Unexpected ACAVCAPS dataset root: {payload.get('dataset_root')!r}")
    if payload.get("public_root_mutation") != "forbidden" or payload.get("scan_mode") != "derived_from_full":
        raise ValueError("ACAVCAPS flat manifest is not a read-only derivation from the full scan")
    if tuple(payload.get("source_stage_order", [])) != EXPECTED_SOURCE_STAGE_ORDER:
        raise ValueError(f"Unexpected ACAVCAPS source stage order: {payload.get('source_stage_order')!r}")
    tars = payload.get("tars")
    if not isinstance(tars, list) or not tars:
        raise ValueError("ACAVCAPS flat manifest must contain a non-empty top-level tars list")
    if "stages" in payload:
        raise ValueError("ACAVCAPS flat manifest must not expose stages as training boundaries")
    if int(payload.get("tar_count", -1)) != len(tars):
        raise ValueError("ACAVCAPS flat manifest tar_count does not match tars length")
    if len(tars) != EXPECTED_TAR_COUNT:
        raise ValueError(f"ACAVCAPS flat manifest must contain all {EXPECTED_TAR_COUNT} tars, got {len(tars)}")
    if int(payload.get("sample_count", -1)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"ACAVCAPS flat manifest sample_count mismatch: expected={EXPECTED_SAMPLE_COUNT} "
            f"actual={payload.get('sample_count')!r}"
        )
    return payload


def _buffer_size(manifest: dict[str, Any]) -> int:
    configured = os.environ.get(BUFFER_ENV, "").strip()
    value = int(configured) if configured else int(manifest.get("sample_shuffle_buffer", DEFAULT_BUFFER_SIZE))
    if value <= 0:
        raise ValueError(f"{BUFFER_ENV} must be positive, got {value}")
    return value


def _max_tars() -> int | None:
    configured = os.environ.get(MAX_TARS_ENV, "").strip()
    if not configured:
        return None
    value = int(configured)
    if value <= 0:
        raise ValueError(f"{MAX_TARS_ENV} must be positive when set")
    return value


def _identity_nodesplitter(urls):
    """Leave each one-tar pipeline visible to all ranks.

    Accelerate's DataLoaderDispatcher owns rank-level batch dispatch for this
    route.  Adding WebDataset rank sharding here would drop or duplicate rows.
    """
    return urls


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


def _audio_member_from_key(key: str) -> str:
    return key if key.lower().endswith(".flac") else f"{key}.flac"


def iter_acavcaps_rows(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    try:
        import webdataset as wds
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"webdataset import failed: {type(exc).__name__}: {exc}") from exc

    dataset_root = Path(str(manifest["dataset_root"])).resolve()
    buffer_size = _buffer_size(manifest)
    max_tars = _max_tars()
    tar_entries = manifest["tars"] if max_tars is None else manifest["tars"][:max_tars]
    for tar_index, tar_entry in enumerate(tar_entries):
        if not isinstance(tar_entry, dict) or not isinstance(tar_entry.get("path"), str):
            raise ValueError(f"Invalid flat ACAVCAPS tar entry at index {tar_index}: {tar_entry!r}")
        if int(tar_entry.get("order_index", -1)) != tar_index:
            raise ValueError(
                f"ACAVCAPS flat tar order_index mismatch: list_index={tar_index} "
                f"entry={tar_entry.get('order_index')!r}"
            )
        tar_path = Path(str(tar_entry["path"])).resolve()
        if dataset_root not in tar_path.parents:
            raise ValueError(f"ACAVCAPS tar is outside the public root: {tar_path}")
        if not tar_path.is_file():
            raise FileNotFoundError(f"ACAVCAPS tar does not exist: {tar_path}")

        dataset = wds.WebDataset(
            str(tar_path),
            shardshuffle=False,
            nodesplitter=_identity_nodesplitter,
        )
        try:
            dataset = dataset.shuffle(buffer_size)
        except TypeError:
            dataset = dataset.shuffle(size=buffer_size)
        for sample in dataset:
            key = str(sample.get("__key__", ""))
            if not key:
                raise ValueError(f"{tar_path}: WebDataset sample has no __key__")
            json_bytes = sample.get("json")
            flac_bytes = sample.get("flac")
            if not isinstance(json_bytes, (bytes, bytearray, memoryview)):
                raise TypeError(f"{tar_path}:{key}: json is not bytes")
            if not isinstance(flac_bytes, (bytes, bytearray, memoryview)):
                raise TypeError(f"{tar_path}:{key}: flac is not bytes")
            label = f"flat_tar_order={tar_index}:{tar_path.name}:{key}"
            payload = json.loads(bytes(json_bytes).decode("utf-8"))
            caption = _caption(payload, label)
            source_stage = str(tar_entry.get("source_stage", "unknown"))
            if source_stage not in EXPECTED_SOURCE_STAGE_ORDER:
                raise ValueError(f"Unexpected ACAVCAPS source_stage at {label}: {source_stage!r}")
            category = str(tar_entry.get("category", "unknown"))
            yield {
                "messages": [
                    {"role": "system", "content": _BASE_PLUGIN.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "Listen to the audio and describe it."},
                    {"role": "assistant", "content": caption},
                ],
                "audios": [
                    {
                        "tar_path": str(tar_path),
                        "audio_member": _audio_member_from_key(key),
                        "sample_id": key,
                        "source_stage": source_stage,
                        "source_category": category,
                        "flat_tar_order": tar_index,
                    }
                ],
                "metadata": {
                    "sample_id": key,
                    "tar_path": str(tar_path),
                    "source_stage": source_stage,
                    "source_category": category,
                    "flat_tar_order": tar_index,
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

    class HuginnWhisperACAVCAPSLoader(DatasetLoader):
        def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
            del dataset_syntax, use_hf
            path = _manifest_path(dataset_meta)
            loaded = _load_manifest(path)
            dataset = build_dataset(path)
            print(
                "[HuginnAudioDynamic30sACAVCAPS] loaded flat IterableDataset "
                f"manifest={path} tar_count={loaded['tar_count']} "
                f"sample_count={loaded.get('sample_count')} "
                f"buffer={_buffer_size(loaded)} max_tars={_max_tars() or 'all'}",
                flush=True,
            )
            return dataset

    metadata = DatasetMeta(
        dataset_path=str(manifest),
        dataset_name=DATASET_NAME,
        loader=HuginnWhisperACAVCAPSLoader,
    )
    try:
        register_dataset(metadata, exist_ok=True)
    except TypeError as exc:
        if "exist_ok" not in str(exc):
            raise
        register_dataset(metadata)
    print(
        f"[HuginnAudioDynamic30sACAVCAPS] registered dataset path={manifest} "
        f"name={DATASET_NAME}",
        flush=True,
    )


_register_dataset()
