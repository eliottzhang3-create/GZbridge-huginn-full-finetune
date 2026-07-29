"""Swift dataset registration for the indexed Huginn Whisper dynamic-90s mixture."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HUGINN_LORA_ROOT = REPO_ROOT / "code" / "huginn_lora"
BASE_PLUGIN_PATH = Path(__file__).with_name("huginn_audio_whisper_dynamic90s_swift.py")
DEFAULT_REGISTRY = (
    REPO_ROOT
    / "data"
    / "audio_swift"
    / "huginn_whisper_dynamic90s_multitask"
    / "v1"
    / "pool_registry.json"
)
REGISTRY_ENV = "HUGINN_DYNAMIC90S_POOL_REGISTRY"
SEED_ENV = "HUGINN_DYNAMIC90S_MIXTURE_SEED"
START_POSITION_ENV = "HUGINN_DYNAMIC90S_MIXTURE_START_POSITION"
MAX_SAMPLES_ENV = "HUGINN_DYNAMIC90S_MIXTURE_MAX_SAMPLES"
DATASET_NAME = "huginn_whisper_dynamic90s_mixture"
DEFAULT_SEED = 20260730

if str(HUGINN_LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(HUGINN_LORA_ROOT))

from data_pipeline.dynamic90s_mixture_rows import (  # noqa: E402
    iter_dynamic90s_mixture_rows,
    load_pool_registry,
)


def _load_base_plugin() -> Any:
    spec = importlib.util.spec_from_file_location("huginn_audio_whisper_dynamic90s_base", BASE_PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dynamic-90s base plugin: {BASE_PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_PLUGIN = _load_base_plugin()


def _registry_path(dataset_meta: Any | None = None) -> Path:
    configured = os.environ.get(REGISTRY_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    metadata_path = getattr(dataset_meta, "dataset_path", None)
    if metadata_path:
        return Path(str(metadata_path)).expanduser().resolve()
    return DEFAULT_REGISTRY


def _integer_environment(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    value = int(raw_value) if raw_value else int(default)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _seed() -> int:
    return _integer_environment(SEED_ENV, DEFAULT_SEED, minimum=0)


def _start_position() -> int:
    return _integer_environment(START_POSITION_ENV, 0, minimum=0)


def _max_samples() -> int | None:
    raw_value = os.environ.get(MAX_SAMPLES_ENV, "").strip()
    if not raw_value:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{MAX_SAMPLES_ENV} must be positive when set, got {value}")
    return value


def build_dataset(registry_path: str | Path):
    from datasets import IterableDataset

    resolved_registry = Path(registry_path).expanduser().resolve()
    load_pool_registry(resolved_registry)
    seed = _seed()
    start_position = _start_position()
    max_samples = _max_samples()
    return IterableDataset.from_generator(
        iter_dynamic90s_mixture_rows,
        gen_kwargs={
            "registry_path": str(resolved_registry),
            "seed": seed,
            "start_position": start_position,
            "max_samples": max_samples,
        },
    )


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

    registry = _registry_path()

    class HuginnDynamic90sMixtureLoader(DatasetLoader):
        def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
            del dataset_syntax, use_hf
            path = _registry_path(dataset_meta)
            dataset = build_dataset(path)
            print(
                "[HuginnDynamic90sMixture] loaded IterableDataset "
                f"registry={path} seed={_seed()} start_position={_start_position()} "
                f"max_samples={_max_samples() or 'unbounded'}"
            )
            return dataset

    metadata = DatasetMeta(
        dataset_path=str(registry),
        dataset_name=DATASET_NAME,
        loader=HuginnDynamic90sMixtureLoader,
    )
    try:
        register_dataset(metadata, exist_ok=True)
    except TypeError as exc:
        if "exist_ok" not in str(exc):
            raise
        register_dataset(metadata)
    print(
        f"[HuginnDynamic90sMixture] registered dataset path={registry} "
        f"name={DATASET_NAME} loader={HuginnDynamic90sMixtureLoader.__name__}"
    )


_register_dataset()
