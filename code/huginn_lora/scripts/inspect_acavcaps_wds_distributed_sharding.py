"""Probe ACAVCAPS IterableDataset sharding under the real Accelerate path.

The probe reads WebDataset samples and metadata only. It does not decode FLAC
and does not construct the LoSATok model. It is intentionally run with
``torchrun --nproc_per_node=2`` before any two-GPU formal training.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json"
)
BASE_REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_PATH = BASE_REPO_ROOT / "code" / "huginn_lora" / "plugins" / "huginn_losatok_acavcaps_wds_swift.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--probe_samples_per_rank", type=int, default=256)
    parser.add_argument(
        "--consume_all",
        action="store_true",
        help="Consume the complete selected-tar stream instead of stopping after probe_samples_per_rank.",
    )
    parser.add_argument(
        "--max_tars_per_stage",
        type=int,
        default=2,
        help="Use a small prefix of each stage for a quick probe; set 0 to consume all manifest tars.",
    )
    parser.add_argument("--expected_world_size", type=int, default=2)
    return parser.parse_args()


def load_plugin(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("huginn_losatok_acavcaps_wds_swift_shard_probe", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ACAVCAPS plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_id(row: dict[str, Any]) -> str:
    audios = row.get("audios")
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], dict):
        raise ValueError(f"Unexpected audios field: {audios!r}")
    audio = audios[0]
    key = audio.get("sample_id")
    stage = audio.get("stage")
    tar_path = audio.get("tar_path")
    audio_bytes = audio.get("audio_bytes")
    if not isinstance(key, str) or not key or not isinstance(stage, str) or not isinstance(tar_path, str):
        raise ValueError(f"Invalid ACAVCAPS row identity: {audio!r}")
    if not isinstance(audio_bytes, (bytes, bytearray, memoryview)) or len(audio_bytes) == 0:
        raise ValueError(f"Invalid ACAVCAPS row audio_bytes for {stage}:{key}")
    return f"{stage}|{tar_path}|{key}"


def main() -> int:
    args = parse_args()
    if args.probe_samples_per_rank <= 0 or args.expected_world_size <= 0 or args.max_tars_per_stage < 0:
        raise ValueError("probe_samples_per_rank and expected_world_size must be positive; max_tars must be >= 0")

    manifest = Path(args.manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    if not PLUGIN_PATH.is_file():
        raise FileNotFoundError(f"ACAVCAPS plugin does not exist: {PLUGIN_PATH}")

    os.environ["ACAVCAPS_WDS_MANIFEST"] = str(manifest)
    if args.max_tars_per_stage == 0:
        os.environ.pop("ACAVCAPS_WDS_MAX_TARS_PER_STAGE", None)
    else:
        os.environ["ACAVCAPS_WDS_MAX_TARS_PER_STAGE"] = str(args.max_tars_per_stage)

    plugin = load_plugin(PLUGIN_PATH)
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from swift.dataset import load_dataset

    accelerator = Accelerator()
    if accelerator.num_processes != args.expected_world_size:
        raise RuntimeError(
            f"Unexpected distributed world size: expected={args.expected_world_size} "
            f"actual={accelerator.num_processes}"
        )

    try:
        train_dataset, val_dataset = load_dataset(
            str(manifest),
            split_dataset_ratio=0.0,
            shuffle=False,
            num_proc=1,
            streaming=True,
        )
    except TypeError:
        train_dataset, val_dataset = load_dataset(
            str(manifest),
            split_dataset_ratio=0.0,
            shuffle=False,
            num_proc=1,
        )
    if val_dataset is not None:
        raise RuntimeError(f"Unexpected validation dataset: {type(val_dataset)}")

    loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: batch[0],
    )
    prepared_loader = accelerator.prepare(loader)

    dataloader_config = getattr(accelerator, "dataloader_config", None)
    accelerator_state = getattr(accelerator, "state", None)
    dispatch_batches = getattr(dataloader_config, "dispatch_batches", None)
    if dispatch_batches is None and accelerator_state is not None:
        dispatch_batches = getattr(accelerator_state, "dispatch_batches", None)
    split_batches = getattr(dataloader_config, "split_batches", None)
    if split_batches is None and accelerator_state is not None:
        split_batches = getattr(accelerator_state, "split_batches", None)
    if accelerator.process_index == 0:
        print(
            f"[accelerate] prepared_loader_type={type(prepared_loader).__name__} "
            f"dispatch_batches={dispatch_batches!r} split_batches={split_batches!r} "
            f"even_batches={getattr(dataloader_config, 'even_batches', None)!r}",
            flush=True,
        )

    local_ids: list[str] = []
    for row in prepared_loader:
        if not isinstance(row, dict):
            raise TypeError(f"Prepared loader yielded {type(row).__name__}, expected dict")
        local_ids.append(sample_id(row))
        if not args.consume_all and len(local_ids) >= args.probe_samples_per_rank:
            break

    accelerator.wait_for_everyone()
    gathered: list[list[str] | None] = [None for _ in range(accelerator.num_processes)]
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized; run this script through torchrun")
    dist.all_gather_object(gathered, local_ids)

    rank = accelerator.process_index
    local_unique = len(set(local_ids))
    if local_unique != len(local_ids):
        raise RuntimeError(f"Rank {rank} produced duplicate sample IDs in its probe: rows={len(local_ids)} unique={local_unique}")
    lengths = [len(values or []) for values in gathered]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Ranks did not receive equal probe lengths: lengths={lengths}")

    overlap_pairs: list[tuple[int, int, int]] = []
    for left in range(len(gathered)):
        left_set = set(gathered[left] or [])
        for right in range(left + 1, len(gathered)):
            overlap = left_set.intersection(gathered[right] or [])
            if overlap:
                overlap_pairs.append((left, right, len(overlap)))
    if overlap_pairs:
        raise RuntimeError(f"Distributed IterableDataset probe found cross-rank duplicate samples: {overlap_pairs}")

    print(
        f"[rank={rank}] python={sys.version.split()[0]} platform={platform.platform()} "
        f"accelerate_version={importlib.metadata.version('accelerate')} "
        f"world_size={accelerator.num_processes} local_rows={len(local_ids)} "
        f"first_ids={local_ids[:4]}",
        flush=True,
    )
    accelerator.wait_for_everyone()
    if rank == 0:
        print("========== ACAVCAPS WEBDATASET DISTRIBUTED SHARD INSPECT ==========")
        print(f"[manifest] path={manifest}")
        print(
            f"[config] max_tars_per_stage={args.max_tars_per_stage or 'all'} "
            f"probe_samples_per_rank={args.probe_samples_per_rank} consume_all={args.consume_all}"
        )
        print(f"[distributed] world_size={accelerator.num_processes} probe_lengths={lengths}")
        if dispatch_batches:
            print("[distributed] read_mode=process0_loader_dispatches_batches_to_other_ranks")
        else:
            print("[distributed] read_mode=each_rank_has_prepared_iterable_loader")
        print(f"[distributed] cross_rank_overlap_pairs={overlap_pairs}")
        print("[result] status=PASS accelerate_prepare=pass equal_lengths=true cross_rank_overlap=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
