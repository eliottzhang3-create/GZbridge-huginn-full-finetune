from __future__ import annotations

import argparse
import io
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/"
    "data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_sampled.json"
)
EXPECTED_STAGE_NAMES = ("stage1", "stage2", "stage3")


def header(title: str) -> None:
    print(f"========== {title} ==========", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consume a small complete ACAVCAPS WebDataset stream: the first N tar files "
            "of every stage, with per-tar buffer shuffle and audio decode checks."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--max-tars-per-stage", type=int, default=2)
    parser.add_argument(
        "--decode-every",
        type=int,
        default=512,
        help="Decode the first sample of every tar and then every Nth sample; raw samples are all consumed.",
    )
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--allow-duplicate-keys", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ACAVCAPS manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), list):
        raise ValueError(f"Invalid ACAVCAPS manifest: {path}")
    return payload


def decode_audio_bytes_16k(audio_bytes: bytes, source_label: str) -> tuple[int, int, str]:
    """Decode one sample without importing the model plugin or constructing a model."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed for {source_label}: {error}")
        sample_count = len(result.stdout) // 4
        if sample_count <= 0:
            raise RuntimeError(f"ffmpeg returned an empty waveform for {source_label}")
        return 16000, sample_count, "ffmpeg"

    try:
        import soundfile as sf

        import numpy as np

        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        values = np.asarray(audio)
        if values.size == 0:
            raise RuntimeError("soundfile returned an empty waveform")
        return int(sample_rate), int(values.shape[0]), "soundfile"
    except Exception as exc:  # noqa: BLE001 - expose the remote decoder failure
        raise RuntimeError(f"No usable FLAC decoder for {source_label}: {exc}") from exc


def make_shuffled_tar_dataset(wds: Any, tar_path: str, buffer_size: int) -> Any:
    dataset = wds.WebDataset(tar_path, shardshuffle=False)
    try:
        return dataset.shuffle(buffer_size)
    except TypeError:
        return dataset.shuffle(size=buffer_size)


def caption_from_payload(payload: Any, label: str) -> str:
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: JSON payload is not an object")
    value = payload.get("long")
    if isinstance(value, list):
        captions = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not captions:
            raise ValueError(f"{label}: JSON long list has no non-empty string")
        caption = captions[0]
    elif isinstance(value, str):
        caption = value.strip()
    else:
        raise ValueError(f"{label}: JSON long field has unsupported type {type(value).__name__}")
    if not caption:
        raise ValueError(f"{label}: JSON long caption is empty")
    return caption


def inspect_stream(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        import webdataset as wds
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"webdataset import failed: {type(exc).__name__}: {exc}") from exc

    stages = manifest["stages"]
    actual_stage_names = tuple(str(stage.get("name", "unknown")) for stage in stages)
    if actual_stage_names != EXPECTED_STAGE_NAMES:
        raise ValueError(
            "Manifest stage order must be stage1, stage2, stage3 for the ACAVCAPS curriculum: "
            f"actual={actual_stage_names}"
        )
    dataset_root = Path(str(manifest.get("dataset_root", ""))).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Manifest dataset_root does not exist: {dataset_root}")
    public_root_marker = "/hpc_stor03/public/shared/data/raa/ACAVCAPS"
    if str(dataset_root) != public_root_marker:
        raise ValueError(
            "Unexpected ACAVCAPS dataset_root; refusing to inspect a different tree: "
            f"{dataset_root}"
        )

    total_samples = 0
    total_decoded = 0
    total_duplicate_keys = 0
    stage_reports: list[dict[str, Any]] = []
    previous_stage_names: list[str] = []

    for stage in stages:
        stage_name = str(stage.get("name", "unknown"))
        stage_tars = stage.get("tars")
        if not isinstance(stage_tars, list) or not stage_tars:
            raise ValueError(f"Manifest stage has no tars: {stage_name}")
        selected_tars = stage_tars[: args.max_tars_per_stage]
        if len(selected_tars) != args.max_tars_per_stage:
            raise ValueError(
                f"Stage {stage_name} has only {len(selected_tars)} tars, "
                f"cannot select {args.max_tars_per_stage}"
            )

        stage_samples = 0
        stage_decoded = 0
        tar_reports: list[dict[str, Any]] = []
        print(f"[stage-start] name={stage_name} selected_tar_count={len(selected_tars)}", flush=True)

        for tar_index, tar_entry in enumerate(selected_tars):
            if not isinstance(tar_entry, dict) or not isinstance(tar_entry.get("path"), str):
                raise ValueError(f"Invalid tar entry in {stage_name}: {tar_entry!r}")
            tar_path = str(tar_entry["path"])
            tar_resolved = Path(tar_path).resolve()
            if dataset_root not in tar_resolved.parents:
                raise ValueError(f"Tar path is outside manifest dataset_root: {tar_resolved}")
            if not tar_resolved.is_file():
                raise FileNotFoundError(f"Manifest tar does not exist: {tar_resolved}")
            expected_count = tar_entry.get("json_count")
            print(
                f"[tar-start] stage={stage_name} index={tar_index} path={tar_path} "
                f"expected_count={expected_count if expected_count is not None else 'unknown'} "
                f"buffer_size={args.buffer_size}",
                flush=True,
            )

            seen_keys: set[str] = set()
            first_keys: list[str] = []
            first_caption: str | None = None
            sample_count = 0
            decoded_count = 0
            url_mismatch_count = 0

            dataset = make_shuffled_tar_dataset(wds, tar_path, args.buffer_size)
            for sample in dataset:
                if not isinstance(sample, dict):
                    raise TypeError(f"{stage_name}:{tar_path}: WebDataset sample is {type(sample).__name__}")
                key = str(sample.get("__key__", ""))
                if not key:
                    raise ValueError(f"{stage_name}:{tar_path}: sample has no __key__")
                if key in seen_keys:
                    total_duplicate_keys += 1
                    if not args.allow_duplicate_keys:
                        raise RuntimeError(f"{stage_name}:{tar_path}: duplicate sample key {key}")
                seen_keys.add(key)
                if len(first_keys) < 8:
                    first_keys.append(key)

                raw_url = sample.get("__url__")
                if raw_url is not None and Path(str(raw_url)).name != Path(tar_path).name:
                    url_mismatch_count += 1

                json_bytes = sample.get("json")
                flac_bytes = sample.get("flac")
                if not isinstance(json_bytes, (bytes, bytearray, memoryview)):
                    raise TypeError(f"{stage_name}:{tar_path}:{key}: json is not bytes")
                if not isinstance(flac_bytes, (bytes, bytearray, memoryview)):
                    raise TypeError(f"{stage_name}:{tar_path}:{key}: flac is not bytes")
                try:
                    payload = json.loads(bytes(json_bytes).decode("utf-8"))
                    caption = caption_from_payload(payload, f"{stage_name}:{tar_path}:{key}")
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"Invalid JSON sample {stage_name}:{tar_path}:{key}: {exc}") from exc
                if first_caption is None:
                    first_caption = caption

                if sample_count == 0 or sample_count % args.decode_every == 0:
                    source_label = f"{stage_name}:{tar_path}:{key}"
                    sample_rate, waveform_samples, decoder = decode_audio_bytes_16k(bytes(flac_bytes), source_label)
                    if sample_rate != 16000 and decoder != "ffmpeg":
                        raise RuntimeError(
                            f"{source_label}: decoder did not produce 16 kHz audio: sr={sample_rate}"
                        )
                    if waveform_samples <= 0:
                        raise RuntimeError(f"{source_label}: decoder produced an empty waveform")
                    decoded_count += 1

                sample_count += 1
                if sample_count % args.log_every == 0:
                    print(
                        f"[tar-progress] stage={stage_name} index={tar_index} "
                        f"samples={sample_count} decoded={decoded_count}",
                        flush=True,
                    )

            if expected_count is not None and int(expected_count) != sample_count:
                raise RuntimeError(
                    f"{stage_name}:{tar_path}: sample count mismatch expected={expected_count} actual={sample_count}"
                )
            if sample_count == 0:
                raise RuntimeError(f"{stage_name}:{tar_path}: WebDataset yielded no samples")

            stage_samples += sample_count
            stage_decoded += decoded_count
            tar_reports.append(
                {
                    "path": tar_path,
                    "sample_count": sample_count,
                    "decoded_count": decoded_count,
                    "unique_key_count": len(seen_keys),
                    "duplicate_key_count": sample_count - len(seen_keys),
                    "url_mismatch_count": url_mismatch_count,
                    "first_keys": first_keys,
                    "first_caption": first_caption,
                }
            )
            print(
                f"[tar-done] stage={stage_name} index={tar_index} samples={sample_count} "
                f"decoded={decoded_count} unique_keys={len(seen_keys)} url_mismatches={url_mismatch_count} "
                f"first_keys={first_keys}",
                flush=True,
            )

        stage_reports.append(
            {
                "name": stage_name,
                "tar_count": len(selected_tars),
                "sample_count": stage_samples,
                "decoded_count": stage_decoded,
                "tars": tar_reports,
                "previous_stage_names": list(previous_stage_names),
            }
        )
        previous_stage_names.append(stage_name)
        total_samples += stage_samples
        total_decoded += stage_decoded
        print(
            f"[stage-done] name={stage_name} tar_count={len(selected_tars)} "
            f"samples={stage_samples} decoded={stage_decoded}",
            flush=True,
        )

    return {
        "manifest": manifest.get("dataset_root"),
        "buffer_size": args.buffer_size,
        "max_tars_per_stage": args.max_tars_per_stage,
        "decode_every": args.decode_every,
        "total_sample_count": total_samples,
        "total_decoded_count": total_decoded,
        "duplicate_key_count": total_duplicate_keys,
        "stage_reports": stage_reports,
    }


def main() -> int:
    args = parse_args()
    if args.buffer_size <= 0 or args.max_tars_per_stage <= 0 or args.decode_every <= 0 or args.log_every <= 0:
        raise ValueError("buffer-size, max-tars-per-stage, decode-every, and log-every must be positive")
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)

    header("ACAVCAPS WEBDATASET STREAM INSPECT")
    print(f"[env] python={sys.version.split()[0]}")
    print(f"[env] platform={platform.platform()}")
    print(f"[manifest] path={manifest_path}")
    print(f"[manifest] dataset_root={manifest.get('dataset_root')}")
    print(f"[config] buffer_size={args.buffer_size}")
    print(f"[config] max_tars_per_stage={args.max_tars_per_stage}")
    print(f"[config] decode_every={args.decode_every}")

    report = inspect_stream(manifest, args)
    print(f"[summary] {json.dumps(report, ensure_ascii=False)}")
    header("ACAVCAPS WEBDATASET STREAM INSPECT DONE")
    print(
        f"[result] status=PASS total_samples={report['total_sample_count']} "
        f"total_decoded={report['total_decoded_count']} duplicate_keys={report['duplicate_key_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
