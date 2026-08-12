"""Native SAGE checkpoint and waveform-forward audit.

The official OWL SAGE implementation is imported from --owl-source-root. A
synthetic binaural waveform is always tested; real BiDepth audio is tested
when --audio-root resolves JSON audio_id references. The script never
modifies the checkpoint and is intended to run as a submitted GPU job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_SAGE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/OWL/SAGE/finetuned.pth"
)
DEFAULT_BIDEPTH = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth"
)
DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/owl/"
    "phase1_sage_native_audit.json"
)


def _add_official_source(source_root: Path) -> list[str]:
    added: list[str] = []
    for candidate in (source_root / "src", source_root):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            added.append(str(candidate))
    return added


def _build_sage(source_root: Path) -> nn.Module:
    _add_official_source(source_root)
    module = importlib.import_module("slam_llm.models.SAGE.sage")
    return module.BinauralEncoder(
        num_classes=355,
        drop_path_rate=0.1,
        num_cls_tokens=3,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"SAGE checkpoint root is {type(checkpoint).__name__}")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise TypeError("SAGE checkpoint has no dict-valued 'model' state_dict")
    return checkpoint


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    numeric = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": bool(torch.isfinite(numeric).all().item()),
        "min": float(numeric.min().item()),
        "max": float(numeric.max().item()),
        "mean": float(numeric.mean().item()),
        "std": float(numeric.std(unbiased=False).item()),
    }


def _forward_sample(model: nn.Module, waveform: torch.Tensor, label: str) -> dict[str, Any]:
    with torch.inference_mode():
        output = model(waveform)
    return {"label": label, "input": _tensor_stats(waveform), "output": _tensor_stats(output)}


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    if path.suffix.lower() == ".npy":
        import numpy as np

        array = np.load(path, allow_pickle=False)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"Expected 1D/2D audio array, got shape={array.shape}")
        waveform = torch.from_numpy(array).float()
        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.transpose(0, 1)
        return waveform, 32000

    import soundfile as sf

    array, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(array.transpose(1, 0)).contiguous()
    return waveform, int(sample_rate)


def _audio_candidates(root: Path, audio_id: str) -> list[Path]:
    normalized = audio_id.replace("\\", "/").lstrip("./")
    return [
        root / normalized,
        root / f"{normalized}.wav",
        root / f"{normalized}.flac",
        root / f"{normalized}.mp3",
        root / f"{normalized}.npy",
    ]


def _resolve_audio(audio_roots: list[Path], audio_id: str) -> Path | None:
    for root in audio_roots:
        for candidate in _audio_candidates(root, audio_id):
            if candidate.is_file():
                return candidate
    return None


def _load_json_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((root / "owl-questions").glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise TypeError(f"Unexpected records container: {path}")
        result[f"{path.parent.name}/{path.stem}"] = [
            record for record in records if isinstance(record, dict)
        ]
    return result


def _select_real_samples(
    model: nn.Module,
    bidepth_root: Path,
    audio_roots: list[Path],
    max_samples: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    records = _load_json_records(bidepth_root)
    selected: list[tuple[str, dict[str, Any]]] = []
    for partition in ("stage1-clsdoa/train", "stage2-single/train", "stage3-mixup/train"):
        partition_records = records.get(partition, [])
        single = next(
            (record for record in partition_records if not record.get("audio_id2")),
            None,
        )
        dual = next(
            (record for record in partition_records if record.get("audio_id2")),
            None,
        )
        if single is not None:
            selected.append((f"{partition}/single", single))
        if dual is not None:
            selected.append((f"{partition}/dual", dual))

    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for label, record in selected[:max_samples]:
        audio_id = str(record["audio_id"])
        path = _resolve_audio(audio_roots, audio_id)
        if path is None:
            unresolved.append(f"{label}:{audio_id}")
            continue
        try:
            waveform, sample_rate = _load_audio(path)
            result: dict[str, Any] = {
                "label": label,
                "audio_id": audio_id,
                "path": str(path),
                "sample_rate": sample_rate,
                "question_type": record.get("question_type"),
                "has_second_source": bool(record.get("audio_id2")),
            }
            if waveform.ndim != 2 or waveform.shape[0] != 2:
                result.update({"status": "invalid_channels", "shape": list(waveform.shape)})
            elif sample_rate != 32000:
                result.update({"status": "unsupported_sample_rate", "shape": list(waveform.shape)})
            else:
                device = next(model.parameters()).device
                result.update(
                    _forward_sample(model, waveform.unsqueeze(0).to(device), label)
                )
                result["status"] = "ok"
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - preserve exact sample failure
            results.append(
                {
                    "label": label,
                    "audio_id": audio_id,
                    "path": str(path),
                    "status": "forward_failed",
                    "error": repr(exc),
                }
            )
    return results, unresolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sage-path", type=Path, default=DEFAULT_SAGE)
    parser.add_argument("--owl-source-root", type=Path, required=True)
    parser.add_argument("--bidepth-root", type=Path, default=DEFAULT_BIDEPTH)
    parser.add_argument("--audio-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--real-sample-count", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("========== OWL PHASE 1 SAGE NATIVE AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(f"[torch] version={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[sage] checkpoint={args.sage_path}")
    print(f"[owl] source_root={args.owl_source_root}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.sage_path)
    state = checkpoint["model"]
    model = _build_sage(args.owl_source_root)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    strict_error = None
    try:
        model.load_state_dict(state, strict=True)
        strict_status = "ok"
    except Exception as exc:  # noqa: BLE001 - report compatibility precisely
        strict_status = "failed"
        strict_error = repr(exc)
        report = {
            "status": "incomplete",
            "python": {"version": sys.version, "executable": sys.executable},
            "torch": {"version": torch.__version__, "cuda_available": torch.cuda.is_available()},
            "checkpoint": {
                "path": str(args.sage_path),
                "size_bytes": args.sage_path.stat().st_size,
                "sha256": _sha256_file(args.sage_path),
                "container_keys": [str(key) for key in checkpoint.keys()],
                "state_dict_key_count": len(state),
            },
            "load_state_dict": {
                "strict_status": strict_status,
                "strict_error": strict_error,
                "forward_skipped": True,
            },
            "issues": ["sage_strict_checkpoint_load_failed"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"[load] strict checkpoint load failed; report={args.output}")
        print("[status] incomplete")
        return
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    waveform = torch.randn(1, 2, 32000, device=device, dtype=torch.float32) * 0.01
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synthetic = _forward_sample(model, waveform, "synthetic_2ch_32000hz_1s")
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0

    real_samples: list[dict[str, Any]] = []
    unresolved: list[str] = []
    if args.audio_root:
        real_samples, unresolved = _select_real_samples(
            model, args.bidepth_root, args.audio_root, args.real_sample_count
        )

    report = {
        "status": "ok" if strict_status == "ok" and not any(
            sample.get("status") not in ("ok",) for sample in real_samples
        ) else "incomplete",
        "python": {"version": sys.version, "executable": sys.executable},
        "torch": {"version": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "device": {
            "requested": str(device),
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "peak_allocated_bytes": int(peak_allocated),
            "peak_reserved_bytes": int(peak_reserved),
        },
        "checkpoint": {
            "path": str(args.sage_path),
            "size_bytes": args.sage_path.stat().st_size,
            "sha256": _sha256_file(args.sage_path),
            "container_keys": [str(key) for key in checkpoint.keys()],
            "state_dict_key_count": len(state),
        },
        "architecture": {
            "parameter_count": parameter_count,
            "trainable_parameter_count_before_freeze": trainable_count,
            "expected_input": {"channels": 2, "sample_rate": 32000},
            "expected_output": {"sequence_length": 515, "hidden_size": 768},
        },
        "load_state_dict": {"strict_status": strict_status, "strict_error": strict_error},
        "synthetic_forward": synthetic,
        "real_audio_samples": real_samples,
        "unresolved_audio_examples": unresolved,
        "audit_contract": {
            "checkpoint_unchanged": True,
            "encoder_eval_mode": not model.training,
            "encoder_parameters_all_frozen": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "real_audio_roots_configured": bool(args.audio_root),
        },
    }
    if not args.audio_root:
        report["status"] = "incomplete"
        report["issues"] = ["real_audio_root_not_configured"]
    elif unresolved:
        report.setdefault("issues", []).append("some_audio_references_unresolved")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[synthetic] output_shape={synthetic['output']['shape']}")
    print(f"[real] successful={sum(sample.get('status') == 'ok' for sample in real_samples)} unresolved={len(unresolved)}")
    print(f"[report] {args.output}")
    print(f"[status] {report['status']}")


if __name__ == "__main__":
    main()
