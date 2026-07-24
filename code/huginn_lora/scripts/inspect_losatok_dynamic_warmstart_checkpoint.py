"""Classify a future dynamic LoSATok AudioCaps checkpoint before ACAVCAPS warm-start.

This inspector never merges a full FSDP checkpoint. It only lists adapter/vit
tensor keys or reads FSDP DCP metadata, so it is safe to run before choosing the
weight-warm-start path. A new ACAVCAPS task must not use Trainer resume semantics
unless the user explicitly intends to resume optimizer/data state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_report", required=True)
    return parser.parse_args()


def classify_key(key: str) -> str:
    normalized = key.split("base_model.model.", 1)[-1]
    if "audio_encoder." in normalized:
        return "losatok_encoder"
    if any(token in normalized for token in ("temporal_compressor.", "audio_projector.", "audio_bos", "audio_eos")):
        return "aligner"
    if "lora_" in normalized:
        return "huginn_lora"
    return "other"


def inspect_tensor_file(path: Path) -> dict[str, Any]:
    import torch

    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            return {"path": str(path), "kind": "not_state_dict"}
        keys = [key for key, value in payload.items() if isinstance(key, str) and torch.is_tensor(value)]
        shapes = {key: tuple(payload[key].shape) for key in keys}
    groups: dict[str, int] = {}
    for key in keys:
        group = classify_key(key)
        groups[group] = groups.get(group, 0) + 1
    normalized = {key.split("base_model.model.", 1)[-1] for key in keys}
    return {
        "path": str(path),
        "kind": "tensor_state_dict",
        "tensor_key_count": len(keys),
        "group_counts": groups,
        "boundary_embeddings_present": {
            "audio_bos": any(key.endswith("audio_bos") for key in normalized),
            "audio_eos": any(key.endswith("audio_eos") for key in normalized),
        },
        "shape_preview": {key: shapes[key] for key in keys[:20]},
        "key_preview": keys[:20],
    }


def inspect_dcp(path: Path) -> dict[str, Any]:
    try:
        from torch.distributed.checkpoint import FileSystemReader
    except Exception as exc:
        return {"path": str(path), "kind": "dcp", "error": f"{type(exc).__name__}: {exc}"}
    try:
        metadata = FileSystemReader(str(path)).read_metadata()
    except Exception as exc:
        return {"path": str(path), "kind": "dcp", "error": f"read_metadata {type(exc).__name__}: {exc}"}
    entries = []
    group_counts: dict[str, int] = {}
    for key, value in getattr(metadata, "state_dict_metadata", {}).items():
        key = str(key)
        size = getattr(value, "size", None)
        shape = tuple(int(item) for item in size) if size is not None else None
        dtype = getattr(getattr(value, "properties", None), "dtype", None)
        entries.append({"key": key, "shape": shape, "dtype": str(dtype) if dtype is not None else None})
        group = classify_key(key)
        group_counts[group] = group_counts.get(group, 0) + 1
    storage_paths = sorted({str(getattr(info, "relative_path", "<unknown>")) for info in getattr(metadata, "storage_data", {}).values()})
    return {
        "path": str(path),
        "kind": "dcp",
        "state_key_count": len(entries),
        "group_counts": group_counts,
        "storage_file_count": len(storage_paths),
        "storage_file_preview": storage_paths[:20],
        "key_preview": entries[:40],
    }


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    report: dict[str, Any] = {"checkpoint": str(checkpoint), "files": [], "tensor_reports": [], "dcp_reports": []}
    for path in sorted(checkpoint.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(checkpoint))
        report["files"].append({"path": relative, "size_bytes": path.stat().st_size})
        if path.name in {"adapter_model.safetensors", "vit.safetensors"}:
            report["tensor_reports"].append(inspect_tensor_file(path))
    for path in sorted(checkpoint.rglob("pytorch_model_fsdp_*")):
        if path.is_dir():
            report["dcp_reports"].append(inspect_dcp(path))

    tensor_groups: dict[str, int] = {}
    has_bos = has_eos = False
    for item in report["tensor_reports"]:
        for group, count in item.get("group_counts", {}).items():
            tensor_groups[group] = tensor_groups.get(group, 0) + count
        boundary = item.get("boundary_embeddings_present", {})
        has_bos = has_bos or bool(boundary.get("audio_bos"))
        has_eos = has_eos or bool(boundary.get("audio_eos"))
    report["tensor_group_counts"] = tensor_groups
    report["boundary_embeddings_present"] = {"audio_bos": has_bos, "audio_eos": has_eos}
    if report["tensor_reports"] and has_bos and has_eos:
        report["warm_start_route"] = "adapter_plus_vit_weight_warm_start"
    elif report["dcp_reports"]:
        report["warm_start_route"] = "fsdp2_dcp_requires_dedicated_streaming_restore_before_training"
    else:
        report["warm_start_route"] = "unsupported_or_incomplete_checkpoint_format"

    output = Path(args.output_report).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print("========== LOSATOK DYNAMIC WARM-START CHECKPOINT INSPECT ==========")
    print(f"[checkpoint] path={checkpoint}")
    print(f"[checkpoint] files={len(report['files'])} tensor_reports={len(report['tensor_reports'])} dcp_reports={len(report['dcp_reports'])}")
    print(f"[checkpoint] tensor_groups={tensor_groups}")
    print(f"[checkpoint] boundary_embeddings={report['boundary_embeddings_present']}")
    print(f"[checkpoint] warm_start_route={report['warm_start_route']}")
    print(f"[checkpoint] report={output}")
    if report["warm_start_route"] == "unsupported_or_incomplete_checkpoint_format":
        raise SystemExit("Checkpoint does not expose a supported dynamic LoSATok warm-start format")
    print("[result] status=PASS checkpoint_format_classified=true full_merge=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
