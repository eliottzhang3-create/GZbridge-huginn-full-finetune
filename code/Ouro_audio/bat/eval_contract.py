"""Shared BAT evaluation contracts.

This module deliberately keeps the Phase-I audit path metadata-only.  The
audio renderer is imported and used only by the Phase-II generation smoke.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SAMPLE_RATE = 32_000
TARGET_SAMPLES = 10 * SAMPLE_RATE
RIR_TARGET_SAMPLES = 2 * SAMPLE_RATE
BAT_PROMPT = (
    "Based on the audio you've heard, refer to the instruction and provide a response.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


EVAL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "A",
        "relative_path": "stage1-clsdoa/eval-stage1-classification.json",
        "table4_metric": "Detection mAP",
        "question_types": ("CLASSIFICATION",),
        "source_shape": "single",
        "answer_contract": "AudioSet label list separated by '; '",
    },
    {
        "name": "B",
        "relative_path": "stage1-clsdoa/eval-stage1-doa.json",
        "table4_metric": "DoA / DP",
        "question_types": ("DOA",),
        "source_shape": "single",
        "answer_contract": "left/right, front/behind, above/below; distance",
    },
    {
        "name": "C",
        "relative_path": "stage2-single/eval-stage2-classification.json",
        "table4_metric": "Detection mAP",
        "question_types": ("MIXUP_SINGLE_CLASSIFICATION",),
        "source_shape": "dual",
        "answer_contract": "AudioSet label list separated by '; '",
    },
    {
        "name": "D",
        "relative_path": "stage2-single/eval-stage2-doa.json",
        "table4_metric": "DoA / DP",
        "question_types": ("MIXUP_SINGLE_DOA",),
        "source_shape": "dual",
        "answer_contract": "left/right, front/behind, above/below; distance",
    },
    {
        "name": "E-direction",
        "relative_path": "stage3-mixup/eval-stage3-direction.json",
        "table4_metric": "Direction Accuracy",
        "question_types": ("MIXUP_DIRECTION",),
        "source_shape": "dual",
        "answer_contract": "Yes or No",
    },
    {
        "name": "E-distance",
        "relative_path": "stage3-mixup/eval-stage3-distance.json",
        "table4_metric": "Distance Accuracy",
        "question_types": ("MIXUP_DISTANCE_BOTH",),
        "source_shape": "dual",
        "answer_contract": "Yes or No",
    },
    {
        "name": "E-nonbinary",
        "relative_path": "stage3-mixup/eval-stage3-nonbinary.json",
        "table4_metric": "diagnostic only; excluded from Table 4",
        "question_types": (
            "MIXUP_NONBINARY_DIRECTION",
            "MIXUP_NONBINARY_DISTANCE",
            "MIXUP_NONBINARY_SOURCE",
        ),
        "source_shape": "dual",
        "answer_contract": "non-binary reasoning answer",
    },
)


def present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "none", "null"}


def normalize_ref(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def load_json_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
        container = "list"
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
        container = "dict[data]"
    else:
        raise ValueError(f"Expected a JSON list or {{data: list}}: {path}")
    if not all(isinstance(item, dict) for item in records):
        raise TypeError(f"Expected object records: {path}")
    return records, container


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def stable_eval_id(split_name: str, index: int, record: dict[str, Any]) -> str:
    return f"{split_name}:{index}:{record_digest(record)[:16]}"


def source_shape(record: dict[str, Any]) -> str:
    second_audio = present(record.get("audio_id2"))
    second_reverb = present(record.get("reverb_id2"))
    if second_audio != second_reverb:
        return "partial_second_ids"
    return "dual" if second_audio else "single"


def _audio_candidates(root: Path, reference: str) -> list[Path]:
    relative = normalize_ref(reference)
    candidate = root / relative
    if candidate.suffix:
        return [candidate]
    return [candidate.with_suffix(ext) for ext in (".wav", ".flac", ".mp3", ".ogg")]


def resolve_audio_path(root: Path, reference: str) -> Path | None:
    return next((path for path in _audio_candidates(root, reference) if path.is_file()), None)


def resolve_reverb_path(root: Path, reference: str) -> Path | None:
    relative = normalize_ref(reference)
    candidates = (root / "binaural" / relative, root / relative, root / "mp3d_reverb" / "binaural" / relative)
    return next((path for path in candidates if path.is_file()), None)


def unique_refs(records: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    return sorted({normalize_ref(record[field]) for record in records for field in fields if present(record.get(field))})


def metadata_asset_coverage(records: list[dict[str, Any]], audio_root: Path, reverb_root: Path) -> dict[str, Any]:
    """Check path existence only; never opens audio or RIR files."""
    audio_refs = unique_refs(records, ("audio_id", "audio_id2"))
    reverb_refs = unique_refs(records, ("reverb_id", "reverb_id2"))
    missing_audio = [ref for ref in audio_refs if resolve_audio_path(audio_root, ref) is None]
    missing_reverb = [ref for ref in reverb_refs if resolve_reverb_path(reverb_root, ref) is None]
    return {
        "mode": "metadata_path_existence_only",
        "audio_reference_count": len(audio_refs),
        "audio_files_checked": len(audio_refs),
        "audio_matched_count": len(audio_refs) - len(missing_audio),
        "audio_missing_count": len(missing_audio),
        "audio_missing_examples": missing_audio[:20],
        "reverb_reference_count": len(reverb_refs),
        "reverb_files_checked": len(reverb_refs),
        "reverb_matched_count": len(reverb_refs) - len(missing_reverb),
        "reverb_missing_count": len(missing_reverb),
        "reverb_missing_examples": missing_reverb[:20],
    }


def summarize_eval_records(records: list[dict[str, Any]], spec: dict[str, Any], split_name: str) -> dict[str, Any]:
    raw_types = Counter(str(record.get("question_type", "<missing>")).upper() for record in records)
    shapes = Counter(source_shape(record) for record in records)
    question_ids = Counter(str(record.get("question_id", "<missing>")) for record in records)
    answers = Counter(str(record.get("answer", "")).strip().lower() for record in records)
    invalid_fields: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        required = ("audio_id", "reverb_id", "question", "answer", "question_type", "question_id")
        missing = [field for field in required if not present(record.get(field))]
        shape = source_shape(record)
        raw_type = str(record.get("question_type", "")).upper()
        if shape == "partial_second_ids" or missing or raw_type not in spec["question_types"] or shape != spec["source_shape"]:
            invalid_fields.append({
                "record_index": index,
                "missing_fields": missing,
                "question_type": raw_type,
                "source_shape": shape,
            })
    return {
        "split_name": split_name,
        "official_type": spec["name"],
        "relative_path": spec["relative_path"],
        "table4_metric": spec["table4_metric"],
        "record_count": len(records),
        "raw_question_type_counts": dict(raw_types),
        "source_shape_counts": dict(shapes),
        "unique_question_id_count": len(question_ids),
        "duplicate_question_id_extra_count": sum(max(0, count - 1) for count in question_ids.values()),
        "answer_exact_yes_count": answers.get("yes", 0),
        "answer_exact_no_count": answers.get("no", 0),
        "invalid_contract_record_count": len(invalid_fields),
        "invalid_contract_examples": invalid_fields[:20],
        "answer_contract": spec["answer_contract"],
    }


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(path: Path, hash_limit_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return item
    item["bytes"] = path.stat().st_size
    if item["bytes"] <= hash_limit_bytes:
        item["sha256"] = sha256_file(path)
    else:
        item["sha256"] = None
        item["sha256_skipped_reason"] = f"file larger than {hash_limit_bytes} bytes"
    return item


def parse_yes_no(text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text.strip().lower()).strip(" .,!?:;\n\t")
    matches = re.findall(r"\b(yes|no)\b", normalized)
    unique = sorted(set(matches))
    return {
        "normalized": normalized,
        "strict_exact": normalized in {"yes", "no"},
        "value": unique[0] if len(unique) == 1 else None,
        "status": "ok" if len(unique) == 1 else "invalid_yes_no",
    }


def parse_location(text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    axis_values = {}
    for axis, values in {
        "horizontal": ("left", "right"),
        "depth": ("front", "behind"),
        "vertical": ("above", "below"),
    }.items():
        found = [value for value in values if re.search(rf"\b{value}\b", normalized)]
        axis_values[axis] = found[0] if len(found) == 1 else None
    distance_matches = re.findall(r"(?<![a-z0-9])([0-9]+(?:\.[0-9]+)?)\s*(?:m|meter|meters)\b", normalized)
    distance = float(distance_matches[0]) if len(distance_matches) == 1 else None
    ok = all(value is not None for value in axis_values.values()) and distance is not None
    return {
        "normalized": normalized,
        "direction": axis_values if ok else None,
        "distance_m": distance,
        "status": "ok" if ok else "invalid_location",
    }


class BATEvalAudioRenderer:
    """Renderer with explicit official or checkpoint-matched RIR policy."""

    def __init__(self, audio_root: str | Path, reverb_root: str | Path, rir_policy: str = "official_bat"):
        if rir_policy not in {"official_bat", "checkpoint_matched"}:
            raise ValueError(f"Unknown RIR policy: {rir_policy}")
        self.audio_root = Path(audio_root).expanduser().resolve()
        self.reverb_root = Path(reverb_root).expanduser().resolve()
        self.rir_policy = rir_policy

    @staticmethod
    def _load_source(path: Path):
        import numpy as np
        import soundfile as sf

        value, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 2:
            value = value[:, 0]
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f"Invalid AudioSet waveform: {path} shape={value.shape}")
        return value, int(sample_rate)

    @staticmethod
    def _normalize(value):
        import numpy as np

        rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2))) if value.size else 0.0
        if rms == 0.0:
            return value.astype(np.float32, copy=False)
        return (value * (10.0 ** ((-14.0 - 20.0 * math.log10(rms)) / 20.0))).astype(np.float32)

    @staticmethod
    def _resample(value, sample_rate: int):
        import numpy as np

        if sample_rate == SAMPLE_RATE:
            return value.astype(np.float32, copy=False)
        from scipy import signal

        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        return signal.resample_poly(value, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)

    @staticmethod
    def _crop_or_pad(value):
        import numpy as np

        output = np.zeros((2, TARGET_SAMPLES), dtype=np.float32)
        output[:, : min(TARGET_SAMPLES, value.shape[-1])] = value[:, :TARGET_SAMPLES]
        return output

    @staticmethod
    def _crop_or_pad_rir(value):
        import numpy as np

        output = np.zeros((2, RIR_TARGET_SAMPLES), dtype=np.float32)
        output[:, : min(RIR_TARGET_SAMPLES, value.shape[-1])] = value[:, :RIR_TARGET_SAMPLES]
        return output

    def _render_one(self, audio_id: str, reverb_id: str):
        import numpy as np
        import torch
        from scipy import signal

        audio_path = resolve_audio_path(self.audio_root, audio_id)
        reverb_path = resolve_reverb_path(self.reverb_root, reverb_id)
        if audio_path is None or reverb_path is None:
            raise FileNotFoundError(f"Missing assets audio={audio_id} reverb={reverb_id}")
        audio, sample_rate = self._load_source(audio_path)
        audio = self._normalize(self._resample(audio, sample_rate))[None, :]
        # Read only the NPY header first.  Some extracted RIR files are longer
        # than the 2-second training target; materializing an arbitrarily long
        # raw array here was an avoidable host-memory hazard in online eval.
        rir_array = np.load(reverb_path, mmap_mode="r", allow_pickle=False)
        shape = tuple(int(value) for value in rir_array.shape)
        if len(shape) != 2 or shape[0] != 2:
            raise ValueError(f"Invalid binaural RIR: {reverb_path} shape={shape}")
        if shape[1] <= 0:
            raise ValueError(f"Empty binaural RIR: {reverb_path}")
        if self.rir_policy == "checkpoint_matched":
            # The checkpoint-matched contract needs exactly 2 seconds.  Copy
            # only that prefix and zero-pad without loading any late tail.
            rir = self._crop_or_pad_rir(
                np.asarray(rir_array[:, :RIR_TARGET_SAMPLES], dtype=np.float32)
            )
        else:
            # For official_bat, taps after 10 seconds cannot affect the first
            # 10 seconds of a causal convolution.  Keeping only this prefix is
            # therefore output-equivalent to raw-RIR -> full convolution ->
            # final [2,320000] crop, without materializing the late tail.
            rir = np.asarray(rir_array[:, :TARGET_SAMPLES], dtype=np.float32)
        del rir_array
        if not np.isfinite(rir).all():
            raise ValueError(f"Invalid binaural RIR values: {reverb_path} shape={rir.shape}")

        # Only the first TARGET_SAMPLES output samples survive _crop_or_pad.
        # RIR taps after that point cannot contribute to a causal convolution
        # prefix, so discard them before scipy allocates its full convolution
        # result.  Convolving one ear at a time also avoids scipy's 2-D path
        # allocating an unnecessary two-channel full-length temporary.  This
        # is bit-equivalent to raw-RIR -> full convolution -> first 10 seconds
        # for official_bat, while preventing pathological long RIR files from
        # exhausting host memory in B/D/E evaluation.
        rir_for_output = rir[:, :TARGET_SAMPLES]
        rendered = np.zeros((2, TARGET_SAMPLES), dtype=np.float32)
        for channel in range(2):
            channel_full = signal.fftconvolve(audio[0], rir_for_output[channel], mode="full")
            rendered[channel, : min(TARGET_SAMPLES, channel_full.shape[0])] = channel_full[:TARGET_SAMPLES]
            del channel_full
        if not np.isfinite(rendered).all():
            raise ValueError(f"Non-finite rendered audio: audio={audio_id} reverb={reverb_id}")
        return torch.from_numpy(rendered).float()

    def render_record(self, record: dict[str, Any]):
        import torch

        first = self._render_one(str(record["audio_id"]), str(record["reverb_id"]))
        second_audio = present(record.get("audio_id2"))
        second_reverb = present(record.get("reverb_id2"))
        if second_audio != second_reverb:
            raise ValueError("Partial second-source pair")
        if second_audio:
            second = self._render_one(str(record["audio_id2"]), str(record["reverb_id2"]))
            first = (first + second) / 2.0
        if tuple(first.shape) != (2, TARGET_SAMPLES) or not torch.isfinite(first).all():
            raise ValueError(f"Unexpected final waveform: {tuple(first.shape)}")
        return first
