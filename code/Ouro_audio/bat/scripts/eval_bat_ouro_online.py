#!/usr/bin/env python3
"""Formal BAT Ouro evaluation: generate and score each record online.

One process evaluates exactly one official split (A, B, C, D, E-direction, or
E-distance).  Audio rendering, model generation, parsing, and metric
accumulation happen in the same loop.  A JSONL row and a progress snapshot are
flushed after every example so an interrupted long evaluation leaves usable
diagnostics without requiring a second offline scoring pass.

For A/C, the paper-style 355-class detection metric is computed without an
external embedding service: each AudioSet class name and each generated
answer are represented in the loaded model's ``lm_head.weight`` token-output
space.  Token rows are mean-pooled and L2-normalized, cosine similarity to
each class vector is used as the prediction score, and AP is computed per
class and averaged.  The ground-truth label-to-class mapping is retained only
to construct the required 355-dimensional multi-hot target vector.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from bat.eval_contract import (
    BAT_PROMPT,
    BATEvalAudioRenderer,
    EVAL_SPECS,
    file_inventory,
    load_json_records,
    parse_location,
    parse_yes_no,
    record_digest,
    resolve_audio_path,
    resolve_reverb_path,
    source_shape,
    stable_eval_id,
)
from bat.scripts.smoke_bat_eval_generation import (
    MODEL_SETTINGS,
    as_ids,
    base_model_of,
    binary_answer_prompt_applies,
    freeze_for_evaluation,
    load_adapter,
    tokenizer_from_processor,
)


SUPPORTED_TYPES = ("A", "B", "C", "D", "E-direction", "E-distance")
SPEC_BY_NAME = {spec["name"]: spec for spec in EVAL_SPECS if spec["name"] in SUPPORTED_TYPES}


def default_generation_limits(eval_type: str) -> tuple[int, int]:
    """Return safe defaults for one-record online evaluation."""

    # All BAT evaluation answers are short. Ten new tokens and greedy decoding
    # are sufficient for detection, location, and binary reasoning answers,
    # while avoiding beam-cache amplification.
    if eval_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported evaluation type: {eval_type}")
    return 10, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-type", choices=SUPPORTED_TYPES, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reverb-root", type=Path, required=True)
    parser.add_argument("--spatial-ast-root", type=Path, required=True)
    parser.add_argument("--spatial-ast-checkpoint", type=Path, required=True)
    parser.add_argument("--qformer-source", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0, help="0 means the complete official split")
    # Resolve task-specific defaults after --eval-type is known. The remote
    # launcher leaves these unset unless the user explicitly overrides them.
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--rir-policy", choices=("official_bat", "checkpoint_matched"), default="official_bat")
    parser.add_argument("--binary-answer-prompt", choices=("off", "on", "auto"), default="off")
    parser.add_argument(
        "--detection-mode",
        choices=("model_output_embedding", "official_semantic", "diagnostic_exact"),
        default="model_output_embedding",
        help=(
            "Used only for A/C. model_output_embedding is the local model-lm_head "
            "embedding metric; official_semantic is a backward-compatible alias."
        ),
    )
    parser.add_argument("--label-csv", type=Path, default=None)
    # Kept as a compatibility argument for old launchers.  It is intentionally
    # not read: class vectors are built from this model's lm_head below.
    parser.add_argument("--label-embeddings", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--embedding-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--openai-api-key-env", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fail_if_public(path: Path) -> None:
    normalized = str(path.expanduser()).replace("\\", "/")
    if not path.is_absolute() or normalized.startswith("/hpc_stor03/public"):
        raise ValueError(f"Output must be an absolute private path: {path}")


def import_plugin(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bat_ouro_online_eval_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_environment(args: argparse.Namespace) -> None:
    settings = MODEL_SETTINGS["ouro"]
    os.environ[settings["model_env"]] = str(args.model_path.resolve())
    os.environ["BAT_AUDIO_ROOT"] = str(args.audio_root.resolve())
    os.environ["BAT_REVERB_ROOT"] = str(args.reverb_root.resolve())
    os.environ["BAT_SPATIAL_AST_CODE_ROOT"] = str(args.spatial_ast_root.resolve())
    os.environ["BAT_SPATIAL_AST_CHECKPOINT"] = str(args.spatial_ast_checkpoint.resolve())
    os.environ["BAT_QFORMER_SOURCE"] = str(args.qformer_source.resolve())


def load_template(model: torch.nn.Module, processor: Any) -> Any:
    try:
        from swift import get_template
    except ImportError:
        from swift.template import get_template
    template = get_template(
        template_type="ouro_bat_audio_prefix",
        processor=processor,
        max_length=512,
        use_chat_template=False,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("transformers")
    return template


def build_input(template: Any, record: dict[str, Any], binary_prompt_mode: str) -> tuple[list[int], dict[str, Any]]:
    original = str(record["question"])
    apply_prompt = binary_answer_prompt_applies(record, "ouro", binary_prompt_mode)
    effective = original
    if apply_prompt:
        effective = f'{original}\n\nPlease answer only "yes" or "no".'
    dummy = dict(record)
    dummy["waveform"] = torch.zeros((2, 320_000), dtype=torch.float32)
    encoded = template.encode({
        "messages": [{"role": "user", "content": BAT_PROMPT.format(instruction=effective)}],
        "audios": [dummy],
    })
    ids = as_ids(encoded.get("input_ids"))
    if len(ids) <= 64:
        raise RuntimeError(f"Template produced no text after audio prefix: {len(ids)}")
    return ids, {
        "original_instruction": original,
        "effective_instruction": effective,
        "binary_answer_prompt_applied": apply_prompt,
        "input_length": len(ids),
        "audio_prefix_tokens": 64,
    }


def split_labels(answer: str) -> list[str]:
    # BAT metadata uses ';' as the list separator.  Commas are preserved
    # because some AudioSet class names themselves contain commas.
    return [part.strip().lower() for part in re.split(r"\s*;\s*", answer) if part.strip()]


class DetectionScorer:
    """Online accumulator for the paper-style A/C detection mAP contract.

    ``target`` remains a 355-dimensional multi-hot vector because AP is
    defined per AudioSet class.  Both sides of the semantic comparison use
    the same model-specific output space: the rows of ``lm_head.weight``
    corresponding to the tokenizer's subword pieces.  A phrase is represented
    by the mean of its token rows followed by L2 normalization.
    """

    def __init__(self, args: argparse.Namespace, model: torch.nn.Module, tokenizer: Any) -> None:
        import numpy as np

        self.mode = "model_output_embedding" if args.detection_mode == "official_semantic" else args.detection_mode
        self.label_csv = args.label_csv
        self.model = model
        self.tokenizer = tokenizer
        self.output_weight: torch.Tensor | None = None
        self.embedding_dimension: int | None = None
        self.cache: dict[tuple[int, ...], list[float]] = {}
        self.targets: list[list[int]] = []
        self.predictions: list[list[float]] = []
        self.unknown_target_labels: list[str] = []
        self.labels: list[str] = []
        self.label_to_index: dict[str, int] = {}
        self.label_embeddings: np.ndarray | None = None
        self._load_label_assets()
        if self.mode == "model_output_embedding":
            self._load_model_output_space()
            self._build_label_embeddings()

    def _load_label_assets(self) -> None:
        if self.label_csv is None or not self.label_csv.is_file():
            raise FileNotFoundError(f"Missing AudioSet label CSV: {self.label_csv}")
        with self.label_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty AudioSet label CSV: {self.label_csv}")
        for row in rows:
            label = str(row.get("display_name", row.get("name", ""))).strip().lower()
            if label:
                self.labels.append(label)
        if len(self.labels) != 355:
            raise ValueError(f"Expected 355 AudioSet subset labels, got {len(self.labels)}")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("AudioSet label CSV contains duplicate display names")
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}

    def _load_model_output_space(self) -> None:
        base = base_model_of(self.model)
        output_layer = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
        if output_layer is None or not hasattr(output_layer, "weight"):
            output_layer = getattr(base, "lm_head", None)
        if output_layer is None or not hasattr(output_layer, "weight"):
            raise RuntimeError("Model has no usable lm_head output embedding matrix")
        weight = output_layer.weight.detach()
        if weight.ndim != 2:
            raise RuntimeError(f"Expected lm_head.weight [vocab,hidden], got {tuple(weight.shape)}")
        if not torch.isfinite(weight).all():
            raise RuntimeError("lm_head.weight contains non-finite values")
        self.output_weight = weight
        self.embedding_dimension = int(weight.shape[1])

    def _token_ids(self, text: str) -> tuple[int, ...]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        special_ids = {int(value) for value in (getattr(self.tokenizer, "all_special_ids", None) or [])}
        vocab_size = int(self.output_weight.shape[0]) if self.output_weight is not None else None
        clean = [int(value) for value in ids if int(value) not in special_ids]
        if vocab_size is not None:
            clean = [value for value in clean if 0 <= value < vocab_size]
        return tuple(clean)

    def _embed_token_ids(self, token_ids: tuple[int, ...]) -> list[float]:
        import numpy as np

        if self.output_weight is None or not token_ids:
            raise ValueError("Cannot embed empty text in model output space")
        key = tuple(token_ids)
        if key in self.cache:
            return self.cache[key]
        indices = torch.tensor(token_ids, dtype=torch.long, device=self.output_weight.device)
        vector = self.output_weight.index_select(0, indices).float().mean(dim=0)
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(vector).all() or not torch.isfinite(norm) or float(norm) == 0.0:
            raise ValueError("Model output-space embedding is non-finite or zero-norm")
        normalized = (vector / norm).detach().cpu().numpy().astype(np.float32).tolist()
        self.cache[key] = normalized
        return normalized

    def _build_label_embeddings(self) -> None:
        import numpy as np

        vectors = [self._embed_token_ids(self._token_ids(label)) for label in self.labels]
        embeddings = np.asarray(vectors, dtype=np.float32)
        if embeddings.shape != (355, self.embedding_dimension):
            raise RuntimeError(f"Unexpected model label embedding shape: {embeddings.shape}")
        self.label_embeddings = embeddings

    def add(
        self,
        reference: str,
        generated: str,
        generated_token_ids: Any | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        target = np.zeros((355,), dtype=np.int8)
        target_labels = split_labels(reference)
        unknown = []
        for label in target_labels:
            index = self.label_to_index.get(label)
            if index is None:
                unknown.append(label)
            else:
                target[index] = 1
        self.unknown_target_labels.extend(unknown)
        if self.mode == "model_output_embedding":
            if generated_token_ids is None:
                token_ids = self._token_ids(generated)
            else:
                if isinstance(generated_token_ids, torch.Tensor):
                    values = generated_token_ids.detach().flatten().tolist()
                else:
                    values = list(generated_token_ids)
                special_ids = {int(value) for value in (getattr(self.tokenizer, "all_special_ids", None) or [])}
                token_ids = tuple(int(value) for value in values if int(value) not in special_ids)
            vector = np.asarray(self._embed_token_ids(token_ids), dtype=np.float32)
            scores = (self.label_embeddings @ vector).astype(np.float32)
        else:
            normalized = generated.lower()
            scores = np.asarray(
                [
                    1.0
                    if re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", normalized)
                    else 0.0
                    for label in self.labels
                ],
                dtype=np.float32,
            )
        self.targets.append(target.tolist())
        self.predictions.append(scores.tolist())
        return {
            "target_labels": target_labels,
            "unknown_target_labels": unknown,
            "score_mode": self.mode,
            "embedding_space": "model_lm_head_token_rows" if self.mode == "model_output_embedding" else None,
            "embedding_dimension": self.embedding_dimension if self.mode == "model_output_embedding" else None,
            "embedding_pooling": "mean_token_rows_then_l2_normalize" if self.mode == "model_output_embedding" else None,
            "ground_truth_target_encoding": "355d_multi_hot_from_semicolon_labels",
            "official_metric": False,
        }

    def finalize(self) -> dict[str, Any]:
        import numpy as np
        from sklearn.metrics import average_precision_score

        if not self.targets:
            return {"status": "incomplete", "reason": "no_detection_records"}
        targets = np.asarray(self.targets, dtype=np.int8)
        predictions = np.asarray(self.predictions, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap = average_precision_score(targets, predictions, average=None)
        ap = np.nan_to_num(np.asarray(ap, dtype=np.float64), nan=0.0)
        return {
            "status": "ok",
            "metric": "Detection mAP",
            "mAP": float(np.mean(ap)),
            "mAP_percent": float(np.mean(ap) * 100.0),
            "per_class_ap_count": int(ap.size),
            "record_count": len(self.targets),
            "score_mode": self.mode,
            "official_metric": False,
            "embedding_space": "model_lm_head_token_rows" if self.mode == "model_output_embedding" else None,
            "embedding_dimension": self.embedding_dimension if self.mode == "model_output_embedding" else None,
            "embedding_pooling": "mean_token_rows_then_l2_normalize" if self.mode == "model_output_embedding" else None,
            "ground_truth_target_encoding": "355d_multi_hot_from_semicolon_labels",
            "class_vector_source": "AudioSet class name tokenized with the same model tokenizer, then lm_head rows",
            "label_count": len(self.labels),
            "unknown_target_label_count": len(self.unknown_target_labels),
            "unknown_target_label_examples": sorted(set(self.unknown_target_labels))[:20],
            "unique_token_sequence_embedding_count": len(self.cache),
        }


def location_score(reference: str, generated: str) -> dict[str, Any]:
    gt = parse_location(reference)
    pred = parse_location(generated)
    doa_correct = bool(gt.get("status") == "ok" and pred.get("status") == "ok" and gt.get("direction") == pred.get("direction"))
    distance_valid = bool(gt.get("distance_m") is not None and pred.get("distance_m") is not None)
    distance_abs_error = abs(float(pred["distance_m"]) - float(gt["distance_m"])) if distance_valid else None
    within_half_meter = bool(distance_valid and distance_abs_error <= 0.5)
    return {
        "reference_parser": gt,
        "generated_parser": pred,
        "doa_correct": doa_correct,
        "distance_abs_error_m": distance_abs_error,
        "distance_within_0_5m": within_half_meter,
        "distance_der_error": not within_half_meter,
    }


def binary_score(reference: str, generated: str) -> dict[str, Any]:
    gt = parse_yes_no(reference)
    pred = parse_yes_no(generated)
    correct = bool(gt.get("value") is not None and pred.get("value") is not None and gt["value"] == pred["value"])
    return {
        "reference_parser": gt,
        "generated_parser": pred,
        "correct": correct,
        "parser_policy": "one unique yes/no token; both tokens or no token is invalid",
    }


def is_cuda_oom(exc: BaseException) -> bool:
    """Recognize both PyTorch CUDA OOM types and wrapped OOM messages."""

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)


def release_generation_tensors(
    *,
    output_ids: Any,
    generated_ids: Any,
    waveform: Any,
    input_tensor: Any,
    attention_mask: Any,
    input_ids: Any,
    prompt_audit: Any,
    device: torch.device,
) -> None:
    """Release per-record CPU/GPU objects before the next evaluation row."""

    # generated_ids is normally a view into output_ids. Both names are
    # deliberately deleted so a future materialized-view change cannot retain
    # a complete beam output across records.
    del output_ids, generated_ids, waveform, input_tensor, attention_mask, input_ids, prompt_audit
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def metric_summary(spec_name: str, location_rows: list[dict[str, Any]], binary_rows: list[dict[str, Any]], detection: DetectionScorer | None) -> dict[str, Any]:
    if spec_name in {"B", "D"}:
        total = len(location_rows)
        doa = sum(int(row["doa_correct"]) for row in location_rows)
        within = sum(int(row["distance_within_0_5m"]) for row in location_rows)
        der = total - within
        return {
            "metric": "DoA Accuracy + DP Distance Error Rate",
            "record_count": total,
            "doa_accuracy": doa / total if total else 0.0,
            "doa_accuracy_percent": 100.0 * doa / total if total else 0.0,
            "dp_distance_within_0_5m": within / total if total else 0.0,
            "dp_distance_within_0_5m_percent": 100.0 * within / total if total else 0.0,
            "dp_distance_error_rate": der / total if total else 0.0,
            "dp_distance_error_rate_percent": 100.0 * der / total if total else 0.0,
            "invalid_reference_count": sum(int(row["reference_parser"].get("status") != "ok") for row in location_rows),
            "invalid_prediction_count": sum(int(row["generated_parser"].get("status") != "ok") for row in location_rows),
            "distance_abs_error_mean_m": (
                sum(row["distance_abs_error_m"] for row in location_rows if row["distance_abs_error_m"] is not None)
                / max(1, sum(row["distance_abs_error_m"] is not None for row in location_rows))
            ),
        }
    if spec_name.startswith("E-"):
        total = len(binary_rows)
        correct = sum(int(row["correct"]) for row in binary_rows)
        return {
            "metric": "Binary Accuracy",
            "record_count": total,
            "binary_accuracy": correct / total if total else 0.0,
            "binary_accuracy_percent": 100.0 * correct / total if total else 0.0,
            "invalid_reference_count": sum(int(row["reference_parser"].get("status") != "ok") for row in binary_rows),
            "invalid_prediction_count": sum(int(row["generated_parser"].get("status") != "ok") for row in binary_rows),
        }
    return detection.finalize() if detection is not None else {"status": "incomplete", "reason": "missing_detection_scorer"}


def main() -> None:
    args = parse_args()
    spec = SPEC_BY_NAME[args.eval_type]
    default_max_new_tokens, default_num_beams = default_generation_limits(args.eval_type)
    if args.max_new_tokens is None:
        args.max_new_tokens = default_max_new_tokens
    if args.num_beams is None:
        args.num_beams = default_num_beams
    fail_if_public(args.output_jsonl)
    fail_if_public(args.output_report)
    if args.start_index < 0 or args.max_records < 0 or args.max_new_tokens <= 0 or args.num_beams <= 0:
        raise ValueError("start-index/max-records must be non-negative and generation limits positive")
    if args.max_new_tokens > 10:
        raise ValueError("BAT evaluation is capped at max_new_tokens<=10 to prevent runaway generations")
    if args.num_beams != 1:
        raise ValueError("BAT evaluation requires greedy single-beam generation: num_beams=1")
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing JSONL: {args.output_jsonl}")
    if args.output_report.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {args.output_report}")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal BAT evaluation requires a CUDA job")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    required = (
        args.model_path, args.plugin_path, args.checkpoint, args.qa_root, args.audio_root,
        args.reverb_root, args.spatial_ast_root, args.spatial_ast_checkpoint, args.qformer_source,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    prepare_environment(args)
    plugin = import_plugin(args.plugin_path.resolve())
    settings = MODEL_SETTINGS["ouro"]
    if plugin.MODEL_TYPE != settings["model_type"] or plugin.TEMPLATE_TYPE != settings["template_type"]:
        raise RuntimeError("Ouro plugin registration constants do not match")
    package_report = {name: importlib.metadata.version(name) for name in ("ms-swift", "transformers", "peft")}
    split_path = args.qa_root / spec["relative_path"]
    records, container = load_json_records(split_path)
    end = len(records) if args.max_records == 0 else min(len(records), args.start_index + args.max_records)
    selected = records[args.start_index:end]
    if not selected:
        raise ValueError(f"No records selected: start={args.start_index} end={end} total={len(records)}")
    location_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    output_jsonl = args.output_jsonl.resolve()
    progress_path = output_jsonl.with_name(output_jsonl.name + ".progress.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print("========== BAT OURO ONLINE EVALUATION ==========")
    print(f"[type] {args.eval_type} split={spec['relative_path']} records={len(selected)}")
    print(f"[renderer] policy={args.rir_policy} audio={args.audio_root} reverb={args.reverb_root}")
    print(f"[generation] do_sample=false num_beams={args.num_beams} max_new_tokens={args.max_new_tokens} use_cache=true")
    if args.eval_type in {"A", "C"}:
        resolved_mode = "model_output_embedding" if args.detection_mode == "official_semantic" else args.detection_mode
        print(
            f"[detection] mode={resolved_mode} label_csv={args.label_csv} "
            "embedding_space=model_lm_head_token_rows pooling=mean_then_l2"
        )
    load_started = time.perf_counter()
    try:
        from swift import get_model_processor
    except ImportError:
        from swift.model import get_model_processor
    base_model, processor = get_model_processor(
        str(args.model_path.resolve()),
        model_type=settings["model_type"],
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        load_model=True,
        download_model=False,
        attn_impl="sdpa",
        model_kwargs={"local_files_only": True, "low_cpu_mem_usage": True},
    )
    model = load_adapter(base_model, args.checkpoint)
    freeze_for_evaluation(model)
    model.eval()
    template = load_template(base_model, processor)
    tokenizer = tokenizer_from_processor(processor)
    renderer = BATEvalAudioRenderer(args.audio_root, args.reverb_root, args.rir_policy)
    from bat.scripts.smoke_bat_eval_generation import parameter_contract
    contract = parameter_contract(model, "ouro")
    detection = DetectionScorer(args, model, tokenizer) if args.eval_type in {"A", "C"} else None
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    errors_count = 0
    attempted_records = 0
    successful_records = 0
    aborted_on_cuda_oom = False
    first_cuda_oom: dict[str, Any] | None = None
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for local_index, record in enumerate(selected):
            absolute_index = args.start_index + local_index
            eval_id = stable_eval_id(spec["relative_path"], absolute_index, record)
            started = time.perf_counter()
            # Initialize every temporary explicitly so the finally block is
            # safe even when rendering or input construction fails.
            input_ids: list[int] | None = None
            prompt_audit: dict[str, Any] | None = None
            waveform: Any = None
            input_tensor: Any = None
            attention_mask: Any = None
            output_ids: Any = None
            generated_ids: Any = None
            row: dict[str, Any]
            try:
                for field in ("audio_id", "reverb_id"):
                    if not record.get(field):
                        raise ValueError(f"Missing {field}")
                if resolve_audio_path(args.audio_root, str(record["audio_id"])) is None:
                    raise FileNotFoundError(f"Missing audio: {record['audio_id']}")
                if resolve_reverb_path(args.reverb_root, str(record["reverb_id"])) is None:
                    raise FileNotFoundError(f"Missing reverb: {record['reverb_id']}")
                input_ids, prompt_audit = build_input(template, record, args.binary_answer_prompt)
                waveform = renderer.render_record(record).unsqueeze(0).to(device=device, dtype=torch.float32)
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_tensor)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=input_tensor,
                        attention_mask=attention_mask,
                        audio_waveforms=waveform,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        do_sample=False,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                        use_cache=True,
                        eos_token_id=getattr(base_model.config, "eos_token_id", None),
                        pad_token_id=getattr(base_model.config, "pad_token_id", None) or tokenizer.eos_token_id,
                    )
                torch.cuda.synchronize(device)
                generated_ids = output_ids[0, len(input_ids):]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                generated_token_count = int(generated_ids.numel())
                waveform_shape = list(waveform.shape)
                if args.eval_type in {"B", "D"}:
                    score = location_score(str(record["answer"]), generated_text)
                    location_rows.append(score)
                elif args.eval_type.startswith("E-"):
                    score = binary_score(str(record["answer"]), generated_text)
                    binary_rows.append(score)
                else:
                    score = detection.add(str(record["answer"]), generated_text, generated_ids)
                row = {
                    "status": "ok",
                    "eval_id": eval_id,
                    "record_index": absolute_index,
                    "split": spec["relative_path"],
                    "official_type": args.eval_type,
                    "question_id": str(record.get("question_id")),
                    "question_type": str(record.get("question_type")),
                    "source_shape": source_shape(record),
                    "question": record.get("question"),
                    "reference_answer": record.get("answer"),
                    "generated_text": generated_text,
                    "generated_token_count": generated_token_count,
                    "prompt_audit": prompt_audit,
                    "waveform_shape": waveform_shape,
                    "score": score,
                    "record_digest": record_digest(record),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                successful_records += 1
            except Exception as exc:
                errors_count += 1
                oom = is_cuda_oom(exc)
                error_item = {
                    "eval_id": eval_id,
                    "record_index": absolute_index,
                    "error": repr(exc),
                    "cuda_oom": oom,
                }
                errors.append(error_item)
                row = {
                    "status": "error",
                    "eval_id": eval_id,
                    "record_index": absolute_index,
                    "split": spec["relative_path"],
                    "official_type": args.eval_type,
                    "question_id": str(record.get("question_id")),
                    "error": repr(exc),
                    "record_digest": record_digest(record),
                    "elapsed_seconds": time.perf_counter() - started,
                    "cuda_oom": oom,
                }
                if oom:
                    aborted_on_cuda_oom = True
                    first_cuda_oom = error_item
                    print(
                        f"[abort] first CUDA OOM at index={absolute_index} eval_id={eval_id}; "
                        "stopping evaluation immediately",
                        file=sys.stderr,
                    )
                else:
                    print(f"[error] index={absolute_index} eval_id={eval_id} {exc}", file=sys.stderr)
            finally:
                try:
                    release_generation_tensors(
                        output_ids=output_ids,
                        generated_ids=generated_ids,
                        waveform=waveform,
                        input_tensor=input_tensor,
                        attention_mask=attention_mask,
                        input_ids=input_ids,
                        prompt_audit=prompt_audit,
                        device=device,
                    )
                except Exception as cleanup_exc:
                    # Cleanup must never suppress the row/report write after
                    # an evaluation error, especially after a CUDA OOM.
                    print(f"[cleanup-warning] index={absolute_index} {cleanup_exc}", file=sys.stderr)
                finally:
                    # The helper releases its own references; clear the
                    # caller's references as well, otherwise Python would
                    # retain them until the next loop iteration.
                    output_ids = None
                    generated_ids = None
                    waveform = None
                    input_tensor = None
                    attention_mask = None
                    input_ids = None
                    prompt_audit = None
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            attempted_records += 1
            progress = {
                "status": "aborted_cuda_oom" if aborted_on_cuda_oom else "running",
                "eval_type": args.eval_type,
                "split": spec["relative_path"],
                "total_records": len(selected),
                "attempted_records": attempted_records,
                "completed_records": attempted_records,
                "successful_records": successful_records,
                "remaining_records": len(selected) - attempted_records,
                "last_record_index": absolute_index,
                "errors": errors_count,
                "aborted_on_cuda_oom": aborted_on_cuda_oom,
                "first_cuda_oom": first_cuda_oom,
                "output_jsonl": str(output_jsonl),
                "updated_unix": time.time(),
            }
            progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            if (local_index + 1) % 100 == 0 or local_index == 0:
                print(f"[progress] {local_index + 1}/{len(selected)} last={absolute_index} errors={errors_count}")
            if aborted_on_cuda_oom:
                break
    metrics = metric_summary(args.eval_type, location_rows, binary_rows, detection)
    report = {
        "status": "ok" if not aborted_on_cuda_oom and errors_count == 0 and metrics.get("status", "ok") == "ok" else "incomplete",
        "scope": "formal_online_generate_and_score",
        "official_bat_contract": {
            "eval_type": args.eval_type,
            "relative_path": spec["relative_path"],
            "table4_metric": spec["table4_metric"],
            "no_offline_second_pass": True,
            "renderer_rir_policy": args.rir_policy,
            "audio_root_read_only": str(args.audio_root.resolve()),
        },
        "model": {
            "kind": "ouro",
            "base_path": str(args.model_path.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_adapter": file_inventory(args.checkpoint / "adapter_model.safetensors"),
            "plugin_path": str(args.plugin_path.resolve()),
            "load_seconds": load_seconds,
            "contract": contract,
        },
        "packages": package_report,
        "dataset": {
            "container": container,
            "source_total_records": len(records),
            "start_index": args.start_index,
            "selected_records": len(selected),
            "attempted_records": attempted_records,
            "completed_records": attempted_records,
            "successful_records": successful_records,
            "remaining_records": len(selected) - attempted_records,
            "errors": errors_count,
        },
        "generation": {
            "do_sample": False,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "use_cache": True,
            "binary_answer_prompt": args.binary_answer_prompt,
        },
        "termination": {
            "aborted_on_cuda_oom": aborted_on_cuda_oom,
            "first_cuda_oom": first_cuda_oom,
        },
        "scoring": metrics,
        "outputs": {
            "jsonl": str(output_jsonl),
            "progress": str(progress_path.resolve()),
            "report": str(args.output_report.resolve()),
        },
        "errors": errors[:100],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_name(args.output_report.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output_report)
    progress_path.write_text(
        json.dumps({**report["dataset"], **report["termination"], "status": report["status"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[report] {args.output_report}")
    print(f"[jsonl] {output_jsonl}")
    print(f"[status] {report['status']} metrics={json.dumps(metrics, ensure_ascii=False)}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
