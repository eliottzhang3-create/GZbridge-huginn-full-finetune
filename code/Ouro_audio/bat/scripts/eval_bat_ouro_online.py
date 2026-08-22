#!/usr/bin/env python3
"""Formal BAT Ouro evaluation: generate and score each record online.

One process evaluates exactly one official split (A, B, C, D, E-direction, or
E-distance).  Audio rendering, model generation, parsing, and metric
accumulation happen in the same loop.  A JSONL row and a progress snapshot are
flushed after every example so an interrupted long evaluation leaves usable
diagnostics without requiring a second offline scoring pass.

For A/C, ``official_semantic`` follows the BAT/SLAM evaluation contract:
generated text is embedded and compared by cosine similarity with the 355
AudioSet class embeddings, then AP is computed per class and averaged.  The
semantic backend is intentionally explicit; ``diagnostic_exact`` is available
only as a non-official fallback when no embedding service/assets are present.
"""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--rir-policy", choices=("official_bat", "checkpoint_matched"), default="official_bat")
    parser.add_argument("--binary-answer-prompt", choices=("off", "on", "auto"), default="off")
    parser.add_argument(
        "--detection-mode",
        choices=("official_semantic", "diagnostic_exact"),
        default="official_semantic",
        help="Used only for A/C; diagnostic_exact is explicitly non-official.",
    )
    parser.add_argument("--label-csv", type=Path, default=None)
    parser.add_argument("--label-embeddings", type=Path, default=None)
    # The official SLAM/BAT calculate_map.py calls text-embedding-ada-002;
    # audioset_class_embeds.npy is expected to be in that same embedding space.
    parser.add_argument("--embedding-model", default="text-embedding-ada-002")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
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
    """Online accumulator for the official A/C detection mAP contract."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.mode = args.detection_mode
        self.label_csv = args.label_csv
        self.label_embeddings_path = args.label_embeddings
        self.embedding_model = args.embedding_model
        self.client = None
        self.cache: dict[str, list[float]] = {}
        self.targets: list[list[int]] = []
        self.predictions: list[list[float]] = []
        self.unknown_target_labels: list[str] = []
        self.labels: list[str] = []
        self.label_to_index: dict[str, int] = {}
        self.label_embeddings = None
        if self.mode == "official_semantic":
            if self.label_csv is None or self.label_embeddings_path is None:
                raise ValueError("official_semantic requires --label-csv and --label-embeddings")
            self._load_label_assets()
            api_key = os.environ.get(args.openai_api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing {args.openai_api_key_env}; official A/C mAP requires the BAT text-embedding backend"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("official_semantic requires the openai Python package") from exc
            self.client = OpenAI(api_key=api_key)
        else:
            self._load_label_assets(require_embeddings=False)

    def _load_label_assets(self, require_embeddings: bool = True) -> None:
        import numpy as np

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
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        if len(self.labels) != 355:
            raise ValueError(f"Expected 355 AudioSet subset labels, got {len(self.labels)}")
        if require_embeddings:
            if self.label_embeddings_path is None or not self.label_embeddings_path.is_file():
                raise FileNotFoundError(f"Missing AudioSet label embeddings: {self.label_embeddings_path}")
            embeddings = np.asarray(np.load(self.label_embeddings_path, allow_pickle=False), dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != 355:
                raise ValueError(f"Expected [355,D] label embeddings, got {embeddings.shape}")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if not np.isfinite(embeddings).all() or np.any(norms == 0):
                raise ValueError("Label embeddings contain non-finite or zero-norm rows")
            self.label_embeddings = embeddings / norms

    def _embed(self, text: str) -> list[float]:
        if text in self.cache:
            return self.cache[text]
        response = self.client.embeddings.create(model=self.embedding_model, input=[text])
        vector = list(response.data[0].embedding)
        self.cache[text] = vector
        return vector

    def add(self, reference: str, generated: str) -> dict[str, Any]:
        import numpy as np

        target = np.zeros((355,), dtype=np.int8)
        unknown = []
        for label in split_labels(reference):
            index = self.label_to_index.get(label)
            if index is None:
                unknown.append(label)
            else:
                target[index] = 1
        self.unknown_target_labels.extend(unknown)
        if self.mode == "official_semantic":
            vector = np.asarray(self._embed(generated), dtype=np.float32)
            if vector.ndim != 1 or self.label_embeddings.shape[1] != vector.shape[0]:
                raise ValueError(
                    f"Embedding width mismatch labels={self.label_embeddings.shape[1]} text={vector.shape[0]}"
                )
            vector_norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector).all() or vector_norm == 0:
                raise ValueError("Generated text embedding is invalid")
            scores = (self.label_embeddings @ (vector / vector_norm)).astype(np.float32)
        else:
            normalized = generated.lower()
            scores = np.asarray(
                [1.0 if re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", normalized) else 0.0 for label in self.labels],
                dtype=np.float32,
            )
        self.targets.append(target.tolist())
        self.predictions.append(scores.tolist())
        return {
            "target_labels": split_labels(reference),
            "unknown_target_labels": unknown,
            "score_mode": self.mode,
            "official_metric": self.mode == "official_semantic",
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
            "official_metric": self.mode == "official_semantic",
            "embedding_model": self.embedding_model if self.mode == "official_semantic" else None,
            "label_count": len(self.labels),
            "unknown_target_label_count": len(self.unknown_target_labels),
            "unknown_target_label_examples": sorted(set(self.unknown_target_labels))[:20],
            "unique_text_embedding_count": len(self.cache),
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
    fail_if_public(args.output_jsonl)
    fail_if_public(args.output_report)
    if args.start_index < 0 or args.max_records < 0 or args.max_new_tokens <= 0 or args.num_beams <= 0:
        raise ValueError("start-index/max-records must be non-negative and generation limits positive")
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
    detection = DetectionScorer(args) if args.eval_type in {"A", "C"} else None
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
        print(f"[detection] mode={args.detection_mode} labels={args.label_csv} embeddings={args.label_embeddings}")
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
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    errors_count = 0
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for local_index, record in enumerate(selected):
            absolute_index = args.start_index + local_index
            eval_id = stable_eval_id(spec["relative_path"], absolute_index, record)
            started = time.perf_counter()
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
                if args.eval_type in {"B", "D"}:
                    score = location_score(str(record["answer"]), generated_text)
                    location_rows.append(score)
                elif args.eval_type.startswith("E-"):
                    score = binary_score(str(record["answer"]), generated_text)
                    binary_rows.append(score)
                else:
                    score = detection.add(str(record["answer"]), generated_text)
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
                    "generated_token_count": int(generated_ids.numel()),
                    "prompt_audit": prompt_audit,
                    "waveform_shape": list(waveform.shape),
                    "score": score,
                    "record_digest": record_digest(record),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                errors_count += 1
                errors.append({"eval_id": eval_id, "record_index": absolute_index, "error": repr(exc)})
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
                }
                print(f"[error] index={absolute_index} eval_id={eval_id} {exc}", file=sys.stderr)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            progress = {
                "status": "running",
                "eval_type": args.eval_type,
                "split": spec["relative_path"],
                "total_records": len(selected),
                "completed_records": local_index + 1,
                "last_record_index": absolute_index,
                "errors": errors_count,
                "output_jsonl": str(output_jsonl),
                "updated_unix": time.time(),
            }
            progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            if (local_index + 1) % 100 == 0 or local_index == 0:
                print(f"[progress] {local_index + 1}/{len(selected)} last={absolute_index} errors={errors_count}")
    metrics = metric_summary(args.eval_type, location_rows, binary_rows, detection)
    report = {
        "status": "ok" if errors_count == 0 and metrics.get("status", "ok") == "ok" else "incomplete",
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
            "completed_records": len(selected) - errors_count,
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
    progress_path.write_text(json.dumps({**report["dataset"], "status": report["status"]}, indent=2) + "\n", encoding="utf-8")
    print(f"[report] {args.output_report}")
    print(f"[jsonl] {output_jsonl}")
    print(f"[status] {report['status']} metrics={json.dumps(metrics, ensure_ascii=False)}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
