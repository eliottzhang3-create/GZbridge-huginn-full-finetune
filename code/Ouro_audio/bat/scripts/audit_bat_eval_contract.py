#!/usr/bin/env python3
"""Phase-I metadata-only audit for official BAT evaluation contracts.

This script intentionally does not import ms-swift, the custom plugins,
Spatial-AST, soundfile, scipy, or numpy.  It never opens an AudioSet WAV or an
RIR NPY.  Asset checks are limited to reference normalization and filesystem
path existence; model checks are limited to config/adapter metadata and
checkpoint tensor-key inspection.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from bat.eval_contract import (
    EVAL_SPECS,
    file_inventory,
    load_json_records,
    metadata_asset_coverage,
    sha256_file,
    summarize_eval_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reverb-root", type=Path, required=True)
    parser.add_argument("--spatial-ast-root", type=Path, required=True)
    parser.add_argument("--spatial-ast-checkpoint", type=Path, required=True)
    parser.add_argument("--qformer-source", type=Path, required=True)
    parser.add_argument("--ouro-model-path", type=Path, required=True)
    parser.add_argument("--ouro-plugin-path", type=Path, required=True)
    parser.add_argument("--ouro-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen3-model-path", type=Path, required=True)
    parser.add_argument("--qwen3-plugin-path", type=Path, required=True)
    parser.add_argument("--qwen3-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hash-bytes", type=int, default=64 * 1024 * 1024)
    return parser.parse_args()


def fail_if_public(path: Path) -> None:
    normalized = str(path.expanduser()).replace("\\", "/")
    if not path.is_absolute() or normalized.startswith("/hpc_stor03/public"):
        raise ValueError(f"Output must be an absolute private path: {path}")


def source_inventory(path: Path, required: tuple[str, ...]) -> dict[str, Any]:
    result = {"root": str(path), "exists": path.is_dir(), "required": {}}
    for name in required:
        result["required"][name] = file_inventory(path / name)
    return result


def plugin_inventory(path: Path, expected_model_type: str, expected_template_type: str) -> dict[str, Any]:
    item = file_inventory(path)
    item["model_type_expected"] = expected_model_type
    item["template_type_expected"] = expected_template_type
    if not path.is_file():
        item["status"] = "missing"
        return item
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
        item["ast_parse"] = "ok"
        item["function_count"] = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
        item["class_count"] = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    except SyntaxError as exc:
        item["ast_parse"] = "failed"
        item["error"] = repr(exc)
    model_match = re.search(r'MODEL_TYPE\s*=\s*["\']([^"\']+)', text)
    template_match = re.search(r'TEMPLATE_TYPE\s*=\s*["\']([^"\']+)', text)
    item["model_type_actual"] = model_match.group(1) if model_match else None
    item["template_type_actual"] = template_match.group(1) if template_match else None
    item["status"] = "ok" if item.get("model_type_actual") == expected_model_type and item.get("template_type_actual") == expected_template_type else "incomplete"
    return item


def model_inventory(path: Path, kind: str) -> dict[str, Any]:
    required = ["config.json"]
    if kind == "qwen3":
        required.extend(["model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"])
    else:
        required.extend(["tokenizer.json", "tokenizer_config.json", "configuration_ouro.py", "modeling_ouro.py"])
    result = source_inventory(path, tuple(required))
    weight_files: list[str] = []
    if path.is_dir():
        for pattern in ("*.safetensors", "*.bin", "*.pt"):
            weight_files.extend(item.name for item in path.glob(pattern))
        weight_files = sorted(set(weight_files))
    result["weight_files"] = weight_files
    result["weight_file_count"] = len(weight_files)
    result["status"] = "ok" if result["exists"] and all(item["exists"] for item in result["required"].values()) and weight_files else "incomplete"
    return result


def checkpoint_inventory(path: Path, kind: str, max_hash_bytes: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_dir(), "kind": kind, "issues": []}
    if not path.is_dir():
        result["status"] = "incomplete"
        result["issues"].append("checkpoint_missing")
        return result
    adapter = path / "adapter_model.safetensors"
    adapter_config_path = path / "adapter_config.json"
    state_path = path / "trainer_state.json"
    result["files"] = {
        name: file_inventory(path / name, max_hash_bytes)
        for name in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json")
    }
    if not adapter.is_file():
        result["issues"].append("adapter_model_missing")
    if not adapter_config_path.is_file():
        result["issues"].append("adapter_config_missing")
    if not state_path.is_file():
        result["issues"].append("trainer_state_missing")
    expected_layers = 24 if kind == "ouro" else 36
    expected_lora_tensors = expected_layers * 2 * 2
    keys: list[str] = []
    if adapter.is_file():
        try:
            from safetensors import safe_open

            with safe_open(str(adapter), framework="pt", device="cpu") as handle:
                keys = sorted(handle.keys())
        except Exception as exc:
            result["issues"].append(f"adapter_read_failed:{type(exc).__name__}")
    lora_keys = [key for key in keys if "lora_A" in key or "lora_B" in key]
    qformer_keys = [key for key in keys if "audio_qformer" in key]
    invalid_lora = [
        key for key in lora_keys
        if not re.search(r"layers\.\d+\.self_attn\.(q_proj|v_proj)\.lora_[AB](?:\.|$)", key)
    ]
    result.update({
        "tensor_count": len(keys),
        "lora_tensor_count": len(lora_keys),
        "expected_lora_tensor_count": expected_lora_tensors,
        "qformer_tensor_count": len(qformer_keys),
        "invalid_lora_key_count": len(invalid_lora),
        "invalid_lora_key_examples": invalid_lora[:10],
        "adapter_key_preview": keys[:12],
    })
    if len(lora_keys) != expected_lora_tensors:
        result["issues"].append(f"lora_tensor_count:{len(lora_keys)}!={expected_lora_tensors}")
    if not qformer_keys:
        result["issues"].append("audio_qformer_tensors_missing")
    if invalid_lora:
        result["issues"].append("invalid_lora_targets")
    if adapter_config_path.is_file():
        try:
            config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            target_modules = sorted(str(item) for item in config.get("target_modules", []))
            modules_to_save = sorted(str(item) for item in config.get("modules_to_save", []))
            result["adapter_config"] = {
                "target_modules": target_modules,
                "r": config.get("r"),
                "lora_alpha": config.get("lora_alpha"),
                "lora_dropout": config.get("lora_dropout"),
                "modules_to_save": modules_to_save,
            }
            if target_modules != ["q_proj", "v_proj"]:
                result["issues"].append(f"target_modules:{target_modules}")
            if config.get("r") != 8 or config.get("lora_alpha") != 32:
                result["issues"].append("lora_rank_or_alpha_mismatch")
            if float(config.get("lora_dropout", -1.0)) != 0.05:
                result["issues"].append("lora_dropout_mismatch")
            if modules_to_save != ["audio_qformer"]:
                result["issues"].append(f"modules_to_save:{modules_to_save}")
        except Exception as exc:
            result["issues"].append(f"adapter_config_invalid:{type(exc).__name__}")
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            result["global_step"] = int(state.get("global_step", -1))
            if result["global_step"] != 10500:
                result["issues"].append(f"global_step:{result['global_step']}!=10500")
        except Exception as exc:
            result["issues"].append(f"trainer_state_invalid:{type(exc).__name__}")
    result["status"] = "ok" if not result["issues"] else "incomplete"
    return result


def audit_eval_files(qa_root: Path, audio_root: Path, reverb_root: Path, issues: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not qa_root.is_dir():
        issues.append("qa_root_missing")
    if not audio_root.is_dir():
        issues.append("audio_root_missing")
    if not reverb_root.is_dir():
        issues.append("reverb_root_missing")
    for spec in EVAL_SPECS:
        path = qa_root / spec["relative_path"]
        key = spec["relative_path"]
        if not path.is_file():
            result[key] = {"status": "missing", "path": str(path), "official_type": spec["name"]}
            issues.append(f"eval_missing:{key}")
            continue
        try:
            records, container = load_json_records(path)
            summary = summarize_eval_records(records, spec, path.parent.name)
            coverage = metadata_asset_coverage(records, audio_root, reverb_root)
            item = {
                "status": "ok" if summary["invalid_contract_record_count"] == 0 and not coverage["audio_missing_count"] and not coverage["reverb_missing_count"] else "incomplete",
                "path": str(path),
                "sha256": sha256_file(path),
                "container": container,
                "official_spec": spec,
                "summary": summary,
                "asset_coverage": coverage,
            }
            result[key] = item
            if summary["invalid_contract_record_count"]:
                issues.append(f"eval_contract_records:{key}:{summary['invalid_contract_record_count']}")
            if coverage["audio_missing_count"]:
                issues.append(f"eval_audio_missing:{key}:{coverage['audio_missing_count']}")
            if coverage["reverb_missing_count"]:
                issues.append(f"eval_reverb_missing:{key}:{coverage['reverb_missing_count']}")
        except Exception as exc:
            result[key] = {"status": "invalid", "path": str(path), "error": repr(exc)}
            issues.append(f"eval_invalid:{key}")
    return result


def main() -> None:
    args = parse_args()
    fail_if_public(args.output)
    issues: list[str] = []
    print("========== BAT PHASE 1 EVALUATION CONTRACT AUDIT ==========")
    print("[scope] metadata-only; no audio decode, RIR load, convolution, Spatial-AST import, or model load")
    print(f"[qa] {args.qa_root}")
    print(f"[audio] {args.audio_root} (path existence only)")
    print(f"[reverb] {args.reverb_root} (path existence only)")
    evaluations = audit_eval_files(args.qa_root, args.audio_root, args.reverb_root, issues)

    sources = {
        "spatial_ast": source_inventory(args.spatial_ast_root, ("spatial_ast.py", "data/dataset.py")),
        "spatial_ast_checkpoint": file_inventory(args.spatial_ast_checkpoint, args.max_hash_bytes),
        "qformer": file_inventory(args.qformer_source, args.max_hash_bytes),
        "ouro_plugin": plugin_inventory(args.ouro_plugin_path, "ouro_bat_spatial_ast", "ouro_bat_audio_prefix"),
        "qwen3_plugin": plugin_inventory(args.qwen3_plugin_path, "qwen3_bat_spatial_ast", "qwen3_bat_audio_prefix"),
    }
    if not sources["spatial_ast"]["exists"]:
        issues.append("spatial_ast_root_missing")
    if not sources["spatial_ast"]["required"]["spatial_ast.py"]["exists"]:
        issues.append("spatial_ast_source_missing")
    if not sources["spatial_ast_checkpoint"]["exists"]:
        issues.append("spatial_ast_checkpoint_missing")
    if not sources["qformer"]["exists"]:
        issues.append("qformer_source_missing")
    for name in ("ouro_plugin", "qwen3_plugin"):
        if sources[name].get("status") != "ok":
            issues.append(f"{name}_contract_invalid")

    models = {
        "ouro": model_inventory(args.ouro_model_path, "ouro"),
        "qwen3": model_inventory(args.qwen3_model_path, "qwen3"),
    }
    for name, item in models.items():
        if item["status"] != "ok":
            issues.append(f"{name}_base_model_incomplete")

    checkpoints = {
        "ouro": checkpoint_inventory(args.ouro_checkpoint, "ouro", args.max_hash_bytes),
        "qwen3": checkpoint_inventory(args.qwen3_checkpoint, "qwen3", args.max_hash_bytes),
    }
    for name, item in checkpoints.items():
        if item["status"] != "ok":
            issues.append(f"{name}_checkpoint_contract_invalid")

    report = {
        "status": "ok" if not issues else "incomplete",
        "python": {"version": sys.version, "executable": sys.executable},
        "scope": {
            "metadata_only": True,
            "audio_decode": False,
            "rir_load": False,
            "convolution": False,
            "spatial_ast_import": False,
            "model_load": False,
            "table4_main_types": ["A", "B", "C", "D", "E-direction", "E-distance"],
            "e_nonbinary_included_as": "diagnostic_only",
        },
        "paths": {
            "qa_root": str(args.qa_root.resolve()),
            "audio_root": str(args.audio_root.resolve()),
            "reverb_root": str(args.reverb_root.resolve()),
        },
        "evaluations": evaluations,
        "sources": sources,
        "models": models,
        "checkpoints": checkpoints,
        "generation_contracts": {
            "current_stable_eval": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 24,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
            },
            "research_reference": {
                "do_sample": False,
                "num_beams": 4,
                "max_new_tokens": 200,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
            },
            "note": "Current results use the stable one-beam 24-token contract requested for Ouro evaluation; they are not claimed as strict official-decoder reproduction.",
        },
        "audio_contracts": {
            "official_bat": "raw RIR -> convolution -> final [2,320000] crop/pad",
            "checkpoint_matched": "RIR crop/pad to 2s -> convolution -> final [2,320000] crop/pad",
            "training_current": "checkpoint_matched",
        },
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for key, item in evaluations.items():
        summary = item.get("summary", {})
        coverage = item.get("asset_coverage", {})
        print(f"[summary] {key} records={summary.get('record_count')} type={summary.get('official_type')} "
              f"audio={coverage.get('audio_matched_count')}/{coverage.get('audio_reference_count')} "
              f"reverb={coverage.get('reverb_matched_count')}/{coverage.get('reverb_reference_count')}")
    print(f"[report] {args.output}")
    print(f"[status] {report['status']} issues={issues}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
