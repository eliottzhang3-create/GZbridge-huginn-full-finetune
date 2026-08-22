#!/usr/bin/env python3
"""Phase-II small BAT evaluation generation smoke.

The smoke loads one trained adapter checkpoint, renders only a very small
number of official evaluation records, and performs deterministic generation.
It writes raw generations first-class so the later scorer can be rerun without
repeating the expensive audio/Spatial-AST path.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from importlib.metadata import version
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
    stable_eval_id,
)


MODEL_SETTINGS = {
    "ouro": {
        "model_type": "ouro_bat_spatial_ast",
        "template_type": "ouro_bat_audio_prefix",
        "model_env": "OURO_MODEL_PATH",
        "model_class": "OuroForCausalLM",
        "hidden_size": 2048,
        "plugin_contract_attr": "_ouro_bat_audio_contract",
    },
    "qwen3": {
        "model_type": "qwen3_bat_spatial_ast",
        "template_type": "qwen3_bat_audio_prefix",
        "model_env": "QWEN3_MODEL_PATH",
        "model_class": "Qwen3ForCausalLM",
        "hidden_size": 2560,
        "plugin_contract_attr": "_qwen3_bat_audio_contract",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=tuple(MODEL_SETTINGS), required=True)
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
    parser.add_argument("--max-records-per-split", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--rir-policy", choices=("official_bat", "checkpoint_matched"), default="official_bat")
    parser.add_argument("--include-nonbinary", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--binary-answer-prompt",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Append a yes/no-only instruction to binary evaluation questions. "
            "auto enables it for Qwen3 and disables it for Ouro."
        ),
    )
    return parser.parse_args()


def fail_if_public(path: Path) -> None:
    normalized = str(path.expanduser()).replace("\\", "/")
    if not path.is_absolute() or normalized.startswith("/hpc_stor03/public"):
        raise ValueError(f"Output must be an absolute private path: {path}")


def import_plugin(path: Path, model_kind: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = f"bat_eval_{model_kind}_plugin"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def tokenizer_from_processor(processor: Any) -> Any:
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def as_ids(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Template did not produce a non-empty token list: {type(value).__name__}")
    return [int(item) for item in value]


def load_adapter(base_model: torch.nn.Module, checkpoint: Path) -> torch.nn.Module:
    from peft import PeftModel

    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter_model.safetensors: {checkpoint}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint), is_trainable=False)
    model.set_adapter("default")
    return model


def freeze_for_evaluation(model: torch.nn.Module) -> None:
    """Make the loaded adapter a strictly inference-only model.

    PEFT's ``modules_to_save`` wrappers can keep their parameters marked as
    trainable even when ``is_trainable=False`` is passed to
    ``from_pretrained``.  That is useful for continuing training, but is not
    the contract we want for evaluation.  The checkpoint weights remain
    loaded; this only disables autograd for the evaluation process.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def base_model_of(model: torch.nn.Module) -> torch.nn.Module:
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        return getter()
    return model


def parameter_contract(model: torch.nn.Module, model_kind: str) -> dict[str, Any]:
    base = base_model_of(model)
    spatial = getattr(base, "spatial_ast_encoder", None)
    qformer = getattr(base, "audio_qformer", None)
    if spatial is None or qformer is None:
        raise RuntimeError("Loaded model does not expose spatial_ast_encoder/audio_qformer")
    spatial_trainable = sum(parameter.numel() for parameter in spatial.parameters() if parameter.requires_grad)
    qformer_trainable = sum(parameter.numel() for parameter in qformer.parameters() if parameter.requires_grad)
    native_trainable = 0
    lora_trainable = 0
    total_parameters = 0
    trainable_parameter_count = 0
    trainable_names: list[str] = []
    loaded_lora_parameters = 0
    loaded_qformer_parameters = 0
    for name, parameter in model.named_parameters():
        total_parameters += parameter.numel()
        if "lora_A" in name or "lora_B" in name:
            loaded_lora_parameters += parameter.numel()
        if "audio_qformer" in name:
            loaded_qformer_parameters += parameter.numel()
        if not parameter.requires_grad:
            continue
        trainable_parameter_count += parameter.numel()
        trainable_names.append(name)
        if "lora_A" in name or "lora_B" in name:
            lora_trainable += parameter.numel()
        elif "audio_qformer" not in name:
            native_trainable += parameter.numel()
    if trainable_parameter_count:
        raise RuntimeError(
            "Evaluation model must be fully frozen after adapter loading: "
            f"total={trainable_parameter_count} preview={trainable_names[:8]}"
        )
    contract = getattr(base, MODEL_SETTINGS[model_kind]["plugin_contract_attr"], None)
    if not isinstance(contract, dict):
        raise RuntimeError("BAT plugin audio contract is missing")
    if contract.get("audio_token_count") != 64:
        raise RuntimeError(f"Expected 64 audio tokens, contract={contract}")
    expected_hidden = int(MODEL_SETTINGS[model_kind]["hidden_size"])
    config_hidden = int(getattr(getattr(base, "config", None), "hidden_size", -1))
    hidden_key = "ouro_hidden_size" if model_kind == "ouro" else "qwen3_hidden_size"
    contract_hidden = int(
        contract.get("model_hidden_size", contract.get(hidden_key, contract.get("hidden_size", -1)))
    )
    if config_hidden != expected_hidden or contract_hidden != expected_hidden:
        raise RuntimeError(
            f"{model_kind} hidden-size contract mismatch: expected={expected_hidden} "
            f"config={config_hidden} plugin={contract_hidden}"
        )
    return {
        "total_parameters": total_parameters,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_name_count": len(trainable_names),
        "trainable_name_preview": trainable_names[:8],
        "spatial_ast_trainable_parameters": spatial_trainable,
        "qformer_trainable_parameters": qformer_trainable,
        "native_trainable_parameters": native_trainable,
        "lora_trainable_parameters": lora_trainable,
        "loaded_lora_parameters": loaded_lora_parameters,
        "loaded_qformer_parameters": loaded_qformer_parameters,
        "spatial_ast_dtype_set": sorted({str(parameter.dtype) for parameter in spatial.parameters()}),
        "audio_contract": contract,
    }


def binary_answer_prompt_applies(record: dict[str, Any], model_kind: str, mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "auto" and model_kind != "qwen3":
        return False
    question_type = str(record.get("question_type", "")).upper()
    normalized_answer = " ".join(str(record.get("answer", "")).strip().lower().split())
    return question_type in {"MIXUP_DIRECTION", "MIXUP_DISTANCE_BOTH"} or normalized_answer in {"yes", "no"}


def build_encoded_prompt(
    template: Any,
    record: dict[str, Any],
    model_kind: str,
    binary_answer_prompt_mode: str,
) -> tuple[list[int], dict[str, Any]]:
    # The dummy waveform prevents the template from touching AudioSet/RIR.  The
    # real waveform is injected immediately before generation by the explicit
    # evaluation renderer.
    dummy_record = dict(record)
    dummy_record["waveform"] = torch.zeros((2, 320_000), dtype=torch.float32)
    original_instruction = str(record["question"])
    apply_binary_prompt = binary_answer_prompt_applies(record, model_kind, binary_answer_prompt_mode)
    effective_instruction = original_instruction
    if apply_binary_prompt:
        effective_instruction = (
            f'{original_instruction}\n\nPlease answer only "yes" or "no".'
        )
    encoded = template.encode({
        "messages": [{"role": "user", "content": BAT_PROMPT.format(instruction=effective_instruction)}],
        "audios": [dummy_record],
    })
    input_ids = as_ids(encoded.get("input_ids"))
    if len(input_ids) <= 64:
        raise RuntimeError(f"Prompt has no text after audio prefix: length={len(input_ids)}")
    if not torch.is_tensor(encoded.get("audio_waveform")):
        raise RuntimeError("Template did not produce dummy audio_waveform")
    return input_ids, {
        "template_input_length": len(input_ids),
        "audio_prefix_tokens": 64,
        "dummy_waveform_used_for_template": True,
        "model_kind": model_kind,
        "binary_answer_prompt_mode": binary_answer_prompt_mode,
        "binary_answer_prompt_applied": apply_binary_prompt,
        "original_instruction": original_instruction,
        "effective_instruction": effective_instruction,
    }


def parse_smoke_output(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    raw_type = str(record.get("question_type", "")).upper()
    if raw_type in {"DOA", "MIXUP_SINGLE_DOA"}:
        return {"task_parser": "location", **parse_location(generated_text)}
    if raw_type in {"MIXUP_DIRECTION", "MIXUP_DISTANCE_BOTH"}:
        return {"task_parser": "yes_no", **parse_yes_no(generated_text)}
    return {
        "task_parser": "detection_text_nonempty",
        "normalized": generated_text.strip(),
        "status": "ok" if generated_text.strip() else "empty_generation",
    }


def reference_comparison(record: dict[str, Any], generated_text: str) -> dict[str, Any]:
    """Return descriptive reference-vs-generation information, not a score."""
    reference = str(record.get("answer", ""))
    generated = str(generated_text)
    normalized_reference = " ".join(reference.strip().lower().split())
    normalized_generated = " ".join(generated.strip().lower().split())
    generated_parser = parse_smoke_output(record, generated)
    reference_parser = parse_smoke_output(record, reference)
    comparison: dict[str, Any] = {
        "reference_answer": reference,
        "generated_text": generated,
        "normalized_text_exact_match": normalized_reference == normalized_generated,
        "reference_parser": reference_parser,
        "generated_parser": generated_parser,
        "parser_value_match": None,
        "formal_metric_computed": False,
    }
    if reference_parser.get("task_parser") == "yes_no":
        comparison["parser_value_match"] = (
            reference_parser.get("value") is not None
            and reference_parser.get("value") == generated_parser.get("value")
        )
    elif reference_parser.get("task_parser") == "location":
        comparison["parser_value_match"] = (
            reference_parser.get("direction") is not None
            and reference_parser.get("direction") == generated_parser.get("direction")
            and reference_parser.get("distance_m") is not None
            and reference_parser.get("distance_m") == generated_parser.get("distance_m")
        )
    return comparison


def selected_specs(include_nonbinary: bool) -> list[dict[str, Any]]:
    return [spec for spec in EVAL_SPECS if include_nonbinary or spec["name"] != "E-nonbinary"]


def effective_generation_limits(spec_name: str, max_new_tokens: int, num_beams: int) -> tuple[int, int]:
    """Apply the common bounded generation contract to every BAT split."""

    return min(max_new_tokens, 24), 1


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)


def main() -> None:
    args = parse_args()
    fail_if_public(args.output_jsonl)
    fail_if_public(args.output_report)
    if args.max_records_per_split <= 0 or args.repeat <= 0 or args.max_new_tokens <= 0 or args.num_beams <= 0:
        raise ValueError("max-records-per-split, repeat, max-new-tokens and num-beams must be positive")
    if args.max_new_tokens > 24 or args.num_beams != 1:
        raise ValueError("BAT evaluation smoke requires max_new_tokens<=24 and num_beams=1")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-II generation smoke requires a submitted CUDA job")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device}")
    torch.cuda.set_device(device)
    settings = MODEL_SETTINGS[args.model_kind]
    for path in (args.model_path, args.plugin_path, args.checkpoint, args.qa_root, args.audio_root, args.reverb_root, args.spatial_ast_root, args.spatial_ast_checkpoint, args.qformer_source):
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ[settings["model_env"]] = str(args.model_path.resolve())
    os.environ["BAT_AUDIO_ROOT"] = str(args.audio_root.resolve())
    os.environ["BAT_REVERB_ROOT"] = str(args.reverb_root.resolve())
    os.environ["BAT_SPATIAL_AST_CODE_ROOT"] = str(args.spatial_ast_root.resolve())
    os.environ["BAT_SPATIAL_AST_CHECKPOINT"] = str(args.spatial_ast_checkpoint.resolve())
    os.environ["BAT_QFORMER_SOURCE"] = str(args.qformer_source.resolve())

    expected_packages = {"ms-swift": "4.4.2", "transformers": "4.54.1", "peft": "0.18.1"}
    package_report = {name: version(name) for name in expected_packages}
    mismatches = {name: (expected, package_report[name]) for name, expected in expected_packages.items() if package_report[name] != expected}
    if mismatches:
        raise RuntimeError(f"Unexpected evaluation environment: {mismatches}")

    plugin = import_plugin(args.plugin_path.resolve(), args.model_kind)
    if plugin.MODEL_TYPE != settings["model_type"] or plugin.TEMPLATE_TYPE != settings["template_type"]:
        raise RuntimeError("Plugin registration constants do not match model-kind")
    try:
        from swift import get_model_processor, get_template
    except ImportError:
        from swift.model import get_model_processor
        from swift.template import get_template

    print("========== BAT PHASE 2 EVALUATION GENERATION SMOKE ==========")
    print(f"[model] kind={args.model_kind} base={args.model_path} checkpoint={args.checkpoint}")
    print(f"[audio] policy={args.rir_policy} root={args.audio_root} reverb={args.reverb_root}")
    print(f"[generation] do_sample=false num_beams={args.num_beams} max_new_tokens={args.max_new_tokens} repeat={args.repeat}")
    load_started = time.perf_counter()
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
    if base_model.__class__.__name__ != settings["model_class"]:
        raise RuntimeError(f"Unexpected base model class: {base_model.__class__.__name__}")
    model = load_adapter(base_model, args.checkpoint)
    freeze_for_evaluation(model)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    contract_report = parameter_contract(model, args.model_kind)
    base_config = base_model.config
    template = get_template(
        template_type=settings["template_type"],
        processor=processor,
        max_length=512,
        use_chat_template=False,
        padding_side="right",
        padding_free=False,
        template_backend="swift",
    )
    template.set_mode("transformers")
    tokenizer = tokenizer_from_processor(processor)
    renderer = BATEvalAudioRenderer(args.audio_root, args.reverb_root, args.rir_policy)
    output_rows: list[dict[str, Any]] = []
    issues: list[str] = []
    binary_prompt_count = 0
    aborted_on_cuda_oom = False
    first_cuda_oom: dict[str, Any] | None = None
    specs = selected_specs(args.include_nonbinary)
    for spec in specs:
        effective_max_new_tokens, effective_num_beams = effective_generation_limits(
            spec["name"], args.max_new_tokens, args.num_beams
        )
        split_path = args.qa_root / spec["relative_path"]
        records, _ = load_json_records(split_path)
        for index, record in enumerate(records[: args.max_records_per_split]):
            eval_id = stable_eval_id(spec["relative_path"], index, record)
            input_ids: list[int] | None = None
            template_audit: dict[str, Any] | None = None
            waveform: Any = None
            attention_mask: Any = None
            input_tensor: Any = None
            try:
                input_ids, template_audit = build_encoded_prompt(
                    template,
                    record,
                    args.model_kind,
                    args.binary_answer_prompt,
                )
                binary_prompt_count += int(template_audit["binary_answer_prompt_applied"])
                waveform = renderer.render_record(record).unsqueeze(0).to(device=device, dtype=torch.float32)
                attention_mask = torch.ones((1, len(input_ids)), dtype=torch.long, device=device)
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                repeat_rows: list[dict[str, Any]] = []
                for repeat_index in range(args.repeat):
                    started = time.perf_counter()
                    output_ids: Any = None
                    generated_ids: Any = None
                    try:
                        with torch.inference_mode():
                            output_ids = model.generate(
                                input_ids=input_tensor,
                                attention_mask=attention_mask,
                                audio_waveforms=waveform,
                                max_new_tokens=effective_max_new_tokens,
                                num_beams=effective_num_beams,
                                do_sample=False,
                                top_p=1.0,
                                repetition_penalty=1.0,
                                length_penalty=1.0,
                                use_cache=True,
                                eos_token_id=getattr(base_config, "eos_token_id", None),
                                pad_token_id=getattr(base_config, "pad_token_id", None) or tokenizer.eos_token_id,
                            )
                        torch.cuda.synchronize(device)
                        elapsed = time.perf_counter() - started
                        if output_ids.shape[1] <= len(input_ids):
                            generated_ids = output_ids[0, 0:0]
                        else:
                            generated_ids = output_ids[0, len(input_ids):]
                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        repeat_rows.append({
                            "repeat_index": repeat_index,
                            "generated_token_count": int(generated_ids.numel()),
                            "generated_text": generated_text,
                            "elapsed_seconds": elapsed,
                            "parser": parse_smoke_output(record, generated_text),
                            "reference_comparison": reference_comparison(record, generated_text),
                        })
                    finally:
                        del output_ids, generated_ids
                        gc.collect()
                        torch.cuda.empty_cache()
                deterministic = all(row["generated_text"] == repeat_rows[0]["generated_text"] for row in repeat_rows[1:])
                if not deterministic:
                    issues.append(f"nondeterministic:{eval_id}")
                if any(row["parser"]["status"] in {"empty_generation", "invalid_location", "invalid_yes_no"} for row in repeat_rows):
                    issues.append(f"parser_issue:{eval_id}")
                output_rows.append({
                    "eval_id": eval_id,
                    "split": spec["relative_path"],
                    "official_type": spec["name"],
                    "question_id": str(record.get("question_id")),
                    "question_type": str(record.get("question_type")),
                    "source_shape": "dual" if record.get("audio_id2") else "single",
                    "audio_id": record.get("audio_id"),
                    "reverb_id": record.get("reverb_id"),
                    "audio_id2": record.get("audio_id2"),
                    "reverb_id2": record.get("reverb_id2"),
                    "question": record.get("question"),
                    "effective_instruction": template_audit["effective_instruction"],
                    "binary_answer_prompt_applied": template_audit["binary_answer_prompt_applied"],
                    "ground_truth_answer": record.get("answer"),
                    "record_digest": record_digest(record),
                    "template_audit": template_audit,
                    "waveform_shape": list(waveform.shape),
                    "repeat_count": args.repeat,
                    "deterministic": deterministic,
                    "repeats": repeat_rows,
                    "generation_limits": {
                        "max_new_tokens": effective_max_new_tokens,
                        "num_beams": effective_num_beams,
                    },
                })
                print(f"[sample] {eval_id} type={spec['name']} deterministic={deterministic} text={repeat_rows[0]['generated_text']!r}")
            except Exception as exc:
                issues.append(f"generation_failed:{eval_id}:{type(exc).__name__}")
                output_rows.append({"eval_id": eval_id, "split": spec["relative_path"], "status": "error", "error": repr(exc)})
                if is_cuda_oom(exc):
                    aborted_on_cuda_oom = True
                    first_cuda_oom = {"eval_id": eval_id, "record_index": index, "error": repr(exc)}
                    print(f"[sample-abort] first CUDA OOM at {eval_id}; stopping smoke", file=sys.stderr)
                else:
                    print(f"[sample-error] {eval_id} {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                del waveform, input_tensor, attention_mask, input_ids, template_audit
                gc.collect()
                torch.cuda.empty_cache()
            if aborted_on_cuda_oom:
                break
        if aborted_on_cuda_oom:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_jsonl = args.output_jsonl.with_name(args.output_jsonl.name + ".tmp")
    with temporary_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_jsonl.replace(args.output_jsonl)
    report = {
        "status": "ok" if not issues and output_rows and not aborted_on_cuda_oom else "incomplete",
        "scope": {
            "phase": "II_generation_smoke",
            "official_eval_splits": [spec["relative_path"] for spec in specs],
            "full_eval": False,
            "audio_convolution_executed": True,
            "spatial_ast_executed": True,
            "scoring_executed": False,
        },
        "packages": package_report,
        "model_kind": args.model_kind,
        "model_path": str(args.model_path.resolve()),
        "plugin_path": str(args.plugin_path.resolve()),
        "checkpoint": file_inventory(args.checkpoint / "adapter_model.safetensors"),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "device": {"name": torch.cuda.get_device_name(device), "index": device.index},
        "load_seconds": load_seconds,
        "contract": contract_report,
        "audio_contract": {
            "rir_policy": args.rir_policy,
            "final_waveform_shape": [2, 320000],
            "sample_rate": 32000,
            "normalization": "RMS/loudness target -14 dBFS",
            "dual_source_mix": "(source1 + source2) / 2",
        },
        "generation": {
            "do_sample": False,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "repeat": args.repeat,
            "binary_answer_prompt_mode": args.binary_answer_prompt,
            "binary_answer_prompt_count": binary_prompt_count,
            "binary_answer_prompt_text": 'Please answer only "yes" or "no".',
            "all_eval_types_generation_cap": {"max_new_tokens": 24, "num_beams": 1},
        },
        "termination": {
            "aborted_on_cuda_oom": aborted_on_cuda_oom,
            "first_cuda_oom": first_cuda_oom,
        },
        "sample_count": len(output_rows),
        "successful_sample_count": sum(row.get("status", "ok") != "error" for row in output_rows),
        "sample_reference_comparisons": [
            {
                "eval_id": row.get("eval_id"),
                "split": row.get("split"),
                "official_type": row.get("official_type"),
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "reference_answer": row.get("ground_truth_answer"),
                "generations": [
                    {
                        "repeat_index": repeat.get("repeat_index"),
                        "generated_text": repeat.get("generated_text"),
                        "reference_comparison": repeat.get("reference_comparison"),
                    }
                    for repeat in row.get("repeats", [])
                ],
            }
            for row in output_rows
        ],
        "reference_comparison_scope": {
            "included_in_each_repeat": True,
            "formal_metric_computed": False,
            "note": "Reference comparisons are smoke diagnostics only; they are not Table 4 scores.",
        },
        "output_jsonl": str(args.output_jsonl.resolve()),
        "issues": issues,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.output_report.with_name(args.output_report.name + ".tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(args.output_report)
    print(f"[report] {args.output_report}")
    print(f"[raw] {args.output_jsonl}")
    print(f"[status] {report['status']} issues={issues}")
    if issues or not output_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
