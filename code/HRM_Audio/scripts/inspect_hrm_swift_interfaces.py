#!/usr/bin/env python3
"""Inspect ms-swift 4.4.2 registration/template interfaces without registering HRM."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import torch


DEFAULT_MODEL_PATH = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text"
PLANNED_MODEL_TYPE = "hrm_text_native"
PLANNED_TEMPLATES = ("hrm_text_direct", "hrm_text_synth_cot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--source-hit-limit", type=int, default=120)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def safe_source(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def dataclass_field_names(obj: Any) -> list[str]:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass, got {obj!r}")
    return [item.name for item in fields(obj)]


def require_fields(class_name: str, actual: list[str], required: set[str]) -> None:
    missing = sorted(required - set(actual))
    if missing:
        raise RuntimeError(f"{class_name} lacks required dataclass fields: {missing}; actual={actual}")


def iter_python_files(root: Path) -> Iterable[Path]:
    yield from (path for path in root.rglob("*.py") if path.is_file())


def source_search(root: Path, patterns: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    python_files = list(iter_python_files(root))
    per_pattern_limit = max(1, limit // len(patterns))
    for pattern in patterns:
        pattern_hits = 0
        for path in python_files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern not in line:
                    continue
                hits.append(
                    {
                        "path": str(path.relative_to(root.parent)),
                        "line": line_number,
                        "matched": [pattern],
                        "source": line.strip(),
                    }
                )
                pattern_hits += 1
                if pattern_hits >= per_pattern_limit or len(hits) >= limit:
                    break
            if pattern_hits >= per_pattern_limit or len(hits) >= limit:
                break
        if len(hits) >= limit:
            break
    return hits


def registry_report(modules: list[Any]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    needles = ("hrm", "HrmTextForCausalLM", PLANNED_MODEL_TYPE, *PLANNED_TEMPLATES)
    for module in modules:
        for attribute_name, value in vars(module).items():
            if "MAPPING" not in attribute_name.upper() or not isinstance(value, Mapping):
                continue
            registry_name = f"{module.__name__}.{attribute_name}"
            keys = [str(key) for key in value.keys()]
            matching_entries: list[dict[str, str]] = []
            for key, item in value.items():
                combined = f"{key!s} {item!r}"
                if any(needle.lower() in combined.lower() for needle in needles):
                    matching_entries.append({"key": str(key), "value_repr": repr(item)[:2000]})
            report[registry_name] = {
                "size": len(value),
                "key_sample": keys[:30],
                "matching_entries": matching_entries,
            }
    return report


def exact_identifier_conflicts(registries: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    planned = {PLANNED_MODEL_TYPE, *PLANNED_TEMPLATES}
    conflicts: list[dict[str, str]] = []
    for registry_name, registry in registries.items():
        for entry in registry["matching_entries"]:
            if entry["key"] in planned:
                conflicts.append({"registry": registry_name, **entry})
    return conflicts


def main() -> None:
    args = parse_args()
    if version("ms-swift") != "4.4.2":
        raise RuntimeError(f"Expected ms-swift==4.4.2, got {version('ms-swift')}")
    if version("transformers") != "5.9.0":
        raise RuntimeError(f"Expected transformers==5.9.0, got {version('transformers')}")

    import swift
    import swift.model as swift_model
    import swift.template as swift_template
    from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
    from swift.template import Template, TemplateMeta, register_template
    from transformers import AutoConfig, HrmTextForCausalLM

    model_register_module = importlib.import_module("swift.model.register")
    model_meta_module = importlib.import_module("swift.model.model_meta")
    template_register_module = importlib.import_module("swift.template.register")
    template_meta_module = importlib.import_module("swift.template.template_meta")
    registry_modules = [
        swift_model,
        model_register_module,
        model_meta_module,
        swift_template,
        template_register_module,
        template_meta_module,
    ]
    registries_before = registry_report(registry_modules)

    swift_root = Path(swift.__file__).resolve().parent
    model_path = args.model_path.resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing HRM config: {config_path}")

    print("========== HRM SWIFT INTERFACE INSPECT ==========", flush=True)
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}", flush=True)
    print(f"[platform] {platform.platform()}", flush=True)
    print(f"[swift] version={swift.__version__} root={swift_root}", flush=True)
    print(f"[model-path] {model_path}", flush=True)

    symbols = {
        "Model": Model,
        "ModelGroup": ModelGroup,
        "ModelMeta": ModelMeta,
        "ModelLoader": ModelLoader,
        "register_model": register_model,
        "Template": Template,
        "TemplateMeta": TemplateMeta,
        "register_template": register_template,
    }
    signatures = {name: safe_signature(obj) for name, obj in symbols.items()}
    for name, signature in signatures.items():
        print(f"[signature] {name}{signature}", flush=True)

    register_model_parameters = inspect.signature(register_model).parameters
    register_template_parameters = inspect.signature(register_template).parameters
    if "exist_ok" not in register_model_parameters:
        raise RuntimeError(f"register_model lacks exist_ok: {signatures['register_model']}")
    if "exist_ok" not in register_template_parameters:
        raise RuntimeError(f"register_template lacks exist_ok: {signatures['register_template']}")

    model_fields = dataclass_field_names(Model)
    model_group_fields = dataclass_field_names(ModelGroup)
    model_meta_fields = dataclass_field_names(ModelMeta)
    template_meta_fields = dataclass_field_names(TemplateMeta)
    require_fields("Model", model_fields, {"model_path"})
    require_fields("ModelGroup", model_group_fields, {"models", "template", "requires", "tags"})
    require_fields(
        "ModelMeta",
        model_meta_fields,
        {
            "model_type",
            "model_groups",
            "loader",
            "template",
            "model_arch",
            "architectures",
            "torch_dtype",
            "is_multimodal",
            "requires",
            "tags",
        },
    )
    require_fields(
        "TemplateMeta",
        template_meta_fields,
        {
            "template_type",
            "prefix",
            "prompt",
            "chat_sep",
            "suffix",
            "template_cls",
            "system_prefix",
            "default_system",
            "stop_words",
            "auto_add_bos",
        },
    )
    dataclass_fields_report = {
        "Model": model_fields,
        "ModelGroup": model_group_fields,
        "ModelMeta": model_meta_fields,
        "TemplateMeta": template_meta_fields,
    }
    for name, field_names in dataclass_fields_report.items():
        print(f"[dataclass-fields] {name}={field_names}", flush=True)

    # Constructor trials only. These objects are deliberately not passed to
    # register_model/register_template and therefore cannot mutate Swift mappings.
    planned_model = Model(model_path=str(model_path))
    planned_group = ModelGroup(models=[planned_model])
    planned_meta = ModelMeta(
        model_type=PLANNED_MODEL_TYPE,
        model_groups=[planned_group],
        template=PLANNED_TEMPLATES[0],
        architectures=["HrmTextForCausalLM"],
        torch_dtype=torch.bfloat16,
        requires=["transformers==5.9.0"],
        tags=["hrm", "text", "prefix-lm"],
    )
    planned_template = TemplateMeta(
        template_type=PLANNED_TEMPLATES[0],
        prefix=[],
        prompt=["<|im_start|><|object_ref_start|>{{QUERY}}<|im_end|>"],
        chat_sep=None,
        suffix=[["eos_token_id"]],
        auto_add_bos=False,
        stop_words=[],
    )
    if planned_meta.loader is not ModelLoader:
        raise RuntimeError(f"Default ModelMeta loader is not ModelLoader: {planned_meta.loader}")
    print(f"[constructor] Model={planned_model!r}", flush=True)
    print(f"[constructor] ModelGroup={planned_group!r}", flush=True)
    print(f"[constructor] ModelMeta.loader={planned_meta.loader}", flush=True)
    print(f"[constructor] TemplateMeta={planned_template!r}", flush=True)

    loader_sources = {
        "get_config": safe_source(ModelLoader.get_config),
        "get_processor": safe_source(ModelLoader.get_processor),
        "get_model": safe_source(ModelLoader.get_model),
        "compat_transformers5": safe_source(ModelLoader._compat_transformers5),
    }
    loader_requirements = {
        "get_config_uses_AutoConfig": "AutoConfig" in loader_sources["get_config"],
        "get_processor_uses_AutoTokenizer_or_AutoProcessor": (
            "AutoTokenizer" in loader_sources["get_processor"] or "AutoProcessor" in loader_sources["get_processor"]
        ),
        "get_model_uses_AutoModelForCausalLM": "AutoModelForCausalLM" in loader_sources["get_model"],
        "has_transformers5_compat": "transformers5" in loader_sources["compat_transformers5"].lower(),
    }
    if not all(loader_requirements.values()):
        raise RuntimeError(f"Default ModelLoader capability check failed: {loader_requirements}")
    print(f"[model-loader] {json.dumps(loader_requirements, ensure_ascii=False)}", flush=True)

    template_sources = {
        "encode": safe_source(Template.encode),
        "_encode": safe_source(Template._encode),
        "data_collator": safe_source(Template.data_collator),
        "_data_collator": safe_source(Template._data_collator),
    }
    collator_source = template_sources["_data_collator"]
    template_requirements = {
        "custom_encode_override_available": callable(getattr(Template, "_encode", None)),
        "token_type_ids_in_default_collator": "token_type_ids" in collator_source,
        "labels_in_default_collator": "labels" in collator_source,
        "position_ids_in_default_collator": "position_ids" in collator_source,
    }
    if not all(template_requirements.values()):
        raise RuntimeError(f"Template/collator capability check failed: {template_requirements}")
    print(f"[template-collator] {json.dumps(template_requirements, ensure_ascii=False)}", flush=True)

    forward_signature = inspect.signature(HrmTextForCausalLM.forward)
    forward_parameters = set(forward_signature.parameters)
    required_hrm_inputs = {"input_ids", "inputs_embeds", "token_type_ids", "labels"}
    missing_hrm_inputs = sorted(required_hrm_inputs - forward_parameters)
    if missing_hrm_inputs:
        raise RuntimeError(f"HrmTextForCausalLM.forward lacks {missing_hrm_inputs}: {forward_signature}")
    print(f"[hrm-forward] {forward_signature}", flush=True)

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    layer_semantics = {
        "raw_num_hidden_layers_per_stack": raw_config.get("num_hidden_layers"),
        "runtime_num_layers_per_stack": getattr(config, "num_layers_per_stack", None),
        "runtime_effective_num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "physical_parameter_layers_h_plus_l": 2 * int(getattr(config, "num_layers_per_stack")),
        "recurrent_stack_invocations": int(config.H_cycles) * (int(config.L_cycles) + 1),
    }
    expected_effective_layers = (
        layer_semantics["runtime_num_layers_per_stack"] * layer_semantics["recurrent_stack_invocations"]
    )
    if layer_semantics != {
        "raw_num_hidden_layers_per_stack": 16,
        "runtime_num_layers_per_stack": 16,
        "runtime_effective_num_hidden_layers": 128,
        "physical_parameter_layers_h_plus_l": 32,
        "recurrent_stack_invocations": 8,
    }:
        raise RuntimeError(f"Unexpected HRM layer semantics: {layer_semantics}")
    if expected_effective_layers != layer_semantics["runtime_effective_num_hidden_layers"]:
        raise RuntimeError(f"Inconsistent effective layer count: {layer_semantics}")
    print(f"[hrm-layers] {json.dumps(layer_semantics, ensure_ascii=False)}", flush=True)

    registries = registry_report(registry_modules)
    registry_sizes_before = {name: item["size"] for name, item in registries_before.items()}
    registry_sizes_after = {name: item["size"] for name, item in registries.items()}
    if registry_sizes_before != registry_sizes_after:
        raise RuntimeError(
            "Swift registry sizes changed during constructor-only inspection: "
            f"before={registry_sizes_before}, after={registry_sizes_after}"
        )
    if not registries:
        raise RuntimeError("No Swift MODEL/TEMPLATE mapping objects were discoverable")
    conflicts = exact_identifier_conflicts(registries)
    for registry_name, registry in registries.items():
        if registry["matching_entries"]:
            print(
                f"[registry] {registry_name} size={registry['size']} "
                f"matches={json.dumps(registry['matching_entries'], ensure_ascii=False)}",
                flush=True,
            )
    if conflicts:
        raise RuntimeError(f"Planned Swift identifiers already exist: {conflicts}")
    print(f"[registry-conflicts] exact_planned_identifier_conflicts={conflicts}", flush=True)

    source_patterns = (
        "token_type_ids",
        "gather_keys",
        "_compat_transformers5",
        "AutoModelForCausalLM.from_pretrained",
        "register_model(",
        "register_template(",
    )
    source_hits = source_search(swift_root, source_patterns, args.source_hit_limit)
    if not any("token_type_ids" in hit["matched"] for hit in source_hits):
        raise RuntimeError("Installed Swift source search found no token_type_ids handling")
    print(f"[source-search] hits={len(source_hits)} limit={args.source_hit_limit}", flush=True)
    for hit in source_hits:
        print(
            f"[source-hit] {hit['path']}:{hit['line']} matched={hit['matched']} source={hit['source']}",
            flush=True,
        )

    report = {
        "status": "ok",
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "ms-swift": version("ms-swift"),
            "transformers": version("transformers"),
            "torch": version("torch"),
        },
        "swift": {
            "root": str(swift_root),
            "signatures": signatures,
            "dataclass_fields": dataclass_fields_report,
            "model_loader_capabilities": loader_requirements,
            "template_collator_capabilities": template_requirements,
        },
        "constructor_trials": {
            "model": repr(planned_model),
            "model_group": repr(planned_group),
            "model_meta": repr(planned_meta),
            "template_meta": repr(planned_template),
            "registry_sizes_before": registry_sizes_before,
            "registry_sizes_after": registry_sizes_after,
            "mutated_registries": registry_sizes_before != registry_sizes_after,
        },
        "hrm": {
            "model_path": str(model_path),
            "forward_signature": str(forward_signature),
            "forward_parameters": sorted(forward_parameters),
            "layer_semantics": layer_semantics,
        },
        "planned_registration": {
            "model_type": PLANNED_MODEL_TYPE,
            "templates": list(PLANNED_TEMPLATES),
            "exact_identifier_conflicts": conflicts,
        },
        "registries": registries,
        "source_hits": source_hits,
    }
    atomic_write_json(args.output_report, report)
    print(f"[result] status=OK output_report={args.output_report}", flush=True)


if __name__ == "__main__":
    main()
