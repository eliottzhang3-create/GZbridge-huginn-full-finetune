"""Audit the official BAT/SLAM-LLM Q-Former contract on CPU.

This script does not load Spatial-AST, Ouro, or any checkpoint.  It builds the
same Q-Former projector used by the public BAT example and verifies the exact
shape/parameter contract needed by Ouro-1.4B.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


def parameter_report(module: torch.nn.Module) -> dict[str, Any]:
    all_parameters = list(module.named_parameters())
    trainable = [(name, parameter) for name, parameter in all_parameters if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in all_parameters if not parameter.requires_grad]
    return {
        "all_parameter_count": sum(parameter.numel() for _, parameter in all_parameters),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameter_count": sum(parameter.numel() for _, parameter in frozen),
        "trainable_name_count": len(trainable),
        "frozen_name_count": len(frozen),
        "trainable_name_preview": [name for name, _ in trainable[:20]],
        "dtypes": sorted({str(parameter.dtype) for _, parameter in all_parameters}),
    }


class QFormerConfig:
    def __init__(self, encoder_dim: int, llm_dim: int, layers: int, query_len: int) -> None:
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.qformer_layers = layers
        self.query_len = query_len

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def build_projector(source_path: Path, encoder_dim: int, llm_dim: int, layers: int, query_len: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not source_path.is_file():
        raise FileNotFoundError(f"Q-Former source does not exist: {source_path}")
    source_text = source_path.read_text(encoding="utf-8", errors="replace")
    spec = importlib.util.spec_from_file_location("bat_audited_projector", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    projector_class = getattr(module, "EncoderProjectorQFormer", None)
    if projector_class is None:
        raise AttributeError(f"EncoderProjectorQFormer not found in {source_path}")
    projector = projector_class(QFormerConfig(encoder_dim, llm_dim, layers, query_len))
    source_contract = {
        "path": str(source_path),
        "sha256": __import__("hashlib").sha256(source_text.encode("utf-8")).hexdigest(),
        "contains_blip2_qformer": "Blip2QFormerModel" in source_text,
        "contains_encoder_hidden_size": "encoder_hidden_size" in source_text,
        "contains_cross_attention_call": "encoder_hidden_states=x" in source_text,
        "contains_query_parameter": "self.query = nn.Parameter" in source_text,
        "contains_output_projection": "self.linear = nn.Linear" in source_text,
    }
    return projector, source_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-dim", type=int, default=768)
    parser.add_argument("--llm-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--query-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--encoder-seq-len", type=int, default=515)
    parser.add_argument("--qformer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.output).startswith("/hpc_stor03/public"):
        raise SystemExit(f"Refusing public output path: {args.output}")

    print("========== BAT Q-FORMER CONTRACT AUDIT ==========")
    print(f"[python] version={sys.version.split()[0]} executable={sys.executable}")
    print(
        f"[config] encoder_dim={args.encoder_dim} llm_dim={args.llm_dim} "
        f"layers={args.layers} query_len={args.query_len}"
    )

    projector, source_contract = build_projector(
        args.qformer_source, args.encoder_dim, args.llm_dim, args.layers, args.query_len
    )
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad = True

    hidden_size = int(projector.qformer.config.hidden_size)
    encoder_hidden_size = int(projector.qformer.config.encoder_hidden_size)
    num_hidden_layers = int(projector.qformer.config.num_hidden_layers)
    num_attention_heads = int(projector.qformer.config.num_attention_heads)
    intermediate_size = int(projector.qformer.config.intermediate_size)

    inputs = torch.zeros(args.batch_size, args.encoder_seq_len, args.encoder_dim)
    attention_mask = torch.ones(args.batch_size, args.encoder_seq_len, dtype=torch.long)
    with torch.no_grad():
        outputs = projector(inputs, attention_mask)

    issues: list[str] = []
    if hidden_size != 768:
        issues.append(f"unexpected_qformer_hidden_size:{hidden_size}")
    if encoder_hidden_size != args.encoder_dim:
        issues.append(f"encoder_hidden_size_mismatch:{encoder_hidden_size}")
    if num_hidden_layers != args.layers:
        issues.append(f"qformer_layer_mismatch:{num_hidden_layers}")
    if tuple(outputs.shape) != (args.batch_size, args.query_len, args.llm_dim):
        issues.append(f"output_shape_mismatch:{tuple(outputs.shape)}")
    if tuple(projector.query.shape) != (1, args.query_len, hidden_size):
        issues.append(f"query_shape_mismatch:{tuple(projector.query.shape)}")

    report = {
        "status": "incomplete" if issues else "ok",
        "scope": {
            "spatial_ast_loaded": False,
            "ouro_loaded": False,
            "checkpoint_loaded": False,
            "encoder_training": False,
        },
        "config": {
            "encoder_dim": args.encoder_dim,
            "llm_dim": args.llm_dim,
            "requested_qformer_layers": args.layers,
            "requested_query_len": args.query_len,
            "batch_size": args.batch_size,
            "encoder_seq_len": args.encoder_seq_len,
            "qformer_source": str(args.qformer_source),
        },
        "qformer_source_contract": source_contract,
        "qformer_config": {
            "hidden_size": hidden_size,
            "encoder_hidden_size": encoder_hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "cross_attention_frequency": int(getattr(projector.qformer.config, "cross_attention_frequency", -1)),
        },
        "module_shapes": {
            "learnable_query": list(projector.query.shape),
            "output_projection_weight": list(projector.linear.weight.shape),
            "output_projection_bias": list(projector.linear.bias.shape),
            "output_layernorm_weight": list(projector.norm.weight.shape),
            "forward_output": list(outputs.shape),
        },
        "parameters": parameter_report(projector),
        "contract": {
            "input": "Spatial-AST token sequence [B,S,768] plus encoder attention mask [B,S]",
            "query": "64 learnable query vectors, expanded across batch",
            "fusion": "Q-Former query embeddings attend to Spatial-AST encoder hidden states",
            "output": "64 projected modality tokens [B,64,2048] for Ouro inputs_embeds",
            "trainable": "all Q-Former, query, Linear, and LayerNorm parameters in this module",
            "frozen_elsewhere": "Spatial-AST and Ouro backbone/gate are outside this module and remain frozen",
        },
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[qformer] hidden={hidden_size} encoder_hidden={encoder_hidden_size} layers={num_hidden_layers} heads={num_attention_heads}")
    print(f"[qformer] query_shape={tuple(projector.query.shape)} output_shape={tuple(outputs.shape)}")
    print(f"[parameters] {report['parameters']}")
    print(f"[report] {args.output}")
    print(f"[status] {report['status']} issues={issues}")


if __name__ == "__main__":
    main()
