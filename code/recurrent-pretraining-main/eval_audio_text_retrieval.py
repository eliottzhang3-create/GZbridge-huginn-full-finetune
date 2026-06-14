"""Evaluate audio-caption alignment with retrieval metrics in embedding space."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from audio_alignment_eval_common import (
    add_common_eval_args,
    collect_audio_embeddings_from_dataloader,
    compute_reference_caption_embeddings,
    create_eval_dataloader,
    GroupedClothoEvalDataset,
    load_eval_components,
    normalize_rows,
    sample_indices,
    seed_everything,
    summarize_references,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_eval_args(parser)
    parser.add_argument("--sample_count", default="all", help="Use 'all' or a positive integer.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--max_text_length", type=int, default=192)
    parser.add_argument("--adapter_path_before", default=None)
    parser.add_argument("--failure_sample_count", type=int, default=10)
    return parser.parse_args()


def parse_sample_count(raw_value: str, total_size: int) -> int:
    if raw_value == "all":
        return total_size
    value = int(raw_value)
    if value <= 0:
        raise ValueError("--sample_count must be 'all' or a positive integer.")
    return min(value, total_size)


def compute_retrieval_metrics(similarity_matrix: torch.Tensor) -> dict:
    ranks_a2t = torch.argsort(similarity_matrix, dim=1, descending=True)
    ranks_t2a = torch.argsort(similarity_matrix, dim=0, descending=True).t()
    target = torch.arange(similarity_matrix.shape[0]).unsqueeze(1)

    def _ranks(sorted_indices: torch.Tensor) -> torch.Tensor:
        return (sorted_indices == target).nonzero(as_tuple=False)[:, 1] + 1

    rank_a2t = _ranks(ranks_a2t)
    rank_t2a = _ranks(ranks_t2a)

    def _stats(rank_tensor: torch.Tensor) -> dict[str, float]:
        rank_float = rank_tensor.float()
        return {
            "recall@1": float((rank_tensor <= 1).float().mean().item()),
            "recall@5": float((rank_tensor <= 5).float().mean().item()),
            "recall@10": float((rank_tensor <= 10).float().mean().item()),
            "mrr": float((1.0 / rank_float).mean().item()),
        }

    diag = torch.diagonal(similarity_matrix)
    off_diag_mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
    neg = similarity_matrix[off_diag_mask]
    return {
        "audio_to_text": _stats(rank_a2t),
        "text_to_audio": _stats(rank_t2a),
        "positive_mean": float(diag.mean().item()),
        "positive_std": float(diag.std(unbiased=False).item()),
        "negative_mean": float(neg.mean().item()),
        "negative_std": float(neg.std(unbiased=False).item()),
        "gap": float((diag.mean() - neg.mean()).item()),
        "ranks_audio_to_text": rank_a2t.tolist(),
        "ranks_text_to_audio": rank_t2a.tolist(),
    }


def collect_failure_examples(
    similarity_matrix: torch.Tensor,
    audio_paths: list[str],
    references: list[list[str]],
    rng_seed: int,
    sample_count: int,
) -> list[dict]:
    top_ranked = torch.argsort(similarity_matrix, dim=1, descending=True)
    failed_indices = [idx for idx in range(similarity_matrix.shape[0]) if int(top_ranked[idx, 0]) != idx]
    rng = random.Random(rng_seed)
    chosen = failed_indices if len(failed_indices) <= sample_count else rng.sample(failed_indices, sample_count)

    examples = []
    for idx in chosen:
        retrieved_indices = top_ranked[idx, :5].tolist()
        examples.append(
            {
                "audio_path": audio_paths[idx],
                "ground_truth_references": references[idx],
                "top5_retrieved": [
                    {
                        "candidate_audio_path": audio_paths[cand_idx],
                        "candidate_references": references[cand_idx],
                        "similarity": float(similarity_matrix[idx, cand_idx].item()),
                    }
                    for cand_idx in retrieved_indices
                ],
            }
        )
    return examples


def evaluate_checkpoint(args: argparse.Namespace, checkpoint_dir: str, chosen_indices: list[int]) -> dict:
    components = load_eval_components(
        checkpoint_dir=checkpoint_dir,
        base_model_name=args.base_model_name,
        audio_model_dir=args.audio_model_dir,
        audio_encoder_name=args.audio_encoder_name,
        precision=args.precision,
        device=args.device,
    )
    model = components["model"]
    tokenizer = components["tokenizer"]
    processor = components["processor"]
    device = components["device"]

    _, dataloader, _ = create_eval_dataloader(
        dataset_dir=args.dataset_dir,
        eval_manifest=args.eval_manifest,
        processor=processor,
        target_sample_rate=args.target_sample_rate,
        max_audio_seconds=args.max_audio_seconds,
        batch_size=args.batch_size,
        selected_indices=chosen_indices,
    )
    audio_embeddings, audio_paths, references = collect_audio_embeddings_from_dataloader(
        model=model,
        dataloader=dataloader,
        device=device,
        precision=args.precision,
    )
    text_embeddings = compute_reference_caption_embeddings(
        model=model,
        tokenizer=tokenizer,
        reference_groups=references,
        device=device,
        max_length=args.max_text_length,
        batch_size=args.text_batch_size,
    ).float()

    audio_embeddings = normalize_rows(audio_embeddings)
    text_embeddings = normalize_rows(text_embeddings)
    similarity_matrix = torch.matmul(audio_embeddings, text_embeddings.t()).cpu()
    metrics = compute_retrieval_metrics(similarity_matrix)
    failures = collect_failure_examples(
        similarity_matrix=similarity_matrix,
        audio_paths=audio_paths,
        references=references,
        rng_seed=args.seed,
        sample_count=args.failure_sample_count,
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "audio_paths": audio_paths,
        "references": references,
        "similarity_matrix": similarity_matrix.numpy(),
        "metrics": metrics,
        "failures": failures,
        "base_load": {
            "missing": len(components["base_load"].missing_keys),
            "unexpected": len(components["base_load"].unexpected_keys),
        },
        "delta_load": {
            "missing": len(components["delta_load"].missing_keys),
            "unexpected": len(components["delta_load"].unexpected_keys),
        },
    }


def build_comparison(before_metrics: dict, after_metrics: dict) -> dict:
    keys = ["recall@1", "recall@5", "recall@10", "mrr"]
    comparison = {"audio_to_text": {}, "text_to_audio": {}}
    for direction in ["audio_to_text", "text_to_audio"]:
        for key in keys:
            comparison[direction][key] = {
                "before": before_metrics[direction][key],
                "after": after_metrics[direction][key],
                "delta": after_metrics[direction][key] - before_metrics[direction][key],
            }
    for key in ["positive_mean", "positive_std", "negative_mean", "negative_std", "gap"]:
        comparison[key] = {
            "before": before_metrics[key],
            "after": after_metrics[key],
            "delta": after_metrics[key] - before_metrics[key],
        }
    return comparison


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    probe_dataset = GroupedClothoEvalDataset(
        dataset_dir=args.dataset_dir,
        manifest_name=args.eval_manifest,
        target_sample_rate=args.target_sample_rate,
        max_audio_seconds=args.max_audio_seconds,
    )
    desired_count = parse_sample_count(str(args.sample_count), len(probe_dataset))
    chosen_indices = sample_indices(len(probe_dataset), desired_count, args.seed)

    after_result = evaluate_checkpoint(args, args.checkpoint_dir, chosen_indices)
    before_result = None
    if args.adapter_path_before:
        before_result = evaluate_checkpoint(args, args.adapter_path_before, chosen_indices)

    metrics_payload = {
        "after": after_result["metrics"],
        "after_checkpoint_dir": after_result["checkpoint_dir"],
        "sample_size": len(chosen_indices),
        "sample_indices": chosen_indices,
    }
    if before_result is not None:
        metrics_payload["before"] = before_result["metrics"]
        metrics_payload["before_checkpoint_dir"] = before_result["checkpoint_dir"]
        metrics_payload["comparison"] = build_comparison(before_result["metrics"], after_result["metrics"])

    metrics_path = output_dir / "retrieval_metrics.json"
    matrix_path = output_dir / "similarity_matrix.npy"
    examples_path = output_dir / "retrieval_examples.txt"
    np.save(matrix_path, after_result["similarity_matrix"])
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    for example_idx, example in enumerate(after_result["failures"], start=1):
        lines.append(f"Failure {example_idx}")
        lines.append(f"Audio: {example['audio_path']}")
        lines.append(f"Ground Truth: {summarize_references(example['ground_truth_references'])}")
        lines.append("Top-5 Retrieved:")
        for item in example["top5_retrieved"]:
            lines.append(
                f"  sim={item['similarity']:.4f}\t{item['candidate_audio_path']}\t"
                f"{summarize_references(item['candidate_references'])}"
            )
        lines.append("")
    examples_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[audio-text-retrieval] saved metrics to {metrics_path}")
    print(f"[audio-text-retrieval] saved similarity matrix to {matrix_path}")
    print(f"[audio-text-retrieval] saved examples to {examples_path}")
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
