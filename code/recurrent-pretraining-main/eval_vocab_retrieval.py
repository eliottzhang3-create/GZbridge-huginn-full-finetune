"""Evaluate whether pooled audio embeddings retrieve meaningful vocabulary tokens."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from audio_alignment_eval_common import (
    add_common_eval_args,
    collect_audio_embeddings_from_dataloader,
    create_eval_dataloader,
    load_eval_components,
    normalize_rows,
    sample_indices,
    seed_everything,
    summarize_references,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_eval_args(parser)
    parser.add_argument("--sample_count", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    components = load_eval_components(
        checkpoint_dir=args.checkpoint_dir,
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

    full_dataset, _, _ = create_eval_dataloader(
        dataset_dir=args.dataset_dir,
        eval_manifest=args.eval_manifest,
        processor=processor,
        target_sample_rate=args.target_sample_rate,
        max_audio_seconds=args.max_audio_seconds,
        batch_size=args.batch_size,
    )
    chosen_indices = sample_indices(len(full_dataset), args.sample_count, args.seed)
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
    audio_embeddings = normalize_rows(audio_embeddings)

    token_embeddings = model.get_input_embeddings().weight.detach().float().cpu()
    token_embeddings = normalize_rows(token_embeddings)
    special_ids = set(tokenizer.all_special_ids)

    token_frequency = Counter()
    lines = []
    for sample_idx, (audio_embedding, audio_path, refs) in enumerate(zip(audio_embeddings, audio_paths, references), start=1):
        similarities = torch.matmul(token_embeddings, audio_embedding.cpu())
        ranked_ids = torch.argsort(similarities, descending=True).tolist()

        top_tokens = []
        for token_id in ranked_ids:
            if token_id in special_ids:
                continue
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token is None or token in tokenizer.all_special_tokens:
                continue
            token = token.strip()
            if not token:
                continue
            score = float(similarities[token_id].item())
            top_tokens.append((token, score))
            token_frequency[token] += 1
            if len(top_tokens) >= args.top_k:
                break

        lines.append(f"Sample {sample_idx}")
        lines.append(f"Audio: {audio_path}")
        lines.append(f"Ground Truth Caption: {summarize_references(refs, limit=1)}")
        lines.append("Top-{0} Nearest Tokens:".format(args.top_k))
        for token, score in top_tokens:
            lines.append(f"  {token}\t{score:.4f}")
        lines.append("")

    text_path = output_dir / "vocab_retrieval.txt"
    freq_path = output_dir / "token_frequency.json"
    text_blob = "\n".join(lines)
    text_path.write_text(text_blob, encoding="utf-8")
    with freq_path.open("w", encoding="utf-8") as f:
        json.dump(token_frequency.most_common(), f, ensure_ascii=False, indent=2)

    print(text_blob)
    print(f"[vocab-retrieval] saved text results to {text_path}")
    print(f"[vocab-retrieval] saved token frequency to {freq_path}")


if __name__ == "__main__":
    main()
