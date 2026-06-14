"""Visualize audio and caption embeddings with UMAP."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from audio_alignment_eval_common import (
    add_common_eval_args,
    collect_audio_embeddings_from_dataloader,
    compute_reference_caption_embeddings,
    create_eval_dataloader,
    load_eval_components,
    normalize_rows,
    sample_indices,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_eval_args(parser)
    parser.add_argument("--sample_count", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--max_text_length", type=int, default=192)
    parser.add_argument("--method", default="umap", choices=["umap"])
    parser.add_argument("--line_pair_count", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for eval_visualization.py in swift_huginn.") from exc
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("umap-learn is required for eval_visualization.py in swift_huginn.") from exc

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
    text_embeddings = compute_reference_caption_embeddings(
        model=model,
        tokenizer=tokenizer,
        reference_groups=references,
        device=device,
        max_length=args.max_text_length,
        batch_size=args.text_batch_size,
    ).float()

    audio_embeddings = normalize_rows(audio_embeddings).cpu().numpy()
    text_embeddings = normalize_rows(text_embeddings).cpu().numpy()
    stacked = np.concatenate([audio_embeddings, text_embeddings], axis=0)

    reducer = umap.UMAP(n_components=2, random_state=args.seed)
    coords = reducer.fit_transform(stacked)
    n = len(audio_paths)
    audio_coords = coords[:n]
    text_coords = coords[n:]

    matched_dist = np.linalg.norm(audio_coords - text_coords, axis=1)
    rng = random.Random(args.seed)
    mismatch_indices = list(range(n))
    rng.shuffle(mismatch_indices)
    mismatch_indices = [(i, mismatch_indices[i]) for i in range(n) if mismatch_indices[i] != i]
    mismatch_dist = np.array(
        [np.linalg.norm(audio_coords[i] - text_coords[j]) for i, j in mismatch_indices],
        dtype=np.float32,
    )

    coordinates_path = output_dir / "coordinates.csv"
    with coordinates_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_index", "point_type", "audio_path", "x", "y"])
        for idx, audio_path in enumerate(audio_paths):
            writer.writerow([idx, "audio", audio_path, float(audio_coords[idx, 0]), float(audio_coords[idx, 1])])
            writer.writerow([idx, "caption", audio_path, float(text_coords[idx, 0]), float(text_coords[idx, 1])])

    stats = {
        "checkpoint_dir": args.checkpoint_dir,
        "sample_size": n,
        "matched_distance_mean": float(matched_dist.mean()),
        "matched_distance_std": float(matched_dist.std()),
        "mismatch_distance_mean": float(mismatch_dist.mean()) if mismatch_dist.size else None,
        "mismatch_distance_std": float(mismatch_dist.std()) if mismatch_dist.size else None,
        "distance_gap": float(mismatch_dist.mean() - matched_dist.mean()) if mismatch_dist.size else None,
    }
    stats_path = output_dir / "distance_statistics.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 8))
    plt.scatter(audio_coords[:, 0], audio_coords[:, 1], c="royalblue", marker="o", alpha=0.65, label="Audio")
    plt.scatter(text_coords[:, 0], text_coords[:, 1], c="crimson", marker="^", alpha=0.65, label="Caption")
    line_count = min(args.line_pair_count, n)
    chosen_line_pairs = rng.sample(range(n), line_count) if n > line_count else list(range(n))
    for idx in chosen_line_pairs:
        plt.plot(
            [audio_coords[idx, 0], text_coords[idx, 0]],
            [audio_coords[idx, 1], text_coords[idx, 1]],
            color="gray",
            alpha=0.25,
            linewidth=0.8,
        )
    plt.legend()
    plt.title("UMAP of Audio and Caption Embeddings")
    plt.tight_layout()
    figure_path = output_dir / "umap.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"[visualization] saved figure to {figure_path}")
    print(f"[visualization] saved coordinates to {coordinates_path}")
    print(f"[visualization] saved stats to {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
