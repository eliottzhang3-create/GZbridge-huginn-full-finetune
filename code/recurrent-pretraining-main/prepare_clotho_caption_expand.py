"""Expand concatenated Clotho captions into per-caption training samples."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CAPITAL_WORD_RE = re.compile(r"[A-Z][a-z][^\s]*")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SplitResult:
    method: str
    segments: list[str]
    raw_count: int
    merged: bool


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_sentence_edges(text: str) -> str:
    return normalize_text(text).strip(" \t\r\n")


def split_by_sentence_end(text: str) -> list[str]:
    parts = [strip_sentence_edges(part) for part in SENTENCE_END_RE.split(text)]
    return [part for part in parts if part]


def is_capital_boundary(text: str, boundary_idx: int, min_prefix_chars: int) -> bool:
    prefix = text[:boundary_idx].rstrip()
    suffix = text[boundary_idx:].lstrip()
    if len(prefix) < min_prefix_chars or not suffix:
        return False

    match = CAPITAL_WORD_RE.match(suffix)
    if not match:
        return False

    next_word = match.group(0)
    prev_word = prefix.split()[-1] if prefix.split() else ""
    prev_char = prefix[-1]

    if prev_char in ".!?":
        return True

    if len(next_word) <= 2:
        return False

    if prev_word.isupper() and len(prev_word) <= 4:
        return False

    if prev_word.endswith(",") or prev_word.endswith(":") or prev_word.endswith(";"):
        return False

    lower_prefix = prefix.lower()
    if lower_prefix.endswith("mr") or lower_prefix.endswith("mrs") or lower_prefix.endswith("dr"):
        return False

    return True


def split_by_capital_starts(text: str, min_prefix_chars: int = 24) -> list[str]:
    boundaries: list[int] = []
    for match in re.finditer(r"\s+(?=[A-Z])", text):
        boundary_idx = match.end()
        if is_capital_boundary(text, boundary_idx, min_prefix_chars=min_prefix_chars):
            boundaries.append(boundary_idx)

    if not boundaries:
        return [strip_sentence_edges(text)] if text else []

    segments: list[str] = []
    start = 0
    for boundary in boundaries:
        chunk = strip_sentence_edges(text[start:boundary])
        if chunk:
            segments.append(chunk)
        start = boundary
    tail = strip_sentence_edges(text[start:])
    if tail:
        segments.append(tail)
    return segments


def split_hybrid(text: str, min_prefix_chars: int = 24) -> list[str]:
    coarse = split_by_sentence_end(text)
    refined: list[str] = []
    for part in coarse:
        capital_parts = split_by_capital_starts(part, min_prefix_chars=min_prefix_chars)
        if len(capital_parts) > 1:
            refined.extend(capital_parts)
        else:
            refined.append(part)
    return [part for part in refined if part]


def merge_shortest_neighbors(segments: list[str], target_count: int) -> tuple[list[str], bool]:
    merged = [strip_sentence_edges(segment) for segment in segments if strip_sentence_edges(segment)]
    changed = False
    while len(merged) > target_count and len(merged) >= 2:
        best_idx = min(
            range(len(merged) - 1),
            key=lambda idx: len(merged[idx]) + len(merged[idx + 1]),
        )
        merged[best_idx : best_idx + 2] = [f"{merged[best_idx]} {merged[best_idx + 1]}".strip()]
        changed = True
    return merged, changed


def build_candidates(text: str, target_count: int) -> list[SplitResult]:
    normalized = normalize_text(text)
    raw_candidates = {
        "whole": [normalized] if normalized else [],
        "capital": split_by_capital_starts(normalized),
        "period": split_by_sentence_end(normalized),
        "hybrid": split_hybrid(normalized),
    }

    candidates: list[SplitResult] = []
    seen: set[tuple[str, ...]] = set()
    for method, raw_segments in raw_candidates.items():
        raw_segments = [segment for segment in raw_segments if segment]
        merged_segments, changed = merge_shortest_neighbors(raw_segments, target_count=target_count)
        dedupe_key = tuple(merged_segments)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(
            SplitResult(
                method=method,
                segments=merged_segments,
                raw_count=len(raw_segments),
                merged=changed,
            )
        )
    return candidates


def score_candidate(result: SplitResult, target_count: int) -> tuple[int, int, float, float, float]:
    count = len(result.segments)
    lengths = [len(segment) for segment in result.segments] or [10**9]
    exact_penalty = 0 if count == target_count else 1
    over_penalty = max(0, count - target_count)
    under_bonus = -count if count <= target_count else count
    mean_length = sum(lengths) / len(lengths)
    max_length = max(lengths)
    stdev = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    return (exact_penalty, over_penalty, under_bonus, mean_length + 0.01 * max_length, stdev)


def choose_best_split(text: str, target_count: int) -> SplitResult:
    candidates = build_candidates(text, target_count=target_count)
    if not candidates:
        return SplitResult(method="empty", segments=[], raw_count=0, merged=False)
    return min(candidates, key=lambda item: score_candidate(item, target_count=target_count))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_expanded_records(records: Iterable[dict], target_count: int) -> tuple[list[dict], dict]:
    expanded: list[dict] = []
    stats = {
        "source_records": 0,
        "expanded_records": 0,
        "exact_target_records": 0,
        "under_target_records": 0,
        "over_target_records": 0,
        "method_counts": {},
        "raw_count_histogram": {},
        "final_count_histogram": {},
    }

    for record_idx, record in enumerate(records):
        stats["source_records"] += 1
        text = record.get("caption") or record.get("text") or ""
        best = choose_best_split(text, target_count=target_count)
        final_count = len(best.segments)

        if final_count == target_count:
            stats["exact_target_records"] += 1
        elif final_count < target_count:
            stats["under_target_records"] += 1
        else:
            stats["over_target_records"] += 1

        stats["method_counts"][best.method] = stats["method_counts"].get(best.method, 0) + 1
        stats["raw_count_histogram"][best.raw_count] = stats["raw_count_histogram"].get(best.raw_count, 0) + 1
        stats["final_count_histogram"][final_count] = stats["final_count_histogram"].get(final_count, 0) + 1

        base_payload = dict(record)
        for caption_idx, segment in enumerate(best.segments):
            item = dict(base_payload)
            item["caption"] = segment
            item["source_record_id"] = record.get("source_record_id", record_idx)
            item["caption_index"] = caption_idx
            item["split_method"] = best.method
            item["raw_split_count"] = best.raw_count
            item["final_split_count"] = final_count
            expanded.append(item)

    stats["expanded_records"] = len(expanded)
    return expanded, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL manifest path.")
    parser.add_argument("--output_json", required=True, help="Output JSON array path.")
    parser.add_argument("--target_count", type=int, default=5, help="Desired number of captions per source sample.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_json)

    records = read_jsonl(input_path)
    expanded, stats = iter_expanded_records(records, target_count=args.target_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(expanded, f, ensure_ascii=False, indent=2)

    print(f"[clotho-expand] input_jsonl={input_path}")
    print(f"[clotho-expand] output_json={output_path}")
    print(f"[clotho-expand] source_records={stats['source_records']} expanded_records={stats['expanded_records']}")
    print(
        f"[clotho-expand] exact_target_records={stats['exact_target_records']} "
        f"under_target_records={stats['under_target_records']} "
        f"over_target_records={stats['over_target_records']}"
    )
    print(f"[clotho-expand] method_counts={json.dumps(stats['method_counts'], ensure_ascii=False, sort_keys=True)}")
    print(
        f"[clotho-expand] raw_count_histogram="
        f"{json.dumps(stats['raw_count_histogram'], ensure_ascii=False, sort_keys=True)}"
    )
    print(
        f"[clotho-expand] final_count_histogram="
        f"{json.dumps(stats['final_count_histogram'], ensure_ascii=False, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
