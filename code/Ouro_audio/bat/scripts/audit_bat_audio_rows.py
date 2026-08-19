#!/usr/bin/env python3
"""Audit the production BAT renderer over an ordered JSONL prefix.

This is deliberately model-free and NCCL-free.  It uses the exact
``BATAudioRenderer`` implementation used by the training template and writes
an atomic progress file before and after every source/record.  A native crash
therefore leaves the last source path and renderer phase on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from bat.models.spatial_ast_audio import BATAudioRenderer
from bat_diagnostics import filesystem_stats, process_stats, read_jsonl, require_private_absolute, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reverb-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=50)
    return parser.parse_args()


def audio_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the flat audio descriptor from either supported manifest shape."""
    audios = record.get("audios")
    if audios is not None:
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], dict):
            raise ValueError(
                "Expected exactly one dictionary in prepared-manifest audios: "
                f"type={type(audios).__name__} value={audios!r}"
            )
        return audios[0]
    return record


def source_pairs(record: dict[str, Any]) -> list[tuple[str, str]]:
    record = audio_record(record)
    pairs: list[tuple[str, str]] = []
    for suffix in ("", "2"):
        audio_id = record.get(f"audio_id{suffix}")
        reverb_id = record.get(f"reverb_id{suffix}")
        if audio_id in (None, "", "null") and reverb_id in (None, "", "null"):
            continue
        if audio_id in (None, "", "null") or reverb_id in (None, "", "null"):
            raise ValueError(f"Partial source pair: {record}")
        pairs.append((str(audio_id), str(reverb_id)))
    if not pairs:
        raise ValueError(f"Record has no source pair: {record}")
    return pairs


def record_summary(index: int, line_number: int, record: dict[str, Any]) -> dict[str, Any]:
    audio = audio_record(record)
    return {
        "record_index": index,
        "line_number": line_number,
        "question_id": record.get("question_id"),
        "question_type": record.get("question_type"),
        "bat_type": record.get("bat_type"),
        "source_shape": "dual" if audio.get("audio_id2") not in (None, "", "null") else "single",
    }


def main() -> None:
    args = parse_args()
    output = require_private_absolute(args.output_report)
    progress = require_private_absolute(args.progress_file)
    if args.limit <= 0 or args.start_index < 0:
        raise ValueError("limit must be positive and start-index must be non-negative")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.audio_root.is_dir() or not args.reverb_root.is_dir():
        raise FileNotFoundError(f"Missing input root: audio={args.audio_root} reverb={args.reverb_root}")

    renderer = BATAudioRenderer(args.audio_root.resolve(), args.reverb_root.resolve())
    started = time.time()
    report: dict[str, Any] = {
        "status": "running",
        "manifest": str(args.manifest.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "reverb_root": str(args.reverb_root.resolve()),
        "start_index": args.start_index,
        "limit": args.limit,
        "renderer": "bat.models.spatial_ast_audio.BATAudioRenderer",
        "production_path": "_render_one source path + render_record-equivalent single/dual arithmetic",
        "records_seen": 0,
        "sources_rendered": 0,
        "failures": [],
        "last_completed": None,
        "process_start": process_stats(),
    }
    write_json(output, report)
    write_json(progress, {
        "status": "started",
        "phase": "before_first_record",
        "manifest": str(args.manifest.resolve()),
        "start_index": args.start_index,
        "limit": args.limit,
        "process": process_stats(),
    })
    print("========== BAT AUDIO RENDERER PREFIX AUDIT ==========" , flush=True)
    print(f"[manifest] {args.manifest.resolve()} start={args.start_index} limit={args.limit}", flush=True)
    print(f"[audio] {args.audio_root.resolve()}", flush=True)
    print(f"[reverb] {args.reverb_root.resolve()}", flush=True)

    try:
        for index, (line_number, record) in enumerate(read_jsonl(args.manifest), start=0):
            if index < args.start_index:
                continue
            if report["records_seen"] >= args.limit:
                break
            summary = record_summary(index, line_number, record)
            pairs = source_pairs(record)
            rendered_sources: list[torch.Tensor] = []
            write_json(progress, {"status": "running", "phase": "record_start", **summary, "source_count": len(pairs), "process": process_stats()})
            print(f"[record] index={index} line={line_number} qid={record.get('question_id')} type={record.get('question_type')} sources={len(pairs)}", flush=True)
            for source_slot, (audio_id, reverb_id) in enumerate(pairs):
                audio_path = renderer._resolve_audio(renderer.audio_root, audio_id)
                reverb_path = renderer._resolve_reverb(renderer.reverb_root, reverb_id)
                write_json(progress, {
                    "status": "running",
                    "phase": "source_start",
                    **summary,
                    "source_slot": source_slot,
                    "audio_id": audio_id,
                    "reverb_id": reverb_id,
                    "audio_path": str(audio_path),
                    "reverb_path": str(reverb_path),
                    "process": process_stats(),
                })
                print(f"[source-start] record={index} slot={source_slot} audio={audio_path} reverb={reverb_path}", flush=True)
                waveform = renderer._render_one(audio_id, reverb_id)
                tensor = waveform if torch.is_tensor(waveform) else torch.as_tensor(waveform)
                if tuple(tensor.shape) != (2, 320000):
                    raise RuntimeError(f"Renderer shape mismatch at record={index} slot={source_slot}: {tuple(tensor.shape)}")
                if not bool(torch.isfinite(tensor.float()).all().item()):
                    raise RuntimeError(f"Renderer produced non-finite values at record={index} slot={source_slot}")
                rendered_sources.append(tensor.float())
                report["sources_rendered"] += 1
                write_json(progress, {
                    "status": "running",
                    "phase": "source_done",
                    **summary,
                    "source_slot": source_slot,
                    "audio_id": audio_id,
                    "reverb_id": reverb_id,
                    "audio_path": str(audio_path),
                    "reverb_path": str(reverb_path),
                    "waveform_shape": list(tensor.shape),
                    "waveform_rms": float(torch.sqrt(torch.mean(tensor.float() ** 2)).item()),
                    "process": process_stats(),
                })
                print(f"[source-ok] record={index} slot={source_slot} rms={float(torch.sqrt(torch.mean(tensor.float() ** 2)).item()):.6g}", flush=True)
            # This is the exact arithmetic implemented by render_record, but
            # reuses the already rendered source tensors so the audit does not
            # perform a second expensive AudioSet/RIR convolution.
            mixed = rendered_sources[0]
            if len(rendered_sources) == 2:
                mixed = (mixed + rendered_sources[1]) / 2.0
            if tuple(mixed.shape) != (2, 320000) or not bool(torch.isfinite(mixed.float()).all().item()):
                raise RuntimeError(f"final BAT mix validation failed at record={index}")
            report["records_seen"] += 1
            report["last_completed"] = {**summary, "waveform_shape": list(mixed.shape)}
            write_json(progress, {"status": "running", "phase": "record_done", **summary, "waveform_shape": list(mixed.shape), "process": process_stats()})
            if report["records_seen"] == 1 or report["records_seen"] % args.report_every == 0:
                print(f"[progress] records={report['records_seen']} sources={report['sources_rendered']} rss={process_stats().get('rss_bytes')}", flush=True)
    except BaseException as exc:
        report["status"] = "incomplete"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["elapsed_seconds"] = time.time() - started
        report["process_end"] = process_stats()
        write_json(output, report)
        write_json(progress, {"status": "failed_python", "error": report["failure"], "last_report": report.get("last_completed"), "process": process_stats()})
        raise

    report["status"] = "ok"
    report["elapsed_seconds"] = time.time() - started
    report["process_end"] = process_stats()
    report["filesystems"] = filesystem_stats((args.manifest, args.audio_root, args.reverb_root, output))
    write_json(output, report)
    write_json(progress, {"status": "ok", "records_seen": report["records_seen"], "sources_rendered": report["sources_rendered"], "process": process_stats()})
    print(f"[summary] records={report['records_seen']} sources={report['sources_rendered']} seconds={report['elapsed_seconds']:.2f}", flush=True)
    print(f"[report] {output}", flush=True)
    print("[status] ok", flush=True)


if __name__ == "__main__":
    main()
