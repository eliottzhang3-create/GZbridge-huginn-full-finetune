#!/usr/bin/env python3
"""Run the BAT renderer concurrently in eight independent processes.

``torchrun`` supplies rank environment variables, but this script deliberately
does not initialize a process group and never touches CUDA/NCCL.  It therefore
isolates concurrent AudioSet/RIR/Scipy/native-file behaviour from distributed
communication.  Each rank writes its own atomic progress and report; the
launcher combines them after all ranks exit.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch

from bat.models.spatial_ast_audio import BATAudioRenderer
from audit_bat_audio_rows import source_pairs, record_summary
from bat_diagnostics import process_stats, read_jsonl, require_private_absolute, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reverb-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--progress-prefix", type=Path, required=True)
    parser.add_argument("--global-record-limit", type=int, default=8500)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--world-size-for-combine", type=int, default=8)
    return parser.parse_args()


def rank_world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def run_rank(args: argparse.Namespace, rank: int, world_size: int) -> None:
    output = require_private_absolute(Path(f"{args.output_prefix}.rank{rank}.json"))
    progress = require_private_absolute(Path(f"{args.progress_prefix}.rank{rank}.json"))
    renderer = BATAudioRenderer(args.audio_root.resolve(), args.reverb_root.resolve())
    started = time.time()
    report: dict[str, Any] = {
        "status": "running",
        "rank": rank,
        "world_size": world_size,
        "manifest": str(args.manifest.resolve()),
        "global_record_limit": args.global_record_limit,
        "local_batch_size": args.local_batch_size,
        "records_rendered": 0,
        "sources_rendered": 0,
        "failures": [],
        "process_start": process_stats(),
    }
    write_json(output, report)
    write_json(progress, {"status": "started", "rank": rank, "world_size": world_size, "phase": "before_first_record", "process": process_stats()}, atomic=False)
    try:
        for global_index, (line_number, record) in enumerate(read_jsonl(args.manifest)):
            if global_index >= args.global_record_limit:
                break
            position = global_index % (world_size * args.local_batch_size)
            if not rank * args.local_batch_size <= position < (rank + 1) * args.local_batch_size:
                continue
            summary = record_summary(global_index, line_number, record)
            pairs = source_pairs(record)
            rendered_sources: list[torch.Tensor] = []
            write_json(progress, {"status": "running", "phase": "record_start", **summary, "source_count": len(pairs), "process": process_stats()}, atomic=False)
            for source_slot, (audio_id, reverb_id) in enumerate(pairs):
                audio_path = renderer._resolve_audio(renderer.audio_root, audio_id)
                reverb_path = renderer._resolve_reverb(renderer.reverb_root, reverb_id)
                write_json(progress, {"status": "running", "phase": "source_start", **summary, "source_slot": source_slot, "audio_id": audio_id, "reverb_id": reverb_id, "audio_path": str(audio_path), "reverb_path": str(reverb_path), "process": process_stats()}, atomic=False)
                waveform = renderer._render_one(audio_id, reverb_id)
                tensor = waveform if torch.is_tensor(waveform) else torch.as_tensor(waveform)
                if tuple(tensor.shape) != (2, 320000) or not bool(torch.isfinite(tensor.float()).all().item()):
                    raise RuntimeError(f"Invalid source render rank={rank} global_index={global_index} slot={source_slot}")
                rendered_sources.append(tensor.float())
                report["sources_rendered"] += 1
                write_json(progress, {"status": "running", "phase": "source_done", **summary, "source_slot": source_slot, "waveform_shape": list(tensor.shape), "process": process_stats()}, atomic=False)
            mixed = rendered_sources[0]
            if len(rendered_sources) == 2:
                mixed = (mixed + rendered_sources[1]) / 2.0
            if tuple(mixed.shape) != (2, 320000) or not bool(torch.isfinite(mixed.float()).all().item()):
                raise RuntimeError(f"Invalid final render rank={rank} global_index={global_index}")
            report["records_rendered"] += 1
            write_json(progress, {"status": "running", "phase": "record_done", **summary, "waveform_shape": list(mixed.shape), "process": process_stats()}, atomic=False)
            if report["records_rendered"] == 1 or report["records_rendered"] % 50 == 0:
                print(f"[rank{rank}] records={report['records_rendered']} sources={report['sources_rendered']}", flush=True)
    except BaseException as exc:
        report["status"] = "incomplete"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["process_end"] = process_stats()
        report["elapsed_seconds"] = time.time() - started
        write_json(output, report)
        write_json(progress, {"status": "failed_python", "rank": rank, "error": report["failure"], "process": process_stats()}, atomic=False)
        raise
    report["status"] = "ok"
    report["process_end"] = process_stats()
    report["elapsed_seconds"] = time.time() - started
    write_json(output, report)
    write_json(progress, {"status": "ok", "rank": rank, "records_rendered": report["records_rendered"], "sources_rendered": report["sources_rendered"], "process": process_stats()}, atomic=False)


def combine(args: argparse.Namespace) -> None:
    output = require_private_absolute(args.output_prefix)
    reports: list[dict[str, Any]] = []
    missing: list[str] = []
    for rank in range(args.world_size_for_combine):
        path = Path(f"{args.output_prefix}.rank{rank}.json")
        if not path.is_file():
            missing.append(str(path))
            continue
        import json
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    failures = [report for report in reports if report.get("status") != "ok"]
    combined = {
        "status": "ok" if not missing and not failures and len(reports) == args.world_size_for_combine else "incomplete",
        "manifest": str(args.manifest.resolve()),
        "world_size": args.world_size_for_combine,
        "global_record_limit": args.global_record_limit,
        "rank_count": len(reports),
        "missing_rank_reports": missing,
        "total_records_rendered": sum(int(report.get("records_rendered", 0)) for report in reports),
        "total_sources_rendered": sum(int(report.get("sources_rendered", 0)) for report in reports),
        "rank_reports": reports,
    }
    write_json(output, combined)
    print(f"[combined] ranks={len(reports)} records={combined['total_records_rendered']} status={combined['status']}", flush=True)
    print(f"[report] {output}", flush=True)
    if combined["status"] != "ok":
        raise RuntimeError(f"Renderer process audit incomplete: missing={missing} failures={len(failures)}")


def main() -> None:
    args = parse_args()
    rank, world_size = rank_world()
    if args.combine:
        combine(args)
    else:
        if world_size < 2:
            raise RuntimeError("This audit must be launched with torchrun using at least two processes")
        run_rank(args, rank, world_size)


if __name__ == "__main__":
    main()
