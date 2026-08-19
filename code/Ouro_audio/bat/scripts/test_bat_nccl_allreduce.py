#!/usr/bin/env python3
"""Standalone NCCL all-reduce health/latency test for BAT's 8-GPU topology."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from bat_diagnostics import require_private_absolute, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--tensor-elements", type=int, default=1 << 20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = require_private_absolute(args.output_report)
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL test requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("Launch this test with torchrun and at least two ranks")
    if args.warmup < 0 or args.iterations <= 0 or args.tensor_elements <= 0:
        raise ValueError("warmup/iterations/tensor-elements must be positive where applicable")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    tensor = torch.ones(args.tensor_elements, device=f"cuda:{local_rank}", dtype=torch.float32)
    for _ in range(args.warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    dist.barrier()
    latencies: list[float] = []
    for _ in range(args.iterations):
        tensor.fill_(float(rank + 1))
        started = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        latencies.append(time.perf_counter() - started)
        expected = float(world_size * (world_size + 1) // 2)
        if not bool(torch.allclose(tensor, torch.full_like(tensor, expected))):
            raise RuntimeError(f"Incorrect all-reduce result on rank={rank}")
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "p50_seconds": statistics.median(latencies),
        "p95_seconds": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "mean_seconds": statistics.mean(latencies),
        "max_seconds": max(latencies),
        "min_seconds": min(latencies),
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size if rank == 0 else []
    dist.gather_object(local, gathered if rank == 0 else None, dst=0)
    dist.barrier()
    if rank == 0:
        report = {
            "status": "ok",
            "backend": "nccl",
            "world_size": world_size,
            "tensor_elements": args.tensor_elements,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "nccl_environment": {key: os.environ.get(key) for key in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NCCL_SHM_DISABLE", "NCCL_CUMEM_HOST_ENABLE")},
            "ranks": gathered,
        }
        write_json(output, report)
        print(f"[summary] world={world_size} tensor_elements={args.tensor_elements} p50_rank0={local['p50_seconds']:.6f}s", flush=True)
        print(f"[report] {output}", flush=True)
        print("[status] ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
