from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny random-data MLP training script.")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--output-dim", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    model = SimpleMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        depth=args.depth,
    ).to(device=device, dtype=dtype if device.type == "cuda" else torch.float32)

    inputs = torch.randn(args.num_samples, args.input_dim, device=device, dtype=model.net[0].weight.dtype)
    targets = torch.randn(args.num_samples, args.output_dim, device=device, dtype=model.net[0].weight.dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    criterion = nn.MSELoss()

    print("========== huginnI ==========")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"steps={args.steps} batch_size={args.batch_size} num_samples={args.num_samples}")
    print(
        f"model_dims=input:{args.input_dim} hidden:{args.hidden_dim} output:{args.output_dim} depth:{args.depth}"
    )
    print(f"parameters={sum(p.numel() for p in model.parameters())}")

    start_time = time.time()
    for step in range(1, args.steps + 1):
        batch_indices = torch.randint(0, args.num_samples, (args.batch_size,), device=device)
        batch_x = inputs[batch_indices]
        batch_y = targets[batch_indices]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
            preds = model(batch_x)
            loss = criterion(preds, batch_y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if step == 1 or step % args.log_interval == 0:
            elapsed = time.time() - start_time
            msg = f"step={step} loss={loss.item():.6f} elapsed_s={elapsed:.2f}"
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / float(1024**3)
                reserved = torch.cuda.memory_reserved(device) / float(1024**3)
                max_alloc = torch.cuda.max_memory_allocated(device) / float(1024**3)
                msg += (
                    f" mem_alloc_gb={allocated:.3f}"
                    f" mem_reserved_gb={reserved:.3f}"
                    f" max_mem_alloc_gb={max_alloc:.3f}"
                )
            print(msg, flush=True)


if __name__ == "__main__":
    main()
