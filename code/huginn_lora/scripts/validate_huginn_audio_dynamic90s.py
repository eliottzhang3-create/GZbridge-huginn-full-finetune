#!/usr/bin/env python3
"""Validate the Whisper dynamic-90s audio-prefix contract with simulated audio.

This test intentionally does not load Whisper, Huginn, a dataset, or a
checkpoint. It validates the length arithmetic, single Conv1d compressor,
per-batch prefix padding, -100 prefix labels, and a differentiable training
step using a frozen simulated encoder and a small language head.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30.0
MAX_AUDIO_SECONDS = 90.0
DISCARD_AUDIO_SECONDS = 120.0
WHISPER_MEL_RATE = 100.0
WHISPER_ENCODER_RATE = 50.0
COMPRESSOR_KERNEL = 6
COMPRESSOR_STRIDE = 6
BOUNDARY_TOKENS = 2


@dataclass(frozen=True)
class SimulatedAudioPlan:
    requested_seconds: float
    included_seconds: float
    chunk_seconds: tuple[float, ...]
    mel_frames: tuple[int, ...]
    encoder_frames: tuple[int, ...]
    audio_tokens: tuple[int, ...]
    discarded: bool

    @property
    def total_audio_tokens(self) -> int:
        return sum(self.audio_tokens)

    @property
    def prefix_tokens(self) -> int:
        return self.total_audio_tokens + BOUNDARY_TOKENS


def simulate_audio_plan(duration_seconds: float) -> SimulatedAudioPlan:
    if duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")
    if duration_seconds > DISCARD_AUDIO_SECONDS:
        return SimulatedAudioPlan(
            requested_seconds=duration_seconds,
            included_seconds=0.0,
            chunk_seconds=(),
            mel_frames=(),
            encoder_frames=(),
            audio_tokens=(),
            discarded=True,
        )

    requested_samples = int(round(duration_seconds * SAMPLE_RATE))
    included_samples = min(requested_samples, int(round(MAX_AUDIO_SECONDS * SAMPLE_RATE)))
    chunk_samples = int(round(CHUNK_SECONDS * SAMPLE_RATE))
    sample_chunks: list[int] = []
    for start in range(0, included_samples, chunk_samples):
        sample_chunks.append(min(chunk_samples, included_samples - start))

    included_seconds = included_samples / SAMPLE_RATE
    chunks = [samples / SAMPLE_RATE for samples in sample_chunks]
    mel_frames = tuple(max(1, samples // (SAMPLE_RATE // int(WHISPER_MEL_RATE))) for samples in sample_chunks)
    encoder_frames = tuple(frames // 2 for frames in mel_frames)
    audio_tokens = tuple(max(0, (frames - COMPRESSOR_KERNEL) // COMPRESSOR_STRIDE + 1) for frames in encoder_frames)
    return SimulatedAudioPlan(
        requested_seconds=duration_seconds,
        included_seconds=included_seconds,
        chunk_seconds=tuple(chunks),
        mel_frames=mel_frames,
        encoder_frames=encoder_frames,
        audio_tokens=audio_tokens,
        discarded=False,
    )


class SimulatedDynamicAudioModel(nn.Module):
    """Small differentiable stand-in for the changed audio-prefix path."""

    def __init__(self, hidden_size: int = 32, vocab_size: int = 97):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.temporal_compressor = nn.Conv1d(
            1280,
            1280,
            kernel_size=COMPRESSOR_KERNEL,
            stride=COMPRESSOR_STRIDE,
            padding=0,
        )
        self.input_norm = nn.LayerNorm(1280)
        self.w1 = nn.Linear(1280, 64)
        self.w2 = nn.Linear(1280, 64)
        self.c_proj = nn.Linear(64, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.audio_bos = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01)
        self.audio_eos = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01)
        self.text_embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.input_norm(hidden)
        return self.output_norm(self.c_proj(self.w1(hidden) * F.silu(self.w2(hidden))))

    def build_batch_prefix(self, plans: list[SimulatedAudioPlan]) -> tuple[torch.Tensor, torch.Tensor]:
        if any(plan.discarded for plan in plans):
            raise ValueError("Discarded samples must be filtered before batching")

        prefix_lengths = [plan.prefix_tokens for plan in plans]
        max_prefix_length = max(prefix_lengths)
        prefix = torch.zeros((len(plans), max_prefix_length, self.hidden_size))
        prefix_mask = torch.zeros((len(plans), max_prefix_length), dtype=torch.bool)

        for sample_index, plan in enumerate(plans):
            chunks: list[torch.Tensor] = [self.audio_bos]
            for chunk_index, encoder_frames in enumerate(plan.encoder_frames):
                if encoder_frames < COMPRESSOR_KERNEL:
                    continue
                # Simulated frozen Whisper output. Its first dimension follows
                # the real encoder's approximately 50 frames/sec output.
                generator = torch.Generator().manual_seed(1000 + sample_index * 10 + chunk_index)
                encoder_hidden = torch.randn(
                    (1, encoder_frames, 1280),
                    generator=generator,
                )
                compressed = self.temporal_compressor(encoder_hidden.transpose(1, 2)).transpose(1, 2)
                chunks.append(self.project(compressed))
            chunks.append(self.audio_eos)
            sample_prefix = torch.cat(chunks, dim=1)
            prefix[sample_index, : sample_prefix.size(1)] = sample_prefix[0]
            prefix_mask[sample_index, : sample_prefix.size(1)] = True

        return prefix, prefix_mask

    def training_step(self, plans: list[SimulatedAudioPlan], text_length: int = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix, prefix_mask = self.build_batch_prefix(plans)
        text_ids = torch.randint(self.vocab_size, (len(plans), text_length))
        text_embeds = self.text_embedding(text_ids)
        combined = torch.cat([prefix, text_embeds], dim=1)
        logits = self.lm_head(combined)

        labels = torch.randint(self.vocab_size, (len(plans), combined.size(1)))
        labels[:, : prefix.size(1)] = -100
        labels[:, prefix.size(1) :] = text_ids
        has_audio_padding = prefix_mask.sum(dim=1).lt(prefix.size(1))
        labels[has_audio_padding, prefix.size(1)] = -100
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return loss, prefix_mask, labels


def validate_individual_lengths(durations: list[float]) -> list[SimulatedAudioPlan]:
    plans = [simulate_audio_plan(duration) for duration in durations]
    for plan in plans:
        if plan.discarded:
            if plan.requested_seconds <= DISCARD_AUDIO_SECONDS:
                raise AssertionError(f"Unexpected discard: {plan}")
            print(f"[discard] duration={plan.requested_seconds:.3f}s")
            continue
        if len(plan.chunk_seconds) > 3:
            raise AssertionError(f"More than three chunks: {plan}")
        # Durations below 30s are intentionally dynamic; only exact 30/60/90s
        # boundaries have fixed reference counts below.
        expected = plan.total_audio_tokens
        if plan.requested_seconds == 30.0 and expected != 250:
            raise AssertionError(f"30s must produce 250 tokens: {plan}")
        if plan.requested_seconds == 60.0 and expected != 500:
            raise AssertionError(f"60s must produce 500 tokens: {plan}")
        if plan.requested_seconds == 90.0 and expected != 750:
            raise AssertionError(f"90s must produce 750 tokens: {plan}")
        if 90.0 < plan.requested_seconds <= 120.0 and expected != 750:
            raise AssertionError(f"90-120s audio must be truncated to 750 tokens: {plan}")
        print(
            f"[length] requested={plan.requested_seconds:.3f}s included={plan.included_seconds:.3f}s "
            f"chunks={plan.chunk_seconds} mel={plan.mel_frames} encoder={plan.encoder_frames} "
            f"audio_tokens={plan.audio_tokens} prefix_tokens={plan.prefix_tokens}"
        )
    return plans


def validate_batch_and_backward(model: SimulatedDynamicAudioModel, plans: list[SimulatedAudioPlan]) -> None:
    plans = [plan for plan in plans if not plan.discarded]
    loss, prefix_mask, labels = model.training_step(plans)
    loss.backward()
    expected_max_prefix = max(plan.prefix_tokens for plan in plans)
    expected_padding = [expected_max_prefix - plan.prefix_tokens for plan in plans]
    actual_padding = (~prefix_mask).sum(dim=1).tolist()
    if actual_padding != expected_padding:
        raise AssertionError(f"Batch padding mismatch: expected={expected_padding} actual={actual_padding}")
    if not torch.all(labels[:, :expected_max_prefix] == -100):
        raise AssertionError("Every padded batch prefix position must be masked with -100 labels")
    has_audio_padding = prefix_mask.sum(dim=1).lt(expected_max_prefix)
    if not torch.all(labels[has_audio_padding, expected_max_prefix] == -100):
        raise AssertionError("The first text target after a padded audio prefix must be masked with -100")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise AssertionError(f"Trainable parameter did not receive a gradient: {name}")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Non-finite gradient: {name}")
    print(
        f"[backward] batch={len(plans)} max_prefix={expected_max_prefix} "
        f"padding={actual_padding} loss={float(loss.detach()):.6f} gradients=finite"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--durations",
        nargs="*",
        type=float,
        default=[0.1, 1.0, 15.0, 29.99, 30.0, 30.01, 45.0, 60.0, 75.0, 90.0, 91.0, 119.0, 120.0, 120.01],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    plans = validate_individual_lengths(args.durations)
    model = SimulatedDynamicAudioModel()
    validate_batch_and_backward(model, plans)
    print("========== HUGINN AUDIO DYNAMIC90S VALIDATION PASSED ==========")


if __name__ == "__main__":
    main()
