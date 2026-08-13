"""Executable BAT training contract.

The values in this module mirror Table 5 of the BAT paper.  They are kept in
one place so a shell launcher cannot silently drift from the paper.  The
published BAT shell does not expose a separate ``epoch partitioning factor``
argument; therefore the value is preserved and reported, but is not invented
as an extra data multiplier or an undocumented scheduler change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BatStage:
    name: str
    dataset_dir: str
    epochs: int
    bat_types: tuple[str, ...]


@dataclass(frozen=True)
class BatTrainingConfig:
    sound_source: str = "AudioSet-20K"
    audio_normalization: bool = True
    augmentation: bool = False
    weighted_sampling: bool = False
    optimizer: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.05
    learning_rate: float = 0.001
    scheduler: str = "half-cycle cosine decay"
    warmup_epochs: int = 2
    epoch_partitioning_factor: int = 10
    per_device_batch_size: int = 2
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    audio_token_count: int = 64
    qformer_layers: int = 8
    stages: tuple[BatStage, ...] = (
        BatStage("I", "stage1-clsdoa", 2, ("A", "B")),
        BatStage("II", "stage2-single", 2, ("A", "B", "C", "D")),
        BatStage("III", "stage3-mixup", 3, ("A", "B", "C", "D", "E")),
    )

    def validate(self) -> None:
        if self.sound_source != "AudioSet-20K":
            raise ValueError(f"BAT source must be AudioSet-20K, got {self.sound_source!r}")
        if not self.audio_normalization or self.augmentation or self.weighted_sampling:
            raise ValueError("BAT requires normalization on, augmentation off, weighted sampling off")
        if self.optimizer != "AdamW" or (self.beta1, self.beta2) != (0.9, 0.95):
            raise ValueError("BAT optimizer contract is AdamW with betas (0.9, 0.95)")
        if self.weight_decay != 0.05 or self.learning_rate != 0.001:
            raise ValueError("BAT optimizer contract requires weight_decay=0.05 and lr=0.001")
        if self.scheduler != "half-cycle cosine decay":
            raise ValueError(f"Unexpected BAT scheduler: {self.scheduler!r}")
        if self.warmup_epochs != 2 or self.epoch_partitioning_factor != 10:
            raise ValueError("BAT requires warmup_epochs=2 and epoch_partitioning_factor=10")
        if self.per_device_batch_size != 2:
            raise ValueError("BAT batch size contract requires per_device_batch_size=2")
        if self.lora_rank != 8 or self.lora_alpha != 32 or self.lora_dropout != 0.05:
            raise ValueError("BAT official LoRA contract requires r=8, alpha=32, dropout=0.05")
        if self.lora_target_modules != ("q_proj", "v_proj"):
            raise ValueError("BAT official target modules are q_proj and v_proj")
        if tuple(stage.epochs for stage in self.stages) != (2, 2, 3):
            raise ValueError("BAT stage epochs must be (2, 2, 3)")

    def stage(self, name: str) -> BatStage:
        self.validate()
        for stage in self.stages:
            if stage.name == name or stage.dataset_dir == name:
                return stage
        raise KeyError(f"Unknown BAT stage: {name!r}")

    def schedule(self, dataset_size: int, world_size: int = 1, gradient_accumulation_steps: int = 1,
                 stage_name: str = "I") -> Mapping[str, int | float | str]:
        """Return the actual Trainer step budget for a stage.

        ``epoch_partitioning_factor`` is reported as a paper contract field.
        The paper and the official public BAT launcher do not define an
        executable transformation for it, so it must not alter exposure count
        here.  This makes the resulting steps auditable instead of fabricating
        a meaning that is not in the source.
        """
        self.validate()
        stage = self.stage(stage_name)
        if dataset_size <= 0 or world_size <= 0 or gradient_accumulation_steps <= 0:
            raise ValueError("dataset_size, world_size and gradient_accumulation_steps must be positive")
        effective_batch = self.per_device_batch_size * world_size * gradient_accumulation_steps
        steps_per_epoch = max(1, math.ceil(dataset_size / effective_batch))
        total_steps = steps_per_epoch * stage.epochs
        warmup_steps = min(total_steps, steps_per_epoch * self.warmup_epochs)
        return {
            "stage": stage.name,
            "dataset_size": dataset_size,
            "world_size": world_size,
            "per_device_batch_size": self.per_device_batch_size,
            "effective_batch_size": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "epochs": stage.epochs,
            "total_steps": total_steps,
            "warmup_epochs": self.warmup_epochs,
            "warmup_steps": warmup_steps,
            "scheduler": self.scheduler,
            "epoch_partitioning_factor": self.epoch_partitioning_factor,
            "epoch_partitioning_semantics": "reported_only_public_source_undefined",
        }


BAT_TRAINING = BatTrainingConfig()
BAT_TRAINING.validate()
