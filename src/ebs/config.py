from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    mnist_root: str = "data/mnist"
    n_samples: int = 12000
    event_prevalence_targets: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.40, 1.00]
    )
    censoring_targets: list[float] = field(default_factory=lambda: [0.00, 0.25, 0.50, 0.75])
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    split_seed: int = 123
    generator_seed_offset: int = 10_000
    cache_dir: str = "results/cache"


@dataclass
class TrainConfig:
    batch_sizes: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    batching_policies: list[str] = field(
        default_factory=lambda: [
            "random",
            "event_quota_wor_25",
            "event_quota_wor_50",
            "event_quota_wor_75",
            "event_quota_wr_25",
            "event_quota_wr_50",
            "event_quota_wr_75",
        ]
    )
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    min_lr: float = 1e-5
    grad_clip_norm: float = 10.0
    device: str = "cpu"
    log_every_batches: int = 50
    save_batch_logs: bool = False
    save_epoch_logs: bool = False
    save_run_meta: bool = False
    save_run_summary: bool = True


@dataclass
class ModelConfig:
    backbone: str = "cox_mlp"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2


@dataclass
class OutputConfig:
    results_dir: str = "results"
    run_dirname: str = "runs"
    aggregate_dirname: str = "aggregates"
    figure_dirname: str = "figures"
    table_dirname: str = "tables"
    interpretation_filename: str = "interpretation.md"


@dataclass
class ExperimentConfig:
    name: str = "small_batch_cox"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return _from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(raw)


def _from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    dataset = DatasetConfig(**raw.get("dataset", {}))
    train = TrainConfig(**raw.get("train", {}))
    model = ModelConfig(**raw.get("model", {}))
    output = OutputConfig(**raw.get("output", {}))
    name = raw.get("name", "small_batch_cox")
    return ExperimentConfig(name=name, dataset=dataset, train=train, model=model, output=output)
