#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebs.config.schema import ExperimentConfig
from ebs.data.mnist_survival import generate_synthetic_survival_from_mnist
from ebs.data.splits import get_or_create_fixed_splits
from ebs.train.engine import RunContext, run_training
from ebs.utils.io import ensure_dir, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EventBatchSurv factorial grid.")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--grid-config", default="configs/grid_event_censor.yaml")
    p.add_argument("--smoke", action="store_true", help="Run a tiny smoke subset.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_merged_config(args.base_config, args.grid_config)

    runs_root = ensure_dir(Path(cfg.output.results_dir) / cfg.output.run_dirname)
    aggregates_root = ensure_dir(Path(cfg.output.results_dir) / cfg.output.aggregate_dirname)

    if args.smoke:
        cfg.dataset.event_prevalence_targets = cfg.dataset.event_prevalence_targets[:1]
        cfg.dataset.censoring_targets = cfg.dataset.censoring_targets[:1]
        cfg.train.batch_sizes = cfg.train.batch_sizes[:2]
        cfg.train.seeds = cfg.train.seeds[:2]

    run_rows: list[dict[str, float | int | str]] = []
    for event_target in cfg.dataset.event_prevalence_targets:
        for censor_target in cfg.dataset.censoring_targets:
            for seed in cfg.train.seeds:
                data_seed = cfg.dataset.generator_seed_offset + seed
                surv = generate_synthetic_survival_from_mnist(
                    mnist_root=cfg.dataset.mnist_root,
                    n_samples=cfg.dataset.n_samples,
                    event_prevalence_target=event_target,
                    censoring_target=censor_target,
                    seed=data_seed,
                )
                split_file = (
                    Path(cfg.dataset.cache_dir)
                    / "splits"
                    / f"evt_{event_target:.2f}_cen_{censor_target:.2f}_seed_{seed}.json"
                )
                split = get_or_create_fixed_splits(
                    n_samples=cfg.dataset.n_samples,
                    train_fraction=cfg.dataset.train_fraction,
                    val_fraction=cfg.dataset.val_fraction,
                    seed=cfg.dataset.split_seed + seed,
                    cache_file=split_file,
                )

                dataset_meta = {
                    "event_target": event_target,
                    "censor_target": censor_target,
                    "seed": seed,
                    "realized_event_rate": surv.realized_event_rate,
                    "realized_censoring_susceptible": surv.realized_censoring_susceptible,
                }
                dataset_meta_file = (
                    Path(cfg.dataset.cache_dir)
                    / "datasets"
                    / f"evt_{event_target:.2f}_cen_{censor_target:.2f}_seed_{seed}.json"
                )
                write_json(dataset_meta_file, dataset_meta)

                for batch_size in cfg.train.batch_sizes:
                    for policy in cfg.train.batching_policies:
                        run_name = (
                            f"evt_{event_target:.2f}__cen_{censor_target:.2f}"
                            f"__b_{batch_size}__policy_{policy}__seed_{seed}"
                        )
                        out_dir = runs_root / run_name
                        summary = run_training(
                            cfg=cfg,
                            context=RunContext(
                                event_target=event_target,
                                censor_target=censor_target,
                                batch_size=batch_size,
                                batching_policy=policy,
                                seed=seed,
                                output_dir=out_dir,
                            ),
                            x=surv.x,
                            time_obs=surv.time,
                            event_obs=surv.event,
                            train_idx=split.train_idx,
                            val_idx=split.val_idx,
                        )
                        summary["run_name"] = run_name
                        summary["realized_dataset_event_rate"] = surv.realized_event_rate
                        summary["realized_censoring_susceptible"] = surv.realized_censoring_susceptible
                        run_rows.append(summary)
                        print(json.dumps({"status": "completed_run", "run_name": run_name}))

    write_csv(aggregates_root / "run_summaries.csv", run_rows)
    pd.DataFrame(run_rows).to_json(aggregates_root / "run_summaries.json", orient="records", indent=2)


def load_merged_config(base_config: str, grid_config: str) -> ExperimentConfig:
    with Path(base_config).open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    with Path(grid_config).open("r", encoding="utf-8") as f:
        grid = yaml.safe_load(f) or {}
    merged = deep_merge(base, grid)
    return ExperimentConfig.from_dict(merged)


def deep_merge(left: dict, right: dict) -> dict:
    out = dict(left)
    for k, v in right.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
