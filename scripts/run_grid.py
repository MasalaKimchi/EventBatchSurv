#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebs.config import ExperimentConfig
from ebs.data import generate_synthetic_survival_from_mnist, get_or_create_fixed_splits
from ebs.engine import RunContext, run_training
from ebs.io import ensure_dir, write_csv, write_json
from ebs.policies import normalize_batching_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EventBatchSurv factorial grid.")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--grid-config", default="configs/grid_event_censor.yaml")
    p.add_argument("--smoke", action="store_true", help="Run a tiny smoke subset.")
    p.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Override config seeds with a contiguous range of this length.",
    )
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed used together with --num-seeds.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_merged_config(args.base_config, args.grid_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = Path(args.grid_config).stem
    compact_run_dir = ensure_dir(Path(cfg.output.results_dir) / f"{run_prefix}_{timestamp}")
    runs_root = ensure_dir(Path(cfg.output.results_dir) / cfg.output.run_dirname)
    aggregates_root = ensure_dir(Path(cfg.output.results_dir) / cfg.output.aggregate_dirname)

    # Compact run layout: we only keep consolidated summaries in one timestamped folder.
    cfg.train.save_run_summary = False

    if args.smoke:
        cfg.dataset.event_prevalence_targets = cfg.dataset.event_prevalence_targets[:1]
        cfg.dataset.censoring_targets = cfg.dataset.censoring_targets[:1]
        cfg.train.batch_sizes = cfg.train.batch_sizes[:2]
        cfg.train.seeds = cfg.train.seeds[:2]
    if args.num_seeds is not None:
        if args.num_seeds <= 0:
            raise ValueError("--num-seeds must be >= 1")
        cfg.train.seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

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
                    for raw_policy in cfg.train.batching_policies:
                        policy = normalize_batching_policy(raw_policy)
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
                            test_idx=split.test_idx,
                        )
                        summary["run_name"] = run_name
                        summary["compact_run_id"] = f"{run_prefix}_{timestamp}"
                        summary["realized_dataset_event_rate"] = surv.realized_event_rate
                        summary["realized_censoring_susceptible"] = surv.realized_censoring_susceptible
                        run_rows.append(summary)
                        print(json.dumps({"status": "completed_run", "run_name": run_name}))

    run_df = pd.DataFrame(run_rows)
    # Write concise, one-file-per-format summaries for this run only.
    write_csv(compact_run_dir / "run_summaries.csv", run_rows)
    run_df.to_json(compact_run_dir / "run_summaries.json", orient="records", indent=2)
    _write_seed_aggregates(run_df, compact_run_dir / "seed_aggregates.csv")
    write_json(
        compact_run_dir / "manifest.json",
        {
            "run_id": f"{run_prefix}_{timestamp}",
            "created_at": timestamp,
            "grid_config": args.grid_config,
            "base_config": args.base_config,
            "n_runs": int(len(run_rows)),
        },
    )

    # Keep legacy aggregate outputs for downstream scripts.
    write_csv(aggregates_root / "run_summaries.csv", run_rows)
    run_df.to_json(aggregates_root / "run_summaries.json", orient="records", indent=2)
    _write_seed_aggregates(run_df, aggregates_root / "seed_aggregates.csv")


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


def _write_seed_aggregates(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    group_cols = ["event_target", "censor_target", "batch_size", "batching_policy"]
    metric_col = "best_val_c_index"

    grouped = (
        df.groupby(group_cols)[metric_col]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_runs",
                "mean": "best_val_c_index_mean",
                "std": "best_val_c_index_std",
            }
        )
    )
    grouped["best_val_c_index_sem"] = grouped["best_val_c_index_std"] / np.sqrt(grouped["n_runs"])
    grouped["best_val_c_index_ci95"] = 1.96 * grouped["best_val_c_index_sem"]
    grouped["best_val_c_index_ci95_lo"] = grouped["best_val_c_index_mean"] - grouped["best_val_c_index_ci95"]
    grouped["best_val_c_index_ci95_hi"] = grouped["best_val_c_index_mean"] + grouped["best_val_c_index_ci95"]
    grouped.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
