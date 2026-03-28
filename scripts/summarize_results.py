#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ebs.analysis import summarize_with_significance
from ebs.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate raw EventBatchSurv logs.")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    agg_dir = ensure_dir(results_dir / "aggregates")
    table_dir = ensure_dir(results_dir / "tables")

    run_summaries = _load_run_summaries(results_dir)
    if run_summaries.empty:
        raise RuntimeError("No run summaries found. Run scripts/run_grid.py first.")

    merged = _attach_empirical_columns(run_summaries=run_summaries, results_dir=results_dir)
    merged["bp"] = merged["batch_size"] * merged["train_event_rate"]
    merged.to_csv(agg_dir / "run_summaries_enriched.csv", index=False)

    summary, tests = summarize_with_significance(merged, metric="best_val_c_index")
    summary.to_csv(agg_dir / "condition_summary.csv", index=False)
    tests.to_csv(agg_dir / "paired_tests.csv", index=False)

    _write_main_table(merged, table_dir / "table_main.csv")
    _write_theory_table(merged, table_dir / "table_theory.csv")
    print(f"Wrote aggregates to {agg_dir} and tables to {table_dir}")


def _load_run_summaries(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "aggregates" / "run_summaries.csv"
    json_path = results_dir / "aggregates" / "run_summaries.json"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        return pd.read_json(json_path)
    return pd.DataFrame()


def _attach_empirical_columns(run_summaries: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    concise_cols = {
        "empirical_zero_event_prob",
        "empirical_weak_info_prob",
        "theoretical_zero_event_prob",
        "theoretical_weak_info_prob",
        "train_event_rate",
        "sampler_feasible",
    }
    if concise_cols.issubset(set(run_summaries.columns)):
        return run_summaries.copy()

    # Backward-compatibility fallback for legacy runs.
    runs_dir = results_dir / "runs"
    empirical_rows = []
    for run_dir in sorted(runs_dir.glob("*")):
        batch_log = run_dir / "batch_logs.jsonl"
        meta_file = run_dir / "run_meta.json"
        if not batch_log.exists() or not meta_file.exists():
            continue
        bdf = pd.read_json(batch_log, lines=True)
        with meta_file.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        empirical_rows.append(
            {
                "run_name": run_dir.name,
                "empirical_zero_event_prob": float((bdf["event_count"] == 0).mean()),
                "empirical_weak_info_prob": float((bdf["event_count"] <= 1).mean()),
                "theoretical_zero_event_prob": float(meta["theoretical_zero_event_prob"]),
                "theoretical_weak_info_prob": float(meta["theoretical_weak_info_prob"]),
                "train_event_rate": float(meta["train_event_rate"]),
                "sampler_feasible": bool(meta.get("sampler_feasible", True)),
            }
        )
    empirical_df = pd.DataFrame(empirical_rows)
    return run_summaries.merge(empirical_df, on="run_name", how="left")


def _write_main_table(df: pd.DataFrame, out_file: Path) -> None:
    cols = ["event_target", "censor_target", "batch_size", "batching_policy"]
    agg = (
        df.groupby(cols)["best_val_c_index"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "best_val_c_index_mean", "std": "best_val_c_index_std"})
    )
    agg.to_csv(out_file, index=False)


def _write_theory_table(df: pd.DataFrame, out_file: Path) -> None:
    cols = ["event_target", "censor_target", "batch_size", "batching_policy"]
    agg = (
        df.groupby(cols)[
            [
                "empirical_zero_event_prob",
                "theoretical_zero_event_prob",
                "empirical_weak_info_prob",
                "theoretical_weak_info_prob",
                "bp",
            ]
        ]
        .mean()
        .reset_index()
    )
    agg["zero_event_gap"] = agg["empirical_zero_event_prob"] - agg["theoretical_zero_event_prob"]
    agg["weak_info_gap"] = agg["empirical_weak_info_prob"] - agg["theoretical_weak_info_prob"]
    agg.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
