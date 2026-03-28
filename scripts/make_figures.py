#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ebs.utils.io import ensure_dir

sns.set_theme(style="whitegrid", context="talk")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate EventBatchSurv manuscript-ready figures.")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    fig_dir = ensure_dir(results_dir / "figures")
    run_df = pd.read_csv(results_dir / "aggregates" / "run_summaries_enriched.csv")
    _plot_heatmaps(run_df, fig_dir)
    _plot_gain_map(run_df, fig_dir)
    _plot_loss_curves(results_dir / "runs", fig_dir)
    _plot_event_histograms(results_dir / "runs", fig_dir)
    _plot_grad_variance(run_df, fig_dir)
    print(f"Wrote figures to {fig_dir}")


def _plot_heatmaps(df: pd.DataFrame, fig_dir: Path) -> None:
    for policy in sorted(df["batching_policy"].unique()):
        d = (
            df[df["batching_policy"] == policy]
            .groupby(["event_target", "batch_size"])["best_val_c_index"]
            .mean()
            .reset_index()
        )
        pivot = d.pivot(index="event_target", columns="batch_size", values="best_val_c_index")
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot.sort_index(ascending=False), annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"Final C-index Heatmap ({policy})")
        plt.xlabel("Batch Size")
        plt.ylabel("Event Prevalence Target")
        plt.tight_layout()
        plt.savefig(fig_dir / f"heatmap_{policy}.png", dpi=240)
        plt.close()


def _plot_gain_map(df: pd.DataFrame, fig_dir: Path) -> None:
    mean_df = (
        df.groupby(["event_target", "batch_size", "batching_policy"])["best_val_c_index"]
        .mean()
        .reset_index()
    )
    pivot = mean_df.pivot_table(
        index=["event_target", "batch_size"], columns="batching_policy", values="best_val_c_index"
    ).reset_index()
    for col in ["event_aware_min1", "event_aware_min2_feasible"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot["gain_min1_vs_random"] = pivot["event_aware_min1"] - pivot["random"]
    pivot["gain_min2_vs_random"] = pivot["event_aware_min2_feasible"] - pivot["random"]
    g = pivot.groupby(["event_target", "batch_size"])[["gain_min1_vs_random", "gain_min2_vs_random"]].max()
    g = g.reset_index()
    gain_pivot = g.pivot(index="event_target", columns="batch_size", values="gain_min1_vs_random")
    plt.figure(figsize=(8, 5))
    sns.heatmap(gain_pivot.sort_index(ascending=False), annot=True, fmt=".3f", cmap="coolwarm", center=0.0)
    plt.title("Largest Event-Aware Gain (min1 vs random)")
    plt.xlabel("Batch Size")
    plt.ylabel("Event Prevalence Target")
    plt.tight_layout()
    plt.savefig(fig_dir / "gain_map_min1_vs_random.png", dpi=240)
    plt.close()


def _plot_loss_curves(runs_dir: Path, fig_dir: Path) -> None:
    rows = []
    for run_dir in sorted(runs_dir.glob("*")):
        ep = run_dir / "epoch_logs.jsonl"
        if not ep.exists():
            continue
        run_name = run_dir.name
        parts = run_name.split("__")
        if len(parts) < 5:
            continue
        batch_size = int(parts[2].replace("b_", ""))
        policy = parts[3].replace("policy_", "")
        d = pd.read_json(ep, lines=True)
        d["batch_size"] = batch_size
        d["batching_policy"] = policy
        rows.append(d[["epoch", "train_loss", "batch_size", "batching_policy"]])
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    agg = (
        df.groupby(["epoch", "batch_size", "batching_policy"])["train_loss"]
        .mean()
        .reset_index()
        .sort_values("epoch")
    )
    g = sns.relplot(
        data=agg,
        kind="line",
        x="epoch",
        y="train_loss",
        hue="batching_policy",
        col="batch_size",
        col_wrap=2,
        height=4,
        aspect=1.2,
    )
    g.fig.suptitle("Training Loss Curves by Batch Size and Policy", y=1.02)
    plt.savefig(fig_dir / "training_loss_curves.png", dpi=240, bbox_inches="tight")
    plt.close()


def _plot_event_histograms(runs_dir: Path, fig_dir: Path) -> None:
    rows = []
    for run_dir in sorted(runs_dir.glob("*")):
        bp = run_dir / "batch_logs.jsonl"
        if not bp.exists():
            continue
        run_name = run_dir.name
        policy = run_name.split("__")[3].replace("policy_", "")
        bdf = pd.read_json(bp, lines=True)
        bdf["batching_policy"] = policy
        rows.append(bdf[["event_count", "batching_policy"]])
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="event_count", hue="batching_policy", bins=20, stat="probability", common_norm=False)
    plt.title("Event Counts Per Batch")
    plt.tight_layout()
    plt.savefig(fig_dir / "event_count_histograms.png", dpi=240)
    plt.close()


def _plot_grad_variance(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="batch_size",
        y="final_grad_norm_var",
        hue="batching_policy",
        style="event_target",
        markers=True,
        dashes=False,
    )
    plt.title("Gradient Norm Variance vs Batch Size")
    plt.tight_layout()
    plt.savefig(fig_dir / "gradient_norm_variance.png", dpi=240)
    plt.close()


if __name__ == "__main__":
    main()
