#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ebs.io import ensure_dir
from ebs.policies import normalize_batching_policy

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
    if "batching_policy" in run_df.columns:
        run_df["batching_policy"] = run_df["batching_policy"].astype(str).map(normalize_batching_policy)
    if "test_harrell_c_index" not in run_df.columns and "test_c_index" in run_df.columns:
        run_df["test_harrell_c_index"] = run_df["test_c_index"]
    if "test_uno_c_index" not in run_df.columns and "test_ipcw_c_index" in run_df.columns:
        run_df["test_uno_c_index"] = run_df["test_ipcw_c_index"]
    _plot_heatmaps(run_df, fig_dir)
    _plot_gain_map(run_df, fig_dir)
    _plot_test_delta_forest(run_df, fig_dir)
    _plot_paired_seed_slopes(run_df, fig_dir)
    _plot_val_test_gap(run_df, fig_dir)
    _plot_mechanism_scatter(run_df, fig_dir)
    _plot_improvement_heatmaps(run_df, fig_dir)
    _plot_loss_curves(results_dir / "runs", fig_dir)
    _plot_event_histograms(results_dir / "runs", fig_dir)
    _plot_grad_variance(run_df, fig_dir)
    print(f"Wrote figures to {fig_dir}")


def _plot_heatmaps(df: pd.DataFrame, fig_dir: Path) -> None:
    for metric, metric_label in [
        ("test_harrell_c_index", "Test Harrell C-index"),
        ("test_uno_c_index", "Test Uno C-index"),
    ]:
        if metric not in df.columns:
            continue
        for policy in sorted(df["batching_policy"].unique()):
            d = (
                df[df["batching_policy"] == policy]
                .groupby(["event_target", "batch_size"])[metric]
                .mean()
                .reset_index()
            )
            pivot = d.pivot(index="event_target", columns="batch_size", values=metric)
            plt.figure(figsize=(8, 5))
            sns.heatmap(pivot.sort_index(ascending=False), annot=True, fmt=".3f", cmap="viridis")
            plt.title(f"{metric_label} Heatmap ({policy})")
            plt.xlabel("Batch Size")
            plt.ylabel("Event Prevalence Target")
            plt.tight_layout()
            plt.savefig(fig_dir / f"heatmap_{metric}_{policy}.png", dpi=240)
            plt.close()


def _plot_gain_map(df: pd.DataFrame, fig_dir: Path) -> None:
    nonrandom = _nonrandom_policies(df)
    if not nonrandom:
        return
    for metric, metric_label in [
        ("test_harrell_c_index", "Test Harrell"),
        ("test_uno_c_index", "Test Uno"),
    ]:
        if metric not in df.columns:
            continue
        mean_df = (
            df.groupby(["event_target", "batch_size", "batching_policy"])[metric]
            .mean()
            .reset_index()
        )
        pivot = mean_df.pivot_table(index=["event_target", "batch_size"], columns="batching_policy", values=metric).reset_index()
        if "random" not in pivot.columns:
            continue
        for policy in nonrandom:
            if policy not in pivot.columns:
                continue
            gain_col = f"gain_{policy}_vs_random"
            pivot[gain_col] = pivot[policy] - pivot["random"]
            g = pivot.groupby(["event_target", "batch_size"])[gain_col].mean().reset_index()
            gain_pivot = g.pivot(index="event_target", columns="batch_size", values=gain_col)
            plt.figure(figsize=(8, 5))
            sns.heatmap(gain_pivot.sort_index(ascending=False), annot=True, fmt=".3f", cmap="coolwarm", center=0.0)
            plt.title(f"Gain map ({metric_label}: {policy} - random)")
            plt.xlabel("Batch Size")
            plt.ylabel("Event Prevalence Target")
            plt.tight_layout()
            plt.savefig(fig_dir / f"gain_map_{metric}_{policy}_vs_random.png", dpi=240)
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
        policy = normalize_batching_policy(parts[3].replace("policy_", ""))
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
        policy = normalize_batching_policy(run_name.split("__")[3].replace("policy_", ""))
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


def _plot_test_delta_forest(df: pd.DataFrame, fig_dir: Path) -> None:
    rows = []
    for policy in _nonrandom_policies(df):
        paired = _paired_policy(df, policy=policy, metric="test_uno_c_index")
        if paired.empty:
            continue
        paired["delta"] = paired[policy] - paired["random"]
        agg = (
            paired.groupby(["event_target", "censor_target", "batch_size"])["delta"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["sem"] = agg["std"] / agg["count"].pow(0.5)
        agg["ci95"] = 1.96 * agg["sem"]
        agg["policy"] = policy
        rows.append(agg)
    if not rows:
        return
    forest = pd.concat(rows, ignore_index=True)
    forest["label"] = (
        "e="
        + forest["event_target"].astype(str)
        + ", c="
        + forest["censor_target"].astype(str)
        + ", b="
        + forest["batch_size"].astype(str)
    )
    forest = forest.sort_values(["policy", "event_target", "censor_target", "batch_size"])

    g = sns.FacetGrid(forest, row="policy", sharex=True, sharey=False, height=6, aspect=1.8)

    def draw(data: pd.DataFrame, **kwargs) -> None:
        y = range(len(data))
        plt.errorbar(
            data["mean"],
            y,
            xerr=data["ci95"],
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            capsize=2,
            alpha=0.9,
        )
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
        plt.yticks(y, data["label"])
        plt.xlabel("Delta Test Uno C-index vs random (95% CI)")
        plt.ylabel("Condition")

    g.map_dataframe(draw)
    g.set_titles("{row_name}")
    g.fig.subplots_adjust(hspace=0.25)
    g.savefig(fig_dir / "forest_test_uno_delta_vs_random.png", dpi=240, bbox_inches="tight")
    plt.close(g.fig)


def _plot_paired_seed_slopes(df: pd.DataFrame, fig_dir: Path) -> None:
    primary = _primary_policy(df)
    if primary is None:
        return
    sub = df[df["batching_policy"].isin(["random", primary])].copy()
    if sub.empty:
        return
    wide = sub.pivot_table(
        index=["event_target", "censor_target", "batch_size", "seed"],
        columns="batching_policy",
        values="test_uno_c_index",
    ).dropna()
    if wide.empty:
        return
    if "random" not in wide.columns or primary not in wide.columns:
        return
    rows = []
    for idx, r in wide.reset_index().iterrows():
        rows.append(
            {
                "event_target": r["event_target"],
                "censor_target": r["censor_target"],
                "batch_size": r["batch_size"],
                "seed": int(r["seed"]),
                "policy": "random",
                "score": float(r["random"]),
            }
        )
        rows.append(
            {
                "event_target": r["event_target"],
                "censor_target": r["censor_target"],
                "batch_size": r["batch_size"],
                "seed": int(r["seed"]),
                "policy": primary,
                "score": float(r[primary]),
            }
        )
    long = pd.DataFrame(rows)
    long["facet"] = (
        "e="
        + long["event_target"].astype(str)
        + ", b="
        + long["batch_size"].astype(str)
        + ", c="
        + long["censor_target"].astype(str)
    )
    g = sns.FacetGrid(long, col="facet", col_wrap=4, sharey=False, height=3.5)
    g.map_dataframe(
        sns.lineplot,
        x="policy",
        y="score",
        units="seed",
        estimator=None,
        alpha=0.45,
        linewidth=1.0,
        color="#3b82f6",
    )
    g.set_axis_labels("Policy", "Test Uno C-index")
    g.set_titles("{col_name}")
    g.savefig(fig_dir / f"paired_seed_slopes_test_uno_{primary}_vs_random.png", dpi=240, bbox_inches="tight")
    plt.close(g.fig)


def _plot_val_test_gap(df: pd.DataFrame, fig_dir: Path) -> None:
    if "test_uno_c_index" not in df.columns:
        return
    d = df.copy()
    d["val_test_gap"] = d["best_val_c_index"] - d["test_uno_c_index"]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=d, x="batching_policy", y="val_test_gap")
    sns.stripplot(
        data=d,
        x="batching_policy",
        y="val_test_gap",
        hue="batch_size",
        alpha=0.25,
        dodge=True,
        size=3,
    )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Validation-to-test gap by policy")
    plt.ylabel("best_val_c_index - test_uno_c_index")
    plt.legend(title="batch_size", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "val_test_gap_by_policy.png", dpi=240)
    plt.close()


def _plot_mechanism_scatter(df: pd.DataFrame, fig_dir: Path) -> None:
    primary = _primary_policy(df)
    if primary is None:
        return
    deltas = _paired_policy(df, policy=primary, metric="test_uno_c_index")
    if deltas.empty:
        return
    deltas["metric_delta"] = deltas[primary] - deltas["random"]
    probs = (
        df[df["batching_policy"] == "random"][
            ["event_target", "censor_target", "batch_size", "seed", "empirical_zero_event_prob", "empirical_weak_info_prob"]
        ]
        .copy()
        .drop_duplicates()
    )
    merged = deltas.merge(probs, on=["event_target", "censor_target", "batch_size", "seed"], how="inner")
    if merged.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.regplot(
        data=merged,
        x="empirical_zero_event_prob",
        y="metric_delta",
        scatter_kws={"alpha": 0.35, "s": 30},
        line_kws={"color": "red"},
        ax=axes[0],
    )
    axes[0].set_title("Mechanism: delta vs zero-event frequency")
    axes[0].set_xlabel("Empirical zero-event batch probability (random)")
    axes[0].set_ylabel(f"Test Uno C-index delta ({primary} - random)")
    sns.regplot(
        data=merged,
        x="empirical_weak_info_prob",
        y="metric_delta",
        scatter_kws={"alpha": 0.35, "s": 30},
        line_kws={"color": "red"},
        ax=axes[1],
    )
    axes[1].set_title("Mechanism: delta vs weak-info frequency")
    axes[1].set_xlabel("Empirical weak-info batch probability (random)")
    plt.tight_layout()
    plt.savefig(fig_dir / "mechanism_scatter_deltas.png", dpi=240)
    plt.close()


def _plot_improvement_heatmaps(df: pd.DataFrame, fig_dir: Path) -> None:
    for policy in _nonrandom_policies(df):
        paired = _paired_policy(df, policy=policy, metric="test_uno_c_index")
        if paired.empty:
            continue
        paired["delta"] = paired[policy] - paired["random"]
        for batch_size in sorted(paired["batch_size"].unique()):
            bdf = paired[paired["batch_size"] == batch_size]
            if bdf.empty:
                continue
            grid = (
                bdf.groupby(["event_target", "censor_target"])["delta"]
                .mean()
                .reset_index()
                .pivot(index="event_target", columns="censor_target", values="delta")
            )
            plt.figure(figsize=(7, 5))
            sns.heatmap(grid.sort_index(ascending=False), annot=True, fmt=".3f", cmap="coolwarm", center=0.0)
            plt.title(f"Test Uno improvement: {policy} vs random (b={batch_size})")
            plt.xlabel("Censor target")
            plt.ylabel("Event target")
            plt.tight_layout()
            plt.savefig(fig_dir / f"improvement_heatmap_test_uno_{policy}_b{int(batch_size)}.png", dpi=240)
            plt.close()


def _paired_policy(df: pd.DataFrame, *, policy: str, metric: str) -> pd.DataFrame:
    sub = df[df["batching_policy"].isin(["random", policy])].copy()
    if sub.empty:
        return pd.DataFrame()
    if metric not in sub.columns:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index=["event_target", "censor_target", "batch_size", "seed"],
        columns="batching_policy",
        values=metric,
    )
    needed = {"random", policy}
    if not needed.issubset(set(pivot.columns)):
        return pd.DataFrame()
    return pivot.reset_index()


def _nonrandom_policies(df: pd.DataFrame) -> list[str]:
    return [p for p in sorted(df["batching_policy"].astype(str).unique()) if p != "random"]


def _primary_policy(df: pd.DataFrame) -> str | None:
    preferred = [
        "event_quota_wor_25",
        "event_quota_wr_25",
    ]
    existing = set(df["batching_policy"].astype(str).unique())
    for p in preferred:
        if p in existing:
            return p
    candidates = [p for p in sorted(existing) if p != "random"]
    return candidates[0] if candidates else None


if __name__ == "__main__":
    main()
