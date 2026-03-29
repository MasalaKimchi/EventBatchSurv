#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ebs.io import ensure_dir
from ebs.policies import FAMILY_ORDER, batching_family, normalize_batching_policy

sns.set_theme(style="whitegrid", context="talk")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate concise family-level sampler summaries.")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    table_dir = ensure_dir(results_dir / "tables")
    fig_dir = ensure_dir(results_dir / "figures")

    run_df = _load_run_summaries(results_dir)
    selected = _select_family_representatives(run_df)
    summary = _build_family_summary(selected)

    csv_path = table_dir / "table_family_summary.csv"
    fig_path = fig_dir / "family_summary_test_uno.png"
    report_path = results_dir / "family_report.md"

    summary.to_csv(csv_path, index=False)
    _plot_family_summary(summary, fig_path)
    report_path.write_text(_build_report(summary, csv_path=csv_path, fig_path=fig_path), encoding="utf-8")
    print(f"Wrote {csv_path}, {fig_path}, and {report_path}")


def _load_run_summaries(results_dir: Path) -> pd.DataFrame:
    enriched = results_dir / "aggregates" / "run_summaries_enriched.csv"
    raw_csv = results_dir / "aggregates" / "run_summaries.csv"
    raw_json = results_dir / "aggregates" / "run_summaries.json"
    if enriched.exists():
        df = pd.read_csv(enriched)
    elif raw_csv.exists():
        df = pd.read_csv(raw_csv)
    elif raw_json.exists():
        df = pd.read_json(raw_json)
    else:
        raise RuntimeError("No run summaries found. Run scripts/run_grid.py first.")

    df = df.copy()
    df["batching_policy"] = df["batching_policy"].astype(str).map(normalize_batching_policy)
    df["batching_family"] = df["batching_policy"].map(batching_family)
    if "test_harrell_c_index" not in df.columns and "test_c_index" in df.columns:
        df["test_harrell_c_index"] = df["test_c_index"]
    if "test_uno_c_index" not in df.columns and "test_ipcw_c_index" in df.columns:
        df["test_uno_c_index"] = df["test_ipcw_c_index"]
    if "bp" not in df.columns and "train_event_rate" in df.columns:
        df["bp"] = df["batch_size"] * df["train_event_rate"]
    if "empirical_zero_event_prob" not in df.columns:
        df["empirical_zero_event_prob"] = np.nan
    if "empirical_weak_info_prob" not in df.columns:
        df["empirical_weak_info_prob"] = np.nan
    if "requested_event_fraction" not in df.columns:
        df["requested_event_fraction"] = np.nan
    return df


def _select_family_representatives(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    keys = ["event_target", "censor_target", "batch_size", "seed", "batching_family"]
    scored = df.copy()
    scored["_best_val_rank"] = scored["best_val_c_index"].astype(float).fillna(-np.inf)
    scored["_requested_fraction_rank"] = scored["requested_event_fraction"].astype(float).fillna(-1.0)
    scored = scored.sort_values(
        keys + ["_best_val_rank", "_requested_fraction_rank", "batching_policy"],
        ascending=[True, True, True, True, True, False, False, True],
    )
    selected = scored.groupby(keys, dropna=False, sort=False).head(1).copy()
    selected = selected.drop(columns=["_best_val_rank", "_requested_fraction_rank"])
    return selected


def _attach_random_deltas(selected: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["event_target", "censor_target", "batch_size", "seed"]
    random_cols = join_cols + ["test_uno_c_index", "test_harrell_c_index", "test_brier_proxy"]
    random_df = (
        selected[selected["batching_family"] == "random"][random_cols]
        .rename(
            columns={
                "test_uno_c_index": "random_test_uno_c_index",
                "test_harrell_c_index": "random_test_harrell_c_index",
                "test_brier_proxy": "random_test_brier_proxy",
            }
        )
        .drop_duplicates(join_cols)
    )
    out = selected.merge(random_df, on=join_cols, how="left")
    out["delta_vs_random_test_uno_c_index"] = out["test_uno_c_index"] - out["random_test_uno_c_index"]
    out["delta_vs_random_test_harrell_c_index"] = out["test_harrell_c_index"] - out["random_test_harrell_c_index"]
    out["delta_vs_random_test_brier_proxy"] = out["test_brier_proxy"] - out["random_test_brier_proxy"]
    random_mask = out["batching_family"] == "random"
    out.loc[random_mask, "delta_vs_random_test_uno_c_index"] = 0.0
    out.loc[random_mask, "delta_vs_random_test_harrell_c_index"] = 0.0
    out.loc[random_mask, "delta_vs_random_test_brier_proxy"] = 0.0
    return out


def _metric_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_sem": float("nan"),
            f"{prefix}_ci95": float("nan"),
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    sem = float(std / np.sqrt(n)) if n > 1 and np.isfinite(std) else float("nan")
    ci95 = float(1.96 * sem) if np.isfinite(sem) else float("nan")
    return {
        f"{prefix}_n": n,
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_sem": sem,
        f"{prefix}_ci95": ci95,
    }


def _policy_histogram(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    if counts.empty:
        return "{}"
    return json.dumps({k: int(v) for k, v in counts.sort_index().items()}, sort_keys=True)


def _summarize_group(group: pd.DataFrame, *, scope: str, bp_regime: str) -> dict[str, object]:
    counts = group["batching_policy"].astype(str).value_counts()
    mode_policy = counts.index[0] if not counts.empty else ""
    mode_count = int(counts.iloc[0]) if not counts.empty else 0
    row: dict[str, object] = {
        "scope": scope,
        "bp_regime": bp_regime,
        "event_target": float(group["event_target"].iloc[0]) if scope == "condition" else float("nan"),
        "censor_target": float(group["censor_target"].iloc[0]) if scope == "condition" else float("nan"),
        "batch_size": int(group["batch_size"].iloc[0]) if scope == "condition" else pd.NA,
        "batching_family": str(group["batching_family"].iloc[0]),
        "n_selected_runs": int(len(group)),
        "selected_policy_mode": mode_policy,
        "selected_policy_mode_count": mode_count,
        "selected_policy_histogram": _policy_histogram(group["batching_policy"]),
        "mean_bp": float(group["bp"].mean()) if "bp" in group.columns else float("nan"),
        "mean_empirical_zero_event_prob": float(group["empirical_zero_event_prob"].mean()),
        "mean_empirical_weak_info_prob": float(group["empirical_weak_info_prob"].mean()),
    }
    for column, prefix in [
        ("best_val_c_index", "best_val_c_index"),
        ("test_uno_c_index", "test_uno_c_index"),
        ("test_harrell_c_index", "test_harrell_c_index"),
        ("test_brier_proxy", "test_brier_proxy"),
        ("delta_vs_random_test_uno_c_index", "delta_vs_random_test_uno_c_index"),
        ("delta_vs_random_test_harrell_c_index", "delta_vs_random_test_harrell_c_index"),
        ("delta_vs_random_test_brier_proxy", "delta_vs_random_test_brier_proxy"),
    ]:
        row.update(_metric_stats(group[column], prefix))
    return row


def _build_family_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()

    selected = _attach_random_deltas(selected)
    rows: list[dict[str, object]] = []

    condition_keys = ["event_target", "censor_target", "batch_size", "batching_family"]
    for _, group in selected.groupby(condition_keys, dropna=False, sort=False):
        rows.append(_summarize_group(group, scope="condition", bp_regime="all"))

    selected["bp_regime"] = np.where(selected["bp"] < 1.0, "lt1", "ge1")
    for regime in ["all", "lt1", "ge1"]:
        regime_df = selected if regime == "all" else selected[selected["bp_regime"] == regime]
        if regime_df.empty:
            continue
        for family, group in regime_df.groupby("batching_family", dropna=False, sort=False):
            if family not in FAMILY_ORDER:
                continue
            rows.append(_summarize_group(group, scope="overall", bp_regime=regime))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["family_order"] = out["batching_family"].map({name: idx for idx, name in enumerate(FAMILY_ORDER)})
    out = out.sort_values(
        ["scope", "bp_regime", "event_target", "censor_target", "batch_size", "family_order"],
        na_position="last",
    ).drop(columns=["family_order"])
    return out.reset_index(drop=True)


def _plot_family_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    overall = summary[(summary["scope"] == "overall") & (summary["bp_regime"] == "all")].copy()
    regime = summary[
        (summary["scope"] == "overall")
        & (summary["bp_regime"].isin(["lt1", "ge1"]))
        & (summary["batching_family"] != "random")
    ].copy()
    if overall.empty:
        return

    overall["batching_family"] = pd.Categorical(overall["batching_family"], categories=FAMILY_ORDER, ordered=True)
    regime["batching_family"] = pd.Categorical(
        regime["batching_family"], categories=[f for f in FAMILY_ORDER if f != "random"], ordered=True
    )
    regime["bp_regime"] = pd.Categorical(regime["bp_regime"], categories=["lt1", "ge1"], ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(
        overall["batching_family"].astype(str),
        overall["test_uno_c_index_mean"],
        yerr=overall["test_uno_c_index_ci95"].fillna(0.0),
        color=["#4c78a8", "#f58518", "#54a24b"],
        capsize=4,
    )
    axes[0].set_title("Family-Level Test Uno")
    axes[0].set_ylabel("Mean test Uno C-index")
    axes[0].set_xlabel("Batching family")
    axes[0].tick_params(axis="x", rotation=15)

    if regime.empty:
        axes[1].axis("off")
    else:
        width = 0.35
        families = [f for f in FAMILY_ORDER if f != "random"]
        x = np.arange(len(families))
        plotted = False
        for offset, bp_regime, color in [(-width / 2, "lt1", "#72b7b2"), (width / 2, "ge1", "#e45756")]:
            sub = (
                regime[regime["bp_regime"] == bp_regime]
                .set_index("batching_family")
                .reindex(families)
            )
            if sub["delta_vs_random_test_uno_c_index_mean"].isna().all():
                continue
            axes[1].bar(
                x + offset,
                sub["delta_vs_random_test_uno_c_index_mean"].fillna(0.0),
                width=width,
                yerr=sub["delta_vs_random_test_uno_c_index_ci95"].fillna(0.0),
                label=bp_regime,
                color=color,
                capsize=4,
            )
            plotted = True
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(families, rotation=15)
        axes[1].set_title("Delta vs Random")
        axes[1].set_ylabel("Mean delta test Uno C-index")
        axes[1].set_xlabel("Batching family")
        if plotted:
            axes[1].legend(title="bp regime")
        else:
            axes[1].text(0.5, 0.5, "No paired regime deltas", ha="center", va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _build_report(summary: pd.DataFrame, *, csv_path: Path, fig_path: Path) -> str:
    if summary.empty:
        return "# Family Summary\n\nNo family-level results were available.\n"

    overall = summary[(summary["scope"] == "overall") & (summary["bp_regime"] == "all")].copy()
    low_high = summary[
        (summary["scope"] == "overall")
        & (summary["bp_regime"].isin(["lt1", "ge1"]))
        & (summary["batching_family"] != "random")
    ].copy()
    overall = overall.sort_values("batching_family", key=lambda s: s.map({name: idx for idx, name in enumerate(FAMILY_ORDER)}))
    low_high = low_high.sort_values(["bp_regime", "batching_family"])

    lines = [
        "# Family Summary",
        "",
        f"- CSV: `{csv_path}`",
        f"- Figure: `{fig_path}`",
        "",
        "## Overall Mean Test Uno",
    ]
    for _, row in overall.iterrows():
        lines.append(
            "- "
            + f"{row['batching_family']}: {row['test_uno_c_index_mean']:.4f}"
            + (f" +/- {row['test_uno_c_index_ci95']:.4f}" if np.isfinite(row["test_uno_c_index_ci95"]) else "")
            + f" | selected policies {row['selected_policy_histogram']}"
        )

    lines.append("")
    lines.append("## Delta Vs Random")
    for _, row in low_high.iterrows():
        lines.append(
            "- "
            + f"{row['batching_family']} ({row['bp_regime']}): "
            + f"{row['delta_vs_random_test_uno_c_index_mean']:.4f}"
            + (f" +/- {row['delta_vs_random_test_uno_c_index_ci95']:.4f}" if np.isfinite(row["delta_vs_random_test_uno_c_index_ci95"]) else "")
        )

    lines.append("")
    lines.append("## Reading Guide")
    lines.append("- `event_quota` collapses all `event_quota_*` variants after within-family selection by validation C-index.")
    lines.append("- `riskset_anchor` collapses all `riskset_anchor_*` variants after within-family selection by validation C-index.")
    lines.append("- Condition rows in the CSV keep `event_target`, `censor_target`, and `batch_size`; overall rows summarize all selected runs by family.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
