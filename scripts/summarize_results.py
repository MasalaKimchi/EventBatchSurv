#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ebs.analysis import summarize_with_paired_effects
from ebs.io import ensure_dir
from ebs.policies import normalize_batching_policy


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

    merged = _require_embedded_columns(run_summaries)
    merged = _ensure_test_metric_columns(merged)
    merged["bp"] = merged["batch_size"] * merged["train_event_rate"]
    merged.to_csv(agg_dir / "run_summaries_enriched.csv", index=False)

    _write_enhanced_tests(merged, agg_dir)
    _write_main_table(merged, table_dir / "table_main.csv")
    _write_theory_table(merged, table_dir / "table_theory.csv")
    _write_feasibility_tables(merged, table_dir)
    _write_high_value_targets(merged, table_dir / "table_high_value_targets.csv")
    print(f"Wrote aggregates to {agg_dir} and tables to {table_dir}")


def _load_run_summaries(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "aggregates" / "run_summaries.csv"
    json_path = results_dir / "aggregates" / "run_summaries.json"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        return pd.read_json(json_path)
    return pd.DataFrame()


def _require_embedded_columns(run_summaries: pd.DataFrame) -> pd.DataFrame:
    required = {
        "empirical_zero_event_prob",
        "empirical_weak_info_prob",
        "theoretical_zero_event_prob",
        "theoretical_weak_info_prob",
        "train_event_rate",
        "sampler_feasible",
    }
    missing = sorted(required.difference(set(run_summaries.columns)))
    if missing:
        raise RuntimeError(
            "Run summaries are missing embedded diagnostics: "
            + ", ".join(missing)
            + ". Re-run experiments with the current training engine."
        )
    return run_summaries.copy()


def _ensure_test_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "batching_policy" in out.columns:
        out["batching_policy"] = out["batching_policy"].astype(str).map(normalize_batching_policy)
    if "test_harrell_c_index" not in out.columns and "test_c_index" in out.columns:
        out["test_harrell_c_index"] = out["test_c_index"]
    if "test_uno_c_index" not in out.columns and "test_ipcw_c_index" in out.columns:
        out["test_uno_c_index"] = out["test_ipcw_c_index"]
    required = {"test_harrell_c_index", "test_uno_c_index", "test_brier_proxy"}
    missing = [c for c in sorted(required) if c not in out.columns]
    if missing:
        raise RuntimeError(
            "Missing test metrics in run summaries: "
            + ", ".join(missing)
            + ". Re-run experiments with updated training engine."
        )
    return out


def _write_main_table(df: pd.DataFrame, out_file: Path) -> None:
    cols = ["event_target", "censor_target", "batch_size", "batching_policy"]
    agg = (
        df.groupby(cols)[["test_harrell_c_index", "test_uno_c_index", "best_val_c_index"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["_".join(str(c) for c in col if c) for col in agg.columns]
    for metric in ["test_harrell_c_index", "test_uno_c_index", "best_val_c_index"]:
        n_col = f"{metric}_count"
        std_col = f"{metric}_std"
        sem_col = f"{metric}_sem"
        ci_col = f"{metric}_ci95"
        agg[sem_col] = agg[std_col] / agg[n_col].clip(lower=1).pow(0.5)
        agg[ci_col] = 1.96 * agg[sem_col]
        agg[f"{metric}_ci95_lo"] = agg[f"{metric}_mean"] - agg[ci_col]
        agg[f"{metric}_ci95_hi"] = agg[f"{metric}_mean"] + agg[ci_col]
    agg = agg.rename(
        columns={
            "test_harrell_c_index_count": "n_seeds_harrell",
            "test_uno_c_index_count": "n_seeds_uno",
            "best_val_c_index_count": "n_seeds_val",
        }
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


def _write_enhanced_tests(df: pd.DataFrame, agg_dir: Path) -> None:
    harrell_summary, harrell_tests = summarize_with_paired_effects(
        df,
        metric="test_harrell_c_index",
        practical_delta=0.01,
    )
    harrell_tests = _ensure_enhanced_test_columns(harrell_tests)
    harrell_summary.to_csv(agg_dir / "condition_summary_test_harrell_enhanced.csv", index=False)
    harrell_tests.to_csv(agg_dir / "paired_tests_test_harrell_enhanced.csv", index=False)

    uno_summary, uno_tests = summarize_with_paired_effects(
        df,
        metric="test_uno_c_index",
        practical_delta=0.01,
    )
    uno_tests = _ensure_enhanced_test_columns(uno_tests)
    uno_summary.to_csv(agg_dir / "condition_summary_test_uno_enhanced.csv", index=False)
    uno_tests.to_csv(agg_dir / "paired_tests_test_uno_enhanced.csv", index=False)

    brier_summary, brier_tests = summarize_with_paired_effects(
        df,
        metric="test_brier_proxy",
        practical_delta=0.01,
    )
    brier_tests = _ensure_enhanced_test_columns(brier_tests)
    brier_summary.to_csv(agg_dir / "condition_summary_test_brier_enhanced.csv", index=False)
    brier_tests.to_csv(agg_dir / "paired_tests_test_brier_enhanced.csv", index=False)


def _write_feasibility_tables(df: pd.DataFrame, table_dir: Path) -> None:
    cols = ["event_target", "censor_target", "batch_size", "batching_policy"]
    feasible_summary = (
        df.groupby(cols + ["sampler_feasible"], dropna=False)["test_uno_c_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "test_uno_c_index_mean", "std": "test_uno_c_index_std", "count": "n"})
    )
    feasible_summary.to_csv(table_dir / "table_feasibility_test_uno.csv", index=False)


def _write_high_value_targets(df: pd.DataFrame, out_file: Path) -> None:
    primary = _primary_policy(df)
    rows: list[dict[str, float | str]] = []
    if primary is not None:
        rows.append(
            {
                "target": f"{primary}_gain_when_bp_lt_1_test_uno",
                "value": _policy_gain_by_bp(df, metric="test_uno_c_index", policy=primary, bp_lt=True),
                "note": "Primary low-information regime signal; positive is desirable.",
            }
        )
        rows.append(
            {
                "target": f"{primary}_gain_when_bp_ge_1_test_uno",
                "value": _policy_gain_by_bp(df, metric="test_uno_c_index", policy=primary, bp_lt=False),
                "note": "Tradeoff regime check; near-zero or negative can indicate diminishing returns.",
            }
        )
    rows.append(
        {
            "target": "uno_minus_harrell_mean_gap_high_censoring",
            "value": _uno_harrell_gap(df, censor_threshold=0.5),
            "note": "Reliability divergence under heavy censoring (Uno expected more stable).",
        }
    )
    rows.append(
        {
            "target": "mean_abs_val_minus_test_uno_gap",
            "value": float((df["best_val_c_index"] - df["test_uno_c_index"]).abs().mean()),
            "note": "Generalization gap diagnostic.",
        }
    )
    for policy in sorted(df["batching_policy"].astype(str).unique()):
        if policy == "random":
            continue
        rows.append(
            {
                "target": f"{policy}_feasible_rate",
                "value": _feasible_rate(df, policy=policy),
                "note": "Feasibility rate for this policy under current event prevalence and batch-size settings.",
            }
        )
    pd.DataFrame(rows).to_csv(out_file, index=False)


def _policy_gain_by_bp(df: pd.DataFrame, *, metric: str, policy: str, bp_lt: bool) -> float:
    sub = df[df["bp"] < 1.0].copy() if bp_lt else df[df["bp"] >= 1.0].copy()
    if sub.empty:
        return float("nan")
    p = (
        sub[sub["batching_policy"].isin(["random", policy])]
        .pivot_table(
            index=["event_target", "censor_target", "batch_size", "seed"],
            columns="batching_policy",
            values=metric,
        )
        .dropna()
    )
    if "random" not in p.columns or policy not in p.columns:
        return float("nan")
    return float((p[policy] - p["random"]).mean())


def _uno_harrell_gap(df: pd.DataFrame, *, censor_threshold: float) -> float:
    sub = df[df["censor_target"] >= censor_threshold]
    if sub.empty:
        return float("nan")
    return float((sub["test_uno_c_index"] - sub["test_harrell_c_index"]).mean())


def _feasible_rate(df: pd.DataFrame, *, policy: str) -> float:
    sub = df[df["batching_policy"] == policy]
    if sub.empty or "sampler_feasible" not in sub.columns:
        return float("nan")
    return float(sub["sampler_feasible"].astype(float).mean())


def _primary_policy(df: pd.DataFrame) -> str | None:
    preferred = [
        "riskset_anchor_25",
        "event_quota_wr_25",
        "event_quota_wor_25",
    ]
    existing = set(df["batching_policy"].astype(str).unique())
    for p in preferred:
        if p in existing:
            return p
    candidates = [p for p in sorted(existing) if p != "random"]
    return candidates[0] if candidates else None


def _ensure_enhanced_test_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "event_target",
        "censor_target",
        "batch_size",
        "left_policy",
        "right_policy",
        "n_pairs",
        "mean_delta",
        "std_delta",
        "sem_delta",
        "ci95_delta",
        "ci95_delta_lo",
        "ci95_delta_hi",
        "cohen_dz",
        "ttest_stat",
        "ttest_pvalue",
        "wilcoxon_stat",
        "wilcoxon_pvalue",
        "ttest_qvalue_bh",
        "wilcoxon_qvalue_bh",
        "significant_p05",
        "significant_q05",
        "significant_wilcoxon_q05",
        "practical_delta_threshold",
        "practically_significant",
    ]
    if df.empty:
        return pd.DataFrame(columns=expected)
    out = df.copy()
    for col in expected:
        if col not in out.columns:
            out[col] = float("nan")
    return out[expected]


if __name__ == "__main__":
    main()
