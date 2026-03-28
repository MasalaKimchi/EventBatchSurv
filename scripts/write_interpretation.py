#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ebs.io import ensure_dir
from ebs.policies import normalize_batching_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-generate interpretation markdown.")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    agg = pd.read_csv(results_dir / "aggregates" / "run_summaries_enriched.csv")
    paired_path = results_dir / "aggregates" / "paired_tests_test_uno_enhanced.csv"
    try:
        paired = pd.read_csv(paired_path)
    except EmptyDataError:
        paired = pd.DataFrame()
    out_file = ensure_dir(results_dir) / "interpretation.md"
    out_file.write_text(_build_report(agg, paired), encoding="utf-8")
    print(f"Wrote {out_file}")


def _build_report(df: pd.DataFrame, paired: pd.DataFrame) -> str:
    if "batching_policy" in df.columns:
        df["batching_policy"] = df["batching_policy"].astype(str).map(normalize_batching_policy)
    if "test_harrell_c_index" not in df.columns and "test_c_index" in df.columns:
        df["test_harrell_c_index"] = df["test_c_index"]
    if "test_uno_c_index" not in df.columns and "test_ipcw_c_index" in df.columns:
        df["test_uno_c_index"] = df["test_ipcw_c_index"]

    mean_by_policy = df.groupby("batching_policy")["test_uno_c_index"].mean().to_dict()
    mean_harrell_by_policy = df.groupby("batching_policy")["test_harrell_c_index"].mean().to_dict()
    low_bp = df[df["bp"] < 1.0]
    high_bp = df[df["bp"] >= 1.0]
    primary_policy = _primary_policy(df)

    def policy_gain(sub: pd.DataFrame, policy: str = "event_quota_wor_25") -> float:
        if sub.empty:
            return float("nan")
        p = (
            sub.groupby(["event_target", "censor_target", "batch_size", "seed", "batching_policy"])[
                "test_uno_c_index"
            ]
            .mean()
            .reset_index()
            .pivot_table(
                index=["event_target", "censor_target", "batch_size", "seed"],
                columns="batching_policy",
                values="test_uno_c_index",
            )
            .dropna()
        )
        if "random" not in p.columns or policy not in p.columns:
            return float("nan")
        return float((p[policy] - p["random"]).mean())

    gain_low = policy_gain(low_bp, primary_policy) if primary_policy is not None else float("nan")
    gain_high = policy_gain(high_bp, primary_policy) if primary_policy is not None else float("nan")
    zero_gap = float((df["empirical_zero_event_prob"] - df["theoretical_zero_event_prob"]).abs().mean())
    weak_gap = float((df["empirical_weak_info_prob"] - df["theoretical_weak_info_prob"]).abs().mean())

    grad_var = (
        df.groupby("batching_policy")["final_grad_norm_var"].mean().sort_values().to_dict()
        if "final_grad_norm_var" in df.columns
        else {}
    )

    sig_share = _safe_boolean_mean(paired, "significant_q05")
    practical_share = _safe_boolean_mean(paired, "practically_significant")
    wilcoxon_sig_share = _safe_boolean_mean(paired, "significant_wilcoxon_q05")
    calib_gain = (
        _paired_mean_gain(df, metric="test_brier_proxy", policy=primary_policy)
        if primary_policy is not None
        else float("nan")
    )
    uno_harrell_gap_high_censor = float(
        (df[df["censor_target"] >= 0.5]["test_uno_c_index"] - df[df["censor_target"] >= 0.5]["test_harrell_c_index"]).mean()
    )
    feasible_breakout = (
        df.groupby(["batching_policy", "sampler_feasible"], dropna=False)["test_uno_c_index"].mean().to_dict()
    )
    return f"""# EventBatchSurv Interpretation

## High-level outcome
- Mean held-out test Uno c-index by policy: {mean_by_policy}
- Mean held-out test Harrell c-index by policy: {mean_harrell_by_policy}
- Primary policy for focused comparisons vs random: {primary_policy}
- Primary-policy gain (vs random) when `bp < 1`: {gain_low:.4f}
- Primary-policy gain (vs random) when `bp >= 1`: {gain_high:.4f}
- Calibration proxy shift (test Brier proxy, primary vs random): {calib_gain:.4f} (negative is better)
- Uno minus Harrell gap at high censoring (`censor_target >= 0.5`): {uno_harrell_gap_high_censor:.4f}

## Theory checks
- Average absolute gap between empirical and theoretical zero-event frequency: {zero_gap:.4f}
- Average absolute gap between empirical and theoretical weakly informative frequency (`E<=1`): {weak_gap:.4f}
- If these gaps are small, the observed zero-event degeneracy tracks the binomial approximation `(1-p)^b`.

## Optimization stability
- Mean gradient-norm variance by policy: {grad_var}
- Lower gradient variance with event-aware batching indicates improved optimization stability under scarce events.

## Statistical evidence
- Fraction of paired policy comparisons with BH-corrected `q < 0.05`: {sig_share:.3f}
- Wilcoxon sensitivity check share with BH-corrected `q < 0.05`: {wilcoxon_sig_share:.3f}
- Fraction of comparisons above practical threshold (`|delta| >= 0.01`): {practical_share:.3f}
- Inspect `results/aggregates/paired_tests_test_uno_enhanced.csv` and `paired_tests_test_brier_enhanced.csv` for condition-specific effect sizes and corrected q-values.

## Feasibility accounting
- Test Uno means by policy and `sampler_feasible`: {feasible_breakout}
- Compare `results/tables/table_feasibility_test_uno.csv` to avoid mixing feasible/infeasible regimes for strict event constraints.

## Bias-variance tradeoff interpretation
- Positive low-`bp` gains with reduced weakly informative batches support the claim that event-enriched sampling helps when batches are under-informative.
- Diminishing or negative gains at higher `bp` suggest a potential bias-variance tradeoff from event enrichment when random batches are already informative.
"""


def _paired_mean_gain(df: pd.DataFrame, *, metric: str, policy: str) -> float:
    p = (
        df[df["batching_policy"].isin(["random", policy])]
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


def _safe_boolean_mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return float("nan")
    return float((df[column] == True).mean())


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
