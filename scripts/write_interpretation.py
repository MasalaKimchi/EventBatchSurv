#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ebs.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-generate interpretation markdown.")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    agg = pd.read_csv(results_dir / "aggregates" / "run_summaries_enriched.csv")
    paired = pd.read_csv(results_dir / "aggregates" / "paired_tests.csv")
    out_file = ensure_dir(results_dir) / "interpretation.md"
    out_file.write_text(_build_report(agg, paired), encoding="utf-8")
    print(f"Wrote {out_file}")


def _build_report(df: pd.DataFrame, paired: pd.DataFrame) -> str:
    mean_by_policy = df.groupby("batching_policy")["best_val_c_index"].mean().to_dict()
    low_bp = df[df["bp"] < 1.0]
    high_bp = df[df["bp"] >= 1.0]

    def policy_gain(sub: pd.DataFrame, policy: str = "event_aware_min1") -> float:
        if sub.empty:
            return float("nan")
        p = (
            sub.groupby(["event_target", "censor_target", "batch_size", "seed", "batching_policy"])[
                "best_val_c_index"
            ]
            .mean()
            .reset_index()
            .pivot_table(
                index=["event_target", "censor_target", "batch_size", "seed"],
                columns="batching_policy",
                values="best_val_c_index",
            )
            .dropna()
        )
        if "random" not in p.columns or policy not in p.columns:
            return float("nan")
        return float((p[policy] - p["random"]).mean())

    gain_low = policy_gain(low_bp, "event_aware_min1")
    gain_high = policy_gain(high_bp, "event_aware_min1")
    zero_gap = float((df["empirical_zero_event_prob"] - df["theoretical_zero_event_prob"]).abs().mean())
    weak_gap = float((df["empirical_weak_info_prob"] - df["theoretical_weak_info_prob"]).abs().mean())

    grad_var = (
        df.groupby("batching_policy")["final_grad_norm_var"].mean().sort_values().to_dict()
        if "final_grad_norm_var" in df.columns
        else {}
    )

    sig_share = float((paired["ttest_pvalue"] < 0.05).mean()) if not paired.empty else float("nan")
    return f"""# EventBatchSurv Interpretation

## High-level outcome
- Mean best validation c-index by policy: {mean_by_policy}
- Event-aware gain (min1 vs random) when `bp < 1`: {gain_low:.4f}
- Event-aware gain (min1 vs random) when `bp >= 1`: {gain_high:.4f}

## Theory checks
- Average absolute gap between empirical and theoretical zero-event frequency: {zero_gap:.4f}
- Average absolute gap between empirical and theoretical weakly informative frequency (`E<=1`): {weak_gap:.4f}
- If these gaps are small, the observed zero-event degeneracy tracks the binomial approximation `(1-p)^b`.

## Optimization stability
- Mean gradient-norm variance by policy: {grad_var}
- Lower gradient variance with event-aware batching indicates improved optimization stability under scarce events.

## Statistical evidence
- Fraction of paired policy comparisons with `p < 0.05`: {sig_share:.3f}
- Inspect `results/aggregates/paired_tests.csv` for condition-specific significance and effect sizes.

## Bias-variance tradeoff interpretation
- Positive low-`bp` gains with reduced weakly informative batches support the claim that event-enriched sampling helps when batches are under-informative.
- Diminishing or negative gains at higher `bp` suggest a potential bias-variance tradeoff from event enrichment when random batches are already informative.
"""


if __name__ == "__main__":
    main()
