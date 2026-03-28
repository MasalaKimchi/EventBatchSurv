from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def summarize_with_significance(
    df: pd.DataFrame,
    *,
    metric: str = "best_val_c_index",
    group_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_cols is None:
        group_cols = ["event_target", "censor_target", "batch_size", "batching_policy"]
    summary = (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_seeds", "mean": f"{metric}_mean", "std": f"{metric}_std"})
    )

    key_cols = ["event_target", "censor_target", "batch_size"]
    tests: list[dict[str, float | str]] = []
    for key, g in df.groupby(key_cols, dropna=False):
        for left, right in itertools.combinations(sorted(g["batching_policy"].unique()), 2):
            left_df = g[g["batching_policy"] == left][["seed", metric]]
            right_df = g[g["batching_policy"] == right][["seed", metric]]
            joined = left_df.merge(right_df, on="seed", suffixes=("_left", "_right"))
            if joined.empty:
                continue
            delta = joined[f"{metric}_left"] - joined[f"{metric}_right"]
            try:
                t_stat, p_val = stats.ttest_rel(joined[f"{metric}_left"], joined[f"{metric}_right"])
            except Exception:
                t_stat, p_val = np.nan, np.nan
            tests.append(
                {
                    "event_target": key[0],
                    "censor_target": key[1],
                    "batch_size": key[2],
                    "left_policy": left,
                    "right_policy": right,
                    "n_pairs": int(len(joined)),
                    "mean_delta": float(np.mean(delta)),
                    "std_delta": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                    "ttest_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                    "ttest_pvalue": float(p_val) if np.isfinite(p_val) else np.nan,
                }
            )
    return summary, pd.DataFrame(tests)
