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


def summarize_with_paired_effects(
    df: pd.DataFrame,
    *,
    metric: str,
    group_cols: list[str] | None = None,
    key_cols: list[str] | None = None,
    practical_delta: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_cols is None:
        group_cols = ["event_target", "censor_target", "batch_size", "batching_policy"]
    if key_cols is None:
        key_cols = ["event_target", "censor_target", "batch_size"]

    summary = (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_seeds", "mean": f"{metric}_mean", "std": f"{metric}_std"})
    )
    summary[f"{metric}_sem"] = summary[f"{metric}_std"] / np.sqrt(summary["n_seeds"].clip(lower=1))
    summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_sem"]
    summary[f"{metric}_ci95_lo"] = summary[f"{metric}_mean"] - summary[f"{metric}_ci95"]
    summary[f"{metric}_ci95_hi"] = summary[f"{metric}_mean"] + summary[f"{metric}_ci95"]

    tests: list[dict[str, float | str | int | bool]] = []
    for key, g in df.groupby(key_cols, dropna=False):
        for left, right in itertools.combinations(sorted(g["batching_policy"].unique()), 2):
            left_df = g[g["batching_policy"] == left][["seed", metric]]
            right_df = g[g["batching_policy"] == right][["seed", metric]]
            joined = left_df.merge(right_df, on="seed", suffixes=("_left", "_right"))
            if joined.empty:
                continue

            left_vals = joined[f"{metric}_left"].to_numpy(dtype=np.float64)
            right_vals = joined[f"{metric}_right"].to_numpy(dtype=np.float64)
            delta = left_vals - right_vals
            n_pairs = int(len(delta))
            mean_delta = float(np.mean(delta))
            std_delta = float(np.std(delta, ddof=1)) if n_pairs > 1 else float("nan")
            sem_delta = float(std_delta / np.sqrt(n_pairs)) if n_pairs > 1 and np.isfinite(std_delta) else float("nan")
            ci95 = float(1.96 * sem_delta) if np.isfinite(sem_delta) else float("nan")
            dz = float(mean_delta / std_delta) if n_pairs > 1 and np.isfinite(std_delta) and std_delta > 0 else float("nan")

            try:
                t_stat, p_val = stats.ttest_rel(left_vals, right_vals)
            except Exception:
                t_stat, p_val = np.nan, np.nan
            try:
                w_stat, w_pval = stats.wilcoxon(left_vals, right_vals, zero_method="wilcox")
            except Exception:
                w_stat, w_pval = np.nan, np.nan

            row: dict[str, float | str | int | bool] = {
                "left_policy": left,
                "right_policy": right,
                "n_pairs": n_pairs,
                "mean_delta": mean_delta,
                "std_delta": std_delta,
                "sem_delta": sem_delta,
                "ci95_delta": ci95,
                "ci95_delta_lo": mean_delta - ci95 if np.isfinite(ci95) else float("nan"),
                "ci95_delta_hi": mean_delta + ci95 if np.isfinite(ci95) else float("nan"),
                "cohen_dz": dz,
                "ttest_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "ttest_pvalue": float(p_val) if np.isfinite(p_val) else np.nan,
                "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "wilcoxon_pvalue": float(w_pval) if np.isfinite(w_pval) else np.nan,
                "practical_delta_threshold": float(practical_delta),
                "practically_significant": bool(abs(mean_delta) >= practical_delta),
            }
            for idx, col in enumerate(key_cols):
                row[col] = key[idx]
            tests.append(row)

    tests_df = pd.DataFrame(tests)
    if tests_df.empty:
        return summary, tests_df

    tests_df["ttest_qvalue_bh"] = _benjamini_hochberg(tests_df["ttest_pvalue"].to_numpy(dtype=np.float64))
    tests_df["wilcoxon_qvalue_bh"] = _benjamini_hochberg(tests_df["wilcoxon_pvalue"].to_numpy(dtype=np.float64))
    tests_df["significant_p05"] = tests_df["ttest_pvalue"] < 0.05
    tests_df["significant_q05"] = tests_df["ttest_qvalue_bh"] < 0.05
    tests_df["significant_wilcoxon_q05"] = tests_df["wilcoxon_qvalue_bh"] < 0.05
    return summary, tests_df


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    qvals = np.full_like(pvals, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(pvals)
    if not finite_mask.any():
        return qvals

    finite = pvals[finite_mask]
    order = np.argsort(finite)
    ranked = finite[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adjusted[i] = prev
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(m, dtype=np.float64)
    restored[order] = adjusted
    qvals[finite_mask] = restored
    return qvals
