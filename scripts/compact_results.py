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

from ebs.io import ensure_dir
from ebs.policies import FAMILY_ORDER, batching_family, normalize_batching_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a compact, decision-oriented results summary.")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    compact_dir = ensure_dir(results_dir / "compact")

    run_df = _load_run_summaries(results_dir)
    selected = _select_family_representatives(run_df)
    compact = _build_compact_summary(selected)

    csv_path = compact_dir / "summary.csv"
    report_path = compact_dir / "report.md"
    compact.to_csv(csv_path, index=False)
    report_path.write_text(_build_report(compact, csv_path), encoding="utf-8")
    print(f"Wrote {csv_path} and {report_path}")


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

    out = df.copy()
    out["batching_policy"] = out["batching_policy"].astype(str).map(normalize_batching_policy)
    out["batching_family"] = out["batching_policy"].map(batching_family)
    if "test_harrell_c_index" not in out.columns and "test_c_index" in out.columns:
        out["test_harrell_c_index"] = out["test_c_index"]
    if "test_uno_c_index" not in out.columns and "test_ipcw_c_index" in out.columns:
        out["test_uno_c_index"] = out["test_ipcw_c_index"]
    if "bp" not in out.columns and "train_event_rate" in out.columns:
        out["bp"] = out["batch_size"] * out["train_event_rate"]
    for col in ["empirical_zero_event_prob", "empirical_weak_info_prob", "requested_event_fraction"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _select_family_representatives(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    keys = ["event_target", "censor_target", "batch_size", "seed", "batching_family"]
    scored = df.copy()
    scored["_best_val_rank"] = scored["best_val_c_index"].astype(float).fillna(-np.inf)
    scored["_fraction_rank"] = scored["requested_event_fraction"].astype(float).fillna(-1.0)
    scored = scored.sort_values(
        keys + ["_best_val_rank", "_fraction_rank", "batching_policy"],
        ascending=[True, True, True, True, True, False, False, True],
    )
    out = scored.groupby(keys, dropna=False, sort=False).head(1).copy()
    return out.drop(columns=["_best_val_rank", "_fraction_rank"])


def _attach_random_deltas(df: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["event_target", "censor_target", "batch_size", "seed"]
    random_cols = join_cols + ["test_uno_c_index", "test_harrell_c_index"]
    random_df = (
        df[df["batching_family"] == "random"][random_cols]
        .rename(
            columns={
                "test_uno_c_index": "random_test_uno_c_index",
                "test_harrell_c_index": "random_test_harrell_c_index",
            }
        )
        .drop_duplicates(join_cols)
    )
    out = df.merge(random_df, on=join_cols, how="left")
    out["delta_vs_random_test_uno_c_index"] = out["test_uno_c_index"] - out["random_test_uno_c_index"]
    out["delta_vs_random_test_harrell_c_index"] = out["test_harrell_c_index"] - out["random_test_harrell_c_index"]
    random_mask = out["batching_family"] == "random"
    out.loc[random_mask, "delta_vs_random_test_uno_c_index"] = 0.0
    out.loc[random_mask, "delta_vs_random_test_harrell_c_index"] = 0.0
    return out


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, float("nan")
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(arr.size)
    return mean, float(1.96 * sem)


def _positive_rate(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float((arr > 0).mean())


def _policy_mix(values: pd.Series) -> str:
    counts = values.astype(str).value_counts().sort_index()
    return json.dumps({k: int(v) for k, v in counts.items()}, sort_keys=True)


def _summarize_group(group: pd.DataFrame, *, scope: str, bp_regime: str) -> dict[str, object]:
    test_uno_mean, test_uno_ci95 = _mean_ci(group["test_uno_c_index"])
    delta_uno_mean, delta_uno_ci95 = _mean_ci(group["delta_vs_random_test_uno_c_index"])
    test_h_mean, _ = _mean_ci(group["test_harrell_c_index"])
    delta_h_mean, _ = _mean_ci(group["delta_vs_random_test_harrell_c_index"])
    policy_counts = group["batching_policy"].astype(str).value_counts()
    representative = policy_counts.index[0] if not policy_counts.empty else ""
    return {
        "scope": scope,
        "bp_regime": bp_regime,
        "event_target": float(group["event_target"].iloc[0]) if scope == "condition" else float("nan"),
        "censor_target": float(group["censor_target"].iloc[0]) if scope == "condition" else float("nan"),
        "batch_size": int(group["batch_size"].iloc[0]) if scope == "condition" else pd.NA,
        "batching_family": str(group["batching_family"].iloc[0]),
        "representative_policy": representative,
        "policy_mix": _policy_mix(group["batching_policy"]),
        "n_runs": int(len(group)),
        "mean_bp": float(group["bp"].mean()) if "bp" in group.columns else float("nan"),
        "test_uno_mean": test_uno_mean,
        "test_uno_ci95": test_uno_ci95,
        "delta_vs_random_test_uno_mean": delta_uno_mean,
        "delta_vs_random_test_uno_ci95": delta_uno_ci95,
        "test_harrell_mean": test_h_mean,
        "delta_vs_random_test_harrell_mean": delta_h_mean,
        "zero_event_prob_mean": float(group["empirical_zero_event_prob"].mean()),
        "weak_info_prob_mean": float(group["empirical_weak_info_prob"].mean()),
        "better_than_random_rate": _positive_rate(group["delta_vs_random_test_uno_c_index"]),
    }


def _build_compact_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()

    selected = _attach_random_deltas(selected)
    rows: list[dict[str, object]] = []

    for _, group in selected.groupby(["event_target", "censor_target", "batch_size", "batching_family"], dropna=False):
        rows.append(_summarize_group(group, scope="condition", bp_regime="all"))

    selected["bp_regime"] = np.where(selected["bp"] < 1.0, "lt1", "ge1")
    for regime in ["all", "lt1", "ge1"]:
        regime_df = selected if regime == "all" else selected[selected["bp_regime"] == regime]
        if regime_df.empty:
            continue
        for _, group in regime_df.groupby("batching_family", dropna=False):
            rows.append(_summarize_group(group, scope="overall", bp_regime=regime))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    family_rank = {name: idx for idx, name in enumerate(FAMILY_ORDER)}
    out["family_rank"] = out["batching_family"].map(family_rank)
    out = out.sort_values(
        ["scope", "bp_regime", "event_target", "censor_target", "batch_size", "family_rank"],
        na_position="last",
    )
    return out.drop(columns=["family_rank"]).reset_index(drop=True)


def _extreme_condition_line(compact: pd.DataFrame, family: str, *, reverse: bool) -> str | None:
    sub = compact[(compact["scope"] == "condition") & (compact["batching_family"] == family)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("delta_vs_random_test_uno_mean", ascending=not reverse)
    row = sub.iloc[0]
    return (
        f"{family} e={row['event_target']:.2f}, c={row['censor_target']:.2f}, "
        f"b={int(row['batch_size'])}: {row['delta_vs_random_test_uno_mean']:+.4f}"
    )


def _format_overall_line(overall: pd.DataFrame) -> str:
    parts: list[str] = []
    for _, row in overall.iterrows():
        parts.append(
            f"{row['batching_family']} {row['test_uno_mean']:.4f} "
            f"({row['delta_vs_random_test_uno_mean']:+.4f} vs random, zero-event {row['zero_event_prob_mean']:.3f})"
        )
    return "; ".join(parts)


def _format_regime_line(compact: pd.DataFrame, regime: str) -> str | None:
    regime_df = compact[(compact["scope"] == "overall") & (compact["bp_regime"] == regime)].copy()
    if regime_df.empty:
        return None
    regime_df = regime_df.sort_values("batching_family", key=lambda s: s.map({k: i for i, k in enumerate(FAMILY_ORDER)}))
    parts: list[str] = []
    for _, row in regime_df.iterrows():
        if row["batching_family"] == "random":
            continue
        parts.append(f"{row['batching_family']} {row['delta_vs_random_test_uno_mean']:+.4f}")
    if not parts:
        return None
    label = "`bp < 1`" if regime == "lt1" else "`bp >= 1`"
    return f"{label}: " + "; ".join(parts)


def _build_report(compact: pd.DataFrame, csv_path: Path) -> str:
    if compact.empty:
        return "# Compact Results\n\nNo runs were available.\n"

    overall = compact[(compact["scope"] == "overall") & (compact["bp_regime"] == "all")].copy()
    overall = overall.sort_values("test_uno_mean", ascending=False)
    lines = ["# Compact Results", "", f"- CSV: `{csv_path}`"]
    lines.append(f"- Overall: {_format_overall_line(overall)}")

    for regime in ["lt1", "ge1"]:
        regime_line = _format_regime_line(compact, regime)
        if regime_line is not None:
            lines.append(f"- {regime_line}")

    best_parts = [
        line
        for family in [f for f in FAMILY_ORDER if f != "random"]
        for line in [_extreme_condition_line(compact, family, reverse=True)]
        if line is not None
    ]
    if best_parts:
        lines.append("- Best corners: " + "; ".join(best_parts))

    worst_parts = [
        line
        for family in [f for f in FAMILY_ORDER if f != "random"]
        for line in [_extreme_condition_line(compact, family, reverse=False)]
        if line is not None
    ]
    if worst_parts:
        lines.append("- Worst corners: " + "; ".join(worst_parts))

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
