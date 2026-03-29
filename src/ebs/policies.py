from __future__ import annotations

import re
from dataclasses import dataclass

FAMILY_ORDER = ["random", "event_quota", "riskset_anchor"]


def normalize_batching_policy(policy: str) -> str:
    key = str(policy).strip()
    if key == "random":
        return "random"
    m = re.fullmatch(r"event_quota_(wor|wr)_(\d{1,3})", key)
    if m:
        pct = int(m.group(2))
        if pct < 0 or pct > 100:
            raise ValueError(f"Invalid quota percentage in policy: {policy}")
        return f"event_quota_{m.group(1)}_{pct}"
    m = re.fullmatch(r"riskset_(?:anchor|quota)_(\d{1,3})", key)
    if m:
        pct = int(m.group(1))
        if pct < 0 or pct > 100:
            raise ValueError(f"Invalid quota percentage in policy: {policy}")
        return f"riskset_anchor_{pct}"
    raise ValueError(
        f"Unknown batching policy: {policy}. "
        "Known policies: random, event_quota_wor_<pct>, event_quota_wr_<pct>, riskset_anchor_<pct>."
    )


@dataclass(frozen=True)
class PolicySpec:
    name: str
    mode: str
    event_fraction: float
    with_replacement: bool
    min_events_per_batch: int
    strict_feasible: bool


def parse_batching_policy(policy: str) -> PolicySpec:
    name = normalize_batching_policy(policy)
    if name == "random":
        return PolicySpec(
            name=name,
            mode="random",
            event_fraction=0.0,
            with_replacement=False,
            min_events_per_batch=0,
            strict_feasible=False,
        )
    m = re.fullmatch(r"event_quota_(wor|wr)_(\d{1,3})", name)
    if m:
        frac = int(m.group(2)) / 100.0
        return PolicySpec(
            name=name,
            mode="quota",
            event_fraction=frac,
            with_replacement=(m.group(1) == "wr"),
            min_events_per_batch=0,
            strict_feasible=(m.group(1) == "wor"),
        )
    m = re.fullmatch(r"riskset_anchor_(\d{1,3})", name)
    if m:
        frac = int(m.group(1)) / 100.0
        return PolicySpec(
            name=name,
            mode="riskset_anchor",
            event_fraction=frac,
            with_replacement=True,
            min_events_per_batch=0,
            strict_feasible=True,
        )
    raise ValueError(f"Unknown batching policy: {policy}")


def batching_family(policy: str) -> str:
    name = normalize_batching_policy(policy)
    if name == "random":
        return "random"
    if name.startswith("event_quota_"):
        return "event_quota"
    if name.startswith("riskset_anchor_"):
        return "riskset_anchor"
    raise ValueError(f"Unknown batching policy family: {policy}")
