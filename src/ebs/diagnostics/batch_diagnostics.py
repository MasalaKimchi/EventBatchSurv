from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def comparable_pairs_count(time: torch.Tensor, event: torch.Tensor) -> int:
    t = time.detach().cpu().numpy()
    e = event.detach().cpu().numpy().astype(np.int64)
    n = len(t)
    count = 0
    for i in range(n):
        if e[i] != 1:
            continue
        count += int(np.sum(t[i] < t))
    return int(count)


def grad_norm(model: torch.nn.Module) -> float:
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        sq_sum += float(torch.sum(p.grad.detach() ** 2).cpu().item())
    return float(np.sqrt(max(sq_sum, 0.0)))


@dataclass
class BatchDiagnosticSummary:
    zero_event_fraction: float
    one_event_fraction: float
    mean_events_per_batch: float
    mean_comparable_pairs: float
    grad_norm_mean: float
    grad_norm_var: float
    grad_norm_cv: float


def summarize_epoch_batch_diagnostics(
    *, event_counts: list[int], comparable_pairs: list[int], grad_norms: list[float]
) -> BatchDiagnosticSummary:
    ec = np.array(event_counts, dtype=np.float64) if event_counts else np.array([0.0])
    cp = np.array(comparable_pairs, dtype=np.float64) if comparable_pairs else np.array([0.0])
    gn = np.array(grad_norms, dtype=np.float64) if grad_norms else np.array([0.0])
    gmean = float(gn.mean())
    gvar = float(gn.var())
    return BatchDiagnosticSummary(
        zero_event_fraction=float((ec == 0).mean()),
        one_event_fraction=float((ec == 1).mean()),
        mean_events_per_batch=float(ec.mean()),
        mean_comparable_pairs=float(cp.mean()),
        grad_norm_mean=gmean,
        grad_norm_var=gvar,
        grad_norm_cv=float(np.sqrt(gvar) / (gmean + 1e-12)),
    )
