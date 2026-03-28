from __future__ import annotations

import numpy as np
import torch
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex


def cox_partial_log_likelihood_loss(
    risk_scores: torch.Tensor, time: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    return neg_partial_log_likelihood(
        log_hz=risk_scores,
        event=event.bool(),
        time=time.float(),
        ties_method="efron",
        reduction="mean",
        checks=True,
    )


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    metric = ConcordanceIndex()
    try:
        return float(
            metric(
                estimate=torch.as_tensor(risk, dtype=torch.float32),
                event=torch.as_tensor(event, dtype=torch.bool),
                time=torch.as_tensor(time, dtype=torch.float32),
            ).item()
        )
    except Exception:
        return float("nan")


def uno_c_index_if_available(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    try:
        from sksurv.metrics import concordance_index_ipcw
        from sksurv.util import Surv
    except Exception:
        return float("nan")
    y = Surv.from_arrays(event.astype(bool), time.astype(float))
    try:
        value = concordance_index_ipcw(y, y, risk.astype(float))[0]
        return float(value)
    except Exception:
        return float("nan")


def brier_at_median_time(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    tau = float(np.median(time))
    if not np.isfinite(tau):
        return float("nan")

    time_t = torch.as_tensor(time, dtype=torch.float32)
    risk_t = torch.as_tensor(risk, dtype=torch.float32)
    event_t = torch.as_tensor(event, dtype=torch.bool)

    # Convert Cox risk score to survival probabilities with an exponential PH approximation.
    # This is only used to feed a calibration-sensitive metric.
    mean_rate = torch.reciprocal(time_t.clamp_min(1e-6)).mean()
    rate = torch.exp(risk_t - risk_t.mean()) * mean_rate
    new_time = torch.as_tensor([tau], dtype=torch.float32)
    surv_prob = torch.exp(-(rate[:, None] * new_time[None, :])).clamp(0.0, 1.0)

    try:
        bs = BrierScore()
        val = bs(
            estimate=surv_prob,
            event=event_t,
            time=time_t,
            new_time=new_time,
            instate=False,
        )
        return float(val.squeeze().item())
    except Exception:
        return float("nan")
