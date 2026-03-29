from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ebs.config import ExperimentConfig
from ebs.data import SurvivalTensorDataset
from ebs.diagnostics import comparable_pairs_count, grad_norm, summarize_epoch_batch_diagnostics
from ebs.io import append_jsonl, ensure_dir
from ebs.metrics import brier_at_median_time, cox_partial_log_likelihood_loss, harrell_c_index, ipcw_c_index
from ebs.model import CoxMLP, CoxResNet18
from ebs.policies import normalize_batching_policy, parse_batching_policy
from ebs.samplers import EventQuotaBatchSampler, RandomBatchSampler, RiskSetAnchorBatchSampler


@dataclass
class RunContext:
    event_target: float
    censor_target: float
    batch_size: int
    batching_policy: str
    seed: int
    output_dir: Path


def run_training(
    *,
    cfg: ExperimentConfig,
    context: RunContext,
    x: np.ndarray,
    time_obs: np.ndarray,
    event_obs: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float | int | bool | str]:
    torch.manual_seed(context.seed)
    np.random.seed(context.seed)
    normalized_policy = normalize_batching_policy(context.batching_policy)

    dataset = SurvivalTensorDataset(x=x, time=time_obs, event=event_obs)
    train_subset = Subset(dataset, indices=train_idx.tolist())
    val_subset = Subset(dataset, indices=val_idx.tolist())
    test_subset = Subset(dataset, indices=test_idx.tolist())
    train_events = event_obs[train_idx]
    # Batch samplers for a Subset must emit local positions [0..len(train_subset)-1].
    train_local_idx = np.arange(train_idx.shape[0], dtype=np.int64)
    sampler = _build_train_sampler(
        policy=normalized_policy,
        indices=train_local_idx,
        events=np.array(train_events, dtype=np.int64),
        times=np.array(time_obs[train_idx], dtype=np.float64),
        batch_size=context.batch_size,
        seed=context.seed,
    )
    train_loader = DataLoader(train_subset, batch_sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=1024, shuffle=False)

    device = torch.device(cfg.train.device)
    backbone = cfg.model.backbone.lower()
    model = _build_model(cfg=cfg, x=x).to(device)
    optimizer = _build_optimizer(cfg=cfg, model=model)
    scheduler = _build_scheduler(cfg=cfg, optimizer=optimizer)

    write_any_artifacts = (
        cfg.train.save_epoch_logs or cfg.train.save_batch_logs
    )
    run_dir = ensure_dir(context.output_dir) if write_any_artifacts else context.output_dir
    epoch_log_path = run_dir / "epoch_logs.jsonl"
    batch_log_path = run_dir / "batch_logs.jsonl"

    if not cfg.train.save_epoch_logs and epoch_log_path.exists():
        epoch_log_path.unlink()
    if not cfg.train.save_batch_logs and batch_log_path.exists():
        batch_log_path.unlink()

    best_val = -np.inf
    best_epoch = -1
    best_time_s = np.nan
    best_state_dict: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    per_epoch_rows: list[dict[str, float | int | str]] = []
    total_batches = 0
    zero_event_batches = 0
    weak_info_batches = 0
    run_meta = _run_meta_dict(context, train_events, sampler)

    for epoch in range(cfg.train.epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        batch_losses: list[float] = []
        event_counts: list[int] = []
        one_batch_comparable_pairs: list[int] = []
        grad_norms: list[float] = []
        epoch_start = time.perf_counter()
        for batch_idx, (bx, bt, be) in enumerate(train_loader):
            bx = _prepare_features(bx=bx.to(device), backbone=backbone)
            bt = bt.to(device)
            be = be.to(device)
            optimizer.zero_grad(set_to_none=True)
            risk = model(bx)
            loss = cox_partial_log_likelihood_loss(risk, bt, be)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            gnorm = grad_norm(model)
            optimizer.step()
            batch_losses.append(float(loss.item()))
            ecount = int(be.sum().item())
            event_counts.append(ecount)
            pairs = comparable_pairs_count(bt, be)
            one_batch_comparable_pairs.append(pairs)
            grad_norms.append(gnorm)
            total_batches += 1
            zero_event_batches += int(ecount == 0)
            weak_info_batches += int(ecount <= 1)

            if cfg.train.save_batch_logs:
                append_jsonl(
                    batch_log_path,
                    {
                        "epoch": epoch,
                        "batch_index": batch_idx,
                        "batch_size": int(bx.shape[0]),
                        "event_count": ecount,
                        "one_event": int(ecount == 1),
                        "zero_event": int(ecount == 0),
                        "comparable_pairs": pairs,
                        "grad_norm": gnorm,
                    },
                )

        scheduler.step()
        val_metrics = evaluate(model=model, loader=val_loader, device=device, backbone=backbone)
        summary = summarize_epoch_batch_diagnostics(
            event_counts=event_counts, comparable_pairs=one_batch_comparable_pairs, grad_norms=grad_norms
        )
        elapsed_epoch = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            "val_loss": float(val_metrics["cox_loss"]),
            "val_c_index": float(val_metrics["c_index"]),
            "val_ipcw_c_index": float(val_metrics["ipcw_c_index"]),
            "val_harrell_c_index": float(val_metrics["c_index"]),
            "val_uno_c_index": float(val_metrics["ipcw_c_index"]),
            "val_brier_proxy": float(val_metrics["brier_proxy"]),
            "zero_event_fraction": summary.zero_event_fraction,
            "one_event_fraction": summary.one_event_fraction,
            "mean_events_per_batch": summary.mean_events_per_batch,
            "mean_comparable_pairs": summary.mean_comparable_pairs,
            "grad_norm_mean": summary.grad_norm_mean,
            "grad_norm_var": summary.grad_norm_var,
            "grad_norm_cv": summary.grad_norm_cv,
            "epoch_time_s": elapsed_epoch,
        }
        per_epoch_rows.append(row)
        if cfg.train.save_epoch_logs:
            append_jsonl(epoch_log_path, row)

        current_val = float(val_metrics["c_index"])
        if np.isfinite(current_val) and current_val > best_val:
            best_val = current_val
            best_epoch = epoch
            best_time_s = time.perf_counter() - started
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    total_time_s = time.perf_counter() - started
    test_metrics = {"c_index": float("nan"), "ipcw_c_index": float("nan"), "brier_proxy": float("nan")}
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        test_metrics = evaluate(model=model, loader=test_loader, device=device, backbone=backbone)

    empirical_zero = float(zero_event_batches / total_batches) if total_batches > 0 else float("nan")
    empirical_weak = float(weak_info_batches / total_batches) if total_batches > 0 else float("nan")
    result = {
        "event_target": context.event_target,
        "censor_target": context.censor_target,
        "batch_size": context.batch_size,
        "batching_policy": normalized_policy,
        "seed": context.seed,
        "best_epoch": best_epoch,
        "best_val_c_index": float(best_val),
        "best_time_s": float(best_time_s),
        "total_time_s": float(total_time_s),
        "final_train_loss": float(per_epoch_rows[-1]["train_loss"]),
        "final_val_loss": float(per_epoch_rows[-1]["val_loss"]),
        "final_val_ipcw_c_index": float(per_epoch_rows[-1]["val_ipcw_c_index"]),
        "final_val_brier_proxy": float(per_epoch_rows[-1]["val_brier_proxy"]),
        "test_harrell_c_index": float(test_metrics["c_index"]),
        "test_uno_c_index": float(test_metrics["ipcw_c_index"]),
        "test_c_index": float(test_metrics["c_index"]),
        "test_ipcw_c_index": float(test_metrics["ipcw_c_index"]),
        "test_brier_proxy": float(test_metrics["brier_proxy"]),
        "final_zero_event_fraction": float(per_epoch_rows[-1]["zero_event_fraction"]),
        "final_one_event_fraction": float(per_epoch_rows[-1]["one_event_fraction"]),
        "final_grad_norm_var": float(per_epoch_rows[-1]["grad_norm_var"]),
        "empirical_zero_event_prob": empirical_zero,
        "empirical_weak_info_prob": empirical_weak,
        "train_event_rate": float(run_meta["train_event_rate"]),
        "theoretical_zero_event_prob": float(run_meta["theoretical_zero_event_prob"]),
        "theoretical_weak_info_prob": float(run_meta["theoretical_weak_info_prob"]),
        "sampler_feasible": bool(run_meta["sampler_feasible"]),
        "sampler_mode": str(run_meta["sampler_mode"]),
        "sampler_with_replacement": bool(run_meta["sampler_with_replacement"]),
        "requested_event_fraction": float(run_meta["requested_event_fraction"]),
        "requested_min_events_per_batch": int(run_meta["requested_min_events_per_batch"]),
    }
    return result


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, backbone: str) -> dict[str, float]:
    model.eval()
    all_time: list[np.ndarray] = []
    all_event: list[np.ndarray] = []
    all_risk: list[np.ndarray] = []
    with torch.no_grad():
        for bx, bt, be in loader:
            bx = _prepare_features(bx=bx.to(device), backbone=backbone)
            risk = model(bx).cpu().numpy()
            all_risk.append(risk)
            all_time.append(bt.numpy())
            all_event.append(be.numpy())
    time_arr = np.concatenate(all_time).astype(np.float64)
    event_arr = np.concatenate(all_event).astype(np.int64)
    risk_arr = np.concatenate(all_risk).astype(np.float64)

    t = torch.from_numpy(time_arr.astype(np.float32))
    e = torch.from_numpy(event_arr.astype(np.int64))
    r = torch.from_numpy(risk_arr.astype(np.float32))
    cox_loss = float(cox_partial_log_likelihood_loss(r, t, e).item())
    return {
        "cox_loss": cox_loss,
        "c_index": harrell_c_index(time_arr, event_arr, risk_arr),
        "ipcw_c_index": ipcw_c_index(time_arr, event_arr, risk_arr),
        "brier_proxy": brier_at_median_time(time_arr, event_arr, risk_arr),
    }


def _build_model(cfg: ExperimentConfig, x: np.ndarray) -> torch.nn.Module:
    backbone = cfg.model.backbone.lower()
    if backbone == "cox_mlp":
        return CoxMLP(
            input_dim=int(np.prod(x.shape[1:])),
            hidden_dims=cfg.model.hidden_dims,
            dropout=cfg.model.dropout,
        )
    if backbone == "resnet18":
        return CoxResNet18()
    raise ValueError(f"Unsupported model backbone: {cfg.model.backbone}")


def _prepare_features(bx: torch.Tensor, backbone: str) -> torch.Tensor:
    if backbone == "cox_mlp":
        return bx.float().view(bx.shape[0], -1)
    if backbone == "resnet18":
        x = bx.float()
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected 3D/4D tensor for ResNet18, got shape {tuple(x.shape)}")
        return x
    raise ValueError(f"Unsupported model backbone: {backbone}")


def _build_train_sampler(
    *,
    policy: str,
    indices: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    batch_size: int,
    seed: int,
) -> torch.utils.data.Sampler[list[int]]:
    spec = parse_batching_policy(policy)
    if spec.mode == "random":
        return RandomBatchSampler(indices=indices, batch_size=batch_size, seed=seed)
    if spec.mode == "quota":
        return EventQuotaBatchSampler(
            indices=indices,
            events=events,
            batch_size=batch_size,
            event_fraction=spec.event_fraction,
            seed=seed,
            with_replacement=spec.with_replacement,
            strict_feasible=spec.strict_feasible,
        )
    if spec.mode == "riskset_anchor":
        return RiskSetAnchorBatchSampler(
            indices=indices,
            events=events,
            times=times,
            batch_size=batch_size,
            event_fraction=spec.event_fraction,
            seed=seed,
            strict_feasible=spec.strict_feasible,
        )
    raise ValueError(f"Unknown batching policy: {policy}")


def _build_optimizer(cfg: ExperimentConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")


def _build_scheduler(
    cfg: ExperimentConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    if cfg.train.scheduler.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.train.epochs, 1), eta_min=cfg.train.min_lr
        )
    if cfg.train.scheduler.lower() == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")


def _run_meta_dict(
    context: RunContext,
    train_events: np.ndarray,
    sampler: torch.utils.data.Sampler[list[int]],
) -> dict[str, float | int | bool | str]:
    train_event_rate = float(np.mean(train_events))
    normalized_policy = normalize_batching_policy(context.batching_policy)
    spec = parse_batching_policy(context.batching_policy)
    return {
        "event_target": context.event_target,
        "censor_target": context.censor_target,
        "batch_size": context.batch_size,
        "batching_policy": normalized_policy,
        "seed": context.seed,
        "train_event_rate": train_event_rate,
        "theoretical_zero_event_prob": float((1.0 - train_event_rate) ** context.batch_size),
        "theoretical_weak_info_prob": float(
            (1.0 - train_event_rate) ** context.batch_size
            + context.batch_size * train_event_rate * ((1.0 - train_event_rate) ** (context.batch_size - 1))
        ),
        "sampler_feasible": bool(getattr(sampler, "feasible", True)),
        "sampler_mode": spec.mode,
        "sampler_with_replacement": bool(getattr(sampler, "with_replacement", False)),
        "requested_event_fraction": float(getattr(sampler, "event_fraction", np.nan)),
        "requested_min_events_per_batch": int(getattr(sampler, "min_events_per_batch", 0)),
    }
