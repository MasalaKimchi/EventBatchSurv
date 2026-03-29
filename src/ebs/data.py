from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms


@dataclass
class SyntheticSurvivalData:
    x: np.ndarray
    time: np.ndarray
    event: np.ndarray


def load_mnist_arrays(mnist_root: str) -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=mnist_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=mnist_root, train=False, download=True, transform=transform)

    x_train = train.data.numpy().astype(np.float32) / 255.0
    x_test = test.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    y_test = test.targets.numpy().astype(np.int64)

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y


def generate_synthetic_survival_from_mnist(
    *,
    mnist_root: str,
    n_samples: int,
    event_prevalence_target: float,
    censoring_target: float,
    seed: int,
) -> SyntheticSurvivalData:
    rng = np.random.default_rng(seed)
    x_all, digits_all = load_mnist_arrays(mnist_root)
    idx = rng.choice(len(x_all), size=n_samples, replace=False)
    x = x_all[idx]
    digits = digits_all[idx]

    flat = x.reshape(n_samples, -1)
    mean_intensity = flat.mean(axis=1)
    stroke_energy = (flat**2).mean(axis=1)
    latent = (
        0.55 * (digits.astype(np.float32) / 9.0)
        + 0.30 * mean_intensity
        + 0.15 * stroke_energy
        + rng.normal(0.0, 0.05, size=n_samples)
    )
    latent = (latent - latent.mean()) / (latent.std() + 1e-8)

    susceptible = rng.binomial(1, p=float(np.clip(event_prevalence_target, 0.0, 1.0)), size=n_samples)
    base_scale = np.exp(-latent)
    event_time = rng.exponential(scale=np.maximum(base_scale, 1e-3), size=n_samples) + 1e-3
    horizon = np.quantile(event_time, 0.98)
    event_time[susceptible == 0] = horizon * 5.0

    censor_scale = _calibrate_censor_scale(
        event_time=event_time,
        susceptible=susceptible.astype(bool),
        target=float(np.clip(censoring_target, 0.0, 1.0)),
        rng=rng,
    )
    censor_time = rng.exponential(scale=np.maximum(censor_scale, 1e-6), size=n_samples) + 1e-3
    observed_time = np.minimum(event_time, censor_time)
    observed_event = (event_time <= censor_time) & (susceptible == 1)

    return SyntheticSurvivalData(
        x=x.astype(np.float32),
        time=observed_time.astype(np.float32),
        event=observed_event.astype(np.int64),
    )


def _calibrate_censor_scale(
    *,
    event_time: np.ndarray,
    susceptible: np.ndarray,
    target: float,
    rng: np.random.Generator,
    n_iter: int = 24,
) -> float:
    if target <= 0.0:
        return float(event_time.max() * 100.0)
    if target >= 1.0:
        return 1e-5
    sus_event = event_time[susceptible]
    if sus_event.size == 0:
        return 1.0

    lo, hi = 1e-4, float(np.quantile(sus_event, 0.95) * 5.0 + 1e-3)
    best = hi
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        probe = rng.exponential(scale=mid, size=sus_event.size)
        censor_frac = float((probe < sus_event).mean())
        if censor_frac < target:
            hi = mid
        else:
            lo = mid
            best = mid
    return best


class SurvivalTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, time: np.ndarray, event: np.ndarray):
        self.x = torch.from_numpy(x)
        self.time = torch.from_numpy(time.astype(np.float32))
        self.event = torch.from_numpy(event.astype(np.int64))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.time[idx], self.event[idx]


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def get_or_create_fixed_splits(
    *,
    n_samples: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    cache_file: str | Path,
) -> SplitIndices:
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if int(payload.get("n_samples", -1)) == int(n_samples):
            return SplitIndices(
                train_idx=np.array(payload["train_idx"], dtype=np.int64),
                val_idx=np.array(payload["val_idx"], dtype=np.int64),
                test_idx=np.array(payload["test_idx"], dtype=np.int64),
            )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(round(train_fraction * n_samples))
    n_val = int(round(val_fraction * n_samples))
    n_train = min(max(n_train, 1), n_samples - 2)
    n_val = min(max(n_val, 1), n_samples - n_train - 1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    payload = {
        "seed": seed,
        "n_samples": n_samples,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return SplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
