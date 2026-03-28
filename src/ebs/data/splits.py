from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
