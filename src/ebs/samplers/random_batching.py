from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class RandomBatchSampler(Sampler[list[int]]):
    def __init__(self, indices: np.ndarray, batch_size: int, seed: int, drop_last: bool = False):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(self.indices)
        for start in range(0, len(perm), self.batch_size):
            batch = perm[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch.tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return math.ceil(len(self.indices) / self.batch_size)
