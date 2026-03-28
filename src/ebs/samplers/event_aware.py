from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class EventAwareBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        indices: np.ndarray,
        events: np.ndarray,
        batch_size: int,
        min_events_per_batch: int,
        seed: int,
        drop_last: bool = False,
        strict_feasible: bool = True,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.events = np.asarray(events, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.min_events_per_batch = int(min_events_per_batch)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.strict_feasible = strict_feasible
        self._feasible = self._check_feasible()

    @property
    def feasible(self) -> bool:
        return self._feasible

    def _check_feasible(self) -> bool:
        n_batches = len(self.indices) // self.batch_size
        if not self.drop_last and (len(self.indices) % self.batch_size) > 0:
            n_batches += 1
        n_events = int(self.events[self.indices].sum())
        return n_events >= n_batches * self.min_events_per_batch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        if self.strict_feasible and not self._feasible:
            perm = rng.permutation(self.indices)
            for start in range(0, len(perm), self.batch_size):
                batch = perm[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch.tolist()
            return

        event_indices = self.indices[self.events[self.indices] == 1].copy()
        censor_indices = self.indices[self.events[self.indices] == 0].copy()
        event_indices = rng.permutation(event_indices)
        censor_indices = rng.permutation(censor_indices)

        e_ptr, c_ptr = 0, 0
        total = len(self.indices)
        produced = 0
        while produced < total:
            remaining = total - produced
            current_bs = min(self.batch_size, remaining)
            if current_bs < self.batch_size and self.drop_last:
                break

            batch: list[int] = []
            remaining_events = len(event_indices) - e_ptr
            ensure_events = min(self.min_events_per_batch, current_bs, remaining_events)
            if not self.strict_feasible:
                ensure_events = min(ensure_events, self.min_events_per_batch)

            for _ in range(ensure_events):
                batch.append(int(event_indices[e_ptr]))
                e_ptr += 1

            # Fill from censored first to keep scarce events spread across batches.
            while len(batch) < current_bs and c_ptr < len(censor_indices):
                batch.append(int(censor_indices[c_ptr]))
                c_ptr += 1
            while len(batch) < current_bs and e_ptr < len(event_indices):
                batch.append(int(event_indices[e_ptr]))
                e_ptr += 1

            if len(batch) < current_bs:
                leftovers: list[int] = []
                if c_ptr < len(censor_indices):
                    leftovers.extend(censor_indices[c_ptr:].tolist())
                    c_ptr = len(censor_indices)
                if e_ptr < len(event_indices):
                    leftovers.extend(event_indices[e_ptr:].tolist())
                    e_ptr = len(event_indices)
                rng.shuffle(leftovers)
                need = current_bs - len(batch)
                batch.extend(int(i) for i in leftovers[:need])

            produced += len(batch)
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return math.ceil(len(self.indices) / self.batch_size)
