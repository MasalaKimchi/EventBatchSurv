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
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
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
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.min_events_per_batch < 0:
            raise ValueError("min_events_per_batch must be >= 0")
        if self.indices.ndim != 1:
            raise ValueError("indices must be a 1D array")
        if self.events.ndim != 1:
            raise ValueError("events must be a 1D array")
        if len(self.indices) == 0:
            raise ValueError("indices must be non-empty")
        if np.any(self.indices < 0) or np.any(self.indices >= len(self.events)):
            raise ValueError("indices contain out-of-range values for events array")
        selected_events = self.events[self.indices]
        if np.any((selected_events != 0) & (selected_events != 1)):
            raise ValueError("events must be binary (0/1) for all selected indices")
        self._feasible = self._check_feasible()

    @property
    def feasible(self) -> bool:
        return self._feasible

    def _check_feasible(self) -> bool:
        n_full_batches = len(self.indices) // self.batch_size
        required_events = n_full_batches * self.min_events_per_batch
        remainder = len(self.indices) % self.batch_size
        if not self.drop_last and remainder > 0:
            required_events += min(self.min_events_per_batch, remainder)
        n_events = int(self.events[self.indices].sum())
        return n_events >= required_events

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
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

            if not batch:
                raise RuntimeError("Sampler produced an empty batch; check events and indices inputs.")
            produced += len(batch)
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return math.ceil(len(self.indices) / self.batch_size)


class EventQuotaBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        indices: np.ndarray,
        events: np.ndarray,
        batch_size: int,
        event_fraction: float,
        seed: int,
        with_replacement: bool,
        drop_last: bool = False,
        strict_feasible: bool = True,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.events = np.asarray(events, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.event_fraction = float(event_fraction)
        self.seed = int(seed)
        self.with_replacement = bool(with_replacement)
        self.drop_last = bool(drop_last)
        self.strict_feasible = bool(strict_feasible)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.event_fraction < 0.0 or self.event_fraction > 1.0:
            raise ValueError("event_fraction must be in [0, 1]")
        if self.indices.ndim != 1:
            raise ValueError("indices must be a 1D array")
        if self.events.ndim != 1:
            raise ValueError("events must be a 1D array")
        if len(self.indices) == 0:
            raise ValueError("indices must be non-empty")
        if np.any(self.indices < 0) or np.any(self.indices >= len(self.events)):
            raise ValueError("indices contain out-of-range values for events array")
        selected_events = self.events[self.indices]
        if np.any((selected_events != 0) & (selected_events != 1)):
            raise ValueError("events must be binary (0/1) for all selected indices")
        self._feasible = self._check_feasible()

    @property
    def feasible(self) -> bool:
        return self._feasible

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batch_sizes(self) -> list[int]:
        total = len(self.indices)
        if self.drop_last:
            n_full = total // self.batch_size
            return [self.batch_size] * n_full
        n_full = total // self.batch_size
        rem = total % self.batch_size
        out = [self.batch_size] * n_full
        if rem > 0:
            out.append(rem)
        return out

    def _required_events_total(self) -> int:
        return int(sum(math.ceil(self.event_fraction * bs) for bs in self._batch_sizes()))

    def _check_feasible(self) -> bool:
        event_pool = self.indices[self.events[self.indices] == 1]
        required = self._required_events_total()
        if self.with_replacement:
            return required == 0 or len(event_pool) > 0
        return len(event_pool) >= required

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.strict_feasible and not self._feasible:
            perm = rng.permutation(self.indices)
            for start in range(0, len(perm), self.batch_size):
                batch = perm[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch.tolist()
            return

        event_pool = self.indices[self.events[self.indices] == 1].copy()
        censor_pool = self.indices[self.events[self.indices] == 0].copy()
        if self.with_replacement:
            all_pool = self.indices.copy()
            for bs in self._batch_sizes():
                required = int(math.ceil(self.event_fraction * bs))
                batch: list[int] = []
                if required > 0 and len(event_pool) > 0:
                    ev = rng.choice(event_pool, size=required, replace=True)
                    batch.extend(int(i) for i in ev.tolist())
                remaining = bs - len(batch)
                if remaining > 0:
                    source = censor_pool if len(censor_pool) > 0 else all_pool
                    rest = rng.choice(source, size=remaining, replace=True)
                    batch.extend(int(i) for i in rest.tolist())
                rng.shuffle(batch)
                yield batch
            return

        event_pool = rng.permutation(event_pool)
        censor_pool = rng.permutation(censor_pool)
        e_ptr = 0
        c_ptr = 0
        for bs in self._batch_sizes():
            required = int(math.ceil(self.event_fraction * bs))
            batch: list[int] = []
            for _ in range(required):
                if e_ptr >= len(event_pool):
                    break
                batch.append(int(event_pool[e_ptr]))
                e_ptr += 1
            while len(batch) < bs and c_ptr < len(censor_pool):
                batch.append(int(censor_pool[c_ptr]))
                c_ptr += 1
            while len(batch) < bs and e_ptr < len(event_pool):
                batch.append(int(event_pool[e_ptr]))
                e_ptr += 1
            if len(batch) < bs:
                leftovers: list[int] = []
                if c_ptr < len(censor_pool):
                    leftovers.extend(censor_pool[c_ptr:].tolist())
                    c_ptr = len(censor_pool)
                if e_ptr < len(event_pool):
                    leftovers.extend(event_pool[e_ptr:].tolist())
                    e_ptr = len(event_pool)
                if leftovers:
                    rng.shuffle(leftovers)
                    batch.extend(int(i) for i in leftovers[: bs - len(batch)])
            yield batch

    def __len__(self) -> int:
        return len(self._batch_sizes())
