from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


def _batch_sizes(total: int, batch_size: int, *, drop_last: bool) -> list[int]:
    if drop_last:
        return [batch_size] * (total // batch_size)
    n_full = total // batch_size
    rem = total % batch_size
    sizes = [batch_size] * n_full
    if rem > 0:
        sizes.append(rem)
    return sizes


def _iter_random_batches(
    indices: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
    drop_last: bool,
) -> Iterator[list[int]]:
    perm = rng.permutation(indices)
    for start in range(0, len(perm), batch_size):
        batch = perm[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        yield batch.tolist()


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
        yield from _iter_random_batches(
            self.indices,
            batch_size=self.batch_size,
            rng=rng,
            drop_last=self.drop_last,
        )

    def __len__(self) -> int:
        return len(_batch_sizes(len(self.indices), self.batch_size, drop_last=self.drop_last))


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
        return _batch_sizes(len(self.indices), self.batch_size, drop_last=self.drop_last)

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
            yield from _iter_random_batches(
                self.indices,
                batch_size=self.batch_size,
                rng=rng,
                drop_last=self.drop_last,
            )
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


class RiskSetAnchorBatchSampler(Sampler[list[int]]):
    """Event-anchor batches whose filler samples come from anchor risk sets."""

    def __init__(
        self,
        *,
        indices: np.ndarray,
        events: np.ndarray,
        times: np.ndarray,
        batch_size: int,
        event_fraction: float,
        seed: int,
        drop_last: bool = False,
        strict_feasible: bool = True,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.events = np.asarray(events, dtype=np.int64)
        self.times = np.asarray(times, dtype=np.float64)
        self.batch_size = int(batch_size)
        self.event_fraction = float(event_fraction)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.strict_feasible = bool(strict_feasible)
        self.with_replacement = True
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.event_fraction < 0.0 or self.event_fraction > 1.0:
            raise ValueError("event_fraction must be in [0, 1]")
        if self.indices.ndim != 1:
            raise ValueError("indices must be a 1D array")
        if self.events.ndim != 1:
            raise ValueError("events must be a 1D array")
        if self.times.ndim != 1:
            raise ValueError("times must be a 1D array")
        if len(self.indices) == 0:
            raise ValueError("indices must be non-empty")
        if np.any(self.indices < 0) or np.any(self.indices >= len(self.events)) or np.any(self.indices >= len(self.times)):
            raise ValueError("indices contain out-of-range values for events/times arrays")
        selected_events = self.events[self.indices]
        if np.any((selected_events != 0) & (selected_events != 1)):
            raise ValueError("events must be binary (0/1) for all selected indices")
        self._event_pool = self.indices[self.events[self.indices] == 1].copy()
        sorted_order = np.argsort(self.times[self.indices], kind="mergesort")
        self._sorted_indices = self.indices[sorted_order]
        self._sorted_times = self.times[self._sorted_indices]
        self._feasible = self._check_feasible()

    @property
    def feasible(self) -> bool:
        return self._feasible

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batch_sizes(self) -> list[int]:
        return _batch_sizes(len(self.indices), self.batch_size, drop_last=self.drop_last)

    def _check_feasible(self) -> bool:
        batch_sizes = self._batch_sizes()
        if not batch_sizes:
            return True
        max_required = max(int(math.ceil(self.event_fraction * bs)) for bs in batch_sizes)
        return max_required == 0 or len(self._event_pool) >= max_required

    def _riskset_candidates(self, anchor_idx: int, excluded: np.ndarray) -> np.ndarray:
        anchor_time = self.times[anchor_idx]
        start = int(np.searchsorted(self._sorted_times, anchor_time, side="left"))
        candidates = self._sorted_indices[start:]
        if candidates.size == 0:
            return candidates
        if excluded.size > 0:
            candidates = candidates[~np.isin(candidates, excluded)]
        return candidates

    def _union_riskset_candidates(self, anchors: np.ndarray, excluded: np.ndarray) -> np.ndarray:
        pools: list[np.ndarray] = []
        for anchor_idx in anchors.tolist():
            pool = self._riskset_candidates(int(anchor_idx), excluded)
            if pool.size > 0:
                pools.append(pool)
        if not pools:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(pools))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.strict_feasible and not self._feasible:
            yield from _iter_random_batches(
                self.indices,
                batch_size=self.batch_size,
                rng=rng,
                drop_last=self.drop_last,
            )
            return

        all_pool = self.indices.copy()
        for bs in self._batch_sizes():
            required = int(math.ceil(self.event_fraction * bs))
            batch: list[int] = []
            anchors = np.empty(0, dtype=np.int64)
            if required > 0 and len(self._event_pool) > 0:
                anchor_count = min(required, len(self._event_pool))
                anchors = rng.choice(self._event_pool, size=anchor_count, replace=False)
                batch.extend(int(i) for i in anchors.tolist())

            while len(batch) < bs and anchors.size > 0:
                excluded = np.array(batch, dtype=np.int64)
                progress = False
                for anchor_idx in rng.permutation(anchors):
                    candidates = self._riskset_candidates(int(anchor_idx), excluded)
                    if candidates.size == 0:
                        continue
                    chosen = int(rng.choice(candidates))
                    batch.append(chosen)
                    progress = True
                    if len(batch) >= bs:
                        break
                    excluded = np.array(batch, dtype=np.int64)
                if not progress:
                    break

            if len(batch) < bs and anchors.size > 0:
                excluded = np.array(batch, dtype=np.int64)
                union_pool = self._union_riskset_candidates(anchors, excluded)
                if union_pool.size > 0:
                    need = min(bs - len(batch), int(union_pool.size))
                    chosen = rng.choice(union_pool, size=need, replace=False)
                    batch.extend(int(i) for i in np.atleast_1d(chosen).tolist())

            if len(batch) < bs:
                excluded = np.array(batch, dtype=np.int64)
                remaining_pool = all_pool if excluded.size == 0 else all_pool[~np.isin(all_pool, excluded)]
                if remaining_pool.size > 0:
                    need = min(bs - len(batch), int(remaining_pool.size))
                    chosen = rng.choice(remaining_pool, size=need, replace=False)
                    batch.extend(int(i) for i in np.atleast_1d(chosen).tolist())

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return len(self._batch_sizes())
