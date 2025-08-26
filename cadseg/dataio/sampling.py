# cadseg/dataio/sampling.py
from __future__ import annotations
from typing import List, Dict, Iterable, Iterator, Optional, Set
from dataclasses import dataclass
import random

import torch
from torch.utils.data import Sampler

from cadseg.dataio.datasets import TiledDataset, TileInfo


# ----------------------------
# Utilities
# ----------------------------
def _shuffle_inplace(xs: List[int], rng: random.Random) -> None:
    rng.shuffle(xs)


@dataclass
class _Cycler:
    """Cycles through a list of indices, reshuffling at each wrap-around."""
    data: List[int]
    rng: random.Random
    ptr: int = 0

    def next_k(self, k: int, exclude: Optional[Set[int]] = None) -> List[int]:
        out: List[int] = []
        if not self.data or k <= 0:
            return out
        exclude = exclude or set()

        # Try to collect k items, allowing multiple wraps with reshuffle at each wrap
        attempts = 0
        while len(out) < k and attempts < k * 4:  # safety bound
            if self.ptr >= len(self.data):
                self.ptr = 0
                _shuffle_inplace(self.data, self.rng)
            idx = self.data[self.ptr]
            self.ptr += 1
            if idx in exclude:
                attempts += 1
                continue
            out.append(idx)
        return out


@dataclass
class TilePools:
    """Holds per-class positive indices and a negative pool."""
    pos_by_class: List[List[int]]
    negatives: List[int]
    # Classes with at least one positive index
    active_classes: List[int]


def build_tile_pools(dataset: TiledDataset, seed: int = 42) -> TilePools:
    """
    Build per-class positive pools and a negative pool from a TiledDataset.
    A tile may appear in multiple class pools (multi-label positives).
    """
    C = len(dataset.class_names)
    pos_by_class: List[List[int]] = [[] for _ in range(C)]
    negatives: List[int] = []

    for idx, t in enumerate(dataset.tiles):
        if t.has_pos_any:
            for c, has_c in enumerate(t.has_pos_per_class):
                if has_c:
                    pos_by_class[c].append(idx)
        else:
            negatives.append(idx)

    active_classes = [c for c in range(C) if len(pos_by_class[c]) > 0]

    # Shuffle once deterministically (sampler will reshuffle again per epoch)
    rng = random.Random(seed)
    for lst in pos_by_class:
        _shuffle_inplace(lst, rng)
    _shuffle_inplace(negatives, rng)

    return TilePools(pos_by_class=pos_by_class, negatives=negatives, active_classes=active_classes)


# ----------------------------
# Class-balanced batch sampler
# ----------------------------
class ClassBalancedBatchSampler(Sampler[List[int]]):
    """
    Yields batches of tile indices with:
      - A target fraction of positives per batch (pos_fraction).
      - Positive quota spread across the classes that have data (near-uniform).
      - Remaining filled with negatives (if available).
    Notes:
      * A tile with multiple classes can be sampled for any of its classes.
      * Duplicates inside a batch are avoided.
      * Pools reshuffle automatically as we cycle through them.
    """

    def __init__(
        self,
        dataset: TiledDataset,
        batch_size: int,
        *,
        pos_fraction: float = 0.75,          # ~3/4 positives, 1/4 negatives is a good start
        batches_per_epoch: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(data_source=None)
        assert 0.0 <= pos_fraction <= 1.0, "pos_fraction must be in [0,1]"
        assert batch_size >= 1, "batch_size must be >= 1"
        self.ds = dataset
        self.batch_size = int(batch_size)
        self.pos_fraction = float(pos_fraction)
        self.seed = int(seed)

        # Pools
        self.pools = build_tile_pools(dataset, seed=self.seed)
        self.C = len(self.ds.class_names)
        self.rng = random.Random(self.seed)

        # Cyclers
        self.pos_cyclers: List[_Cycler] = [
            _Cycler(self.pools.pos_by_class[c][:], self.rng) for c in range(self.C)
        ]
        self.neg_cycler = _Cycler(self.pools.negatives[:], self.rng)

        # How many batches per epoch? Default: cover all tiles roughly once.
        if batches_per_epoch is None:
            # Heuristic: total tiles / batch_size, but ensure at least #active_class cycles
            total = max(1, len(self.ds.tiles))
            self._num_batches = max(1, total // self.batch_size)
        else:
            self._num_batches = int(batches_per_epoch)

    def __len__(self) -> int:
        return self._num_batches

    def _per_class_quota(self, k_pos: int) -> List[int]:
        """Split k_pos as evenly as possible across active classes."""
        active = self.pools.active_classes
        if not active or k_pos <= 0:
            return [0] * self.C
        base = k_pos // len(active)
        rem = k_pos % len(active)
        quota = [0] * self.C
        # distribute remainder first
        for i, c in enumerate(active):
            quota[c] = base + (1 if i < rem else 0)
        return quota

    def __iter__(self) -> Iterator[List[int]]:
        # New epoch → reshuffle base order to vary sampling rotation
        epoch_seed = self.rng.randrange(1 << 30)
        rng_epoch = random.Random(epoch_seed)

        # Also reshuffle the underlying lists once per epoch (cyclers will reshuffle on wrap too)
        for cyc in self.pos_cyclers:
            _shuffle_inplace(cyc.data, rng_epoch)
            cyc.ptr = 0
        _shuffle_inplace(self.neg_cycler.data, rng_epoch)
        self.neg_cycler.ptr = 0

        for _ in range(self._num_batches):
            # Compute positive/negative counts for this batch
            k_pos = int(round(self.batch_size * self.pos_fraction))
            k_pos = min(k_pos, self.batch_size)
            k_neg = self.batch_size - k_pos

            # Allocate positives across active classes
            quota = self._per_class_quota(k_pos)

            batch: List[int] = []
            seen: Set[int] = set()

            # Draw positives per class
            for c, k in enumerate(quota):
                if k <= 0:
                    continue
                picked = self.pos_cyclers[c].next_k(k, exclude=seen)
                batch.extend(picked)
                seen.update(picked)

            # If we undershot (due to duplicates/empty pools), try to fill remaining from any positive class
            missing = k_pos - len(batch)
            if missing > 0:
                active = [c for c in self.pools.active_classes if len(self.pos_cyclers[c].data) > 0]
                # round-robin draw
                rr = 0
                while missing > 0 and active:
                    c = active[rr % len(active)]
                    drawn = self.pos_cyclers[c].next_k(1, exclude=seen)
                    if drawn:
                        batch.append(drawn[0])
                        seen.add(drawn[0])
                        missing -= 1
                    rr += 1
                    if rr > self.batch_size * 4:
                        break  # safety

            # Fill with negatives
            if k_neg > 0 and len(self.neg_cycler.data) > 0:
                neg = self.neg_cycler.next_k(k_neg, exclude=seen)
                batch.extend(neg)
                seen.update(neg)

            # If still short (e.g., no negatives), backfill from any pool
            while len(batch) < self.batch_size:
                # Prefer positives if available, otherwise negatives
                src = self.pools.active_classes if self.pools.active_classes else []
                picked = []
                for c in src:
                    picked = self.pos_cyclers[c].next_k(1, exclude=seen)
                    if picked:
                        break
                if not picked and len(self.neg_cycler.data) > 0:
                    picked = self.neg_cycler.next_k(1, exclude=seen)
                if not picked:
                    # Nothing left to sample; break to avoid infinite loop
                    break
                batch.append(picked[0])
                seen.add(picked[0])

            yield batch


# ----------------------------
# Debug helpers
# ----------------------------
def estimate_epoch_class_histogram(
    sampler: ClassBalancedBatchSampler, dataset: TiledDataset, max_batches: Optional[int] = 100
) -> Dict[str, int]:
    """
    Simulate one epoch (or subset) and count how many samples in the yielded batches
    contain positives for each class. Useful to verify balance.
    """
    counts = {name: 0 for name in dataset.class_names}
    n = 0
    for b, batch in enumerate(iter(sampler)):
        if max_batches is not None and b >= max_batches:
            break
        for idx in batch:
            t: TileInfo = dataset.tiles[idx]
            for c, has in enumerate(t.has_pos_per_class):
                if has:
                    counts[dataset.class_names[c]] += 1
        n += 1
    counts["_simulated_batches"] = n
    return counts
