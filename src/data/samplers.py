from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from src.data.sequence_npz_dataset import SequenceNPZDataset


class DateBatchSampler(Sampler[list[int]]):
    """Yield mini-batches that keep each trade_date cross-section together."""

    def __init__(
        self,
        dataset: SequenceNPZDataset,
        max_samples_per_batch: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.max_samples_per_batch = max_samples_per_batch
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        dates = np.asarray(self.dataset.trade_date).astype(str)
        for trade_date in np.unique(dates):
            indices = np.flatnonzero(dates == trade_date).astype(int).tolist()
            if not indices:
                continue
            if self.max_samples_per_batch is None or len(indices) <= self.max_samples_per_batch:
                batches.append(indices)
                continue
            for start in range(0, len(indices), self.max_samples_per_batch):
                chunk = indices[start : start + self.max_samples_per_batch]
                if len(chunk) == self.max_samples_per_batch or not self.drop_last:
                    batches.append(chunk)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = [batch.copy() for batch in self._batches]
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(batches)
        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)
