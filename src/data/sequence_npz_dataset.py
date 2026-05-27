from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceNPZDataset(Dataset):
    """Dataset for model-ready sequence NPZ files.

    Expected NPZ keys:
        X: float32 array [N, lookback, num_features]
        y: float32 array [N]
        trade_date: string array [N]
        ts_code: string array [N]
        split: string array [N]
        feature_names: string array [num_features]
    """

    REQUIRED_KEYS = ("X", "y", "trade_date", "ts_code", "split", "feature_names")
    VALID_SPLITS = ("train", "validation", "test")

    def __init__(self, npz_path: str | Path, split: str):
        if split not in self.VALID_SPLITS:
            valid = ", ".join(self.VALID_SPLITS)
            raise ValueError(f"Unknown split '{split}'. Expected one of: {valid}.")

        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Missing sequence dataset: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)
        missing = [key for key in self.REQUIRED_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"NPZ dataset is missing required keys: {missing}")

        split_values = data["split"].astype(str)
        mask = split_values == split
        if not mask.any():
            raise ValueError(f"No samples found for split '{split}' in {self.npz_path}")

        self.split = split
        self.X = data["X"][mask].astype("float32", copy=False)
        self.y = data["y"][mask].astype("float32", copy=False)
        self.trade_date = data["trade_date"][mask].astype(str)
        self.ts_code = data["ts_code"][mask].astype(str)
        self.feature_names = data["feature_names"].astype(str).tolist()

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"Expected X to be 3D [N, T, F], got shape {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"Expected y to be 1D [N], got shape {self.y.shape}")

        sample_count = self.X.shape[0]
        lengths = {
            "y": len(self.y),
            "trade_date": len(self.trade_date),
            "ts_code": len(self.ts_code),
        }
        mismatched = {name: length for name, length in lengths.items() if length != sample_count}
        if mismatched:
            raise ValueError(
                f"Metadata lengths must match X samples ({sample_count}); mismatched: {mismatched}"
            )

        feature_count = self.X.shape[2]
        if len(self.feature_names) != feature_count:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) does not match X feature dimension ({feature_count})"
            )

    @property
    def lookback(self) -> int:
        return int(self.X.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.X.shape[2])

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "x": torch.from_numpy(self.X[index]),
            "y": torch.tensor(self.y[index], dtype=torch.float32),
            "trade_date": str(self.trade_date[index]),
            "ts_code": str(self.ts_code[index]),
        }
