from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch


def _to_1d_numpy(values: Any, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}")
    return array


def _metrics_frame(
    pred: Sequence[float] | np.ndarray | torch.Tensor,
    target: Sequence[float] | np.ndarray | torch.Tensor,
    dates: Sequence[str] | np.ndarray,
) -> pd.DataFrame:
    pred_array = _to_1d_numpy(pred, "pred").astype("float64", copy=False)
    target_array = _to_1d_numpy(target, "target").astype("float64", copy=False)
    date_array = _to_1d_numpy(dates, "dates").astype(str, copy=False)

    if not (len(pred_array) == len(target_array) == len(date_array)):
        raise ValueError(
            "pred, target, and dates must have the same length; "
            f"got {len(pred_array)}, {len(target_array)}, {len(date_array)}"
        )

    frame = pd.DataFrame(
        {
            "trade_date": date_array,
            "pred": pred_array,
            "target": target_array,
        }
    )
    finite_mask = np.isfinite(frame["pred"].to_numpy()) & np.isfinite(frame["target"].to_numpy())
    return frame.loc[finite_mask].reset_index(drop=True)


def _safe_corr(group: pd.DataFrame, pred_col: str, target_col: str, min_count: int) -> float:
    if len(group) < min_count:
        return np.nan
    pred = group[pred_col]
    target = group[target_col]
    if pred.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
        return np.nan
    return float(pred.corr(target, method="pearson"))


def _daily_cross_section_diagnostics(frame: pd.DataFrame, min_count: int) -> dict[str, int | str]:
    if frame.empty:
        return {
            "valid_sample_count": 0,
            "valid_date_count": 0,
            "eligible_daily_count": 0,
            "pred_constant_daily_count": 0,
            "target_constant_daily_count": 0,
            "daily_status": "no_valid_samples",
            "daily_skip_reason": "No finite prediction/target pairs are available.",
        }

    grouped = frame.groupby("trade_date", sort=True)
    eligible_daily_count = 0
    pred_constant_daily_count = 0
    target_constant_daily_count = 0

    for _, group in grouped:
        if len(group) < min_count:
            continue
        eligible_daily_count += 1
        if group["pred"].nunique(dropna=True) <= 1:
            pred_constant_daily_count += 1
        if group["target"].nunique(dropna=True) <= 1:
            target_constant_daily_count += 1

    if eligible_daily_count == 0:
        status = "no_valid_daily_cross_section"
        reason = f"No trade date has at least min_count={min_count} finite samples."
    elif pred_constant_daily_count == eligible_daily_count:
        status = "prediction_collapse"
        reason = "All eligible trade dates have constant predictions, so IC/RankIC is undefined."
    elif target_constant_daily_count == eligible_daily_count:
        status = "target_collapse"
        reason = "All eligible trade dates have constant targets, so IC/RankIC is undefined."
    else:
        status = "ok"
        reason = ""

    return {
        "valid_sample_count": int(len(frame)),
        "valid_date_count": int(frame["trade_date"].nunique(dropna=True)),
        "eligible_daily_count": int(eligible_daily_count),
        "pred_constant_daily_count": int(pred_constant_daily_count),
        "target_constant_daily_count": int(target_constant_daily_count),
        "daily_status": status,
        "daily_skip_reason": reason,
    }


def compute_daily_ic(
    pred: Sequence[float] | np.ndarray | torch.Tensor,
    target: Sequence[float] | np.ndarray | torch.Tensor,
    dates: Sequence[str] | np.ndarray,
    min_count: int = 20,
) -> pd.Series:
    """Compute daily Pearson IC indexed by trade_date."""

    if min_count <= 1:
        raise ValueError(f"min_count must be greater than 1, got {min_count}")
    frame = _metrics_frame(pred, target, dates)
    if frame.empty:
        return pd.Series(dtype="float64", name="ic")

    values = {
        trade_date: _safe_corr(group, "pred", "target", min_count)
        for trade_date, group in frame.groupby("trade_date", sort=True)
    }
    daily = pd.Series(values, dtype="float64").dropna()
    daily.name = "ic"
    return daily


def compute_daily_rank_ic(
    pred: Sequence[float] | np.ndarray | torch.Tensor,
    target: Sequence[float] | np.ndarray | torch.Tensor,
    dates: Sequence[str] | np.ndarray,
    min_count: int = 20,
) -> pd.Series:
    """Compute daily Spearman RankIC indexed by trade_date."""

    if min_count <= 1:
        raise ValueError(f"min_count must be greater than 1, got {min_count}")
    frame = _metrics_frame(pred, target, dates)
    if frame.empty:
        return pd.Series(dtype="float64", name="rank_ic")

    frame["pred_rank"] = frame.groupby("trade_date")["pred"].rank(method="average")
    frame["target_rank"] = frame.groupby("trade_date")["target"].rank(method="average")
    values = {
        trade_date: _safe_corr(group, "pred_rank", "target_rank", min_count)
        for trade_date, group in frame.groupby("trade_date", sort=True)
    }
    daily = pd.Series(values, dtype="float64").dropna()
    daily.name = "rank_ic"
    return daily


def compute_icir(daily_values: pd.Series | Sequence[float] | np.ndarray) -> float:
    values = pd.Series(daily_values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return float("nan")
    std = values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(values.mean() / std)


def summarize_daily_ic(
    pred: Sequence[float] | np.ndarray | torch.Tensor,
    target: Sequence[float] | np.ndarray | torch.Tensor,
    dates: Sequence[str] | np.ndarray,
    min_count: int = 20,
) -> dict[str, float | int | str]:
    frame = _metrics_frame(pred, target, dates)
    daily_ic = compute_daily_ic(pred, target, dates, min_count=min_count)
    daily_rank_ic = compute_daily_rank_ic(pred, target, dates, min_count=min_count)
    diagnostics = _daily_cross_section_diagnostics(frame, min_count=min_count)
    if len(daily_ic):
        diagnostics["daily_status"] = "ok"
        diagnostics["daily_skip_reason"] = ""
    elif diagnostics["daily_status"] == "ok":
        diagnostics["daily_status"] = "no_valid_daily_correlation"
        diagnostics["daily_skip_reason"] = (
            "Eligible daily cross-sections exist, but all IC values are undefined after correlation checks."
        )

    return {
        "daily_count": int(len(daily_ic)),
        "rank_daily_count": int(len(daily_rank_ic)),
        "ic_mean": float(daily_ic.mean()) if len(daily_ic) else float("nan"),
        "ic_std": float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan"),
        "icir": compute_icir(daily_ic),
        "rank_ic_mean": float(daily_rank_ic.mean()) if len(daily_rank_ic) else float("nan"),
        "rank_ic_std": float(daily_rank_ic.std(ddof=1)) if len(daily_rank_ic) > 1 else float("nan"),
        "rank_icir": compute_icir(daily_rank_ic),
        **diagnostics,
    }
