from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "trade_date",
    "ts_code",
    "pred_score",
    "label_rel_return",
    "split",
}


def parse_k_values(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one K value is required.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All K values must be positive.")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate daily Top-K portfolio proxy metrics from prediction parquet files."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.parquet with trade_date, ts_code, pred_score, label_rel_return, split.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for topk_metrics.json, topk_daily.csv, and topk_quantiles.csv. Defaults to prediction parent.",
    )
    parser.add_argument("--k", type=parse_k_values, default=parse_k_values("10,20,30"))
    parser.add_argument("--quantiles", type=int, default=10, help="Number of prediction quantile buckets per date.")
    parser.add_argument("--min-daily-count", type=int, default=20)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_series(values: pd.Series) -> dict[str, float | int]:
    clean = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "daily_count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "positive_rate": float("nan"),
        }
    std = clean.std(ddof=1) if len(clean) > 1 else float("nan")
    return {
        "daily_count": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(std),
        "ir": float(clean.mean() / std) if std and math.isfinite(std) else float("nan"),
        "positive_rate": float((clean > 0).mean()),
    }


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    clean = frame.copy()
    clean["trade_date"] = clean["trade_date"].astype(str)
    clean["ts_code"] = clean["ts_code"].astype(str)
    clean["split"] = clean["split"].astype(str)
    clean["pred_score"] = pd.to_numeric(clean["pred_score"], errors="coerce")
    clean["label_rel_return"] = pd.to_numeric(clean["label_rel_return"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["trade_date", "ts_code", "split", "pred_score", "label_rel_return"])
    return clean


def daily_topk(split: str, frame: pd.DataFrame, k_values: list[int], min_daily_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group.sort_values("pred_score", ascending=False)
        n = int(len(group))
        if n < min_daily_count:
            continue
        market_mean = float(group["label_rel_return"].mean())
        for k in k_values:
            if n < k:
                continue
            top = group.head(k)
            bottom = group.tail(k)
            top_mean = float(top["label_rel_return"].mean())
            bottom_mean = float(bottom["label_rel_return"].mean())
            rows.append(
                {
                    "split": split,
                    "trade_date": trade_date,
                    "k": int(k),
                    "daily_count": n,
                    "top_mean": top_mean,
                    "bottom_mean": bottom_mean,
                    "long_short_spread": top_mean - bottom_mean,
                    "top_excess_vs_daily_mean": top_mean - market_mean,
                    "bottom_excess_vs_daily_mean": bottom_mean - market_mean,
                }
            )
    return pd.DataFrame(rows)


def quantile_proxy(split: str, frame: pd.DataFrame, quantiles: int, min_daily_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if quantiles < 2:
        return pd.DataFrame(rows)

    for trade_date, group in frame.groupby("trade_date", sort=True):
        n = int(len(group))
        if n < max(min_daily_count, quantiles):
            continue
        ranked = group.sort_values("pred_score", ascending=True).copy()
        ranked["bucket"] = pd.qcut(
            ranked["pred_score"].rank(method="first"),
            q=quantiles,
            labels=False,
            duplicates="drop",
        )
        for bucket, bucket_frame in ranked.groupby("bucket", sort=True):
            rows.append(
                {
                    "split": split,
                    "trade_date": trade_date,
                    "bucket": int(bucket) + 1,
                    "quantiles": int(quantiles),
                    "count": int(len(bucket_frame)),
                    "mean_label_rel_return": float(bucket_frame["label_rel_return"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_summary(daily: pd.DataFrame, quantiles: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if daily.empty:
        return summary

    for split, split_daily in daily.groupby("split", sort=True):
        split_summary: dict[str, Any] = {}
        for k, group in split_daily.groupby("k", sort=True):
            split_summary[f"top_{int(k)}"] = {
                "top": summarize_series(group["top_mean"]),
                "bottom": summarize_series(group["bottom_mean"]),
                "long_short_spread": summarize_series(group["long_short_spread"]),
                "top_excess_vs_daily_mean": summarize_series(group["top_excess_vs_daily_mean"]),
                "bottom_excess_vs_daily_mean": summarize_series(group["bottom_excess_vs_daily_mean"]),
            }

        if not quantiles.empty:
            q = quantiles[quantiles["split"] == split]
            if not q.empty:
                bucket_mean = (
                    q.groupby("bucket", sort=True)["mean_label_rel_return"]
                    .mean()
                    .rename("mean_label_rel_return")
                    .reset_index()
                )
                low = bucket_mean.loc[bucket_mean["bucket"].idxmin(), "mean_label_rel_return"]
                high = bucket_mean.loc[bucket_mean["bucket"].idxmax(), "mean_label_rel_return"]
                split_summary["quantile_mean_by_bucket"] = {
                    str(int(row["bucket"])): float(row["mean_label_rel_return"])
                    for _, row in bucket_mean.iterrows()
                }
                split_summary["quantile_high_minus_low"] = float(high - low)

        summary[str(split)] = split_summary
    return summary


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = validate_predictions(pd.read_parquet(predictions_path))
    daily_frames: list[pd.DataFrame] = []
    quantile_frames: list[pd.DataFrame] = []

    for split, split_frame in predictions.groupby("split", sort=True):
        daily_frames.append(daily_topk(split, split_frame, args.k, args.min_daily_count))
        quantile_frames.append(quantile_proxy(split, split_frame, args.quantiles, args.min_daily_count))

    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    quantiles = pd.concat(quantile_frames, ignore_index=True) if quantile_frames else pd.DataFrame()
    summary = {
        "predictions": str(predictions_path),
        "rows": int(len(predictions)),
        "k_values": args.k,
        "quantiles": int(args.quantiles),
        "min_daily_count": int(args.min_daily_count),
        "summary": build_summary(daily, quantiles),
    }

    daily.to_csv(output_dir / "topk_daily.csv", index=False)
    quantiles.to_csv(output_dir / "topk_quantiles.csv", index=False)
    (output_dir / "topk_metrics.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
