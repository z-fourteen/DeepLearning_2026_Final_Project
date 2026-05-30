from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STYLE_COLUMNS = [
    "lag1_log_total_mv",
    "lag1_log_circ_mv",
    "lag1_amount_log",
    "lag1_amount_rank_pct",
    "lag1_turnover_rate",
    "lag1_turnover_20d_mean",
    "lag1_turnover_20d_std",
    "lag1_turnover_cost_proxy",
    "lag1_illiquidity_proxy",
    "lag1_ret_20d_std",
    "lag1_ret_60d_std",
    "lag1_max_drawdown_20d",
    "lag1_beta_20d",
    "lag1_beta_60d",
    "lag1_dist_to_limit_up",
    "lag1_dist_to_limit_down",
    "lag1_near_limit_up_2pct",
    "lag1_near_limit_down_2pct",
    "lag1_limit_touch_up",
    "lag1_limit_touch_down",
    "lag1_is_limit_up",
    "lag1_is_limit_down",
    "lag1_listed_trading_days",
]


BOOL_COLUMNS = {
    "lag1_near_limit_up_2pct",
    "lag1_near_limit_down_2pct",
    "lag1_limit_touch_up",
    "lag1_limit_touch_down",
    "lag1_is_limit_up",
    "lag1_is_limit_down",
}


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one K value is required.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("K values must be positive.")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Top/Bottom K prediction groups.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.parquet.")
    parser.add_argument(
        "--dataset",
        default="data/mart/datasets/dataset_v20260526.parquet",
        help="Dataset parquet containing style and realized return fields.",
    )
    parser.add_argument(
        "--basic",
        default="data/lake/raw/basic/basic_b5ea92fdf45d.parquet",
        help="Basic stock info parquet with industry column.",
    )
    parser.add_argument("--output-dir", help="Output directory. Defaults to prediction parent.")
    parser.add_argument("--split", default="test", help="Split to diagnose, e.g. validation or test.")
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("10,30"))
    parser.add_argument("--min-daily-count", type=int, default=20)
    parser.add_argument("--worst-n", type=int, default=30)
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


def load_frame(predictions_path: Path, dataset_path: Path, basic_path: Path, split: str) -> pd.DataFrame:
    predictions = pd.read_parquet(predictions_path)
    dataset = pd.read_parquet(dataset_path)
    basic = pd.read_parquet(basic_path)

    for frame in (predictions, dataset, basic):
        if "trade_date" in frame.columns:
            frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)

    predictions = predictions[predictions["split"].astype(str) == split].copy()
    predictions["pred_score"] = pd.to_numeric(predictions["pred_score"], errors="coerce")

    wanted = ["trade_date", "ts_code", "pred_score", "split"]
    dataset_cols = [
        "trade_date",
        "ts_code",
        "future_return",
        "benchmark_future_return",
        "label_rel_return",
        *[col for col in STYLE_COLUMNS if col in dataset.columns],
    ]
    merged = predictions[wanted].merge(
        dataset[dataset_cols],
        on=["trade_date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    if "industry" in basic.columns:
        merged = merged.merge(basic[["ts_code", "industry"]], on="ts_code", how="left")
    else:
        merged["industry"] = "unknown"

    numeric_cols = ["pred_score", "future_return", "benchmark_future_return", "label_rel_return"]
    numeric_cols += [col for col in STYLE_COLUMNS if col in merged.columns]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged.replace([np.inf, -np.inf], np.nan)


def assign_groups(frame: pd.DataFrame, k_values: list[int], min_daily_count: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group.sort_values("pred_score", ascending=False).reset_index(drop=True)
        if len(group) < min_daily_count:
            continue
        for k in k_values:
            if len(group) < k:
                continue
            top = group.head(k).copy()
            top["diagnostic_group"] = f"top_{k}"
            bottom = group.tail(k).copy()
            bottom["diagnostic_group"] = f"bottom_{k}"
            rows.extend([top, bottom])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_group(grouped: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "future_return",
        "label_rel_return",
        "pred_score",
        *[col for col in STYLE_COLUMNS if col in grouped.columns],
    ]
    for name, group in grouped.groupby("diagnostic_group", sort=True):
        row: dict[str, Any] = {
            "diagnostic_group": name,
            "rows": int(len(group)),
            "dates": int(group["trade_date"].nunique()),
            "unique_stocks": int(group["ts_code"].nunique()),
            "mean_future_return": float(group["future_return"].mean()),
            "mean_label_rel_return": float(group["label_rel_return"].mean()),
        }
        for col in metrics:
            if col in BOOL_COLUMNS:
                continue
            series = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(series.mean())
            row[f"{col}_median"] = float(series.median())
            row[f"{col}_p10"] = float(series.quantile(0.1))
            row[f"{col}_p90"] = float(series.quantile(0.9))
        for col in sorted(BOOL_COLUMNS & set(group.columns)):
            row[f"{col}_rate"] = float(group[col].fillna(False).astype(bool).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_industry(grouped: pd.DataFrame) -> pd.DataFrame:
    if "industry" not in grouped.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for name, group in grouped.groupby("diagnostic_group", sort=True):
        counts = group["industry"].fillna("unknown").value_counts(dropna=False)
        total = int(counts.sum())
        for rank, (industry, count) in enumerate(counts.head(10).items(), start=1):
            rows.append(
                {
                    "diagnostic_group": name,
                    "rank": rank,
                    "industry": str(industry),
                    "count": int(count),
                    "share": float(count / total) if total else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def worst_contributors(grouped: pd.DataFrame, worst_n: int) -> pd.DataFrame:
    columns = [
        "diagnostic_group",
        "trade_date",
        "ts_code",
        "industry",
        "pred_score",
        "future_return",
        "label_rel_return",
        "lag1_log_total_mv",
        "lag1_amount_log",
        "lag1_turnover_rate",
        "lag1_ret_20d_std",
        "lag1_is_limit_up",
        "lag1_is_limit_down",
    ]
    available = [col for col in columns if col in grouped.columns]
    rows: list[pd.DataFrame] = []
    for name, group in grouped.groupby("diagnostic_group", sort=True):
        rows.append(group.nsmallest(worst_n, "future_return")[available])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=available)


def spread_diagnostics(grouped: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary = summarize_group(grouped).set_index("diagnostic_group")
    pairs = [(10, 30)]
    compare_cols = [
        "future_return_mean",
        "label_rel_return_mean",
        "lag1_log_total_mv_mean",
        "lag1_amount_log_mean",
        "lag1_turnover_rate_mean",
        "lag1_ret_20d_std_mean",
        "lag1_illiquidity_proxy_mean",
        "lag1_is_limit_up_rate",
        "lag1_is_limit_down_rate",
        "lag1_near_limit_up_2pct_rate",
        "lag1_near_limit_down_2pct_rate",
    ]
    for small, large in pairs:
        for side in ["top", "bottom"]:
            left = f"{side}_{small}"
            right = f"{side}_{large}"
            if left not in summary.index or right not in summary.index:
                continue
            row: dict[str, Any] = {"comparison": f"{left}_minus_{right}"}
            for col in compare_cols:
                if col in summary.columns:
                    row[col] = float(summary.loc[left, col] - summary.loc[right, col])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frame(predictions_path, Path(args.dataset), Path(args.basic), args.split)
    grouped = assign_groups(frame, args.k, args.min_daily_count)
    group_summary = summarize_group(grouped)
    industry = summarize_industry(grouped)
    worst = worst_contributors(grouped, args.worst_n)
    spread = spread_diagnostics(grouped)

    prefix = f"topk_diagnosis_{args.split}"
    group_summary.to_csv(output_dir / f"{prefix}_groups.csv", index=False)
    industry.to_csv(output_dir / f"{prefix}_industry.csv", index=False)
    worst.to_csv(output_dir / f"{prefix}_worst.csv", index=False)
    spread.to_csv(output_dir / f"{prefix}_spread.csv", index=False)

    report = {
        "predictions": str(predictions_path),
        "dataset": str(args.dataset),
        "basic": str(args.basic),
        "split": args.split,
        "rows_after_merge": int(len(frame)),
        "diagnostic_rows": int(len(grouped)),
        "k_values": args.k,
        "outputs": {
            "groups": str(output_dir / f"{prefix}_groups.csv"),
            "industry": str(output_dir / f"{prefix}_industry.csv"),
            "worst": str(output_dir / f"{prefix}_worst.csv"),
            "spread": str(output_dir / f"{prefix}_spread.csv"),
        },
        "top10_minus_top30": spread.to_dict(orient="records"),
    }
    (output_dir / f"{prefix}.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
