from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTROL_SETS = {
    "full_style": [
        "lag1_log_circ_mv",
        "lag1_log_total_mv",
        "lag1_amount_log",
        "lag1_amount_rank_pct",
        "lag1_turnover_rate_f",
        "lag1_turnover_20d_mean",
        "lag1_ret_20d_std",
        "lag1_ret_60d_std",
        "lag1_ret_20d",
        "lag1_ret_60d_mean",
        "lag1_beta_20d",
        "lag1_beta_60d",
        "lag1_pb_winsor",
        "lag1_pe_ttm_winsor",
    ],
    "industry_proxy_full_style": [
        "lag1_industry_turnover_rank",
        "lag1_industry_amount_rank",
        "lag1_industry_pb_rank",
        "lag1_industry_mv_rank",
        "lag1_log_circ_mv",
        "lag1_log_total_mv",
        "lag1_amount_log",
        "lag1_amount_rank_pct",
        "lag1_turnover_rate_f",
        "lag1_ret_20d_std",
        "lag1_ret_20d",
        "lag1_beta_60d",
        "lag1_pb_winsor",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily cross-sectionally residualized prediction scores.")
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet",
    )
    parser.add_argument("--mart", default="data/mart/datasets/dataset_v20260526.parquet")
    parser.add_argument("--control-set", choices=sorted(CONTROL_SETS), default="industry_proxy_full_style")
    parser.add_argument(
        "--output",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_resid_industry_proxy_full_style/predictions.parquet",
    )
    parser.add_argument("--min-count", type=int, default=40)
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


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.astype("float64").replace([np.inf, -np.inf], np.nan)
    result = result.fillna(result.median(numeric_only=True))
    std = result.std(ddof=0).replace(0, np.nan)
    return ((result - result.mean()) / std).fillna(0.0)


def residualize_group(group: pd.DataFrame, controls: list[str], min_count: int) -> pd.DataFrame:
    result = group.copy()
    result["pred_score_raw"] = result["pred_score"]
    available = [column for column in controls if column in group.columns]
    frame = group[["pred_score", *available]].dropna(subset=["pred_score"])
    if len(frame) < max(min_count, len(available) + 5) or not available:
        resid = group["pred_score"] - group["pred_score"].mean()
    else:
        x = zscore(frame[available])
        y = pd.to_numeric(frame["pred_score"], errors="coerce").astype("float64")
        design = np.column_stack([np.ones(len(frame)), x.to_numpy(dtype="float64")])
        coef = np.linalg.lstsq(design, y.to_numpy(dtype="float64"), rcond=None)[0]
        fitted = design @ coef
        resid = pd.Series(np.nan, index=group.index, dtype="float64")
        resid.loc[frame.index] = y.to_numpy(dtype="float64") - fitted
        resid = resid.fillna(group["pred_score"] - group["pred_score"].mean())
    std = resid.std(ddof=0)
    result["pred_score"] = (resid - resid.mean()) / std if std and pd.notna(std) else resid - resid.mean()
    return result


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    controls = CONTROL_SETS[args.control_set]
    predictions = pd.read_parquet(args.predictions)
    mart_cols = set(pd.read_parquet(args.mart, engine="pyarrow").columns)
    selected = ["trade_date", "ts_code", *[column for column in controls if column in mart_cols]]
    mart = pd.read_parquet(args.mart, columns=selected)
    data = predictions.merge(mart, on=["trade_date", "ts_code"], how="left")
    for column in ["trade_date", "ts_code", "split"]:
        data[column] = data[column].astype(str)
    for column in ["pred_score", *selected[2:]]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    out = (
        data.groupby(["split", "trade_date"], group_keys=False)
        .apply(lambda group: residualize_group(group, controls, args.min_count))
        .reset_index(drop=True)
    )
    out["model_name"] = out.get("model_name", "gru_baseline").astype(str) + "_resid_" + args.control_set
    keep_cols = [
        column
        for column in [
            "trade_date",
            "ts_code",
            "pred_score",
            "pred_score_raw",
            "label_rel_return",
            "split",
            "model_name",
        ]
        if column in out.columns
    ]
    out[keep_cols].to_parquet(output_path, index=False)
    manifest = {
        "input": args.predictions,
        "mart": args.mart,
        "control_set": args.control_set,
        "controls": [column for column in controls if column in mart_cols],
        "output": str(output_path),
        "rows": int(len(out)),
        "date_count": int(out["trade_date"].nunique()),
    }
    output_path.with_name("manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
