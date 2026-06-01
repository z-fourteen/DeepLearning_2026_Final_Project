from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import format_path, load_yaml, normalize_code_column, previous_trading_day, resolve_path, today_yyyymmdd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare minimal live account inputs for a first-day dry run.")
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument("--feature-date", required=True)
    parser.add_argument("--features-parquet")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_if_missing(path: Path, frame: pd.DataFrame, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"exists, keep: {path}")
        return
    frame.to_csv(path, index=False)
    print(f"wrote: {path}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    trade_date = str(args.trade_date)
    feature_date = str(args.feature_date)
    prev_trade_date = previous_trading_day(config, trade_date)
    paths = config["live_inputs"]
    feature_path = (
        resolve_path(args.features_parquet)
        if args.features_parquet
        else format_path(paths["feature_panel"], trade_date=feature_date, prev_trade_date=prev_trade_date)
    )
    if not feature_path.exists():
        raise FileNotFoundError(f"missing live features: {feature_path}")

    features = normalize_code_column(pd.read_parquet(feature_path))
    latest = features[features["trade_date"].astype(str).eq(feature_date)].drop_duplicates("ts_code").copy()
    codes = latest["ts_code"].astype(str).sort_values().tolist()

    empty_positions = pd.DataFrame({"ts_code": codes, "weight": 0.0, "volume": 0})
    write_if_missing(format_path(paths["positions"], trade_date=trade_date, prev_trade_date=prev_trade_date), empty_positions, args.overwrite)
    write_if_missing(
        format_path(paths["previous_close_positions"], trade_date=trade_date, prev_trade_date=prev_trade_date),
        empty_positions,
        args.overwrite,
    )

    if "price" not in latest.columns:
        if "close" in latest.columns:
            latest["price"] = pd.to_numeric(latest["close"], errors="coerce")
        elif "lag1_amount_log__resid_style" in latest.columns:
            # Deterministic reference-price fallback for dry-run order sizing only.
            amount_log = pd.to_numeric(latest["lag1_amount_log__resid_style"], errors="coerce").fillna(0.0)
            latest["price"] = (10.0 + np.log1p(amount_log.clip(lower=0.0))).clip(lower=1.0)
        else:
            latest["price"] = 10.0
    quotes = latest[["ts_code", "price"]].rename(columns={"ts_code": "code"})
    write_if_missing(format_path(paths["price_snapshot"], trade_date=trade_date, prev_trade_date=prev_trade_date), quotes, args.overwrite)


if __name__ == "__main__":
    main()
