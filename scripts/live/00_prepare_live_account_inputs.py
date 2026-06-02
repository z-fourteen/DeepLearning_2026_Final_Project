from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    apply_universe_filter,
    assert_universe,
    format_path,
    load_yaml,
    normalize_code_column,
    previous_trading_day,
    resolve_path,
    today_yyyymmdd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare minimal live account inputs and quote snapshots."
    )
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


def read_live_features(path: Path, config: dict, feature_date: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing live features: {path}")
    if str(path).endswith(".csv"):
        features = normalize_code_column(pd.read_csv(path))
    else:
        features = normalize_code_column(pd.read_parquet(path))
    features = apply_universe_filter(features, config, "account input features stage 00")
    latest = (
        features[features["trade_date"].astype(str).eq(feature_date)]
        .drop_duplicates("ts_code")
        .copy()
    )
    if latest.empty:
        raise ValueError(f"no feature rows for feature_date={feature_date}")
    return latest


def normalize_positions(frame: pd.DataFrame, config: dict, label: str) -> pd.DataFrame:
    frame = normalize_code_column(frame)
    assert_universe(frame, config, label)
    if "volume" not in frame.columns and "shares" in frame.columns:
        frame = frame.rename(columns={"shares": "volume"})
    if "weight" not in frame.columns:
        frame["weight"] = 0.0
    if "volume" not in frame.columns:
        frame["volume"] = 0
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype(int)
    return frame[["ts_code", "weight", "volume"]].copy()


def inherited_or_empty_positions(
    previous_close_path: Path,
    latest: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if previous_close_path.exists():
        inherited = normalize_positions(
            pd.read_csv(previous_close_path),
            config,
            "previous close positions for account input",
        )
        print(f"inherit positions from: {previous_close_path}")
        return inherited
    codes = latest["ts_code"].astype(str).sort_values().tolist()
    print("previous close positions missing; create empty first-day template")
    return pd.DataFrame({"ts_code": codes, "weight": 0.0, "volume": 0})


def attach_reference_prices(latest: pd.DataFrame, feature_date: str) -> pd.DataFrame:
    latest = latest.copy()
    raw_daily_path = Path("A股数据") / "daily" / f"{feature_date}.csv"
    if "price" not in latest.columns:
        if raw_daily_path.exists():
            try:
                raw_daily = normalize_code_column(pd.read_csv(raw_daily_path))
                raw_daily["price"] = pd.to_numeric(raw_daily["close"], errors="coerce")
                raw_prices = raw_daily[["ts_code", "price"]].drop_duplicates("ts_code")
                latest = latest.merge(
                    raw_prices, on="ts_code", how="left", suffixes=("_old", "")
                )
                if "price_old" in latest.columns:
                    latest = latest.drop(columns=["price_old"])
            except Exception as exc:
                print(f"Failed to read raw prices from {raw_daily_path}: {exc}; using fallback prices.")
                latest["price"] = np.nan
        else:
            latest["price"] = np.nan

    missing_price = latest["price"].isna()
    if missing_price.any():
        if "close" in latest.columns:
            latest.loc[missing_price, "price"] = pd.to_numeric(
                latest.loc[missing_price, "close"], errors="coerce"
            )
        elif "lag1_amount_log__resid_style" in latest.columns:
            amount_log = pd.to_numeric(
                latest.loc[missing_price, "lag1_amount_log__resid_style"],
                errors="coerce",
            ).fillna(0.0)
            latest.loc[missing_price, "price"] = (
                10.0 + np.log1p(amount_log.clip(lower=0.0))
            ).clip(lower=1.0)
        else:
            latest.loc[missing_price, "price"] = 10.0
    return latest


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
        else format_path(
            paths["feature_panel"],
            trade_date=feature_date,
            prev_trade_date=prev_trade_date,
        )
    )

    latest = read_live_features(feature_path, config, feature_date)
    previous_close_path = format_path(
        paths["previous_close_positions"],
        trade_date=trade_date,
        prev_trade_date=prev_trade_date,
    )
    positions = inherited_or_empty_positions(previous_close_path, latest, config)
    positions_path = format_path(
        paths["positions"], trade_date=trade_date, prev_trade_date=prev_trade_date
    )
    write_if_missing(positions_path, positions, args.overwrite)

    if previous_close_path.exists():
        print(f"exists, keep previous close positions: {previous_close_path}")
    else:
        write_if_missing(previous_close_path, positions, overwrite=False)

    latest = attach_reference_prices(latest, feature_date)
    quotes = latest[["ts_code", "price"]].rename(columns={"ts_code": "code"})
    write_if_missing(
        format_path(
            paths["price_snapshot"],
            trade_date=trade_date,
            prev_trade_date=prev_trade_date,
        ),
        quotes,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
