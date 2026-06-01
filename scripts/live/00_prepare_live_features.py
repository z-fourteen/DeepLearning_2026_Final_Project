from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live.common import load_yaml, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live stage 0.5: materialize data/live/features/features_DATE.parquet for inference."
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--data-version", default="v20260526")
    parser.add_argument("--feature-date", required=True)
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--mart-features", help="Override mart features parquet.")
    parser.add_argument("--output", help="Override live feature panel output path.")
    parser.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help="If mart features are missing, build a best-effort panel from raw daily CSVs.",
    )
    return parser.parse_args()


def normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "trade_date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "trade_date"})
    if "ts_code" not in frame.columns:
        for column in ["code", "symbol"]:
            if column in frame.columns:
                frame = frame.rename(columns={column: "ts_code"})
                break
    if "trade_date" not in frame.columns or "ts_code" not in frame.columns:
        raise ValueError("feature panel must contain trade_date/date and ts_code/code/symbol")
    frame["trade_date"] = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def fill_compatible_feature_aliases(panel: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    panel = panel.copy()
    aliases = {
        "lag1_turnover_cost_proxy__resid_style": "lag1_turnover_cost_proxy",
        "lag1_turnover_20d_std__resid_style": "lag1_turnover_20d_std",
        "lag1_turnover_60d_std__resid_style": "lag1_turnover_60d_std",
        "lag1_amount_rank_pct__resid_style": "lag1_amount_rank_pct",
        "lag1_amount_log__resid_style": "lag1_amount_log",
    }
    derived: list[str] = []
    for target, source in aliases.items():
        if target in expected_features and target not in panel.columns and source in panel.columns:
            panel[target] = panel[source]
            derived.append(f"{target}<={source}")
    if derived:
        print("Derived compatible live feature aliases: " + ", ".join(derived))
    return panel


def build_from_mart(mart_path: Path, feature_date: str, lookback: int, expected_features: list[str]) -> pd.DataFrame:
    if not mart_path.exists():
        raise FileNotFoundError(f"missing mart features: {mart_path}")
    panel = normalize_panel(pd.read_parquet(mart_path))
    panel = fill_compatible_feature_aliases(panel, expected_features)
    missing = [feature for feature in expected_features if feature not in panel.columns]
    if missing:
        raise ValueError(f"mart feature panel missing expected model features: {missing}")
    panel = panel[["trade_date", "ts_code", *expected_features, *[c for c in ["amount", "next_amount"] if c in panel.columns]]]
    if "next_amount" not in panel.columns:
        if "amount" in panel.columns:
            panel["next_amount"] = pd.to_numeric(panel["amount"], errors="coerce").fillna(0.0)
        elif "lag1_amount_log__resid_style" in panel.columns:
            amount_log = pd.to_numeric(panel["lag1_amount_log__resid_style"], errors="coerce").fillna(0.0)
            panel["next_amount"] = np.expm1(amount_log.clip(lower=0.0, upper=30.0))
        else:
            panel["next_amount"] = 0.0
    panel = panel[panel["trade_date"].le(feature_date)].sort_values(["ts_code", "trade_date"])

    valid_codes: list[str] = []
    rows: list[pd.DataFrame] = []
    for code, group in panel.groupby("ts_code", sort=False):
        if group["trade_date"].iloc[-1] != feature_date:
            continue
        tail = group.tail(lookback)
        if len(tail) != lookback:
            continue
        values = tail[expected_features].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            continue
        rows.append(tail)
        valid_codes.append(str(code))

    if not rows:
        raise ValueError(f"no valid {lookback}-day live sequences ending at feature_date={feature_date}")
    result = pd.concat(rows, ignore_index=True)
    print(f"Prepared live feature panel from mart: rows={len(result)} stocks={len(valid_codes)}")
    return result


def raw_daily_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "A股数据" / "daily",
        PROJECT_ROOT / "data" / "lake" / "raw" / "daily",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("cannot locate raw daily directory")


def read_raw_daily(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def build_raw_fallback(feature_date: str, lookback: int, expected_features: list[str]) -> pd.DataFrame:
    root = raw_daily_dir()
    if root.name == "daily" and any(root.glob("trade_date=*")):
        files = sorted(root.glob("trade_date=*/*.parquet"))
        dated_files = [(path.parent.name.split("=", 1)[1], path) for path in files]
    else:
        files = sorted([*root.glob("*.csv"), *root.glob("*.parquet")])
        dated_files = [(path.stem, path) for path in files if path.stem.isdigit()]
    dates = [date for date, _path in dated_files if date <= feature_date]
    dates = sorted(set(dates))[-lookback:]
    by_date = {date: path for date, path in dated_files if date in dates}
    if len(by_date) < lookback:
        raise ValueError(f"raw fallback has only {len(by_date)} dates, need lookback={lookback}")

    frames: list[pd.DataFrame] = []
    for date in dates:
        df = read_raw_daily(by_date[date])
        if "trade_date" not in df.columns and "date" not in df.columns:
            df["trade_date"] = date
        df = normalize_panel(df)
        if "close" not in df.columns:
            raise ValueError(f"raw daily file missing close column: {by_date[date]}")
        for column in ["open", "high", "low", "vol", "amount"]:
            if column not in df.columns:
                df[column] = df["close"] if column in {"open", "high", "low"} else 0.0
        df["trade_date"] = date
        frames.append(df[["trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount"]])
    raw = pd.concat(frames, ignore_index=True).sort_values(["ts_code", "trade_date"])

    grouped = raw.groupby("ts_code", group_keys=False)
    raw["ret_1d"] = pd.to_numeric(raw["close"], errors="coerce") / pd.to_numeric(raw["open"], errors="coerce").replace(0, np.nan) - 1
    raw["amount_log"] = np.log1p(pd.to_numeric(raw["amount"], errors="coerce").fillna(0).clip(lower=0))
    raw["close_position"] = (
        (pd.to_numeric(raw["close"], errors="coerce") - pd.to_numeric(raw["low"], errors="coerce"))
        / (pd.to_numeric(raw["high"], errors="coerce") - pd.to_numeric(raw["low"], errors="coerce")).replace(0, np.nan)
    ).clip(0, 1)
    raw["amount_rank_pct"] = raw.groupby("trade_date")["amount"].rank(pct=True)

    # Last-resort deterministic approximations. This path is for operational fallback only.
    raw["lag1_net_mf_strength_20d_mean"] = 0.0
    raw["lag1_net_mf_strength_60d_mean"] = 0.0
    raw["lag1_close_position"] = raw["close_position"].fillna(0.5)
    raw["lag1_excess_ret_10d_mean"] = grouped["ret_1d"].transform(lambda s: s.rolling(10, min_periods=5).mean()).fillna(0.0)
    raw["lag1_excess_ret_1d"] = raw["ret_1d"].fillna(0.0)
    raw["lag1_excess_ret_5d_mean"] = grouped["ret_1d"].transform(lambda s: s.rolling(5, min_periods=3).mean()).fillna(0.0)
    raw["lag1_industry_neutral_ret_1d"] = raw["ret_1d"].fillna(0.0)
    raw["lag1_ret_1d"] = raw["ret_1d"].fillna(0.0)
    raw["lag1_ret_20d"] = grouped["close"].transform(lambda s: s / s.shift(20) - 1).fillna(0.0)
    raw["lag1_ret_5d_mean"] = grouped["ret_1d"].transform(lambda s: s.rolling(5, min_periods=3).mean()).fillna(0.0)
    raw["lag1_bollinger_z_20d"] = 0.0
    raw["lag1_ma_ratio_20_60"] = 1.0
    raw["lag1_macd_hist"] = 0.0
    raw["lag1_turnover_cost_proxy__resid_style"] = 0.0
    raw["lag1_turnover_20d_std__resid_style"] = 0.0
    raw["lag1_turnover_60d_std__resid_style"] = 0.0
    raw["lag1_amount_rank_pct__resid_style"] = raw["amount_rank_pct"].fillna(0.5)
    raw["lag1_amount_log__resid_style"] = raw["amount_log"].fillna(0.0)

    return build_from_mart_like(raw[["trade_date", "ts_code", *expected_features, "amount"]], feature_date, lookback, expected_features)


def build_from_mart_like(panel: pd.DataFrame, feature_date: str, lookback: int, expected_features: list[str]) -> pd.DataFrame:
    pathless = Path("__in_memory__.parquet")
    panel = normalize_panel(panel)
    panel = panel[panel["trade_date"].le(feature_date)].sort_values(["ts_code", "trade_date"])
    rows = []
    for _code, group in panel.groupby("ts_code", sort=False):
        if group["trade_date"].iloc[-1] == feature_date and len(group.tail(lookback)) == lookback:
            rows.append(group.tail(lookback))
    if not rows:
        raise ValueError(f"no valid fallback sequences ending at {feature_date}; source={pathless}")
    result = pd.concat(rows, ignore_index=True)
    print(f"Prepared live feature panel from raw fallback: rows={len(result)} stocks={result['ts_code'].nunique()}")
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    feature_date = str(args.feature_date)
    lookback = int(args.lookback or config["model"]["lookback"])
    expected_features = [str(item) for item in config["model"]["expected_features"]]
    output = (
        resolve_path(args.output)
        if args.output
        else resolve_path(config["live_inputs"]["feature_panel"].format(trade_date=feature_date, prev_trade_date="NA"))
    )
    mart_path = (
        resolve_path(args.mart_features)
        if args.mart_features
        else PROJECT_ROOT / "data" / "mart" / "features_daily" / f"features_daily_{args.data_version}.parquet"
    )

    try:
        panel = build_from_mart(mart_path, feature_date, lookback, expected_features)
        source = str(mart_path)
        source_type = "mart"
    except Exception as exc:
        if not args.allow_raw_fallback:
            raise
        print(f"Mart feature preparation failed: {exc}", file=sys.stderr)
        print("Falling back to raw daily CSV/parquet approximation.", file=sys.stderr)
        panel = build_raw_fallback(feature_date, lookback, expected_features)
        source = "raw_daily_fallback"
        source_type = "raw_fallback"

    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    csv_output = output.with_suffix(".csv")
    panel.to_csv(csv_output, index=False)
    manifest = output.parent / f"manifest_{feature_date}.json"
    write_json(
        manifest,
        {
            "feature_date": feature_date,
            "data_version": args.data_version,
            "lookback": lookback,
            "source": source,
            "source_type": source_type,
            "output_parquet": str(output),
            "output_csv": str(csv_output),
            "rows": int(len(panel)),
            "stocks": int(panel["ts_code"].nunique()),
            "features": expected_features,
        },
    )
    print(f"Live feature panel written: {output}")
    print(f"CSV mirror written: {csv_output}")
    print(f"Manifest written: {manifest}")


if __name__ == "__main__":
    main()
