from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipelines.ingest.agent import load_yaml
from pipelines.state.query import query_security_state


RAW_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "amount_log",
    "vol_log",
    "log_total_mv",
    "log_circ_mv",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe_ttm_winsor",
    "pb_winsor",
    "ps_ttm_winsor",
    "net_mf_amount_to_amount",
    "large_order_imbalance",
    "main_mf_strength",
    "amplitude",
    "close_position",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "gap_open",
    "intraday_return",
    "benchmark_ret_1d",
    "excess_ret_1d",
    "beta_20d",
    "beta_60d",
    "residual_ret_20d",
    "residual_ret_60d",
    "rsi_14d",
    "macd_diff",
    "macd_dea",
    "macd_hist",
    "ma_ratio_5_20",
    "ma_ratio_20_60",
    "price_to_ma20",
    "bollinger_z_20d",
    "industry_ret_1d_mean",
    "industry_neutral_ret_1d",
    "industry_neutral_ret_20d",
    "industry_turnover_rank",
    "industry_amount_rank",
    "industry_pb_rank",
    "industry_mv_rank",
    "is_limit_up",
    "is_limit_down",
    "has_price_limit",
    "limit_ratio",
    "dist_to_limit_up",
    "dist_to_limit_down",
    "listed_trading_days",
    "weekday",
    "month",
    "is_month_end",
]

STATE_FEATURE_COLUMNS = [
    "is_limit_up",
    "is_limit_down",
    "has_price_limit",
    "limit_ratio",
    "limit_up_price",
    "limit_down_price",
    "listed_trading_days",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_registry(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    path = project_root / config["meta"]["file_registry"]
    if not path.exists():
        raise FileNotFoundError(f"Missing file registry: {path}")
    return pd.read_parquet(path)


def read_raw_dataset(project_root: Path, config: dict[str, Any], dataset: str) -> pd.DataFrame:
    registry = read_registry(project_root, config)
    rows = registry[(registry["dataset"] == dataset) & (registry["status"] == "ingested")]
    if rows.empty:
        raise ValueError(f"No raw dataset found: {dataset}")
    frames = []
    for raw_path in rows["raw_path"].dropna().sort_values():
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" in df.columns:
        df = df.copy()
        df["trade_date"] = df["trade_date"].astype("string").str.replace(r"\.0$", "", regex=True)
    return df


def read_pool(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    pool_path = project_root / config["pool"]["output_dir"] / config["pool"]["scd2_file"]
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing pool SCD2 file: {pool_path}")
    pool = pd.read_parquet(pool_path)
    pool["ts_code"] = pool["ts_code"].astype("string")
    pool["effective_from"] = pool["effective_from"].astype("string")
    pool["effective_to"] = pool["effective_to"].astype("string")
    return pool


def read_basic(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    basic = read_raw_dataset(project_root, config, "basic")
    required = ["ts_code", "industry", "market", "list_date"]
    missing = [column for column in required if column not in basic.columns]
    if missing:
        raise ValueError(f"Basic dataset missing columns: {missing}")
    basic = basic[required].copy()
    basic["ts_code"] = basic["ts_code"].astype("string")
    basic["industry"] = basic["industry"].fillna("UNKNOWN").astype("string")
    basic["market"] = basic["market"].fillna("UNKNOWN").astype("string")
    basic["list_date"] = basic["list_date"].astype("string")
    return basic.drop_duplicates("ts_code", keep="last")


def winsorize_by_date(df: pd.DataFrame, columns: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    result = df.copy()
    grouped = result.groupby("trade_date", group_keys=False)
    for column in columns:
        if column not in result.columns:
            continue
        quantiles = grouped[column].transform(lambda s: s.quantile(lower))
        upper_quantiles = grouped[column].transform(lambda s: s.quantile(upper))
        result[f"{column}_winsor"] = result[column].clip(lower=quantiles, upper=upper_quantiles)
    return result


def industry_rank(df: pd.DataFrame, column: str, output: str, ascending: bool = True) -> pd.DataFrame:
    result = df.copy()
    result[output] = result.groupby(["trade_date", "industry"])[column].rank(pct=True, ascending=ascending)
    return result


def filter_by_pool(panel: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    merged = panel.merge(pool[["ts_code", "effective_from", "effective_to"]], on="ts_code", how="inner")
    mask = (merged["trade_date"].astype(str) >= merged["effective_from"]) & (
        merged["trade_date"].astype(str) <= merged["effective_to"]
    )
    return merged.loc[mask, panel.columns].drop_duplicates(["trade_date", "ts_code"])


def build_base_panel(project_root: Path, config: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    daily = normalize_dates(read_raw_dataset(project_root, config, "daily"))
    metric = normalize_dates(read_raw_dataset(project_root, config, "metric"))
    moneyflow = normalize_dates(read_raw_dataset(project_root, config, "moneyflow"))

    daily = daily[(daily["trade_date"] >= start_date) & (daily["trade_date"] <= end_date)].copy()
    metric = metric[(metric["trade_date"] >= start_date) & (metric["trade_date"] <= end_date)].copy()
    moneyflow = moneyflow[(moneyflow["trade_date"] >= start_date) & (moneyflow["trade_date"] <= end_date)].copy()

    for df in [daily, metric, moneyflow]:
        df["ts_code"] = df["ts_code"].astype("string")

    metric_cols = [
        "trade_date",
        "ts_code",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "circ_mv",
    ]
    moneyflow_cols = [
        "trade_date",
        "ts_code",
        "net_mf_amount",
        "net_mf_vol",
        "buy_lg_amount",
        "sell_lg_amount",
    ]
    panel = daily.merge(metric[metric_cols], on=["trade_date", "ts_code"], how="left")
    panel = panel.merge(moneyflow[moneyflow_cols], on=["trade_date", "ts_code"], how="left")
    panel = panel.merge(read_basic(project_root, config), on="ts_code", how="left")
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return filter_by_pool(panel, read_pool(project_root, config))


def read_market_returns(project_root: Path, config: dict[str, Any], benchmark: str) -> pd.DataFrame:
    market = normalize_dates(read_raw_dataset(project_root, config, "market"))
    market["ts_code"] = market["ts_code"].astype("string")
    market = market[market["ts_code"] == benchmark].copy()
    if market.empty:
        raise ValueError(f"Benchmark not found in market raw dataset: {benchmark}")
    for column in ["close", "pre_close"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = market.sort_values("trade_date")
    market["benchmark_ret_1d"] = market["close"] / market["pre_close"] - 1
    return market[["trade_date", "benchmark_ret_1d"]]


def add_features(panel: pd.DataFrame, windows: list[int], benchmark_returns: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    numeric = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "amount",
        "vol",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "net_mf_amount",
        "net_mf_vol",
        "buy_lg_amount",
        "sell_lg_amount",
    ]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ret_1d"] = df["close"] / df["pre_close"] - 1
    df["ret_5d"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(5) - 1)
    df["ret_20d"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(20) - 1)
    df["amount_log"] = np.log1p(df["amount"].fillna(0).clip(lower=0))
    df["vol_log"] = np.log1p(df["vol"].fillna(0).clip(lower=0))
    df["log_total_mv"] = np.log1p(df["total_mv"].clip(lower=0))
    df["log_circ_mv"] = np.log1p(df["circ_mv"].clip(lower=0))
    df["net_mf_amount_to_amount"] = df["net_mf_amount"] / df["amount"].replace(0, np.nan)
    df["large_order_imbalance"] = (df["buy_lg_amount"] - df["sell_lg_amount"]) / (
        df["buy_lg_amount"] + df["sell_lg_amount"]
    ).replace(0, np.nan)
    df["main_mf_strength"] = (df["buy_lg_amount"] - df["sell_lg_amount"]) / df["amount"].replace(0, np.nan)
    df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"].replace(0, np.nan)
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["upper_shadow_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["pre_close"].replace(0, np.nan)
    df["lower_shadow_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["pre_close"].replace(0, np.nan)
    df["gap_open"] = df["open"] / df["pre_close"] - 1
    df["intraday_return"] = df["close"] / df["open"].replace(0, np.nan) - 1
    df = df.merge(benchmark_returns, on="trade_date", how="left")
    df["excess_ret_1d"] = df["ret_1d"] - df["benchmark_ret_1d"]
    grouped = df.groupby("ts_code", group_keys=False)
    close_grouped = grouped["close"]
    df["rsi_14d"] = grouped["ret_1d"].transform(
        lambda s: 100
        - 100
        / (
            1
            + s.clip(lower=0).rolling(14, min_periods=7).mean()
            / (-s.clip(upper=0).rolling(14, min_periods=7).mean()).replace(0, np.nan)
        )
    )
    ema12 = close_grouped.transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
    ema26 = close_grouped.transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
    df["macd_diff"] = ema12 - ema26
    df["macd_dea"] = grouped["macd_diff"].transform(lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
    df["macd_hist"] = 2 * (df["macd_diff"] - df["macd_dea"])
    ma5 = close_grouped.transform(lambda s: s.rolling(5, min_periods=3).mean())
    ma20 = close_grouped.transform(lambda s: s.rolling(20, min_periods=10).mean())
    ma60 = close_grouped.transform(lambda s: s.rolling(60, min_periods=30).mean())
    close_std20 = close_grouped.transform(lambda s: s.rolling(20, min_periods=10).std())
    df["ma_ratio_5_20"] = ma5 / ma20.replace(0, np.nan) - 1
    df["ma_ratio_20_60"] = ma20 / ma60.replace(0, np.nan) - 1
    df["price_to_ma20"] = df["close"] / ma20.replace(0, np.nan) - 1
    df["bollinger_z_20d"] = (df["close"] - ma20) / close_std20.replace(0, np.nan)
    parsed_trade_date = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df["weekday"] = parsed_trade_date.dt.weekday
    df["month"] = parsed_trade_date.dt.month
    df["is_month_end"] = parsed_trade_date.dt.is_month_end
    df = winsorize_by_date(df, ["pe_ttm", "pb", "ps_ttm"])

    for window in windows:
        df[f"ret_{window}d_mean"] = grouped["ret_1d"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean())
        df[f"ret_{window}d_std"] = grouped["ret_1d"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).std())
        df[f"amount_{window}d_mean"] = grouped["amount"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean())
        df[f"turnover_{window}d_mean"] = grouped["turnover_rate"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean())
        df[f"turnover_{window}d_std"] = grouped["turnover_rate"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).std())
        df[f"net_mf_strength_{window}d_mean"] = grouped["net_mf_amount_to_amount"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        )
        df[f"excess_ret_{window}d_mean"] = grouped["excess_ret_1d"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        )
    df["_benchmark_ret_x_ret"] = df["benchmark_ret_1d"] * df["ret_1d"]
    df["_benchmark_ret_sq"] = df["benchmark_ret_1d"] * df["benchmark_ret_1d"]
    grouped = df.groupby("ts_code", group_keys=False)
    for window in [20, 60]:
        mean_x = grouped["benchmark_ret_1d"].transform(lambda s: s.rolling(window, min_periods=max(10, window // 2)).mean())
        mean_y = grouped["ret_1d"].transform(lambda s: s.rolling(window, min_periods=max(10, window // 2)).mean())
        mean_xy = grouped["_benchmark_ret_x_ret"].transform(
            lambda s: s.rolling(window, min_periods=max(10, window // 2)).mean()
        )
        mean_x2 = grouped["_benchmark_ret_sq"].transform(lambda s: s.rolling(window, min_periods=max(10, window // 2)).mean())
        beta = (mean_xy - mean_x * mean_y) / (mean_x2 - mean_x * mean_x).replace(0, np.nan)
        df[f"beta_{window}d"] = beta
        df[f"residual_ret_{window}d"] = df["ret_1d"] - beta * df["benchmark_ret_1d"]
    df = df.drop(columns=["_benchmark_ret_x_ret", "_benchmark_ret_sq"])
    industry_return = df.groupby(["trade_date", "industry"])["ret_1d"].transform("mean")
    df["industry_ret_1d_mean"] = industry_return
    df["industry_neutral_ret_1d"] = df["ret_1d"] - industry_return
    df["industry_neutral_ret_20d"] = df.groupby("ts_code")["industry_neutral_ret_1d"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    df = industry_rank(df, "turnover_rate", "industry_turnover_rank")
    df = industry_rank(df, "amount", "industry_amount_rank")
    df = industry_rank(df, "pb_winsor", "industry_pb_rank")
    df = industry_rank(df, "circ_mv", "industry_mv_rank")
    df["dist_to_limit_up"] = df["limit_up_price"] / df["close"].replace(0, np.nan) - 1
    df["dist_to_limit_down"] = df["close"] / df["limit_down_price"].replace(0, np.nan) - 1
    return df


def read_benchmark(project_root: Path, config: dict[str, Any], benchmark: str, horizon: int) -> pd.DataFrame:
    market = normalize_dates(read_raw_dataset(project_root, config, "market"))
    market["ts_code"] = market["ts_code"].astype("string")
    market = market[market["ts_code"] == benchmark].copy()
    if market.empty:
        raise ValueError(f"Benchmark not found in market raw dataset: {benchmark}")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.sort_values("trade_date")
    # LEAKAGE_ALLOWED_LABEL_SHIFT: benchmark forward return is a label target, not a feature.
    market["benchmark_future_return"] = market["close"].shift(-horizon) / market["close"] - 1
    return market[["trade_date", "benchmark_future_return"]]


def add_labels(features: pd.DataFrame, benchmark: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = features.sort_values(["ts_code", "trade_date"]).copy()
    # LEAKAGE_ALLOWED_LABEL_SHIFT: stock forward return is the supervised label target.
    df["future_return"] = df.groupby("ts_code")["close"].shift(-horizon) / df["close"] - 1
    df = df.merge(benchmark, on="trade_date", how="left")
    df["label_rel_return"] = df["future_return"] - df["benchmark_future_return"]
    return df


def apply_state_filter(
    features: pd.DataFrame,
    project_root: Path,
    config_path: Path,
    data_version: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    pool_path = project_root / "data/lake/core/chinext_pool/chinext_pool_scd2.parquet"
    state = query_security_state(
        config_path=config_path,
        project_root=project_root,
        data_version=data_version,
        start_date=start_date,
        end_date=end_date,
        tradable_only=True,
        pool_path=str(pool_path),
        columns=["trade_date", "ts_code", "is_tradable", *STATE_FEATURE_COLUMNS],
    )
    return features.merge(state.drop(columns=["is_tradable"]), on=["trade_date", "ts_code"], how="inner")


def add_lagged_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    result = df.sort_values(["ts_code", "trade_date"]).copy()
    available_columns = [column for column in feature_columns if column in result.columns]
    lagged_columns: list[str] = []
    grouped = result.groupby("ts_code", group_keys=False)
    for column in available_columns:
        lagged_name = f"lag1_{column}"
        result[lagged_name] = grouped[column].shift(1)
        lagged_columns.append(lagged_name)
    return result, lagged_columns


def validate_no_future_leakage() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    offenders = [
        match.group(0)
        for match in re.finditer(r"\.shift\(\s*-\s*\d+", text)
        if "LEAKAGE_ALLOWED_LABEL_SHIFT" not in text[max(0, match.start() - 120) : match.start()]
    ]
    return "PASS" if not offenders else "FAIL"


def write_outputs(
    df: pd.DataFrame,
    project_root: Path,
    config: dict[str, Any],
    data_version: str,
    lagged_feature_columns: list[str],
) -> dict[str, str]:
    features_dir = project_root / config["mart"]["features_dir"]
    labels_dir = project_root / config["mart"]["labels_dir"]
    datasets_dir = project_root / config["mart"]["datasets_dir"]
    for directory in [features_dir, labels_dir, datasets_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    feature_cols = ["trade_date", "ts_code", *lagged_feature_columns]
    label_cols = ["trade_date", "ts_code", "future_return", "benchmark_future_return", "label_rel_return"]
    feature_path = features_dir / f"features_daily_{data_version}.parquet"
    label_path = labels_dir / f"labels_{data_version}.parquet"
    dataset_path = datasets_dir / f"dataset_{data_version}.parquet"
    df[feature_cols].to_parquet(feature_path, index=False)
    df[label_cols].to_parquet(label_path, index=False)
    df[[*feature_cols, "future_return", "benchmark_future_return", "label_rel_return"]].dropna(
        subset=["label_rel_return"]
    ).to_parquet(dataset_path, index=False)
    return {"features": str(feature_path), "labels": str(label_path), "dataset": str(dataset_path)}


def update_audit(project_root: Path, config: dict[str, Any], data_version: str, summary: dict[str, Any]) -> None:
    path = project_root / config["logs"]["audit_dir"] / f"{data_version}_audit.json"
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(summary)
    payload["last_audit_merge_at"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_mart_agent(
    project_root: Path,
    config_path: Path,
    features_config_path: Path,
    labels_config_path: Path,
    data_version: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    features_config = load_yaml(features_config_path)
    labels_config = load_yaml(labels_config_path)
    horizon = int(labels_config.get("default_horizon", 5))
    benchmark = labels_config.get("benchmark", "399006.SZ")
    windows = [int(w) for w in features_config.get("rolling_windows", [5, 10, 20])]

    panel = build_base_panel(project_root, config, start_date, end_date)
    state_filtered_panel = apply_state_filter(panel, project_root, config_path, data_version, start_date, end_date)
    benchmark_returns = read_market_returns(project_root, config, benchmark)
    featured = add_features(state_filtered_panel, windows, benchmark_returns)
    featured, lagged_feature_columns = add_lagged_features(
        featured,
        [
            *RAW_FEATURE_COLUMNS,
            *[f"ret_{window}d_mean" for window in windows],
            *[f"ret_{window}d_std" for window in windows],
            *[f"amount_{window}d_mean" for window in windows],
            *[f"turnover_{window}d_mean" for window in windows],
            *[f"turnover_{window}d_std" for window in windows],
            *[f"net_mf_strength_{window}d_mean" for window in windows],
            *[f"excess_ret_{window}d_mean" for window in windows],
        ],
    )
    benchmark_df = read_benchmark(project_root, config, benchmark, horizon)
    labeled = add_labels(featured, benchmark_df, horizon)
    dataset = labeled
    outputs = write_outputs(dataset, project_root, config, data_version, lagged_feature_columns)

    summary = {
        "mart_agent": "PASS",
        "features_rows": int(len(dataset)),
        "dataset_rows": int(dataset["label_rel_return"].notna().sum()),
        "feature_columns": int(len(lagged_feature_columns)),
        "label_horizon": horizon,
        "feature_availability": "lag1_close_to_next_session",
        "future_leakage_check": validate_no_future_leakage(),
        "mart_updated_at": utc_now_iso(),
    }
    update_audit(project_root, config, data_version, summary)
    return {"data_version": data_version, "outputs": outputs, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research data mart assets.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--labels-config", default="configs/labels.yaml")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    result = run_mart_agent(
        project_root=project_root,
        config_path=project_root / args.config,
        features_config_path=project_root / args.features_config,
        labels_config_path=project_root / args.labels_config,
        data_version=args.data_version,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
