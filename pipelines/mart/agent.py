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


FEATURE_COLUMNS = [
    "ret_1d",
    "amount_log",
    "vol_log",
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
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return filter_by_pool(panel, read_pool(project_root, config))


def add_features(panel: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = panel.copy()
    numeric = [
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
    df["amount_log"] = np.log1p(df["amount"].fillna(0).clip(lower=0))
    df["vol_log"] = np.log1p(df["vol"].fillna(0).clip(lower=0))

    grouped = df.groupby("ts_code", group_keys=False)
    for window in windows:
        df[f"ret_{window}d_mean"] = grouped["ret_1d"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean())
        df[f"ret_{window}d_std"] = grouped["ret_1d"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).std())
        df[f"amount_{window}d_mean"] = grouped["amount"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean())
    return df


def read_benchmark(project_root: Path, config: dict[str, Any], benchmark: str) -> pd.DataFrame:
    market = normalize_dates(read_raw_dataset(project_root, config, "market"))
    market["ts_code"] = market["ts_code"].astype("string")
    market = market[market["ts_code"] == benchmark].copy()
    if market.empty:
        raise ValueError(f"Benchmark not found in market raw dataset: {benchmark}")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.sort_values("trade_date")
    market["benchmark_future_return"] = market["close"].shift(-5) / market["close"] - 1
    return market[["trade_date", "benchmark_future_return"]]


def add_labels(features: pd.DataFrame, benchmark: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = features.sort_values(["ts_code", "trade_date"]).copy()
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
        columns=["trade_date", "ts_code", "is_tradable"],
    )
    return features.merge(state[["trade_date", "ts_code"]], on=["trade_date", "ts_code"], how="inner")


def validate_no_future_leakage() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    allowed = "future_return"
    offenders = [
        match.group(0)
        for match in re.finditer(r"\.shift\(\s*-\s*\d+", text)
        if allowed not in text[max(0, match.start() - 80) : match.start()]
    ]
    return "PASS" if not offenders else "FAIL"


def write_outputs(df: pd.DataFrame, project_root: Path, config: dict[str, Any], data_version: str) -> dict[str, str]:
    features_dir = project_root / config["mart"]["features_dir"]
    labels_dir = project_root / config["mart"]["labels_dir"]
    datasets_dir = project_root / config["mart"]["datasets_dir"]
    for directory in [features_dir, labels_dir, datasets_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    feature_cols = ["trade_date", "ts_code", *[col for col in df.columns if col in FEATURE_COLUMNS or re.match(r".*_(5|10|20|60)d_.*", col)]]
    label_cols = ["trade_date", "ts_code", "future_return", "benchmark_future_return", "label_rel_return"]
    feature_path = features_dir / f"features_daily_{data_version}.parquet"
    label_path = labels_dir / f"labels_{data_version}.parquet"
    dataset_path = datasets_dir / f"dataset_{data_version}.parquet"
    df[feature_cols].to_parquet(feature_path, index=False)
    df[label_cols].to_parquet(label_path, index=False)
    df.dropna(subset=["label_rel_return"]).to_parquet(dataset_path, index=False)
    return {"features": str(feature_path), "labels": str(label_path), "dataset": str(dataset_path)}


def update_audit(project_root: Path, config: dict[str, Any], data_version: str, summary: dict[str, Any]) -> None:
    path = project_root / config["logs"]["audit_dir"] / f"{data_version}_audit.json"
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(summary)
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
    featured = add_features(panel, windows)
    benchmark_df = read_benchmark(project_root, config, benchmark)
    labeled = add_labels(featured, benchmark_df, horizon)
    dataset = apply_state_filter(labeled, project_root, config_path, data_version, start_date, end_date)
    outputs = write_outputs(dataset, project_root, config, data_version)

    summary = {
        "mart_agent": "PASS",
        "features_rows": int(len(dataset)),
        "dataset_rows": int(dataset["label_rel_return"].notna().sum()),
        "feature_columns": int(len([c for c in dataset.columns if c not in ["trade_date", "ts_code"]])),
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
