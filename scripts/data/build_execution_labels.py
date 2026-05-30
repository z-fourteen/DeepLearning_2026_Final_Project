from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "vwap",
]
STATE_COLUMNS = [
    "trade_date",
    "ts_code",
    "is_st",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "is_tradable",
    "price_valid",
    "volume_valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build T+1 execution-aware label table for fill simulation."
    )
    parser.add_argument("--daily-root", default="data/lake/raw/daily")
    parser.add_argument("--state-path", default="data/lake/state/security_daily_state.parquet")
    parser.add_argument("--market-root", default="data/lake/raw/market")
    parser.add_argument("--benchmark", default="399006.SZ")
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument(
        "--output",
        default="data/mart/labels/execution_labels_v20260526.parquet",
    )
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


def infer_partition_date(path: Path) -> str | None:
    for part in path.parts[::-1]:
        if part.startswith("trade_date="):
            return part.split("=", 1)[1]
    return None


def read_daily_files(daily_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(daily_root.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        if "trade_date" not in frame.columns:
            partition_date = infer_partition_date(path)
            if partition_date is not None:
                frame["trade_date"] = partition_date
        missing = [column for column in DAILY_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} missing daily columns: {missing}")
        frames.append(frame[DAILY_COLUMNS].copy())
    if not frames:
        raise FileNotFoundError(f"No parquet files found under {daily_root}")
    daily = pd.concat(frames, ignore_index=True)
    daily["trade_date"] = daily["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    daily["ts_code"] = daily["ts_code"].astype(str)
    for column in ["open", "high", "low", "close", "vol", "amount", "vwap"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = daily.drop_duplicates(["trade_date", "ts_code"], keep="last")
    return daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def read_state(state_path: Path) -> pd.DataFrame:
    state = pd.read_parquet(state_path, columns=STATE_COLUMNS)
    state["trade_date"] = state["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    state["ts_code"] = state["ts_code"].astype(str)
    return state.drop_duplicates(["trade_date", "ts_code"], keep="last")


def read_benchmark(market_root: Path, benchmark: str) -> pd.DataFrame:
    files = sorted((market_root / f"ts_code={benchmark}").glob("*.parquet"))
    if not files:
        files = sorted(market_root.rglob(f"{benchmark}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Benchmark market parquet not found for {benchmark}")
    market = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    required = ["trade_date", "open", "close"]
    missing = [column for column in required if column not in market.columns]
    if missing:
        raise ValueError(f"Benchmark market data missing columns: {missing}")
    market["trade_date"] = market["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    market = market.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    for column in ["open", "close"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    return market[["trade_date", "open", "close"]].rename(
        columns={"open": "benchmark_open", "close": "benchmark_close"}
    )


def add_forward_execution_columns(daily: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    result = daily.sort_values(["ts_code", "trade_date"]).copy()
    grouped = result.groupby("ts_code", group_keys=False)
    result["signal_close"] = result["close"]
    result["next_trade_date"] = grouped["trade_date"].shift(-1)
    result["exit_trade_date"] = grouped["trade_date"].shift(-holding_days)
    result["next_open"] = grouped["open"].shift(-1)
    result["next_vwap"] = grouped["vwap"].shift(-1)
    result["next_close"] = grouped["close"].shift(-1)
    result["next_amount"] = grouped["amount"].shift(-1)
    result["next_vol"] = grouped["vol"].shift(-1)
    result["exit_close"] = grouped["close"].shift(-holding_days)
    result["exit_vwap"] = grouped["vwap"].shift(-holding_days)
    result["next_open_to_exit_close_return"] = result["exit_close"] / result["next_open"] - 1.0
    result["next_vwap_to_exit_vwap_return"] = result["exit_vwap"] / result["next_vwap"] - 1.0
    result["next_open_to_exit_vwap_return"] = result["exit_vwap"] / result["next_open"] - 1.0
    result["signal_close_to_exit_close_return"] = result["exit_close"] / result["signal_close"] - 1.0
    return result


def add_benchmark_execution(market: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    result = market.sort_values("trade_date").copy()
    result["benchmark_next_trade_date"] = result["trade_date"].shift(-1)
    result["benchmark_exit_trade_date"] = result["trade_date"].shift(-holding_days)
    result["benchmark_next_open"] = result["benchmark_open"].shift(-1)
    result["benchmark_exit_close"] = result["benchmark_close"].shift(-holding_days)
    result["benchmark_next_open_to_exit_close_return"] = (
        result["benchmark_exit_close"] / result["benchmark_next_open"] - 1.0
    )
    return result[
        [
            "trade_date",
            "benchmark_next_trade_date",
            "benchmark_exit_trade_date",
            "benchmark_next_open_to_exit_close_return",
        ]
    ]


def build_execution_labels(
    daily: pd.DataFrame,
    state: pd.DataFrame,
    benchmark_exec: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    labels = add_forward_execution_columns(daily, holding_days)

    next_state = state.rename(
        columns={
            "trade_date": "next_trade_date",
            "is_st": "next_is_st",
            "is_suspended": "next_is_suspended",
            "is_limit_up": "next_is_limit_up",
            "is_limit_down": "next_is_limit_down",
            "is_tradable": "next_is_tradable",
            "price_valid": "next_price_valid",
            "volume_valid": "next_volume_valid",
        }
    )
    labels = labels.merge(next_state, on=["next_trade_date", "ts_code"], how="left")
    labels = labels.merge(benchmark_exec, on="trade_date", how="left")

    for column in [
        "next_is_st",
        "next_is_suspended",
        "next_is_limit_up",
        "next_is_limit_down",
        "next_is_tradable",
        "next_price_valid",
        "next_volume_valid",
    ]:
        labels[column] = labels[column].fillna(False).astype(bool)

    valid_next_price = labels["next_open"].notna() & labels["next_open"].gt(0)
    valid_next_vwap = labels["next_vwap"].notna() & labels["next_vwap"].gt(0)
    valid_exit = labels["exit_close"].notna() & labels["exit_close"].gt(0)
    active_next = (
        labels["next_is_tradable"]
        & ~labels["next_is_suspended"]
        & labels["next_price_valid"]
        & labels["next_volume_valid"]
        & labels["next_amount"].fillna(0).gt(0)
        & labels["next_vol"].fillna(0).gt(0)
    )
    labels["buy_executable_t1_open"] = active_next & valid_next_price & ~labels["next_is_limit_up"]
    labels["sell_executable_t1_open"] = active_next & valid_next_price & ~labels["next_is_limit_down"]
    labels["entry_vwap_available_t1"] = active_next & valid_next_vwap
    labels["execution_return_open_to_close5"] = labels["next_open_to_exit_close_return"].where(
        valid_next_price & valid_exit
    )
    labels["execution_return_vwap_to_vwap5"] = labels["next_vwap_to_exit_vwap_return"].where(
        valid_next_vwap & labels["exit_vwap"].notna() & labels["exit_vwap"].gt(0)
    )
    labels["execution_excess_open_to_close5"] = (
        labels["execution_return_open_to_close5"]
        - labels["benchmark_next_open_to_exit_close_return"]
    )
    output_columns = [
        "trade_date",
        "ts_code",
        "signal_close",
        "next_trade_date",
        "exit_trade_date",
        "next_open",
        "next_vwap",
        "next_close",
        "exit_close",
        "exit_vwap",
        "next_amount",
        "next_vol",
        "next_is_st",
        "next_is_suspended",
        "next_is_limit_up",
        "next_is_limit_down",
        "next_is_tradable",
        "next_price_valid",
        "next_volume_valid",
        "buy_executable_t1_open",
        "sell_executable_t1_open",
        "entry_vwap_available_t1",
        "signal_close_to_exit_close_return",
        "execution_return_open_to_close5",
        "execution_return_vwap_to_vwap5",
        "benchmark_next_trade_date",
        "benchmark_exit_trade_date",
        "benchmark_next_open_to_exit_close_return",
        "execution_excess_open_to_close5",
    ]
    return labels[output_columns].copy()


def main() -> None:
    args = parse_args()
    daily_root = Path(args.daily_root)
    state_path = Path(args.state_path)
    market_root = Path(args.market_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    daily = read_daily_files(daily_root)
    state = read_state(state_path)
    benchmark = read_benchmark(market_root, args.benchmark)
    benchmark_exec = add_benchmark_execution(benchmark, args.holding_days)
    labels = build_execution_labels(daily, state, benchmark_exec, args.holding_days)
    labels.to_parquet(output_path, index=False)

    summary = {
        "output": str(output_path),
        "rows": int(len(labels)),
        "trade_dates": int(labels["trade_date"].nunique()),
        "stocks": int(labels["ts_code"].nunique()),
        "date_min": str(labels["trade_date"].min()),
        "date_max": str(labels["trade_date"].max()),
        "holding_days": int(args.holding_days),
        "buy_executable_rate": float(labels["buy_executable_t1_open"].mean()),
        "sell_executable_rate": float(labels["sell_executable_t1_open"].mean()),
        "execution_return_coverage": float(labels["execution_return_open_to_close5"].notna().mean()),
    }
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
