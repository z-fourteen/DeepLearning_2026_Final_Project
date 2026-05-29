from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "trade_date",
    "ts_code",
    "future_return",
    "benchmark_future_return",
    "label_rel_return",
]

EXECUTION_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge close-to-close research labels and execution labels into one canonical table."
    )
    parser.add_argument("--base-labels", default="data/mart/labels/labels_v20260526.parquet")
    parser.add_argument(
        "--execution-labels",
        default="data/mart/labels/execution_labels_v20260526.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/mart/labels/labels_canonical_v20260526.parquet",
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


def read_available(path: Path, columns: list[str]) -> pd.DataFrame:
    available = set(pd.read_parquet(path, engine="pyarrow").columns)
    selected = [column for column in columns if column in available]
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = pd.read_parquet(path, columns=selected)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame.drop_duplicates(["trade_date", "ts_code"], keep="last")


def add_aliases(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    result["label_rel_return_close_to_close5"] = result["label_rel_return"]
    result["next_open_return_5d"] = result["execution_return_open_to_close5"]
    result["next_vwap_return_5d"] = result["execution_return_vwap_to_vwap5"]
    result["buy_executable"] = result["buy_executable_t1_open"]
    result["sell_executable"] = result["sell_executable_t1_open"]
    result["benchmark_next_open_return_5d"] = result["benchmark_next_open_to_exit_close_return"]
    return result


def build_canonical(base_path: Path, execution_path: Path) -> pd.DataFrame:
    base = read_available(base_path, BASE_COLUMNS)
    execution = read_available(execution_path, EXECUTION_COLUMNS)
    labels = base.merge(execution, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    labels = add_aliases(labels)
    ordered = [
        "trade_date",
        "ts_code",
        "future_return",
        "benchmark_future_return",
        "label_rel_return",
        "label_rel_return_close_to_close5",
        "next_open_return_5d",
        "next_vwap_return_5d",
        "execution_excess_open_to_close5",
        "benchmark_next_open_return_5d",
        "buy_executable",
        "sell_executable",
        "buy_executable_t1_open",
        "sell_executable_t1_open",
        "entry_vwap_available_t1",
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
        "signal_close_to_exit_close_return",
        "execution_return_open_to_close5",
        "execution_return_vwap_to_vwap5",
        "benchmark_next_trade_date",
        "benchmark_exit_trade_date",
        "benchmark_next_open_to_exit_close_return",
    ]
    return labels[[column for column in ordered if column in labels.columns]].copy()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = build_canonical(Path(args.base_labels), Path(args.execution_labels))
    labels.to_parquet(output_path, index=False)
    duplicate_count = int(labels.duplicated(["trade_date", "ts_code"]).sum())
    summary = {
        "output": str(output_path),
        "base_labels": args.base_labels,
        "execution_labels": args.execution_labels,
        "rows": int(len(labels)),
        "trade_dates": int(labels["trade_date"].nunique()),
        "stocks": int(labels["ts_code"].nunique()),
        "date_min": str(labels["trade_date"].min()),
        "date_max": str(labels["trade_date"].max()),
        "duplicate_keys": duplicate_count,
        "buy_executable_rate": float(labels["buy_executable"].mean()),
        "sell_executable_rate": float(labels["sell_executable"].mean()),
        "next_open_return_coverage": float(labels["next_open_return_5d"].notna().mean()),
        "next_vwap_return_coverage": float(labels["next_vwap_return_5d"].notna().mean()),
    }
    output_path.with_name(output_path.stem + "_manifest.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
