from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.ingest.agent import load_yaml  # noqa: E402
from pipelines.state.agent import STATE_COLUMNS, available_daily_dates, raw_root, state_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate market state coverage for a date range.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def partition_dates(output_root: Path, data_version: str) -> set[str]:
    dates: set[str] = set()
    if not output_root.exists():
        return dates
    for partition in output_root.glob("trade_date=*"):
        if not partition.is_dir():
            continue
        trade_date = partition.name.split("=", 1)[1]
        try:
            df = pd.read_parquet(partition, columns=["state_version"])
        except Exception:
            continue
        if not df.empty and df["state_version"].astype(str).eq(data_version).any():
            dates.add(trade_date)
    return dates


def validate_state_schema(output_root: Path, sample_dates: list[str]) -> list[str]:
    errors: list[str] = []
    for trade_date in sample_dates:
        partition = output_root / f"trade_date={trade_date}"
        if not partition.exists():
            errors.append(f"missing partition {trade_date}")
            continue
        df = pd.read_parquet(partition)
        if "trade_date" not in df.columns:
            df["trade_date"] = trade_date
        missing = sorted(set(STATE_COLUMNS) - set(df.columns))
        if missing:
            errors.append(f"{trade_date} missing columns {missing}")
            continue
        if df.duplicated(["trade_date", "ts_code", "state_version"]).any():
            errors.append(f"{trade_date} duplicated trade_date+ts_code+state_version")
        if df["is_tradable"].isna().any():
            errors.append(f"{trade_date} null is_tradable")
        if (df["is_tradable"] & ~df["price_valid"]).any():
            errors.append(f"{trade_date} tradable rows with price_valid=false")
        if (df["is_tradable"] & ~df["volume_valid"]).any():
            errors.append(f"{trade_date} tradable rows with volume_valid=false")
    return errors


def main() -> None:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    raw_dates = available_daily_dates(raw_root(PROJECT_ROOT, config))
    start_date = args.start_date or (raw_dates[0] if raw_dates else None)
    end_date = args.end_date or (raw_dates[-1] if raw_dates else None)
    expected = [date for date in raw_dates if (not start_date or date >= start_date) and (not end_date or date <= end_date)]
    actual = partition_dates(state_root(PROJECT_ROOT, config), args.data_version)
    expected_set = set(expected)
    covered = sorted(actual & expected_set)
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)
    sample_dates = sorted(set(covered[:3] + covered[-3:]))
    schema_errors = validate_state_schema(state_root(PROJECT_ROOT, config), sample_dates)
    coverage_check = "PASS" if not missing and not schema_errors else "FAIL"

    result = {
        "data_version": args.data_version,
        "start_date": start_date,
        "end_date": end_date,
        "expected_trade_dates": len(expected),
        "covered_trade_dates": len(covered),
        "missing_trade_dates": len(missing),
        "extra_trade_dates": len(extra),
        "schema_checked_dates": sample_dates,
        "schema_errors": schema_errors,
        "coverage_check": coverage_check,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and coverage_check != "PASS":
        raise ValueError("Market state coverage validation failed")


if __name__ == "__main__":
    main()
