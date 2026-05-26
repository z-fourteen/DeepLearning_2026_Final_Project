from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipelines.ingest.agent import load_yaml
from pipelines.state.agent import STATE_COLUMNS, available_daily_dates, raw_root, state_root


def state_partitions(output_root: Path, start_date: str, end_date: str) -> list[Path]:
    if not output_root.exists():
        raise FileNotFoundError(f"Missing state root: {output_root}")
    partitions = []
    for partition in output_root.glob("trade_date=*"):
        if not partition.is_dir():
            continue
        trade_date = partition.name.split("=", 1)[1]
        if start_date <= trade_date <= end_date:
            partitions.append(partition)
    return sorted(partitions)


def read_partitions(partitions: Iterable[Path], columns: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for partition in partitions:
        read_columns = [column for column in columns if column != "trade_date"] if columns else None
        df = pd.read_parquet(partition, columns=read_columns)
        if "trade_date" not in df.columns:
            df["trade_date"] = partition.name.split("=", 1)[1]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=columns or STATE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def validate_requested_coverage(
    config: dict,
    project_root: Path,
    output_root: Path,
    data_version: str,
    start_date: str,
    end_date: str,
) -> None:
    raw_dates = available_daily_dates(raw_root(project_root, config))
    expected = {date for date in raw_dates if start_date <= date <= end_date}
    actual: set[str] = set()
    for partition in output_root.glob("trade_date=*"):
        if not partition.is_dir():
            continue
        trade_date = partition.name.split("=", 1)[1]
        if trade_date < start_date or trade_date > end_date:
            continue
        df = pd.read_parquet(partition, columns=["state_version"])
        if not df.empty and df["state_version"].astype(str).eq(data_version).any():
            actual.add(trade_date)
    missing = sorted(expected - actual)
    if missing:
        preview = missing[:5]
        raise ValueError(f"State coverage missing {len(missing)} trade dates, first missing={preview}")


def load_pool_filter(project_root: Path, pool_path: str | None, start_date: str, end_date: str) -> pd.DataFrame | None:
    if not pool_path:
        return None
    path = Path(pool_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"Missing pool file: {path}")
    pool = pd.read_parquet(path)
    required = {"ts_code", "effective_from", "effective_to"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Pool file missing columns: {sorted(missing)}")
    pool = pool.copy()
    pool["ts_code"] = pool["ts_code"].astype("string")
    pool["effective_from"] = pool["effective_from"].astype("string")
    pool["effective_to"] = pool["effective_to"].astype("string")
    return pool[(pool["effective_from"] <= end_date) & (pool["effective_to"] >= start_date)][
        ["ts_code", "effective_from", "effective_to"]
    ]


def apply_pool_filter(state: pd.DataFrame, pool: pd.DataFrame | None) -> pd.DataFrame:
    if pool is None or state.empty:
        return state
    merged = state.merge(pool, on="ts_code", how="inner")
    mask = (merged["trade_date"].astype(str) >= merged["effective_from"]) & (
        merged["trade_date"].astype(str) <= merged["effective_to"]
    )
    return merged.loc[mask, state.columns].drop_duplicates(["trade_date", "ts_code", "state_version"])


def query_security_state(
    config_path: Path,
    project_root: Path,
    data_version: str,
    start_date: str,
    end_date: str,
    tradable_only: bool = False,
    require_price_valid: bool = False,
    require_volume_valid: bool = False,
    ts_codes: list[str] | None = None,
    pool_path: str | None = None,
    columns: list[str] | None = None,
    validate_coverage: bool = True,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    output_root = state_root(project_root, config)
    if validate_coverage:
        validate_requested_coverage(config, project_root, output_root, data_version, start_date, end_date)

    requested_columns = columns[:] if columns else STATE_COLUMNS[:]
    required_for_filters = ["trade_date", "ts_code", "state_version", "is_tradable", "price_valid", "volume_valid"]
    for column in required_for_filters:
        if column not in requested_columns:
            requested_columns.append(column)

    state = read_partitions(state_partitions(output_root, start_date, end_date), columns=requested_columns)
    if state.empty:
        return state
    state = state[state["state_version"].astype(str) == data_version]
    state = state[(state["trade_date"].astype(str) >= start_date) & (state["trade_date"].astype(str) <= end_date)]

    if ts_codes:
        allowed = set(ts_codes)
        state = state[state["ts_code"].astype(str).isin(allowed)]
    state = apply_pool_filter(state, load_pool_filter(project_root, pool_path, start_date, end_date))
    if tradable_only:
        state = state[state["is_tradable"].eq(True)]
    if require_price_valid:
        state = state[state["price_valid"].eq(True)]
    if require_volume_valid:
        state = state[state["volume_valid"].eq(True)]

    if columns:
        return state[columns]
    return state[STATE_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the security daily state layer.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--tradable-only", action="store_true")
    parser.add_argument("--require-price-valid", action="store_true")
    parser.add_argument("--require-volume-valid", action="store_true")
    parser.add_argument("--ts-code", action="append", dest="ts_codes")
    parser.add_argument("--pool-path")
    parser.add_argument("--columns", help="Comma separated output columns.")
    parser.add_argument("--output")
    parser.add_argument("--no-coverage-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    columns = [item.strip() for item in args.columns.split(",")] if args.columns else None
    state = query_security_state(
        config_path=project_root / args.config,
        project_root=project_root,
        data_version=args.data_version,
        start_date=args.start_date,
        end_date=args.end_date,
        tradable_only=args.tradable_only,
        require_price_valid=args.require_price_valid,
        require_volume_valid=args.require_volume_valid,
        ts_codes=args.ts_codes,
        pool_path=args.pool_path,
        columns=columns,
        validate_coverage=not args.no_coverage_check,
    )
    summary = {
        "rows": int(len(state)),
        "trade_dates": int(state["trade_date"].nunique()) if "trade_date" in state.columns and not state.empty else 0,
        "ts_codes": int(state["ts_code"].nunique()) if "ts_code" in state.columns and not state.empty else 0,
    }
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = project_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        state.to_parquet(output, index=False)
        summary["output"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
