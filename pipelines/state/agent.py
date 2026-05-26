from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


STATE_COLUMNS = [
    "trade_date",
    "ts_code",
    "is_st",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "is_tradable",
    "listed_days",
    "volume_valid",
    "price_valid",
    "state_version",
    "created_at",
]

@dataclass(frozen=True)
class BuildSummary:
    state_rows: int
    st_filtered: int
    invalid_price_rows: int
    suspended_rows: int
    trade_dates: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def state_root(project_root: Path, config: dict[str, Any]) -> Path:
    configured = config.get("state", {}).get("output_dir")
    if configured:
        return project_root / configured
    return project_root / config["lake"]["state_dir"] / "security_daily_state.parquet"


def raw_root(project_root: Path, config: dict[str, Any]) -> Path:
    return project_root / config["lake"]["raw_dir"]


def read_parquet_files(paths: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def dataset_files(raw_dir: Path, dataset: str) -> list[Path]:
    dataset_dir = raw_dir / dataset
    if not dataset_dir.exists():
        return []
    return sorted(path for path in dataset_dir.rglob("*.parquet") if path.is_file())


def trade_date_from_partition(path: Path) -> str | None:
    for part in path.parts:
        match = re.fullmatch(r"trade_date=(\d{8})", part)
        if match:
            return match.group(1)
    return None


def available_daily_dates(raw_dir: Path) -> list[str]:
    dates = {
        date
        for path in dataset_files(raw_dir, "daily")
        for date in [trade_date_from_partition(path)]
        if date
    }
    return sorted(dates)


def existing_state_dates(output_root: Path, state_version: str) -> set[str]:
    if not output_root.exists():
        return set()
    dates: set[str] = set()
    for partition in output_root.glob("trade_date=*"):
        if not partition.is_dir():
            continue
        trade_date = partition.name.split("=", 1)[1]
        try:
            df = pd.read_parquet(partition, columns=["state_version"])
        except Exception:
            continue
        if not df.empty and (df["state_version"].astype(str) == state_version).any():
            dates.add(trade_date)
    return dates


def read_state_partition(partition: Path) -> pd.DataFrame:
    df = pd.read_parquet(partition)
    if "trade_date" not in df.columns:
        df["trade_date"] = partition.name.split("=", 1)[1]
    return df[STATE_COLUMNS]


def partition_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    mtimes = [child.stat().st_mtime for child in path.rglob("*") if child.is_file()]
    return max(mtimes) if mtimes else path.stat().st_mtime


def raw_date_mtime(raw_dir: Path, trade_date: str) -> float:
    mtimes: list[float] = []
    for dataset in ["daily", "stock_st"]:
        partition = raw_dir / dataset / f"trade_date={trade_date}"
        if partition.exists():
            mtimes.extend(path.stat().st_mtime for path in partition.glob("*.parquet"))
    return max(mtimes) if mtimes else 0.0


def incremental_dates(raw_dir: Path, output_root: Path, state_version: str) -> list[str]:
    dates = available_daily_dates(raw_dir)
    existing = existing_state_dates(output_root, state_version)
    affected: list[str] = []
    for trade_date in dates:
        partition = output_root / f"trade_date={trade_date}"
        if trade_date not in existing:
            affected.append(trade_date)
            continue
        if raw_date_mtime(raw_dir, trade_date) > partition_mtime(partition):
            affected.append(trade_date)
    return affected


def normalize_date(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.replace(r"\.0$", "", regex=True)
    return values.mask(values.isin(["<NA>", "nan", "NaN", "None"]), pd.NA)


def load_basic(raw_dir: Path) -> pd.DataFrame:
    basic = read_parquet_files(dataset_files(raw_dir, "basic"))
    if basic.empty:
        return pd.DataFrame(columns=["ts_code", "list_date"])
    basic = basic.copy()
    basic["ts_code"] = basic["ts_code"].astype("string")
    basic["list_date"] = normalize_date(basic["list_date"])
    return basic[["ts_code", "list_date"]].drop_duplicates("ts_code", keep="last")


def load_trade_calendar(raw_dir: Path) -> pd.DataFrame:
    cal = read_parquet_files(dataset_files(raw_dir, "trade_cal"))
    if cal.empty:
        return pd.DataFrame(columns=["cal_date", "is_open", "trade_index"])
    cal = cal.copy()
    cal["cal_date"] = normalize_date(cal["cal_date"])
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    cal = cal.loc[cal["is_open"] == 1, ["cal_date"]].dropna().drop_duplicates()
    cal = cal.sort_values("cal_date").reset_index(drop=True)
    cal["trade_index"] = range(len(cal))
    return cal


def load_daily(raw_dir: Path, trade_date: str) -> pd.DataFrame:
    partition = raw_dir / "daily" / f"trade_date={trade_date}"
    daily = read_parquet_files(sorted(partition.glob("*.parquet")) if partition.exists() else [])
    if daily.empty:
        return pd.DataFrame()
    daily = daily.copy()
    daily["trade_date"] = normalize_date(daily["trade_date"]).fillna(trade_date)
    daily["ts_code"] = daily["ts_code"].astype("string")
    numeric_columns = ["open", "high", "low", "close", "pre_close", "vol", "pct_chg"]
    for column in numeric_columns:
        if column not in daily.columns:
            daily[column] = pd.NA
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    return daily.drop_duplicates(["trade_date", "ts_code"], keep="last")


def load_stock_st(raw_dir: Path, trade_date: str) -> pd.DataFrame:
    partition = raw_dir / "stock_st" / f"trade_date={trade_date}"
    st = read_parquet_files(sorted(partition.glob("*.parquet")) if partition.exists() else [])
    if st.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code"])
    st = st.copy()
    st["trade_date"] = normalize_date(st["trade_date"]).fillna(trade_date)
    st["ts_code"] = st["ts_code"].astype("string")
    return st.loc[st["trade_date"] == trade_date, ["trade_date", "ts_code"]].drop_duplicates()


def compute_listed_days(state: pd.DataFrame, trade_cal: pd.DataFrame) -> pd.Series:
    if trade_cal.empty:
        trade_dt = pd.to_datetime(state["trade_date"], format="%Y%m%d", errors="coerce")
        list_dt = pd.to_datetime(state["list_date"], format="%Y%m%d", errors="coerce")
        days = (trade_dt - list_dt).dt.days
        return days.fillna(0).clip(lower=0).astype("int64")

    cal_index = trade_cal.set_index("cal_date")["trade_index"]
    trade_index = state["trade_date"].map(cal_index)
    list_index = state["list_date"].map(cal_index)
    missing_list = list_index.isna() & state["list_date"].notna()
    if missing_list.any():
        dates = trade_cal["cal_date"].to_numpy()
        positions = pd.Series(
            dates.searchsorted(state.loc[missing_list, "list_date"].to_numpy()),
            index=state.index[missing_list],
        )
        list_index.loc[missing_list] = positions
    days = trade_index - list_index + 1
    return days.fillna(0).clip(lower=0).astype("int64")


def build_state_for_date(
    raw_dir: Path,
    trade_date: str,
    data_version: str,
    basic: pd.DataFrame,
    trade_cal: pd.DataFrame,
    created_at: str,
    min_listed_days: int,
) -> pd.DataFrame:
    daily = load_daily(raw_dir, trade_date)
    if daily.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)

    st = load_stock_st(raw_dir, trade_date)
    state = daily.merge(basic, on="ts_code", how="left")
    state = state.merge(st.assign(is_st=True), on=["trade_date", "ts_code"], how="left")
    state["is_st"] = state["is_st"].eq(True)

    price_columns = ["open", "high", "low", "close", "pre_close"]
    state["price_valid"] = state[price_columns].notna().all(axis=1) & (state["close"] > 0)
    state["volume_valid"] = state["vol"].notna() & (state["vol"] > 0)
    state["is_suspended"] = (~state["volume_valid"]).astype(bool)
    state["is_limit_up"] = (state["pct_chg"] >= 9.8).fillna(False).astype(bool)
    state["is_limit_down"] = (state["pct_chg"] <= -9.8).fillna(False).astype(bool)
    state["listed_days"] = compute_listed_days(state, trade_cal)
    state["is_tradable"] = (
        (~state["is_st"])
        & (~state["is_suspended"])
        & state["price_valid"]
        & state["volume_valid"]
        & (state["listed_days"] >= min_listed_days)
    ).astype(bool)
    state["state_version"] = data_version
    state["created_at"] = created_at
    return state[STATE_COLUMNS].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def validate_state(state: pd.DataFrame) -> None:
    if state.empty:
        return
    duplicate = state.duplicated(["trade_date", "ts_code", "state_version"])
    if duplicate.any():
        sample = state.loc[duplicate, ["trade_date", "ts_code", "state_version"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate state keys found: {sample}")
    if state["is_tradable"].isna().any():
        raise ValueError("is_tradable contains null values")
    if (state["listed_days"] < 0).any():
        raise ValueError("listed_days contains negative values")
    if (state["is_tradable"] & ~state["price_valid"]).any():
        raise ValueError("price_valid=false rows cannot be tradable")
    if (state["is_tradable"] & ~state["volume_valid"]).any():
        raise ValueError("volume_valid=false rows cannot be tradable")


def replace_date_partition(output_root: Path, trade_date: str, state: pd.DataFrame) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    partition = output_root / f"trade_date={trade_date}"
    if partition.exists():
        shutil.rmtree(partition)
    partition.mkdir(parents=True, exist_ok=True)
    state.drop(columns=["trade_date"]).to_parquet(partition / "part.parquet", index=False)


def state_audit_fields(data_version: str, summary: BuildSummary) -> dict[str, Any]:
    return {
        "data_version": data_version,
        "state_agent": "PASS",
        "state_rows": summary.state_rows,
        "state_trade_dates": summary.trade_dates,
        "st_filtered": summary.st_filtered,
        "invalid_price_rows": summary.invalid_price_rows,
        "suspended_rows": summary.suspended_rows,
        "state_updated_at": utc_now_iso(),
    }


def write_audit(project_root: Path, config: dict[str, Any], data_version: str, summary: BuildSummary) -> None:
    audit_dir = project_root / config["logs"]["audit_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {"agent": "market_state", **state_audit_fields(data_version, summary), "created_at": utc_now_iso()}
    with (audit_dir / f"{data_version}_state_audit.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    main_audit_path = audit_dir / f"{data_version}_audit.json"
    if main_audit_path.exists():
        with main_audit_path.open("r", encoding="utf-8") as f:
            main_payload = json.load(f)
    else:
        main_payload = {"data_version": data_version}
    main_payload.update(state_audit_fields(data_version, summary))
    with main_audit_path.open("w", encoding="utf-8") as f:
        json.dump(main_payload, f, ensure_ascii=False, indent=2)


def resolve_trade_dates(
    raw_dir: Path,
    output_root: Path,
    data_version: str,
    trade_date: str | None,
    incremental: bool,
    backfill: bool,
) -> list[str]:
    if trade_date:
        if not re.fullmatch(r"\d{8}", trade_date):
            raise ValueError(f"--trade-date must be YYYYMMDD, got {trade_date}")
        return [trade_date]
    if backfill:
        return available_daily_dates(raw_dir)
    if incremental:
        return incremental_dates(raw_dir, output_root, data_version)
    return available_daily_dates(raw_dir)


def run_market_state(
    config_path: Path,
    project_root: Path,
    data_version: str,
    trade_date: str | None = None,
    incremental: bool = False,
    backfill: bool = False,
    min_listed_days: int = 0,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if min_listed_days == 0:
        min_listed_days = int(config.get("state", {}).get("min_listed_days", 0))
    raw_dir = raw_root(project_root, config)
    output_root = state_root(project_root, config)
    trade_dates = resolve_trade_dates(raw_dir, output_root, data_version, trade_date, incremental, backfill)

    basic = load_basic(raw_dir)
    trade_cal = load_trade_calendar(raw_dir)
    created_at = utc_now_iso()

    totals = BuildSummary(0, 0, 0, 0, 0)
    for current_date in trade_dates:
        state = build_state_for_date(raw_dir, current_date, data_version, basic, trade_cal, created_at, min_listed_days)
        validate_state(state)
        if state.empty:
            continue
        replace_date_partition(output_root, current_date, state)
        totals = BuildSummary(
            state_rows=totals.state_rows + int(len(state)),
            st_filtered=totals.st_filtered + int(state["is_st"].sum()),
            invalid_price_rows=totals.invalid_price_rows + int((~state["price_valid"]).sum()),
            suspended_rows=totals.suspended_rows + int(state["is_suspended"].sum()),
            trade_dates=totals.trade_dates + 1,
        )

    write_audit(project_root, config, data_version, totals)
    return {
        "data_version": data_version,
        "output_path": str(output_root),
        "summary": {
            "state_agent": "PASS",
            "state_rows": totals.state_rows,
            "state_trade_dates": totals.trade_dates,
            "st_filtered": totals.st_filtered,
            "invalid_price_rows": totals.invalid_price_rows,
            "suspended_rows": totals.suspended_rows,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the market security daily state layer.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to the data config YAML.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--data-version", required=True, help="State version, for example v20260526.")
    parser.add_argument("--incremental", action="store_true", help="Build only missing or raw-newer trade_date partitions.")
    parser.add_argument("--trade-date", help="Build exactly one YYYYMMDD trade date.")
    parser.add_argument("--backfill", action="store_true", help="Rebuild all available daily trade dates.")
    parser.add_argument("--min-listed-days", type=int, default=0, help="Minimum listed_days required for is_tradable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    result = run_market_state(
        config_path=project_root / args.config,
        project_root=project_root,
        data_version=args.data_version,
        trade_date=args.trade_date,
        incremental=args.incremental,
        backfill=args.backfill,
        min_listed_days=args.min_listed_days,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
