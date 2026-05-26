from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipelines.ingest.agent import load_yaml


OPEN_ENDED_DATE = "99991231"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_file_registry(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    registry_path = project_root / config["meta"]["file_registry"]
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing file registry: {registry_path}")
    return pd.read_parquet(registry_path)


def read_raw_dataset(project_root: Path, registry: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = registry[(registry["dataset"] == dataset) & (registry["status"] == "ingested")]
    if rows.empty:
        raise ValueError(f"No ingested raw files found for dataset={dataset}")

    frames: list[pd.DataFrame] = []
    for raw_path in rows["raw_path"].dropna().sort_values():
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"Raw file registered but missing: {path}")
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_index_weight(index_weight: pd.DataFrame, index_code: str) -> pd.DataFrame:
    required = {"index_code", "con_code", "trade_date", "weight"}
    missing = required - set(index_weight.columns)
    if missing:
        raise ValueError(f"index_weight missing columns: {sorted(missing)}")

    df = index_weight.loc[index_weight["index_code"].astype(str) == index_code].copy()
    if df.empty:
        raise ValueError(f"No index_weight rows found for index_code={index_code}")

    df = df.rename(columns={"con_code": "ts_code", "weight": "index_weight"})
    df["index_code"] = df["index_code"].astype("string")
    df["ts_code"] = df["ts_code"].astype("string")
    df["trade_date"] = df["trade_date"].astype("string").str.replace(r"\.0$", "", regex=True)
    df["index_weight"] = pd.to_numeric(df["index_weight"], errors="coerce")
    df = df.dropna(subset=["ts_code", "trade_date"])
    df = df.drop_duplicates(["index_code", "ts_code", "trade_date"], keep="last")
    return df[["index_code", "ts_code", "trade_date", "index_weight"]]


def build_snapshot_table(index_weight: pd.DataFrame) -> pd.DataFrame:
    snapshots = index_weight.rename(columns={"trade_date": "source_trade_date"}).copy()
    snapshots = snapshots.sort_values(["source_trade_date", "index_code", "ts_code"]).reset_index(drop=True)
    return snapshots


def build_scd2_intervals(snapshots: pd.DataFrame, data_version: str) -> pd.DataFrame:
    if snapshots.empty:
        raise ValueError("Cannot build SCD2 intervals from empty snapshots")

    dates = sorted(snapshots["source_trade_date"].dropna().astype(str).unique())
    intervals: list[dict[str, Any]] = []
    active: dict[tuple[str, str], dict[str, Any]] = {}
    created_at = utc_now_iso()

    for idx, trade_date in enumerate(dates):
        snapshot = snapshots[snapshots["source_trade_date"].astype(str) == trade_date]
        current_keys = set(zip(snapshot["index_code"].astype(str), snapshot["ts_code"].astype(str)))
        previous_keys = set(active)

        for key in sorted(previous_keys - current_keys):
            interval = active.pop(key)
            interval["effective_to"] = dates[idx - 1]
            interval["is_active"] = False
            intervals.append(interval)

        for row in snapshot.itertuples(index=False):
            key = (str(row.index_code), str(row.ts_code))
            weight = float(row.index_weight) if pd.notna(row.index_weight) else None
            if key in active:
                active[key]["index_weight"] = weight
                active[key]["source_trade_date"] = str(row.source_trade_date)
            else:
                active[key] = {
                    "ts_code": str(row.ts_code),
                    "index_code": str(row.index_code),
                    "effective_from": str(row.source_trade_date),
                    "effective_to": OPEN_ENDED_DATE,
                    "index_weight": weight,
                    "source_trade_date": str(row.source_trade_date),
                    "pool_version": data_version,
                    "is_active": True,
                    "created_at": created_at,
                }

    intervals.extend(active.values())
    scd = pd.DataFrame(intervals)
    if scd.empty:
        raise ValueError("SCD2 interval build produced no rows")
    return scd.sort_values(["index_code", "ts_code", "effective_from"]).reset_index(drop=True)


def enrich_with_basic(scd: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    keep = ["ts_code", "name", "industry", "market", "list_date"]
    missing = [column for column in keep if column not in basic.columns]
    if missing:
        raise ValueError(f"basic missing columns: {missing}")

    dim = basic[keep].drop_duplicates("ts_code", keep="last").copy()
    dim["ts_code"] = dim["ts_code"].astype("string")
    dim["list_date"] = dim["list_date"].astype("string").str.replace(r"\.0$", "", regex=True)
    enriched = scd.merge(dim, on="ts_code", how="left")
    ordered = [
        "ts_code",
        "index_code",
        "effective_from",
        "effective_to",
        "index_weight",
        "source_trade_date",
        "name",
        "industry",
        "market",
        "list_date",
        "pool_version",
        "is_active",
        "created_at",
    ]
    return enriched[ordered]


def validate_scd2(scd: pd.DataFrame) -> dict[str, Any]:
    required = {
        "ts_code",
        "index_code",
        "effective_from",
        "effective_to",
        "source_trade_date",
        "pool_version",
        "is_active",
    }
    missing = required - set(scd.columns)
    if missing:
        raise ValueError(f"SCD2 output missing columns: {sorted(missing)}")

    invalid_order = scd[scd["effective_from"].astype(str) > scd["effective_to"].astype(str)]
    if not invalid_order.empty:
        raise ValueError(f"SCD2 contains intervals with effective_from > effective_to: {len(invalid_order)}")

    overlap_count = 0
    for (_index_code, _ts_code), group in scd.groupby(["index_code", "ts_code"], dropna=False):
        ordered = group.sort_values("effective_from")
        previous_to: str | None = None
        for row in ordered.itertuples(index=False):
            current_from = str(row.effective_from)
            if previous_to is not None and current_from <= previous_to:
                overlap_count += 1
            previous_to = str(row.effective_to)
    if overlap_count:
        raise ValueError(f"SCD2 interval overlap detected: {overlap_count}")

    active_count = int(scd["is_active"].sum())
    return {
        "survivorship_bias_check": "PASS",
        "pool_intervals": int(len(scd)),
        "pool_active_intervals": active_count,
        "pool_closed_intervals": int(len(scd) - active_count),
        "pool_unique_stocks": int(scd["ts_code"].nunique()),
    }


def output_path(project_root: Path, config: dict[str, Any]) -> Path:
    pool_config = config["pool"]
    return project_root / pool_config["output_dir"] / pool_config["scd2_file"]


def write_pool(scd: pd.DataFrame, project_root: Path, config: dict[str, Any], overwrite: bool) -> Path:
    path = output_path(project_root, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Pool output exists. Use --overwrite to replace: {path}")
    scd.to_parquet(path, index=False)
    return path


def update_audit(project_root: Path, config: dict[str, Any], data_version: str, metrics: dict[str, Any]) -> None:
    audit_dir = project_root / config["logs"]["audit_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{data_version}_audit.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"data_version": data_version}

    payload.update(
        {
            "pool_agent": "PASS",
            "survivorship_bias_check": metrics["survivorship_bias_check"],
            "pool_intervals": metrics["pool_intervals"],
            "pool_active_intervals": metrics["pool_active_intervals"],
            "pool_closed_intervals": metrics["pool_closed_intervals"],
            "pool_unique_stocks": metrics["pool_unique_stocks"],
            "pool_updated_at": utc_now_iso(),
        }
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_pool_agent(
    config_path: Path,
    project_root: Path,
    data_version: str,
    index_code: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    registry = read_file_registry(project_root, config)
    index_code = index_code or config["pool"]["default_index_code"]

    index_weight = read_raw_dataset(project_root, registry, "index_weight")
    basic = read_raw_dataset(project_root, registry, "basic")

    normalized = normalize_index_weight(index_weight, index_code)
    snapshots = build_snapshot_table(normalized)
    scd = build_scd2_intervals(snapshots, data_version)
    scd = enrich_with_basic(scd, basic)
    metrics = validate_scd2(scd)
    path = output_path(project_root, config)

    if not dry_run:
        path = write_pool(scd, project_root, config, overwrite=overwrite)
        update_audit(project_root, config, data_version, metrics)

    return {
        "data_version": data_version,
        "index_code": index_code,
        "output_path": str(path),
        "dry_run": dry_run,
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical SCD2 stock pool.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--index-code")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    result = run_pool_agent(
        config_path=project_root / args.config,
        project_root=project_root,
        data_version=args.data_version,
        index_code=args.index_code,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
