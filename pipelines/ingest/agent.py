from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REGISTRY_COLUMNS = [
    "dataset",
    "source_path",
    "filename",
    "md5",
    "modified_time",
    "file_size",
    "row_count",
    "trade_date_min",
    "trade_date_max",
    "raw_path",
    "status",
    "last_ingest_time",
]


@dataclass(frozen=True)
class SourceFile:
    dataset: str
    path: Path
    layout: str
    date_from: str
    date_column: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_directories(config: dict[str, Any], project_root: Path) -> None:
    dirs = [
        "configs",
        "data/lake/raw/basic",
        "data/lake/raw/trade_cal",
        "data/lake/raw/daily",
        "data/lake/raw/stock_st",
        "data/lake/raw/index_weight",
        config["lake"]["core_dir"],
        config["lake"]["state_dir"],
        config["lake"]["audit_dir"],
        config["mart"]["features_dir"],
        config["mart"]["labels_dir"],
        config["mart"]["datasets_dir"],
        config["cache"]["rolling_features_dir"],
        "meta",
        config["logs"]["audit_dir"],
        "pipelines/ingest",
        "pipelines/pool",
        "pipelines/state",
        "pipelines/feature",
        "pipelines/label",
        "pipelines/backtest",
        "outputs",
        "scripts",
    ]
    for directory in dirs:
        (project_root / directory).mkdir(parents=True, exist_ok=True)


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    registry = pd.read_parquet(path)
    for column in REGISTRY_COLUMNS:
        if column not in registry.columns:
            registry[column] = None
    return registry[REGISTRY_COLUMNS]


def write_registry(path: Path, registry: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(path, index=False)


def discover_sources(config: dict[str, Any], project_root: Path) -> list[SourceFile]:
    source_root = project_root / config["source"]["root_dir"]
    sources: list[SourceFile] = []
    for dataset, spec in config["ingestion"]["datasets"].items():
        source_path = source_root / spec["path"]
        layout = spec["layout"]
        date_from = spec.get("date_from", "none")
        date_column = spec.get("date_column")

        if layout == "single_file":
            if source_path.exists():
                sources.append(SourceFile(dataset, source_path, layout, date_from, date_column))
            continue

        if not source_path.exists():
            continue

        for file_path in sorted(source_path.glob("*.csv")):
            sources.append(SourceFile(dataset, file_path, layout, date_from, date_column))
    return sources


def get_trade_date_from_filename(path: Path) -> str | None:
    match = re.match(r"^(\d{8})\.csv$", path.name)
    return match.group(1) if match else None


def get_file_signature(source: SourceFile) -> dict[str, Any]:
    stat = source.path.stat()
    return {
        "dataset": source.dataset,
        "source_path": str(source.path),
        "filename": source.path.name,
        "md5": md5_file(source.path),
        "modified_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        "file_size": stat.st_size,
    }


def changed_files(sources: list[SourceFile], registry: pd.DataFrame) -> list[tuple[SourceFile, dict[str, Any], str]]:
    if registry.empty:
        known = {}
    else:
        latest = registry.sort_values("last_ingest_time").drop_duplicates("source_path", keep="last")
        known = latest.set_index("source_path").to_dict("index")

    changed: list[tuple[SourceFile, dict[str, Any], str]] = []
    for source in sources:
        signature = get_file_signature(source)
        previous = known.get(signature["source_path"])
        if previous is None:
            changed.append((source, signature, "new"))
        elif previous.get("md5") != signature["md5"]:
            changed.append((source, signature, "modified"))
    return changed


def read_csv(source: SourceFile, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(source.path, encoding=encoding)
    if source.date_from == "filename":
        trade_date = get_trade_date_from_filename(source.path)
        if trade_date and "trade_date" not in df.columns:
            df["trade_date"] = trade_date
    return df


def date_bounds(df: pd.DataFrame, source: SourceFile) -> tuple[str | None, str | None]:
    date_column = source.date_column or "trade_date"
    if source.date_from == "filename":
        trade_date = get_trade_date_from_filename(source.path)
        return trade_date, trade_date
    if date_column in df.columns and not df.empty:
        values = df[date_column].dropna().astype(str)
        if not values.empty:
            return values.min(), values.max()
    return None, None


def raw_output_path(raw_dir: Path, source: SourceFile, signature: dict[str, Any]) -> Path:
    stem = source.path.stem
    digest = signature["md5"][:12]
    if source.date_from == "filename":
        trade_date = get_trade_date_from_filename(source.path) or "unknown_date"
        return raw_dir / source.dataset / f"trade_date={trade_date}" / f"{stem}_{digest}.parquet"
    return raw_dir / source.dataset / f"{stem}_{digest}.parquet"


def ingest_one(
    source: SourceFile,
    signature: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        trade_date_min, trade_date_max = date_bounds(pd.DataFrame(), source)
        return {
            **signature,
            "row_count": None,
            "trade_date_min": trade_date_min,
            "trade_date_max": trade_date_max,
            "raw_path": str(raw_output_path(project_root / config["lake"]["raw_dir"], source, signature)),
            "status": "dry_run",
            "last_ingest_time": utc_now_iso(),
        }

    encoding = config["source"].get("csv_encoding", "utf-8")
    raw_dir = project_root / config["lake"]["raw_dir"]
    df = read_csv(source, encoding)
    trade_date_min, trade_date_max = date_bounds(df, source)
    output_path = raw_output_path(raw_dir, source, signature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Append-only raw layer violation: {output_path} already exists")
    df.to_parquet(output_path, index=False)

    return {
        **signature,
        "row_count": int(len(df)),
        "trade_date_min": trade_date_min,
        "trade_date_max": trade_date_max,
        "raw_path": str(output_path),
        "status": "ingested",
        "last_ingest_time": utc_now_iso(),
    }


def append_data_version(project_root: Path, config: dict[str, Any], data_version: str, summary: dict[str, Any]) -> None:
    path = project_root / config["meta"]["data_versions"]
    row = pd.DataFrame(
        [
            {
                "data_version": data_version,
                "created_at": utc_now_iso(),
                "agent": "ingestion",
                "new_trade_dates": summary["new_trade_dates"],
                "new_files": summary["new_files"],
                "modified_files": summary["modified_files"],
            }
        ]
    )
    if path.exists():
        versions = pd.read_parquet(path)
        versions = pd.concat([versions, row], ignore_index=True)
    else:
        versions = row
    path.parent.mkdir(parents=True, exist_ok=True)
    versions.to_parquet(path, index=False)


def write_audit(project_root: Path, config: dict[str, Any], data_version: str, summary: dict[str, Any]) -> None:
    audit_dir = project_root / config["logs"]["audit_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_date = data_version.removeprefix("v")
    payload = {
        "data_version": data_version,
        "new_trade_dates": summary["new_trade_dates"],
        "new_stocks": 0,
        "removed_stocks": 0,
        "st_filtered": 0,
        "invalid_price_rows": 0,
        "future_leakage_check": "NOT_RUN",
        "survivorship_bias_check": "NOT_RUN",
        "agent": "ingestion",
        "new_files": summary["new_files"],
        "modified_files": summary["modified_files"],
        "ingested_rows": summary["ingested_rows"],
        "created_at": utc_now_iso(),
    }
    with (audit_dir / f"{audit_date}_audit.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def latest_data_version(records: list[dict[str, Any]]) -> str:
    dates = [
        record["trade_date_max"]
        for record in records
        if record.get("trade_date_max") and re.fullmatch(r"\d{8}", str(record["trade_date_max"]))
    ]
    if dates:
        return f"v{max(dates)}"
    return "v" + datetime.now().strftime("%Y%m%d")


def run_ingestion(config_path: Path, project_root: Path, dry_run: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    ensure_directories(config, project_root)

    registry_path = project_root / config["meta"]["file_registry"]
    registry = read_registry(registry_path)
    sources = discover_sources(config, project_root)
    changes = changed_files(sources, registry)

    records: list[dict[str, Any]] = []
    for source, signature, _change_type in changes:
        records.append(ingest_one(source, signature, config, project_root, dry_run))

    if records and not dry_run:
        updated = pd.concat([registry, pd.DataFrame(records)], ignore_index=True)
        write_registry(registry_path, updated[REGISTRY_COLUMNS])

    changed_trade_dates = {
        record["trade_date_max"]
        for record in records
        if record.get("trade_date_max") and re.fullmatch(r"\d{8}", str(record["trade_date_max"]))
    }
    summary = {
        "discovered_files": len(sources),
        "changed_files": len(changes),
        "new_files": sum(1 for _, _, change_type in changes if change_type == "new"),
        "modified_files": sum(1 for _, _, change_type in changes if change_type == "modified"),
        "new_trade_dates": len(changed_trade_dates),
        "ingested_rows": int(sum(record.get("row_count") or 0 for record in records)),
    }
    data_version = latest_data_version(records)

    if not dry_run:
        append_data_version(project_root, config, data_version, summary)
        write_audit(project_root, config, data_version, summary)

    return {"data_version": data_version, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the incremental raw data ingestion agent.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to the data config YAML.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--dry-run", action="store_true", help="Detect changes without writing raw data or metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    result = run_ingestion(project_root / args.config, project_root, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
