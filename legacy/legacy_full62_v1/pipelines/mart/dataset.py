from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipelines.ingest.agent import load_yaml


def read_mart_dataset(project_root: Path, data_config: dict[str, Any], data_version: str) -> pd.DataFrame:
    dataset_path = project_root / data_config["mart"]["datasets_dir"] / f"dataset_{data_version}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing mart dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df["trade_date"] = df["trade_date"].astype("string")
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def resolve_split_config(project_root: Path, split_name: str | None) -> tuple[str, dict[str, Any]]:
    config = load_yaml(project_root / "configs" / "data" / "splits.yaml")
    name = split_name or config["default_split"]
    split = config["splits"].get(name)
    if split is None:
        raise ValueError(f"Unknown split: {name}")
    if "folds" in split:
        active_fold = split.get("active_fold")
        if not active_fold:
            raise ValueError(f"Split {name} declares folds but no active_fold.")
        fold = split["folds"].get(active_fold)
        if fold is None:
            raise ValueError(f"Unknown active_fold for {name}: {active_fold}")
        fold = dict(fold)
        fold["scheme"] = split.get("scheme", config.get("split_policy", {}).get("scheme"))
        fold["fold_name"] = active_fold
        fold["split_name"] = name
        fold["split_policy"] = config.get("split_policy", {})
        return name, fold
    return name, split


def add_split_column(df: pd.DataFrame, split: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    result["split"] = "unused"
    result["purge_reason"] = ""
    for split_name in ["train", "validation", "test"]:
        start = split[split_name]["start_date"]
        end = split[split_name]["end_date"]
        mask = result["trade_date"].between(start, end)
        result.loc[mask, "split"] = split_name
    for purge_range in split.get("purge_ranges", []):
        mask = result["trade_date"].between(purge_range["start_date"], purge_range["end_date"])
        result.loc[mask, "split"] = "purged"
        result.loc[mask, "purge_reason"] = purge_range.get("reason", "purged")
    return result[result["split"].isin(["train", "validation", "test"])].reset_index(drop=True)


def selected_features(project_root: Path, feature_set: str) -> list[str]:
    features_config = load_yaml(project_root / "configs" / "features.yaml")
    features = features_config.get("feature_sets", {}).get(feature_set, {}).get("selected_features", [])
    if not features:
        raise ValueError(f"Feature set is empty or missing: {feature_set}")
    return list(features)


def build_lgbm_dataset(
    project_root: Path,
    data_version: str,
    feature_set: str,
    label_column: str,
    split_name: str | None = None,
) -> dict[str, Any]:
    data_config = load_yaml(project_root / "configs" / "data" / "data.yaml")
    split_name, split = resolve_split_config(project_root, split_name)
    df = read_mart_dataset(project_root, data_config, data_version)
    features = [feature for feature in selected_features(project_root, feature_set) if feature in df.columns]
    if not features:
        raise ValueError(f"No selected features found in mart dataset for {feature_set}.")
    required_columns = ["trade_date", "ts_code", label_column, *features]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    dataset = add_split_column(df[required_columns].copy(), split)
    dataset = dataset.dropna(subset=[label_column]).reset_index(drop=True)
    output_dir = project_root / data_config["mart"]["datasets_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dataset_lgbm_{feature_set}_{split_name}_{data_version}.parquet"
    dataset.to_parquet(output_path, index=False)
    return {
        "dataset_type": "lgbm",
        "path": str(output_path),
        "rows": int(len(dataset)),
        "features": int(len(features)),
        "feature_set": feature_set,
        "split_name": split_name,
        "split_scheme": split.get("scheme", "single_holdout"),
        "active_fold": split.get("fold_name", ""),
        "purge_ranges": split.get("purge_ranges", []),
        "split_counts": dataset["split"].value_counts().to_dict(),
    }


def build_sequence_dataset(
    project_root: Path,
    data_version: str,
    feature_set: str,
    label_column: str,
    lookback: int,
    split_name: str | None = None,
) -> dict[str, Any]:
    data_config = load_yaml(project_root / "configs" / "data" / "data.yaml")
    split_name, split = resolve_split_config(project_root, split_name)
    df = read_mart_dataset(project_root, data_config, data_version)
    features = [feature for feature in selected_features(project_root, feature_set) if feature in df.columns]
    if not features:
        raise ValueError(f"No selected features found in mart dataset for {feature_set}.")
    required_columns = ["trade_date", "ts_code", label_column, *features]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    panel = add_split_column(df[required_columns].copy(), split)
    panel = panel.dropna(subset=[label_column]).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    for feature in features:
        panel[feature] = pd.to_numeric(panel[feature], errors="coerce")
    panel[features] = panel[features].replace([np.inf, -np.inf], np.nan)

    sequences: list[np.ndarray] = []
    labels: list[float] = []
    trade_dates: list[str] = []
    ts_codes: list[str] = []
    splits: list[str] = []
    for _, group in panel.groupby("ts_code", sort=True):
        values = group[features].to_numpy(dtype="float32")
        y = pd.to_numeric(group[label_column], errors="coerce").to_numpy(dtype="float32")
        for index in range(lookback - 1, len(group)):
            window = values[index - lookback + 1 : index + 1]
            label = y[index]
            if np.isnan(label) or np.isnan(window).any():
                continue
            row = group.iloc[index]
            sequences.append(window)
            labels.append(float(label))
            trade_dates.append(str(row["trade_date"]))
            ts_codes.append(str(row["ts_code"]))
            splits.append(str(row["split"]))

    if sequences:
        x_array = np.stack(sequences).astype("float32")
        y_array = np.asarray(labels, dtype="float32")
    else:
        x_array = np.empty((0, lookback, len(features)), dtype="float32")
        y_array = np.empty((0,), dtype="float32")

    output_dir = project_root / data_config["mart"]["datasets_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dataset_sequence_l{lookback}_{feature_set}_{split_name}_{data_version}.npz"
    np.savez_compressed(
        output_path,
        X=x_array,
        y=y_array,
        trade_date=np.asarray(trade_dates),
        ts_code=np.asarray(ts_codes),
        split=np.asarray(splits),
        feature_names=np.asarray(features),
    )
    split_counts = pd.Series(splits).value_counts().to_dict() if splits else {}
    return {
        "dataset_type": "sequence",
        "path": str(output_path),
        "samples": int(len(y_array)),
        "lookback": int(lookback),
        "features": int(len(features)),
        "feature_set": feature_set,
        "split_name": split_name,
        "split_scheme": split.get("scheme", "single_holdout"),
        "active_fold": split.get("fold_name", ""),
        "purge_ranges": split.get("purge_ranges", []),
        "split_counts": split_counts,
    }


def write_dataset_manifest(project_root: Path, data_version: str, results: list[dict[str, Any]]) -> Path:
    output_dir = project_root / "data" / "mart" / "datasets"
    manifest_path = output_dir / f"dataset_manifest_{data_version}.json"
    manifest_path.write_text(json.dumps({"data_version": data_version, "datasets": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready LGBM and sequence datasets from mart data.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--label-column", default="label_rel_return")
    parser.add_argument("--split-name")
    parser.add_argument("--lgbm-feature-set", default="baseline_lightgbm_fixed")
    parser.add_argument("--sequence-feature-set", default="advanced_sequence_fixed")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[20, 60])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    results: list[dict[str, Any]] = []
    results.append(
        build_lgbm_dataset(
            project_root,
            args.data_version,
            args.lgbm_feature_set,
            args.label_column,
            args.split_name,
        )
    )
    for lookback in args.lookbacks:
        results.append(
            build_sequence_dataset(
                project_root,
                args.data_version,
                args.sequence_feature_set,
                args.label_column,
                lookback,
                args.split_name,
            )
        )
    manifest = write_dataset_manifest(project_root, args.data_version, results)
    print(json.dumps({"manifest": str(manifest), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
