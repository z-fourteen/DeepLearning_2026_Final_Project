from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipelines.ingest.agent import load_yaml


BUILD_MODES = {"alpha_only", "alpha_plus_residual_style"}
BUILD_MODE_SLUGS = {
    "alpha_only": "alpha_only",
    "alpha_plus_residual_style": "alpha_resid_style",
}
MASK_COLUMNS = [
    "strict_tradable",
    "mask_state_missing",
    "mask_state_not_tradable",
    "mask_st",
    "mask_suspended",
    "mask_price_invalid",
    "mask_volume_invalid",
    "mask_locked_limit",
    "mask_low_amount",
    "mask_microcap",
]


def unique_list(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def read_mart_dataset(project_root: Path, data_version: str) -> pd.DataFrame:
    data_config = load_yaml(project_root / "configs" / "data.yaml")
    dataset_path = project_root / data_config["mart"]["datasets_dir"] / f"dataset_{data_version}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing mart dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df["trade_date"] = df["trade_date"].astype("string")
    df = add_industry_from_pool(project_root, data_config, df)
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def add_industry_from_pool(project_root: Path, data_config: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    if "industry" in df.columns:
        return df

    pool_config = data_config.get("pool", {})
    pool_output_dir = pool_config.get("output_dir")
    pool_file = pool_config.get("scd2_file")
    if not pool_output_dir or not pool_file:
        return df

    pool_path = project_root / pool_output_dir / pool_file
    if not pool_path.exists():
        return df

    pool = pd.read_parquet(pool_path)
    required = {"ts_code", "industry", "effective_from", "effective_to"}
    if not required.issubset(pool.columns):
        return df

    result = df.copy()
    result["_row_id"] = np.arange(len(result))
    left = result[["_row_id", "trade_date", "ts_code"]].copy()
    right = pool[["ts_code", "industry", "effective_from", "effective_to"]].copy()
    left["trade_date"] = left["trade_date"].astype("string")
    right["effective_from"] = right["effective_from"].astype("string")
    right["effective_to"] = right["effective_to"].fillna("99991231").astype("string")
    merged = left.merge(right, on="ts_code", how="left")
    active = merged[
        (merged["trade_date"] >= merged["effective_from"])
        & (merged["trade_date"] <= merged["effective_to"])
    ].copy()
    active = active.sort_values(["_row_id", "effective_from"]).drop_duplicates("_row_id", keep="last")
    result = result.merge(active[["_row_id", "industry"]], on="_row_id", how="left").drop(columns=["_row_id"])
    result["industry"] = result["industry"].astype("string").fillna("UNKNOWN")
    return result


def resolve_split_config(project_root: Path, split_name: str | None) -> tuple[str, dict[str, Any]]:
    config = load_yaml(project_root / "configs" / "splits.yaml")
    name = split_name or config["default_split"]
    split = config["splits"].get(name)
    if split is None:
        raise ValueError(f"Unknown split: {name}")
    return name, split


def add_split_column(df: pd.DataFrame, split: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    result["split"] = "unused"
    for split_name in ["train", "validation", "test"]:
        start = split[split_name]["start_date"]
        end = split[split_name]["end_date"]
        result.loc[result["trade_date"].between(start, end), "split"] = split_name
    return result[result["split"].ne("unused")].reset_index(drop=True)


def read_state_table(project_root: Path, data_config: dict[str, Any]) -> pd.DataFrame:
    state_path = project_root / data_config["state"]["output_dir"]
    if not state_path.exists():
        raise FileNotFoundError(f"Missing security state table: {state_path}")
    columns = [
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
    state = pd.read_parquet(state_path, columns=columns)
    state["trade_date"] = state["trade_date"].astype("string")
    state["ts_code"] = state["ts_code"].astype("string")
    return state.drop_duplicates(["trade_date", "ts_code"], keep="last")


def bool_column(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="bool")
    return frame[column].fillna(default).astype("bool")


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def add_strict_tradable_mask(
    project_root: Path,
    data_config: dict[str, Any],
    frame: pd.DataFrame,
    clean_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    mask_config = clean_config.get("strict_tradable_mask", {})
    if not mask_config.get("enabled", False):
        result = frame.copy()
        result["strict_tradable"] = True
        for column in MASK_COLUMNS:
            if column != "strict_tradable":
                result[column] = False
        summary = {
            "enabled": False,
            "input_rows": int(len(result)),
            "kept_rows": int(len(result)),
            "dropped_rows": 0,
            "drop_rate": 0.0,
            "reason_counts": {},
        }
        return result, summary, pd.DataFrame()

    state = read_state_table(project_root, data_config)
    result = frame.merge(state, on=["trade_date", "ts_code"], how="left", indicator="_state_merge")

    state_filters = mask_config.get("state_filters", {})
    result["mask_state_missing"] = result["_state_merge"].ne("both")
    result["mask_state_not_tradable"] = state_filters.get("require_is_tradable", True) & ~bool_column(result, "is_tradable")
    result["mask_st"] = state_filters.get("remove_st_or_star_st", True) & bool_column(result, "is_st")
    result["mask_suspended"] = state_filters.get("remove_suspended", True) & bool_column(result, "is_suspended")
    result["mask_price_invalid"] = state_filters.get("require_price_valid", True) & ~bool_column(result, "price_valid", default=True)
    result["mask_volume_invalid"] = state_filters.get("require_volume_valid", True) & ~bool_column(result, "volume_valid", default=True)
    result["mask_locked_limit"] = (
        state_filters.get("remove_locked_limit_up_or_down_execution_samples", True)
        & (bool_column(result, "is_limit_up") | bool_column(result, "is_limit_down"))
    )

    liquidity_filters = mask_config.get("liquidity_filters", {})
    result["mask_low_amount"] = False
    if liquidity_filters.get("enabled", False):
        amount_column = liquidity_filters.get("amount_column")
        amount = numeric_column(result, amount_column)
        min_amount = liquidity_filters.get("min_amount")
        if min_amount is not None:
            result["mask_low_amount"] = result["mask_low_amount"] | amount.isna() | amount.lt(float(min_amount))
        bottom_quantile = liquidity_filters.get("bottom_quantile_by_date")
        if bottom_quantile is not None and amount_column in result.columns:
            threshold = result.groupby("trade_date", sort=False)[amount_column].transform(
                lambda values: pd.to_numeric(values, errors="coerce").quantile(float(bottom_quantile))
            )
            result["mask_low_amount"] = result["mask_low_amount"] | amount.isna() | amount.lt(threshold)

    size_filters = mask_config.get("size_filters", {})
    result["mask_microcap"] = False
    if size_filters.get("enabled", False):
        size_column = size_filters.get("size_column")
        size = numeric_column(result, size_column)
        bottom_quantile = size_filters.get("bottom_quantile_by_date")
        if bottom_quantile is not None and size_column in result.columns:
            threshold = result.groupby("trade_date", sort=False)[size_column].transform(
                lambda values: pd.to_numeric(values, errors="coerce").quantile(float(bottom_quantile))
            )
            result["mask_microcap"] = result["mask_microcap"] | size.isna() | size.lt(threshold)

    reason_columns = [column for column in MASK_COLUMNS if column != "strict_tradable"]
    result["strict_tradable"] = ~result[reason_columns].any(axis=1)
    result = result.drop(columns=["_state_merge"])

    split_group = ["split"] if "split" in result.columns else []
    by_split = (
        result.groupby(split_group, dropna=False)["strict_tradable"]
        .agg(input_rows="count", kept_rows="sum")
        .reset_index()
        if split_group
        else pd.DataFrame()
    )
    if not by_split.empty:
        by_split["dropped_rows"] = by_split["input_rows"] - by_split["kept_rows"]
        by_split["drop_rate"] = by_split["dropped_rows"] / by_split["input_rows"]

    reason_counts = {column: int(result[column].sum()) for column in reason_columns}
    summary = {
        "enabled": True,
        "input_rows": int(len(result)),
        "kept_rows": int(result["strict_tradable"].sum()),
        "dropped_rows": int((~result["strict_tradable"]).sum()),
        "drop_rate": float((~result["strict_tradable"]).mean()) if len(result) else 0.0,
        "reason_counts": reason_counts,
        "by_split": by_split.to_dict(orient="records") if not by_split.empty else [],
        "config": mask_config,
    }
    log_columns = ["trade_date", "ts_code", *split_group, *MASK_COLUMNS]
    return result, summary, result[log_columns].copy()


def load_clean_feature_config(project_root: Path, config_path: str) -> dict[str, Any]:
    path = project_root / config_path
    if not path.exists():
        raise FileNotFoundError(f"Missing clean feature config: {path}")
    return load_yaml(path)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def design_matrix(group: pd.DataFrame, neutralize_against: dict[str, Any]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    continuous_columns: list[str] = []
    for exposure in neutralize_against.get("style_exposures", []):
        source = first_existing_column(group, list(exposure.get("candidates", [])))
        if source:
            continuous_columns.append(source)

    if continuous_columns:
        continuous = group[continuous_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        continuous = continuous.apply(lambda col: (col - col.mean()) / col.std(ddof=0) if col.std(ddof=0) else col * 0)
        continuous.columns = [f"exposure_{column}" for column in continuous.columns]
        parts.append(continuous)

    industry_config = neutralize_against.get("industry", {})
    industry_column = industry_config.get("column") if industry_config.get("enabled", True) else None
    if industry_column and industry_column in group.columns:
        dummies = pd.get_dummies(group[industry_column].astype("string").fillna("UNKNOWN"), prefix="industry", dtype="float64")
        if dummies.shape[1] > 1:
            parts.append(dummies.iloc[:, 1:])

    if not parts:
        return pd.DataFrame(index=group.index)
    return pd.concat(parts, axis=1)


def residualize_features(
    df: pd.DataFrame,
    features: list[str],
    neutralize_against: dict[str, Any],
    suffix: str,
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)

    residual_columns = [f"{feature}{suffix}" for feature in features]
    residuals = pd.DataFrame(np.nan, index=df.index, columns=residual_columns, dtype="float64")
    for _, group in df.groupby("trade_date", sort=True):
        x = design_matrix(group, neutralize_against)
        if x.empty:
            continue
        valid_x_mask = x.notna().all(axis=1)
        x_valid = x.loc[valid_x_mask]
        x_valid = x_valid.loc[:, x_valid.nunique(dropna=True) > 1]
        if x_valid.empty:
            continue
        y_frame = group.loc[valid_x_mask, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        complete_features = [feature for feature in features if bool(y_frame[feature].notna().all()) and y_frame[feature].nunique() >= 2]
        if complete_features and len(x_valid) > x_valid.shape[1] + 2:
            x_values = np.column_stack([np.ones(len(x_valid)), x_valid.to_numpy(dtype="float64")])
            y_values = y_frame[complete_features].to_numpy(dtype="float64")
            try:
                beta = np.linalg.lstsq(x_values, y_values, rcond=None)[0]
                residuals.loc[y_frame.index, [f"{feature}{suffix}" for feature in complete_features]] = y_values - x_values @ beta
            except np.linalg.LinAlgError:
                pass

        for feature in [feature for feature in features if feature not in complete_features]:
            y = y_frame[feature]
            valid = y.notna()
            x_feature = x_valid.loc[valid]
            x_feature = x_feature.loc[:, x_feature.nunique(dropna=True) > 1]
            if int(valid.sum()) <= x_feature.shape[1] + 2 or y[valid].nunique() < 2:
                continue
            x_values = np.column_stack([np.ones(len(x_feature)), x_feature.to_numpy(dtype="float64")])
            y_values = y.loc[valid].to_numpy(dtype="float64")
            try:
                beta = np.linalg.lstsq(x_values, y_values, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            residuals.loc[y.loc[valid].index, f"{feature}{suffix}"] = y_values - x_values @ beta
    return residuals


def residual_candidate_features(clean_config: dict[str, Any]) -> list[str]:
    residual_config = clean_config.get("residualized_style_features", {})
    families = residual_config.get("candidate_families", {})
    features: list[str] = []
    for family in families.values():
        features.extend(family.get("initial_residual_candidates", []))
    return list(dict.fromkeys(features))


def build_feature_frame(
    df: pd.DataFrame,
    clean_config: dict[str, Any],
    build_mode: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if build_mode not in BUILD_MODES:
        raise ValueError(f"Unsupported build mode: {build_mode}")

    alpha_features = list(clean_config.get("alpha_features", []))
    risk_controls = list(clean_config.get("risk_controls", []))
    tradability_controls = list(clean_config.get("tradability_controls", []))
    missing_alpha = [feature for feature in alpha_features if feature not in df.columns]
    if missing_alpha:
        raise ValueError(f"Missing alpha features: {missing_alpha}")

    result = df.copy()
    model_features = list(alpha_features)
    residual_features: list[str] = []
    if build_mode == "alpha_plus_residual_style":
        residual_config = clean_config.get("residualized_style_features", {})
        suffix = str(residual_config.get("naming", "{feature}__resid_style")).replace("{feature}", "")
        candidates = [feature for feature in residual_candidate_features(clean_config) if feature in result.columns]
        residuals = residualize_features(result, candidates, residual_config.get("neutralize_against", {}), suffix)
        result = pd.concat([result, residuals], axis=1)
        residual_features = list(residuals.columns)
        model_features.extend(residual_features)

    return result, model_features, risk_controls, tradability_controls


def build_sequence_arrays(
    panel: pd.DataFrame,
    model_features: list[str],
    label_column: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    records: list[dict[str, Any]] = []
    for _, group in panel.groupby("ts_code", sort=True):
        values = group[model_features].to_numpy(dtype="float32")
        y = pd.to_numeric(group[label_column], errors="coerce").to_numpy(dtype="float32")
        for index in range(lookback - 1, len(group)):
            window = values[index - lookback + 1 : index + 1]
            label = y[index]
            if np.isnan(label) or np.isnan(window).any():
                continue
            row = group.iloc[index]
            sequences.append(window)
            labels.append(float(label))
            records.append(
                {
                    "trade_date": str(row["trade_date"]),
                    "ts_code": str(row["ts_code"]),
                    "split": str(row["split"]),
                    label_column: float(label),
                }
            )

    if sequences:
        x_array = np.stack(sequences).astype("float32")
        y_array = np.asarray(labels, dtype="float32")
    else:
        x_array = np.empty((0, lookback, len(model_features)), dtype="float32")
        y_array = np.empty((0,), dtype="float32")
    return x_array, y_array, pd.DataFrame(records)


def build_clean_sequence_dataset(
    project_root: Path,
    data_version: str,
    clean_config_path: str,
    build_mode: str,
    label_column: str,
    lookback: int,
    split_name: str | None,
) -> dict[str, Any]:
    data_config = load_yaml(project_root / "configs" / "data.yaml")
    split_name, split = resolve_split_config(project_root, split_name)
    clean_config = load_clean_feature_config(project_root, clean_config_path)
    df = read_mart_dataset(project_root, data_version)
    feature_frame, model_features, risk_controls, tradability_controls = build_feature_frame(
        df,
        clean_config,
        build_mode,
    )

    control_columns = unique_list(
        [column for column in [*risk_controls, *tradability_controls] if column in feature_frame.columns]
    )
    required = unique_list(["trade_date", "ts_code", label_column, *model_features, *control_columns])
    missing = [column for column in required if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"Missing clean dataset columns: {missing}")

    panel = add_split_column(feature_frame[required].copy(), split)
    panel = panel.dropna(subset=[label_column]).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    panel, mask_summary, mask_log = add_strict_tradable_mask(project_root, data_config, panel, clean_config)
    panel = panel[panel["strict_tradable"]].copy().reset_index(drop=True)
    for feature in model_features:
        panel[feature] = pd.to_numeric(panel[feature], errors="coerce")
    panel[model_features] = panel[model_features].replace([np.inf, -np.inf], np.nan)

    x_array, y_array, sample_index = build_sequence_arrays(panel, model_features, label_column, lookback)
    output_dir = project_root / data_config["mart"]["datasets_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    name = clean_config.get("name", Path(clean_config_path).stem)
    dataset_slug = clean_config.get("dataset_slug", name)
    split_slug = split_name.removesuffix("_v1")
    build_slug = BUILD_MODE_SLUGS[build_mode]
    output_stem = f"dataset_seq_l{lookback}_{dataset_slug}_{build_slug}_{split_slug}"
    output_path = output_dir / f"{output_stem}.npz"
    sidecar_path = output_dir / f"{output_stem}_sidecar.parquet"
    manifest_path = output_dir / f"{output_stem}_manifest.json"
    filter_log_path = output_dir / f"{output_stem}_filter_log.csv"

    np.savez_compressed(
        output_path,
        X=x_array,
        y=y_array,
        trade_date=sample_index["trade_date"].to_numpy() if not sample_index.empty else np.asarray([]),
        ts_code=sample_index["ts_code"].to_numpy() if not sample_index.empty else np.asarray([]),
        split=sample_index["split"].to_numpy() if not sample_index.empty else np.asarray([]),
        feature_names=np.asarray(model_features),
        build_mode=np.asarray([build_mode]),
    )

    sidecar_columns = [column for column in [*control_columns, *MASK_COLUMNS] if column in panel.columns]
    if not sample_index.empty and sidecar_columns:
        sidecar = sample_index.merge(
            panel[["trade_date", "ts_code", *sidecar_columns]],
            on=["trade_date", "ts_code"],
            how="left",
        )
    else:
        sidecar = sample_index
    sidecar.to_parquet(sidecar_path, index=False)
    if clean_config.get("strict_tradable_mask", {}).get("outputs", {}).get("write_filter_log", True):
        mask_log.to_csv(filter_log_path, index=False, encoding="utf-8-sig")

    manifest = {
        "dataset_type": "clean_sequence",
        "path": str(output_path),
        "sidecar_path": str(sidecar_path),
        "filter_log_path": str(filter_log_path),
        "data_version": data_version,
        "split_name": split_name,
        "lookback": lookback,
        "feature_set": name,
        "dataset_slug": dataset_slug,
        "build_mode": build_mode,
        "output_stem": output_stem,
        "samples": int(len(y_array)),
        "model_features": model_features,
        "model_feature_count": len(model_features),
        "risk_controls": [column for column in risk_controls if column in feature_frame.columns],
        "tradability_controls": [column for column in tradability_controls if column in feature_frame.columns],
        "sidecar_control_count": len(control_columns),
        "strict_tradable_mask": mask_summary,
        "split_counts": sample_index["split"].value_counts().to_dict() if not sample_index.empty else {},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean model datasets from role-aware feature configs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--clean-config", default="configs/feature_sets/advanced_sequence_clean_v1.yaml")
    parser.add_argument("--build-mode", choices=sorted(BUILD_MODES), default="alpha_only")
    parser.add_argument("--label-column", default="label_rel_return")
    parser.add_argument("--split-name")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[20])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    results = [
        build_clean_sequence_dataset(
            project_root=project_root,
            data_version=args.data_version,
            clean_config_path=args.clean_config,
            build_mode=args.build_mode,
            label_column=args.label_column,
            lookback=lookback,
            split_name=args.split_name,
        )
        for lookback in args.lookbacks
    ]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
