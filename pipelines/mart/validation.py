from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipelines.ingest.agent import load_yaml


@dataclass(frozen=True)
class ValidationConfig:
    data_version: str
    label_column: str
    quantiles: int
    min_cross_section: int
    max_baseline_corr: float
    train_end_date: str | None = None
    eval_start_date: str | None = None
    neutralized_jobs: int = 1
    neutralized_chunk_size: int = 16


STAGES = {
    "all",
    "ic",
    "quality",
    "correlation",
    "regime-ic",
    "quantile",
    "extended-quantile",
    "neutralized",
    "recommendation",
}

QUANTILE_COLUMNS = [
    "feature",
    "quantiles",
    "q1_mean_return",
    "q_top_mean_return",
    "long_short_mean_return",
    "long_short_t_stat",
    "long_short_win_ratio",
    "long_short_max_drawdown",
    "days",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "daily_ic": output_dir / "factor_daily_ic.csv",
        "ic": output_dir / "factor_ic_rankic.csv",
        "quantile": output_dir / "factor_quantile_long_short.csv",
        "quantile_detail": output_dir / "factor_quantile_detail.csv",
        "yearly_quantile": output_dir / "factor_yearly_quantile_long_short.csv",
        "regime_quantile": output_dir / "factor_regime_quantile_long_short.csv",
        "neutralized_ic": output_dir / "factor_neutralized_ic.csv",
        "holdout_quantile": output_dir / "factor_holdout_quantile_long_short.csv",
        "quality": output_dir / "feature_quality.csv",
        "correlation": output_dir / "feature_correlation_top.csv",
        "regime_ic": output_dir / "factor_regime_ic.csv",
        "recommendations": output_dir / "feature_recommendations.csv",
        "summary": output_dir / "factor_validation_summary.json",
        "profile": output_dir / "validation_profile.json",
    }


def safe_path_token(value: str | int | float | None) -> str:
    if value is None or value == "":
        return "all"
    return str(value).replace(".", "p").replace("-", "m").replace("/", "_").replace("\\", "_")


def validation_mode_id(
    label_column: str,
    quantiles: int,
    min_cross_section: int,
    max_baseline_corr: float,
    train_end_date: str | None,
    eval_start_date: str | None,
    skip_quantile: bool,
    skip_extended_quantile: bool,
    skip_neutralized: bool,
) -> str:
    parts = [
        safe_path_token(label_column),
        f"q{quantiles}",
        f"cs{min_cross_section}",
        f"corr{safe_path_token(max_baseline_corr)}",
        f"train_{safe_path_token(train_end_date)}",
        f"eval_{safe_path_token(eval_start_date)}",
        f"quantile_{'off' if skip_quantile else 'on'}",
        f"ext_{'off' if skip_extended_quantile else 'on'}",
        f"neutral_{'off' if skip_neutralized else 'on'}",
    ]
    return "_".join(parts)


def validation_output_dir(project_root: Path, data_version: str, feature_set: str | None, mode_id: str) -> Path:
    scope = feature_set or "all_lag1"
    return project_root / "outputs" / "factor_validation" / data_version / scope / mode_id


def should_run_stage(requested_stage: str, stage: str) -> bool:
    return requested_stage == "all" or requested_stage == stage


def empty_quantile_table() -> pd.DataFrame:
    return pd.DataFrame(columns=QUANTILE_COLUMNS)


def read_csv_checkpoint(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv_checkpoint(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def frame_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024 / 1024)


def append_profile_event(output_dir: Path, event: dict[str, Any]) -> None:
    paths = output_paths(output_dir)
    events: list[dict[str, Any]] = []
    if paths["profile"].exists():
        try:
            events = json.loads(paths["profile"].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            events = []
    events.append(event)
    paths["profile"].write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")


def profile_stage(
    output_dir: Path,
    stage: str,
    status: str,
    started_at: float,
    df: pd.DataFrame,
    features: list[str],
    outputs: list[Path],
) -> None:
    append_profile_event(
        output_dir,
        {
            "stage": stage,
            "status": status,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "rows": int(len(df)),
            "trade_dates": int(df["trade_date"].nunique()) if "trade_date" in df.columns else None,
            "stocks": int(df["ts_code"].nunique()) if "ts_code" in df.columns else None,
            "features": int(len(features)),
            "dataset_memory_mb": round(frame_memory_mb(df), 3),
            "outputs": [str(path) for path in outputs],
            "finished_at": utc_now_iso(),
        },
    )


def recommended_features(table: pd.DataFrame, recommendation: str) -> list[str]:
    if table.empty or "recommendation" not in table.columns or "feature" not in table.columns:
        return []
    return table[table["recommendation"].eq(recommendation)]["feature"].tolist()


def read_dataset(project_root: Path, config: dict[str, Any], data_version: str) -> pd.DataFrame:
    dataset_path = project_root / config["mart"]["datasets_dir"] / f"dataset_{data_version}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing mart dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    if "trade_date" not in df.columns or "ts_code" not in df.columns:
        raise ValueError("Mart dataset must contain trade_date and ts_code.")
    df["trade_date"] = df["trade_date"].astype("string")
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def read_registry(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    path = project_root / config["meta"]["file_registry"]
    if not path.exists():
        raise FileNotFoundError(f"Missing file registry: {path}")
    return pd.read_parquet(path)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" in df.columns:
        df = df.copy()
        df["trade_date"] = df["trade_date"].astype("string").str.replace(r"\.0$", "", regex=True)
    return df


def read_raw_dataset(project_root: Path, config: dict[str, Any], dataset: str) -> pd.DataFrame:
    registry = read_registry(project_root, config)
    rows = registry[(registry["dataset"] == dataset) & (registry["status"] == "ingested")]
    if rows.empty:
        raise ValueError(f"No raw dataset found: {dataset}")
    frames: list[pd.DataFrame] = []
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


def read_benchmark_regime_source(project_root: Path, config: dict[str, Any], benchmark: str) -> pd.DataFrame:
    market = normalize_dates(read_raw_dataset(project_root, config, "market"))
    market["ts_code"] = market["ts_code"].astype("string")
    market = market[market["ts_code"] == benchmark].copy()
    if market.empty:
        raise ValueError(f"Benchmark not found in market raw dataset: {benchmark}")
    for column in ["close", "pre_close", "amount"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = market.sort_values("trade_date")
    market["benchmark_ret_1d"] = market["close"] / market["pre_close"] - 1
    market["benchmark_ret_20d"] = market["close"] / market["close"].shift(20) - 1
    market["benchmark_ret_60d"] = market["close"] / market["close"].shift(60) - 1
    market["benchmark_vol_20d"] = market["benchmark_ret_1d"].rolling(20, min_periods=10).std()
    market["benchmark_amount_rank_60d"] = market["amount"].rolling(60, min_periods=20).rank(pct=True)
    return market[
        [
            "trade_date",
            "benchmark_ret_20d",
            "benchmark_ret_60d",
            "benchmark_vol_20d",
            "benchmark_amount_rank_60d",
        ]
    ]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("lag1_")]


def add_neutralization_exposures(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["exposure_size"] = pd.to_numeric(result.get("lag1_log_circ_mv"), errors="coerce")
    turnover_candidates = ["lag1_turnover_rate_f", "lag1_turnover_rate", "lag1_turnover_5d_mean", "lag1_turnover_60d_mean"]
    turnover = next((column for column in turnover_candidates if column in result.columns), None)
    result["exposure_liquidity"] = pd.to_numeric(result[turnover], errors="coerce") if turnover else np.nan
    return result


def compute_neutralized_residuals(source: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    residual_columns = [f"{feature}__neutral" for feature in features]
    residuals = pd.DataFrame(np.nan, index=source.index, columns=residual_columns, dtype="float64")

    if not features:
        return residuals

    exposure_columns = ["exposure_size", "exposure_liquidity"]
    for _, group in source.groupby("trade_date", sort=True):
        exposures = group[exposure_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        exposure_valid = exposures.notna().all(axis=1)
        if int(exposure_valid.sum()) <= len(exposure_columns) + 2:
            continue
        for feature, residual_feature in zip(features, residual_columns, strict=True):
            y = pd.to_numeric(group[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = exposure_valid & y.notna()
            if int(valid.sum()) <= len(exposure_columns) + 2 or y[valid].nunique() < 2:
                continue
            x_values = exposures.loc[valid, exposure_columns].to_numpy(dtype="float64")
            x_values = np.column_stack([np.ones(len(x_values)), x_values])
            y_values = y.loc[valid].to_numpy(dtype="float64")
            try:
                beta = np.linalg.lstsq(x_values, y_values, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            residuals.loc[y.loc[valid].index, residual_feature] = y_values - x_values @ beta

    return residuals


def neutralized_chunk_worker(args: tuple[pd.DataFrame, list[str]]) -> pd.DataFrame:
    source, features = args
    return compute_neutralized_residuals(source, features)


def build_neutralized_dataset(
    df: pd.DataFrame,
    features: list[str],
    label_column: str,
    neutralized_jobs: int = 1,
    neutralized_chunk_size: int = 16,
) -> pd.DataFrame:
    source = add_neutralization_exposures(df[["trade_date", "ts_code", label_column, *features]].copy())
    skip_features = {"lag1_log_circ_mv", "lag1_turnover_rate", "lag1_turnover_rate_f"}
    neutral_features = [feature for feature in features if feature not in skip_features]
    if not neutral_features:
        return source[["trade_date", "ts_code", label_column]]

    jobs = max(1, int(neutralized_jobs))
    chunk_size = max(1, int(neutralized_chunk_size))
    feature_chunks = chunked(neutral_features, chunk_size)
    if jobs == 1 or len(feature_chunks) == 1:
        residuals = compute_neutralized_residuals(source, neutral_features)
    else:
        worker_count = min(jobs, len(feature_chunks), os.cpu_count() or 1)
        common_columns = ["trade_date", "exposure_size", "exposure_liquidity"]
        tasks = [(source[common_columns + chunk].copy(), chunk) for chunk in feature_chunks]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            residual_frames = list(executor.map(neutralized_chunk_worker, tasks))
        residuals = pd.concat(residual_frames, axis=1)

    return pd.concat([source[["trade_date", "ts_code", label_column]], residuals], axis=1)


def _pearson_ic(group: pd.DataFrame, feature: str, label_column: str, method: str) -> float:
    valid = group[[feature, label_column]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid[feature].nunique() < 2 or valid[label_column].nunique() < 2:
        return np.nan
    return float(valid[feature].corr(valid[label_column], method=method))


def compute_daily_ic_table(df: pd.DataFrame, features: list[str], label_column: str, min_cross_section: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature in features:
        data = df[["trade_date", feature, label_column]].copy()
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data[label_column] = pd.to_numeric(data[label_column], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            continue
        counts = data.groupby("trade_date").size()
        valid_dates = counts[counts >= min_cross_section].index
        data = data[data["trade_date"].isin(valid_dates)]
        if data.empty:
            continue

        pearson = data.groupby("trade_date")[[feature, label_column]].corr()
        pearson = pearson.xs(feature, level=1)[label_column].rename("ic")

        ranked = data.copy()
        ranked[feature] = ranked.groupby("trade_date")[feature].rank(method="average")
        ranked[label_column] = ranked.groupby("trade_date")[label_column].rank(method="average")
        spearman = ranked.groupby("trade_date")[[feature, label_column]].corr()
        spearman = spearman.xs(feature, level=1)[label_column].rename("rank_ic")

        daily = pd.concat([pearson, spearman, counts.rename("n")], axis=1).reset_index()
        daily = daily[daily["trade_date"].isin(valid_dates)]
        daily["feature"] = feature
        records.extend(daily[["trade_date", "feature", "ic", "rank_ic", "n"]].to_dict(orient="records"))
    return pd.DataFrame(records)


def summarize_ic_table(daily_ic: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature in features:
        daily = daily_ic[daily_ic["feature"].eq(feature)]
        if daily.empty:
            records.append(
                {
                    "feature": feature,
                    "ic_mean": np.nan,
                    "ic_std": np.nan,
                    "ic_t_stat": np.nan,
                    "rank_ic_mean": np.nan,
                    "rank_ic_std": np.nan,
                    "rank_ic_t_stat": np.nan,
                    "positive_rank_ic_ratio": np.nan,
                    "days": 0,
                    "avg_cross_section": np.nan,
                }
            )
            continue
        valid_ic = daily["ic"].dropna()
        valid_rank_ic = daily["rank_ic"].dropna()
        ic_std = valid_ic.std(ddof=1)
        rank_std = valid_rank_ic.std(ddof=1)
        records.append(
            {
                "feature": feature,
                "ic_mean": valid_ic.mean(),
                "ic_std": ic_std,
                "ic_t_stat": valid_ic.mean() / ic_std * np.sqrt(valid_ic.count()) if ic_std else np.nan,
                "rank_ic_mean": valid_rank_ic.mean(),
                "rank_ic_std": rank_std,
                "rank_ic_t_stat": valid_rank_ic.mean() / rank_std * np.sqrt(valid_rank_ic.count()) if rank_std else np.nan,
                "positive_rank_ic_ratio": (valid_rank_ic > 0).mean(),
                "days": int(valid_rank_ic.count()),
                "avg_cross_section": daily["n"].mean(),
            }
        )
    table = pd.DataFrame(records)
    table["abs_rank_ic_mean"] = table["rank_ic_mean"].abs()
    return table.sort_values(["days", "abs_rank_ic_mean"], ascending=[False, False]).drop(columns=["abs_rank_ic_mean"])


def compute_ic_table(df: pd.DataFrame, features: list[str], label_column: str, min_cross_section: int) -> pd.DataFrame:
    return summarize_ic_table(compute_daily_ic_table(df, features, label_column, min_cross_section), features)


def assign_group_quantiles(working: pd.DataFrame, feature: str, quantiles: int, min_cross_section: int) -> pd.DataFrame:
    working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature])
    if working.empty:
        return working.assign(quantile=pd.Series(dtype="float64"))
    grouped = working.groupby("trade_date", sort=False)[feature]
    counts = grouped.transform("size")
    unique_counts = grouped.transform("nunique")
    valid = (counts >= max(quantiles, min_cross_section)) & (unique_counts >= 2)
    if not bool(valid.any()):
        return working.iloc[0:0].assign(quantile=pd.Series(dtype="float64"))

    ranked = grouped.rank(method="first")
    bucket = np.floor((ranked - 1) * quantiles / counts) + 1
    result = working.loc[valid].copy()
    result["quantile"] = bucket.loc[valid].clip(1, quantiles).astype("float64")
    return result


def compute_quantile_table(
    df: pd.DataFrame,
    features: list[str],
    label_column: str,
    quantiles: int,
    min_cross_section: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    base = df[["trade_date", label_column, *features]].copy()
    base[label_column] = pd.to_numeric(base[label_column], errors="coerce")
    base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_column])
    for feature in features:
        working = base[["trade_date", feature, label_column]].copy()
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        working = assign_group_quantiles(working, feature, quantiles, min_cross_section)
        if working.empty:
            continue
        daily = working.groupby(["trade_date", "quantile"], as_index=False)[label_column].mean()
        pivot = daily.pivot(index="trade_date", columns="quantile", values=label_column)
        top = float(quantiles)
        bottom = 1.0
        long_short = pivot[top] - pivot[bottom] if top in pivot.columns and bottom in pivot.columns else pd.Series(dtype="float64")
        cumulative = (1 + long_short.fillna(0)).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1 if not cumulative.empty else pd.Series(dtype="float64")
        valid_long_short_days = int(long_short.dropna().count())
        records.append(
            {
                "feature": feature,
                "quantiles": quantiles,
                "q1_mean_return": pivot[bottom].mean() if bottom in pivot.columns else np.nan,
                "q_top_mean_return": pivot[top].mean() if top in pivot.columns else np.nan,
                "long_short_mean_return": long_short.mean() if valid_long_short_days else np.nan,
                "long_short_t_stat": long_short.mean() / long_short.std(ddof=1) * np.sqrt(valid_long_short_days)
                if long_short.std(ddof=1)
                else np.nan,
                "long_short_win_ratio": (long_short > 0).mean() if valid_long_short_days else np.nan,
                "long_short_max_drawdown": drawdown.min() if valid_long_short_days else np.nan,
                "days": valid_long_short_days,
            }
        )
    table = pd.DataFrame(records)
    table["abs_long_short_mean_return"] = table["long_short_mean_return"].abs()
    return table.sort_values(["days", "abs_long_short_mean_return"], ascending=[False, False]).drop(
        columns=["abs_long_short_mean_return"]
    )


def compute_quantile_detail_table(
    df: pd.DataFrame,
    features: list[str],
    label_column: str,
    quantiles: int,
    min_cross_section: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    base = df[["trade_date", label_column, *features]].copy()
    base[label_column] = pd.to_numeric(base[label_column], errors="coerce")
    base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_column])
    for feature in features:
        working = base[["trade_date", feature, label_column]].copy()
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        working = assign_group_quantiles(working, feature, quantiles, min_cross_section)
        if working.empty:
            continue
        working["feature"] = feature
        detail = working.groupby(["feature", "quantile"], as_index=False)[label_column].mean()
        detail["feature"] = feature
        detail = detail.rename(columns={label_column: "mean_return"})
        pivot = detail.sort_values("quantile")
        returns = pivot["mean_return"].to_numpy()
        monotonic_up = bool(np.all(np.diff(returns) >= 0)) if len(returns) == quantiles else False
        monotonic_down = bool(np.all(np.diff(returns) <= 0)) if len(returns) == quantiles else False
        detail["is_monotonic"] = monotonic_up or monotonic_down
        detail["monotonic_direction"] = "up" if monotonic_up else "down" if monotonic_down else "none"
        records.extend(detail.to_dict(orient="records"))
    return pd.DataFrame(records)


def compute_regime_quantile_table(
    df: pd.DataFrame,
    features: list[str],
    label_column: str,
    benchmark_regime: pd.DataFrame,
    quantiles: int,
    min_cross_section: int,
) -> pd.DataFrame:
    with_regime = assign_market_regimes(df, benchmark_regime)
    records: list[pd.DataFrame] = []
    for regime, regime_df in with_regime.groupby("market_regime", sort=True):
        table = compute_quantile_table(regime_df.drop(columns=["market_regime"]), features, label_column, quantiles, min_cross_section)
        table.insert(0, "market_regime", regime)
        records.append(table)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def compute_yearly_quantile_table(
    df: pd.DataFrame,
    features: list[str],
    label_column: str,
    quantiles: int,
    min_cross_section: int,
) -> pd.DataFrame:
    working = df.copy()
    working["year"] = working["trade_date"].astype(str).str.slice(0, 4)
    records: list[pd.DataFrame] = []
    for year, year_df in working.groupby("year", sort=True):
        table = compute_quantile_table(year_df.drop(columns=["year"]), features, label_column, quantiles, min_cross_section)
        table.insert(0, "year", year)
        records.append(table)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def compute_quality_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature in features:
        series = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        records.append(
            {
                "feature": feature,
                "missing_ratio": float(series.isna().mean()),
                "finite_ratio": float(np.isfinite(series).mean()),
                "zero_ratio": float((series == 0).mean()),
                "mean": series.mean(),
                "std": series.std(ddof=1),
                "min": series.min(),
                "p01": series.quantile(0.01),
                "p50": series.quantile(0.50),
                "p99": series.quantile(0.99),
                "max": series.max(),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(records).sort_values(["missing_ratio", "feature"])


def compute_correlation_table(df: pd.DataFrame, features: list[str], top_n: int = 100) -> pd.DataFrame:
    usable = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = usable.corr(method="spearman", min_periods=max(20, len(df) // 100))
    records: list[dict[str, Any]] = []
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            records.append({"feature_left": left, "feature_right": right, "spearman_corr": float(value)})
    if not records:
        return pd.DataFrame(columns=["feature_left", "feature_right", "spearman_corr"])
    table = pd.DataFrame(records)
    return table.reindex(table["spearman_corr"].abs().sort_values(ascending=False).index).head(top_n)


def assign_market_regimes(df: pd.DataFrame, benchmark_regime: pd.DataFrame) -> pd.DataFrame:
    daily = benchmark_regime.copy()
    ret_lower = daily["benchmark_ret_20d"].quantile(1 / 3)
    ret_upper = daily["benchmark_ret_20d"].quantile(2 / 3)
    vol_upper = daily["benchmark_vol_20d"].quantile(2 / 3)
    daily["market_regime"] = np.select(
        [
            daily["benchmark_vol_20d"] >= vol_upper,
            daily["benchmark_ret_20d"] >= ret_upper,
            daily["benchmark_ret_20d"] <= ret_lower,
        ],
        ["high_vol", "bull", "bear"],
        default="sideways",
    )
    return df.merge(daily[["trade_date", "market_regime"]], on="trade_date", how="left")


def compute_regime_ic_table(
    daily_ic: pd.DataFrame,
    df: pd.DataFrame,
    benchmark_regime: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    with_regime = assign_market_regimes(df[["trade_date"]].drop_duplicates(), benchmark_regime)[["trade_date", "market_regime"]]
    daily_with_regime = daily_ic.merge(with_regime.drop_duplicates("trade_date"), on="trade_date", how="left")
    records: list[pd.DataFrame] = []
    for regime, regime_daily in daily_with_regime.groupby("market_regime", sort=True):
        table = summarize_ic_table(regime_daily.drop(columns=["market_regime"]), features)
        table.insert(0, "market_regime", regime)
        records.append(table)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def build_feature_recommendations(
    features: list[str],
    ic_table: pd.DataFrame,
    quantile_table: pd.DataFrame,
    quality_table: pd.DataFrame,
    correlation_table: pd.DataFrame,
    max_baseline_corr: float,
) -> pd.DataFrame:
    baseline_excluded_keywords = [
        "month",
        "weekday",
        "benchmark_ret",
        "beta_",
        "residual_ret",
        "macd_",
        "rsi_",
        "bollinger_",
        "ma_ratio",
        "price_to_ma",
    ]
    table = pd.DataFrame({"feature": features})
    table = table.merge(ic_table, on="feature", how="left")
    table = table.merge(
        quantile_table[
            [
                "feature",
                "long_short_mean_return",
                "long_short_t_stat",
                "long_short_win_ratio",
                "long_short_max_drawdown",
            ]
        ],
        on="feature",
        how="left",
    )
    table = table.merge(quality_table[["feature", "missing_ratio", "unique_values"]], on="feature", how="left")
    table["abs_rank_ic_mean"] = table["rank_ic_mean"].abs()
    table["abs_long_short_mean_return"] = table["long_short_mean_return"].abs()
    table["score"] = (
        table["abs_rank_ic_mean"].fillna(0) * 100
        + table["abs_long_short_mean_return"].fillna(0) * 100
        + table["long_short_win_ratio"].fillna(0) * 0.5
        - table["missing_ratio"].fillna(1) * 0.5
    )
    table = table.sort_values("score", ascending=False).reset_index(drop=True)

    pruning_groups = [
        {"lag1_ret_5d", "lag1_ret_5d_mean"},
        {"lag1_ret_20d", "lag1_ret_20d_mean"},
        {"lag1_large_order_imbalance", "lag1_main_mf_strength"},
        {"lag1_price_to_ma20", "lag1_bollinger_z_20d"},
        {"lag1_macd_diff", "lag1_macd_dea", "lag1_macd_hist"},
        {"lag1_dist_to_limit_up", "lag1_dist_to_limit_down", "lag1_ret_1d"},
    ]
    group_winners: dict[str, str] = {}
    for group in pruning_groups:
        candidates = table[table["feature"].isin(group)]
        if candidates.empty:
            continue
        winner = candidates.sort_values("score", ascending=False).iloc[0]["feature"]
        for feature in group:
            group_winners[feature] = winner

    selected: list[str] = []
    correlated_to_selected: dict[str, str] = {}
    corr_lookup: dict[tuple[str, str], float] = {}
    for row in correlation_table.itertuples(index=False):
        left = getattr(row, "feature_left")
        right = getattr(row, "feature_right")
        corr = float(getattr(row, "spearman_corr"))
        corr_lookup[(left, right)] = corr
        corr_lookup[(right, left)] = corr

    recommendations: list[str] = []
    reasons: list[str] = []
    for row in table.itertuples(index=False):
        feature = getattr(row, "feature")
        missing_ratio = getattr(row, "missing_ratio")
        unique_values = getattr(row, "unique_values")
        rank_ic = getattr(row, "rank_ic_mean")
        ls_return = getattr(row, "long_short_mean_return")
        ls_t = getattr(row, "long_short_t_stat")
        if pd.isna(missing_ratio) or missing_ratio > 0.35 or unique_values < 5:
            recommendations.append("drop")
            reasons.append("low_coverage_or_low_variation")
            continue
        if any(keyword in feature for keyword in baseline_excluded_keywords):
            recommendations.append("advanced")
            reasons.append("reserved_for_advanced_or_sequence_model")
            continue
        group_winner = group_winners.get(feature)
        if group_winner is not None and group_winner != feature:
            recommendations.append("advanced")
            reasons.append(f"pruned_by_collinearity_group:{group_winner}")
            continue
        if pd.isna(rank_ic) and pd.isna(ls_return):
            recommendations.append("drop")
            reasons.append("no_validation_signal")
            continue
        correlated = None
        for chosen in selected:
            corr = corr_lookup.get((feature, chosen))
            if corr is not None and abs(corr) >= max_baseline_corr:
                correlated = chosen
                break
        strong_enough = (
            abs(rank_ic) >= 0.015
            or abs(ls_return) >= 0.006
            or (not pd.isna(ls_t) and abs(ls_t) >= 2.0)
        )
        if strong_enough and correlated is None:
            selected.append(feature)
            recommendations.append("baseline")
            reasons.append("stable_cross_section_signal")
        elif strong_enough and correlated is not None:
            correlated_to_selected[feature] = correlated
            recommendations.append("advanced")
            reasons.append(f"correlated_with_baseline:{correlated}")
        else:
            recommendations.append("advanced")
            reasons.append("weak_but_potential_sequence_signal")
    table["recommendation"] = recommendations
    table["reason"] = reasons
    return table.drop(columns=["abs_rank_ic_mean", "abs_long_short_mean_return"])


def write_outputs(
    output_dir: Path,
    ic_table: pd.DataFrame,
    quantile_table: pd.DataFrame,
    quantile_detail_table: pd.DataFrame,
    yearly_quantile_table: pd.DataFrame,
    regime_quantile_table: pd.DataFrame,
    neutralized_ic_table: pd.DataFrame,
    holdout_quantile_table: pd.DataFrame,
    quality_table: pd.DataFrame,
    correlation_table: pd.DataFrame,
    regime_ic_table: pd.DataFrame,
    recommendation_table: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    ic_table.to_csv(paths["ic"], index=False, encoding="utf-8-sig")
    quantile_table.to_csv(paths["quantile"], index=False, encoding="utf-8-sig")
    quantile_detail_table.to_csv(paths["quantile_detail"], index=False, encoding="utf-8-sig")
    yearly_quantile_table.to_csv(paths["yearly_quantile"], index=False, encoding="utf-8-sig")
    regime_quantile_table.to_csv(paths["regime_quantile"], index=False, encoding="utf-8-sig")
    neutralized_ic_table.to_csv(paths["neutralized_ic"], index=False, encoding="utf-8-sig")
    holdout_quantile_table.to_csv(paths["holdout_quantile"], index=False, encoding="utf-8-sig")
    quality_table.to_csv(paths["quality"], index=False, encoding="utf-8-sig")
    correlation_table.to_csv(paths["correlation"], index=False, encoding="utf-8-sig")
    regime_ic_table.to_csv(paths["regime_ic"], index=False, encoding="utf-8-sig")
    recommendation_table.to_csv(paths["recommendations"], index=False, encoding="utf-8-sig")
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def run_factor_validation(
    project_root: Path,
    config_path: Path,
    data_version: str,
    label_column: str,
    quantiles: int,
    min_cross_section: int,
    max_baseline_corr: float,
    feature_set: str | None = None,
    train_end_date: str | None = None,
    eval_start_date: str | None = None,
    skip_neutralized: bool = False,
    skip_extended_quantile: bool = False,
    skip_quantile: bool = False,
    stage: str = "all",
    resume: bool = False,
    neutralized_jobs: int = 1,
    neutralized_chunk_size: int = 16,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported validation stage: {stage}")
    config = load_yaml(config_path)
    labels_config_path = project_root / "configs" / "labels.yaml"
    labels_config = load_yaml(labels_config_path) if labels_config_path.exists() else {}
    benchmark = labels_config.get("benchmark", "399006.SZ")
    validation_config = ValidationConfig(
        data_version=data_version,
        label_column=label_column,
        quantiles=quantiles,
        min_cross_section=min_cross_section,
        max_baseline_corr=max_baseline_corr,
        train_end_date=train_end_date,
        eval_start_date=eval_start_date,
        neutralized_jobs=neutralized_jobs,
        neutralized_chunk_size=neutralized_chunk_size,
    )
    df = read_dataset(project_root, config, validation_config.data_version)
    if validation_config.label_column not in df.columns:
        raise ValueError(f"Label column not found: {validation_config.label_column}")
    features = feature_columns(df)
    if feature_set:
        configured = load_yaml(project_root / "configs" / "features.yaml")
        selected = configured.get("feature_sets", {}).get(feature_set, {}).get("selected_features", [])
        features = [feature for feature in selected if feature in features]
    if not features:
        raise ValueError("No lag1_ feature columns found in mart dataset.")

    recommendation_df = df
    evaluation_df = df
    if validation_config.train_end_date:
        recommendation_df = df[df["trade_date"].astype(str) <= validation_config.train_end_date].copy()
    if validation_config.eval_start_date:
        evaluation_df = df[df["trade_date"].astype(str) >= validation_config.eval_start_date].copy()
    if recommendation_df.empty:
        raise ValueError("Train-only recommendation window is empty.")
    if evaluation_df.empty:
        raise ValueError("Evaluation window is empty.")

    mode_id = validation_mode_id(
        label_column=validation_config.label_column,
        quantiles=validation_config.quantiles,
        min_cross_section=validation_config.min_cross_section,
        max_baseline_corr=validation_config.max_baseline_corr,
        train_end_date=validation_config.train_end_date,
        eval_start_date=validation_config.eval_start_date,
        skip_quantile=skip_quantile,
        skip_extended_quantile=skip_extended_quantile,
        skip_neutralized=skip_neutralized,
    )
    output_dir = validation_output_dir(project_root, validation_config.data_version, feature_set, mode_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)

    daily_ic_table = read_csv_checkpoint(paths["daily_ic"])
    daily_ic_table = daily_ic_table if daily_ic_table is not None else pd.DataFrame()
    ic_table = read_csv_checkpoint(paths["ic"])
    ic_table = ic_table if ic_table is not None else pd.DataFrame()
    quantile_table = read_csv_checkpoint(paths["quantile"])
    quantile_table = quantile_table if quantile_table is not None else empty_quantile_table()
    quantile_detail_table = read_csv_checkpoint(paths["quantile_detail"])
    quantile_detail_table = quantile_detail_table if quantile_detail_table is not None else pd.DataFrame()
    quality_table = read_csv_checkpoint(paths["quality"])
    quality_table = quality_table if quality_table is not None else pd.DataFrame()
    correlation_table = read_csv_checkpoint(paths["correlation"])
    correlation_table = correlation_table if correlation_table is not None else pd.DataFrame()
    neutralized_ic_table = read_csv_checkpoint(paths["neutralized_ic"])
    neutralized_ic_table = neutralized_ic_table if neutralized_ic_table is not None else pd.DataFrame()
    regime_ic_table = read_csv_checkpoint(paths["regime_ic"])
    regime_ic_table = regime_ic_table if regime_ic_table is not None else pd.DataFrame()
    yearly_quantile_table = read_csv_checkpoint(paths["yearly_quantile"])
    yearly_quantile_table = yearly_quantile_table if yearly_quantile_table is not None else pd.DataFrame()
    regime_quantile_table = read_csv_checkpoint(paths["regime_quantile"])
    regime_quantile_table = regime_quantile_table if regime_quantile_table is not None else pd.DataFrame()
    recommendation_table = read_csv_checkpoint(paths["recommendations"])
    recommendation_table = recommendation_table if recommendation_table is not None else pd.DataFrame()
    holdout_quantile = read_csv_checkpoint(paths["holdout_quantile"])
    holdout_quantile = holdout_quantile if holdout_quantile is not None else empty_quantile_table()

    if should_run_stage(stage, "ic"):
        started_at = time.perf_counter()
        if resume and paths["daily_ic"].exists() and paths["ic"].exists():
            daily_ic_table = pd.read_csv(paths["daily_ic"])
            ic_table = pd.read_csv(paths["ic"])
            profile_stage(output_dir, "ic", "resumed", started_at, df, features, [paths["daily_ic"], paths["ic"]])
        else:
            daily_ic_table = compute_daily_ic_table(df, features, validation_config.label_column, validation_config.min_cross_section)
            ic_table = summarize_ic_table(daily_ic_table, features)
            write_csv_checkpoint(daily_ic_table, paths["daily_ic"])
            write_csv_checkpoint(ic_table, paths["ic"])
            profile_stage(output_dir, "ic", "computed", started_at, df, features, [paths["daily_ic"], paths["ic"]])

    if should_run_stage(stage, "quality"):
        started_at = time.perf_counter()
        if resume and paths["quality"].exists():
            quality_table = pd.read_csv(paths["quality"])
            profile_stage(output_dir, "quality", "resumed", started_at, df, features, [paths["quality"]])
        else:
            quality_table = compute_quality_table(df, features)
            write_csv_checkpoint(quality_table, paths["quality"])
            profile_stage(output_dir, "quality", "computed", started_at, df, features, [paths["quality"]])

    if should_run_stage(stage, "correlation"):
        started_at = time.perf_counter()
        if resume and paths["correlation"].exists():
            correlation_table = pd.read_csv(paths["correlation"])
            profile_stage(output_dir, "correlation", "resumed", started_at, df, features, [paths["correlation"]])
        else:
            correlation_table = compute_correlation_table(df, features)
            write_csv_checkpoint(correlation_table, paths["correlation"])
            profile_stage(output_dir, "correlation", "computed", started_at, df, features, [paths["correlation"]])

    if should_run_stage(stage, "quantile"):
        started_at = time.perf_counter()
        if skip_quantile:
            quantile_table = empty_quantile_table()
            write_csv_checkpoint(quantile_table, paths["quantile"])
            profile_stage(output_dir, "quantile", "skipped", started_at, df, features, [paths["quantile"]])
        elif resume and paths["quantile"].exists():
            quantile_table = pd.read_csv(paths["quantile"])
            profile_stage(output_dir, "quantile", "resumed", started_at, df, features, [paths["quantile"]])
        else:
            quantile_table = compute_quantile_table(
                df,
                features,
                validation_config.label_column,
                validation_config.quantiles,
                validation_config.min_cross_section,
            )
            write_csv_checkpoint(quantile_table, paths["quantile"])
            profile_stage(output_dir, "quantile", "computed", started_at, df, features, [paths["quantile"]])

    if should_run_stage(stage, "extended-quantile"):
        started_at = time.perf_counter()
        if skip_extended_quantile:
            quantile_detail_table = pd.DataFrame()
            yearly_quantile_table = pd.DataFrame()
            regime_quantile_table = pd.DataFrame()
            for path in [paths["quantile_detail"], paths["yearly_quantile"], paths["regime_quantile"]]:
                write_csv_checkpoint(pd.DataFrame(), path)
            profile_stage(
                output_dir,
                "extended-quantile",
                "skipped",
                started_at,
                df,
                features,
                [paths["quantile_detail"], paths["yearly_quantile"], paths["regime_quantile"]],
            )
        elif (
            resume
            and paths["quantile_detail"].exists()
            and paths["yearly_quantile"].exists()
            and paths["regime_quantile"].exists()
        ):
            quantile_detail_table = pd.read_csv(paths["quantile_detail"])
            yearly_quantile_table = pd.read_csv(paths["yearly_quantile"])
            regime_quantile_table = pd.read_csv(paths["regime_quantile"])
            profile_stage(
                output_dir,
                "extended-quantile",
                "resumed",
                started_at,
                df,
                features,
                [paths["quantile_detail"], paths["yearly_quantile"], paths["regime_quantile"]],
            )
        else:
            benchmark_regime = read_benchmark_regime_source(project_root, config, benchmark)
            quantile_detail_table = compute_quantile_detail_table(
                df,
                features,
                validation_config.label_column,
                validation_config.quantiles,
                validation_config.min_cross_section,
            )
            yearly_quantile_table = compute_yearly_quantile_table(
                df,
                features,
                validation_config.label_column,
                validation_config.quantiles,
                validation_config.min_cross_section,
            )
            regime_quantile_table = compute_regime_quantile_table(
                df,
                features,
                validation_config.label_column,
                benchmark_regime,
                validation_config.quantiles,
                validation_config.min_cross_section,
            )
            write_csv_checkpoint(quantile_detail_table, paths["quantile_detail"])
            write_csv_checkpoint(yearly_quantile_table, paths["yearly_quantile"])
            write_csv_checkpoint(regime_quantile_table, paths["regime_quantile"])
            profile_stage(
                output_dir,
                "extended-quantile",
                "computed",
                started_at,
                df,
                features,
                [paths["quantile_detail"], paths["yearly_quantile"], paths["regime_quantile"]],
            )

    if should_run_stage(stage, "neutralized"):
        started_at = time.perf_counter()
        if skip_neutralized:
            neutralized_ic_table = pd.DataFrame()
            write_csv_checkpoint(neutralized_ic_table, paths["neutralized_ic"])
            profile_stage(output_dir, "neutralized", "skipped", started_at, df, features, [paths["neutralized_ic"]])
        elif resume and paths["neutralized_ic"].exists():
            neutralized_ic_table = pd.read_csv(paths["neutralized_ic"])
            profile_stage(output_dir, "neutralized", "resumed", started_at, df, features, [paths["neutralized_ic"]])
        else:
            neutralized = build_neutralized_dataset(
                df,
                features,
                validation_config.label_column,
                neutralized_jobs=validation_config.neutralized_jobs,
                neutralized_chunk_size=validation_config.neutralized_chunk_size,
            )
            neutralized_features = [column for column in neutralized.columns if column.endswith("__neutral")]
            neutralized_ic_table = compute_ic_table(
                neutralized,
                neutralized_features,
                validation_config.label_column,
                validation_config.min_cross_section,
            )
            write_csv_checkpoint(neutralized_ic_table, paths["neutralized_ic"])
            profile_stage(output_dir, "neutralized", "computed", started_at, df, features, [paths["neutralized_ic"]])

    if should_run_stage(stage, "regime-ic"):
        started_at = time.perf_counter()
        if resume and paths["regime_ic"].exists():
            regime_ic_table = pd.read_csv(paths["regime_ic"])
            profile_stage(output_dir, "regime-ic", "resumed", started_at, df, features, [paths["regime_ic"]])
        else:
            if daily_ic_table.empty:
                if paths["daily_ic"].exists():
                    daily_ic_table = pd.read_csv(paths["daily_ic"])
                else:
                    daily_ic_table = compute_daily_ic_table(
                        df,
                        features,
                        validation_config.label_column,
                        validation_config.min_cross_section,
                    )
                    write_csv_checkpoint(daily_ic_table, paths["daily_ic"])
            benchmark_regime = read_benchmark_regime_source(project_root, config, benchmark)
            regime_ic_table = compute_regime_ic_table(daily_ic_table, df, benchmark_regime, features)
            write_csv_checkpoint(regime_ic_table, paths["regime_ic"])
            profile_stage(output_dir, "regime-ic", "computed", started_at, df, features, [paths["regime_ic"]])

    if should_run_stage(stage, "recommendation"):
        started_at = time.perf_counter()
        if resume and paths["recommendations"].exists() and (skip_quantile or paths["holdout_quantile"].exists()):
            recommendation_table = pd.read_csv(paths["recommendations"])
            holdout_quantile = empty_quantile_table() if skip_quantile else pd.read_csv(paths["holdout_quantile"])
            profile_stage(
                output_dir,
                "recommendation",
                "resumed",
                started_at,
                df,
                features,
                [paths["recommendations"], paths["holdout_quantile"]],
            )
        else:
            recommendation_daily_ic = compute_daily_ic_table(
                recommendation_df,
                features,
                validation_config.label_column,
                validation_config.min_cross_section,
            )
            recommendation_ic = summarize_ic_table(recommendation_daily_ic, features)
            recommendation_quantile = (
                empty_quantile_table()
                if skip_quantile
                else compute_quantile_table(
                    recommendation_df,
                    features,
                    validation_config.label_column,
                    validation_config.quantiles,
                    validation_config.min_cross_section,
                )
            )
            recommendation_quality = compute_quality_table(recommendation_df, features)
            recommendation_correlation = compute_correlation_table(recommendation_df, features)
            recommendation_table = build_feature_recommendations(
                features,
                recommendation_ic,
                recommendation_quantile,
                recommendation_quality,
                recommendation_correlation,
                validation_config.max_baseline_corr,
            )
            selected_by_train = recommendation_table[
                recommendation_table["recommendation"].isin(["baseline", "advanced"])
            ]["feature"].tolist()
            holdout_quantile = (
                empty_quantile_table()
                if skip_quantile
                else compute_quantile_table(
                    evaluation_df,
                    selected_by_train,
                    validation_config.label_column,
                    validation_config.quantiles,
                    validation_config.min_cross_section,
                )
            )
            write_csv_checkpoint(recommendation_table, paths["recommendations"])
            write_csv_checkpoint(holdout_quantile, paths["holdout_quantile"])
            profile_stage(
                output_dir,
                "recommendation",
                "computed",
                started_at,
                df,
                features,
                [paths["recommendations"], paths["holdout_quantile"]],
            )
    summary = {
        "data_version": validation_config.data_version,
        "label_column": validation_config.label_column,
        "rows": int(len(df)),
        "trade_dates": int(df["trade_date"].nunique()),
        "stocks": int(df["ts_code"].nunique()),
        "features": int(len(features)),
        "feature_set": feature_set or "all_lag1",
        "validation_mode": mode_id,
        "output_dir": str(output_dir),
        "train_end_date": validation_config.train_end_date,
        "eval_start_date": validation_config.eval_start_date,
        "quantiles": validation_config.quantiles,
        "min_cross_section": validation_config.min_cross_section,
        "max_baseline_corr": validation_config.max_baseline_corr,
        "regime_definition": "historical_benchmark_ret20_vol20_amount60",
        "neutralized_ic_skipped": skip_neutralized,
        "quantile_skipped": skip_quantile,
        "extended_quantile_skipped": skip_extended_quantile,
        "stage": stage,
        "resume": resume,
        "neutralized_jobs": validation_config.neutralized_jobs,
        "neutralized_chunk_size": validation_config.neutralized_chunk_size,
        "baseline_features": recommended_features(recommendation_table, "baseline"),
        "advanced_features": recommended_features(recommendation_table, "advanced"),
        "dropped_features": recommended_features(recommendation_table, "drop"),
        "top_abs_rank_ic": ic_table.head(10).to_dict(orient="records"),
        "top_abs_long_short": quantile_table.head(10).to_dict(orient="records"),
        "top_holdout_long_short": holdout_quantile.head(10).to_dict(orient="records"),
        "generated_at": utc_now_iso(),
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = {key: str(path) for key, path in paths.items()}
    return {"summary": summary, "outputs": outputs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor IC, RankIC, quantile and quality validation for mart data.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--label-column", default="label_rel_return")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--max-baseline-corr", type=float, default=0.85)
    parser.add_argument("--feature-set", choices=["baseline_lightgbm", "advanced_sequence"])
    parser.add_argument("--train-end-date")
    parser.add_argument("--eval-start-date")
    parser.add_argument("--skip-neutralized", action="store_true")
    parser.add_argument("--skip-quantile", action="store_true")
    parser.add_argument("--skip-extended-quantile", action="store_true")
    parser.add_argument("--stage", choices=sorted(STAGES), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--neutralized-jobs", type=int, default=1)
    parser.add_argument("--neutralized-chunk-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    result = run_factor_validation(
        project_root=project_root,
        config_path=project_root / args.config,
        data_version=args.data_version,
        label_column=args.label_column,
        quantiles=args.quantiles,
        min_cross_section=args.min_cross_section,
        max_baseline_corr=args.max_baseline_corr,
        feature_set=args.feature_set,
        train_end_date=args.train_end_date,
        eval_start_date=args.eval_start_date,
        skip_neutralized=args.skip_neutralized,
        skip_quantile=args.skip_quantile,
        skip_extended_quantile=args.skip_extended_quantile,
        stage=args.stage,
        resume=args.resume,
        neutralized_jobs=args.neutralized_jobs,
        neutralized_chunk_size=args.neutralized_chunk_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
