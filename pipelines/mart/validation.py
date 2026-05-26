from __future__ import annotations

import argparse
import json
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_dataset(project_root: Path, config: dict[str, Any], data_version: str) -> pd.DataFrame:
    dataset_path = project_root / config["mart"]["datasets_dir"] / f"dataset_{data_version}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing mart dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    if "trade_date" not in df.columns or "ts_code" not in df.columns:
        raise ValueError("Mart dataset must contain trade_date and ts_code.")
    df["trade_date"] = df["trade_date"].astype("string")
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("lag1_")]


def _pearson_ic(group: pd.DataFrame, feature: str, label_column: str, method: str) -> float:
    valid = group[[feature, label_column]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid[feature].nunique() < 2 or valid[label_column].nunique() < 2:
        return np.nan
    return float(valid[feature].corr(valid[label_column], method=method))


def compute_daily_ic_table(df: pd.DataFrame, features: list[str], label_column: str, min_cross_section: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = df.groupby("trade_date", sort=True)
    for feature in features:
        for trade_date, group in grouped:
            valid_count = int(
                group[[feature, label_column]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .shape[0]
            )
            if valid_count < min_cross_section:
                continue
            records.append(
                {
                    "trade_date": trade_date,
                    "feature": feature,
                    "ic": _pearson_ic(group, feature, label_column, "pearson"),
                    "rank_ic": _pearson_ic(group, feature, label_column, "spearman"),
                    "n": valid_count,
                }
            )
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


def _assign_quantile(values: pd.Series, quantiles: int) -> pd.Series:
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    if valid.nunique() < 2 or len(valid) < quantiles:
        return result
    try:
        result.loc[valid.index] = pd.qcut(valid.rank(method="first"), quantiles, labels=False) + 1
    except ValueError:
        return result
    return result


def compute_quantile_table(
    df: pd.DataFrame,
    features: list[str],
    label_column: str,
    quantiles: int,
    min_cross_section: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature in features:
        working = df[["trade_date", feature, label_column]].copy()
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        working[label_column] = pd.to_numeric(working[label_column], errors="coerce")
        working = working.replace([np.inf, -np.inf], np.nan).dropna()
        if working.empty:
            continue
        working["quantile"] = working.groupby("trade_date", group_keys=False)[feature].transform(
            lambda s: _assign_quantile(s, quantiles)
        )
        working = working.dropna(subset=["quantile"])
        day_sizes = working.groupby("trade_date").size()
        valid_dates = day_sizes[day_sizes >= min_cross_section].index
        working = working[working["trade_date"].isin(valid_dates)]
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


def assign_market_regimes(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    daily = df.groupby("trade_date", as_index=False)[label_column].mean().rename(columns={label_column: "market_proxy_return"})
    lower = daily["market_proxy_return"].quantile(1 / 3)
    upper = daily["market_proxy_return"].quantile(2 / 3)
    daily["market_regime"] = np.select(
        [daily["market_proxy_return"] <= lower, daily["market_proxy_return"] >= upper],
        ["bear", "bull"],
        default="sideways",
    )
    return df.merge(daily[["trade_date", "market_regime"]], on="trade_date", how="left")


def compute_regime_ic_table(
    daily_ic: pd.DataFrame,
    df: pd.DataFrame,
    features: list[str],
    label_column: str,
) -> pd.DataFrame:
    with_regime = assign_market_regimes(df[["trade_date", label_column]], label_column)[["trade_date", "market_regime"]]
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
    quality_table: pd.DataFrame,
    correlation_table: pd.DataFrame,
    regime_ic_table: pd.DataFrame,
    recommendation_table: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ic": output_dir / "factor_ic_rankic.csv",
        "quantile": output_dir / "factor_quantile_long_short.csv",
        "quality": output_dir / "feature_quality.csv",
        "correlation": output_dir / "feature_correlation_top.csv",
        "regime_ic": output_dir / "factor_regime_ic.csv",
        "recommendations": output_dir / "feature_recommendations.csv",
        "summary": output_dir / "factor_validation_summary.json",
    }
    ic_table.to_csv(paths["ic"], index=False, encoding="utf-8-sig")
    quantile_table.to_csv(paths["quantile"], index=False, encoding="utf-8-sig")
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
) -> dict[str, Any]:
    config = load_yaml(config_path)
    validation_config = ValidationConfig(
        data_version=data_version,
        label_column=label_column,
        quantiles=quantiles,
        min_cross_section=min_cross_section,
        max_baseline_corr=max_baseline_corr,
    )
    df = read_dataset(project_root, config, validation_config.data_version)
    if validation_config.label_column not in df.columns:
        raise ValueError(f"Label column not found: {validation_config.label_column}")
    features = feature_columns(df)
    if not features:
        raise ValueError("No lag1_ feature columns found in mart dataset.")

    daily_ic_table = compute_daily_ic_table(df, features, validation_config.label_column, validation_config.min_cross_section)
    ic_table = summarize_ic_table(daily_ic_table, features)
    quantile_table = compute_quantile_table(
        df,
        features,
        validation_config.label_column,
        validation_config.quantiles,
        validation_config.min_cross_section,
    )
    quality_table = compute_quality_table(df, features)
    correlation_table = compute_correlation_table(df, features)
    regime_ic_table = compute_regime_ic_table(daily_ic_table, df, features, validation_config.label_column)
    recommendation_table = build_feature_recommendations(
        features,
        ic_table,
        quantile_table,
        quality_table,
        correlation_table,
        validation_config.max_baseline_corr,
    )
    output_dir = project_root / "outputs" / "factor_validation" / validation_config.data_version
    summary = {
        "data_version": validation_config.data_version,
        "label_column": validation_config.label_column,
        "rows": int(len(df)),
        "trade_dates": int(df["trade_date"].nunique()),
        "stocks": int(df["ts_code"].nunique()),
        "features": int(len(features)),
        "quantiles": validation_config.quantiles,
        "min_cross_section": validation_config.min_cross_section,
        "max_baseline_corr": validation_config.max_baseline_corr,
        "baseline_features": recommendation_table[recommendation_table["recommendation"].eq("baseline")]["feature"].tolist(),
        "advanced_features": recommendation_table[recommendation_table["recommendation"].eq("advanced")]["feature"].tolist(),
        "dropped_features": recommendation_table[recommendation_table["recommendation"].eq("drop")]["feature"].tolist(),
        "top_abs_rank_ic": ic_table.head(10).to_dict(orient="records"),
        "top_abs_long_short": quantile_table.head(10).to_dict(orient="records"),
        "generated_at": utc_now_iso(),
    }
    outputs = write_outputs(
        output_dir,
        ic_table,
        quantile_table,
        quality_table,
        correlation_table,
        regime_ic_table,
        recommendation_table,
        summary,
    )
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
