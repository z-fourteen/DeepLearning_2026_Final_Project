from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.backtest_t1_fill_sim import build_summary, load_data, run_backtest  # noqa: E402


STYLE_COLUMNS = [
    "lag1_log_circ_mv",
    "lag1_log_total_mv",
    "lag1_beta_20d",
    "lag1_beta_60d",
    "lag1_ret_20d",
    "lag1_ret_5d_mean",
    "lag1_ret_20d_mean",
    "lag1_ret_60d_mean",
    "lag1_ret_20d_std",
    "lag1_ret_60d_std",
    "lag1_amplitude",
    "lag1_vol_log",
]
LIQUIDITY_COLUMNS = [
    "lag1_amount_log",
    "lag1_amount_rank_pct",
    "lag1_amount_20d_mean",
    "lag1_amount_60d_mean",
    "lag1_turnover_rate",
    "lag1_turnover_rate_f",
    "lag1_turnover_20d_mean",
    "lag1_turnover_60d_mean",
    "lag1_turnover_cost_proxy",
    "lag1_illiquidity_proxy",
]
TRADABILITY_COLUMNS = [
    "lag1_limit_position",
    "lag1_limit_touch_up",
    "lag1_limit_touch_down",
    "lag1_near_limit_up_2pct",
    "lag1_near_limit_down_2pct",
]
EXECUTION_COLUMNS = [
    "trade_date",
    "ts_code",
    "next_trade_date",
    "exit_trade_date",
    "next_amount",
    "next_vol",
    "buy_executable_t1_open",
    "sell_executable_t1_open",
    "next_is_limit_up",
    "next_is_limit_down",
    "next_is_suspended",
    "execution_return_open_to_close5",
    "benchmark_next_open_to_exit_close_return",
]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_path(path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


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


def safe_ir(series: pd.Series) -> float:
    clean = pd.Series(series, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / std) if std and pd.notna(std) else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    clean = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    equity = (1.0 + clean).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def annualized_return(returns: pd.Series, periods_per_year: float = 252.0 / 5.0) -> float:
    clean = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    cumulative = float((1.0 + clean).prod() - 1.0)
    return float((1.0 + cumulative) ** (periods_per_year / len(clean)) - 1.0)


def summarize_period_group(group: pd.DataFrame) -> dict[str, Any]:
    net = pd.to_numeric(group["net_return"], errors="coerce")
    bench = pd.to_numeric(group["excess_vs_benchmark"], errors="coerce")
    execu = pd.to_numeric(group["excess_vs_executable_universe"], errors="coerce")
    return {
        "periods": int(len(group)),
        "net_mean": float(net.mean()),
        "net_ann": annualized_return(net),
        "net_ir": safe_ir(net),
        "net_max_drawdown": max_drawdown(net),
        "win_rate": float((net > 0).mean()),
        "excess_benchmark_ann": annualized_return(bench),
        "excess_exec_universe_ann": annualized_return(execu),
        "avg_filled_turnover": float(group["filled_turnover"].mean()),
        "avg_position_count": float(group["position_count"].mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the compatibility T+1 route for the frozen final mainline.")
    parser.add_argument("--config", default="configs/backtest/clean_resid_t1_top20_keep2.yaml")
    parser.add_argument("--skip-heavy-sensitivity", action="store_true")
    return parser.parse_args()


def read_mart(mart_path: Path) -> pd.DataFrame:
    wanted = ["trade_date", "ts_code", "industry", *STYLE_COLUMNS, *LIQUIDITY_COLUMNS, *TRADABILITY_COLUMNS]
    columns = set(pd.read_parquet(mart_path, engine="pyarrow").columns)
    selected = [column for column in wanted if column in columns]
    mart = pd.read_parquet(mart_path, columns=selected)
    mart["trade_date"] = mart["trade_date"].astype(str)
    mart["ts_code"] = mart["ts_code"].astype(str)
    if "industry" in mart.columns:
        mart["industry"] = mart["industry"].astype("string").fillna("UNKNOWN")
    else:
        mart["industry"] = "UNKNOWN"
    numeric_cols = [column for column in mart.columns if column not in {"trade_date", "ts_code", "industry"}]
    for column in numeric_cols:
        mart[column] = pd.to_numeric(mart[column], errors="coerce")
    return mart


def read_execution(path: Path) -> pd.DataFrame:
    columns = set(pd.read_parquet(path, engine="pyarrow").columns)
    selected = [column for column in EXECUTION_COLUMNS if column in columns]
    execution = pd.read_parquet(path, columns=selected)
    execution["trade_date"] = execution["trade_date"].astype(str)
    execution["ts_code"] = execution["ts_code"].astype(str)
    for column in ["next_amount", "next_vol", "execution_return_open_to_close5", "benchmark_next_open_to_exit_close_return"]:
        if column in execution.columns:
            execution[column] = pd.to_numeric(execution[column], errors="coerce")
    for column in [
        "buy_executable_t1_open",
        "sell_executable_t1_open",
        "next_is_limit_up",
        "next_is_limit_down",
        "next_is_suspended",
    ]:
        if column in execution.columns:
            execution[column] = execution[column].fillna(False).astype(bool)
    return execution


def selected_codes_for_periods(predictions: pd.DataFrame, periods: pd.DataFrame, k: int, keep: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_periods in periods.groupby("split", sort=True):
        split_predictions = predictions[predictions["split"] == split]
        previous_codes: list[str] = []
        for _, period in split_periods.sort_values("trade_date").iterrows():
            date = str(period["trade_date"])
            day = split_predictions[split_predictions["trade_date"] == date].sort_values("pred_score", ascending=False)
            if len(day) < k:
                continue
            ordered = day["ts_code"].astype(str).tolist()
            keep_rank = min(len(ordered), max(k, int(math.ceil(k * keep))))
            keep_set = set(ordered[:keep_rank])
            selected = [code for code in previous_codes if code in keep_set]
            selected_set = set(selected)
            for code in ordered:
                if len(selected) >= k:
                    break
                if code not in selected_set:
                    selected.append(code)
                    selected_set.add(code)
            previous_codes = selected[:k]
            rank_map = {code: rank + 1 for rank, code in enumerate(ordered)}
            for code in previous_codes:
                rows.append(
                    {
                        "split": split,
                        "trade_date": date,
                        "ts_code": code,
                        "rank": rank_map.get(code),
                        "k": k,
                        "keep_multiplier": keep,
                    }
                )
    return pd.DataFrame(rows)


def run_mainline(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = config["inputs"]
    execution = config["execution"]
    data = load_data(resolve_path(inputs["predictions"]), resolve_path(inputs["execution_labels"]))
    periods = run_backtest(
        frame=data,
        k_values=[int(execution["k"])],
        keep_multipliers=[float(execution["keep_multiplier"])],
        cost_bps=float(execution["cost_bps"]),
        slippage_bps=float(execution["slippage_bps"]),
        portfolio_nav=float(execution["portfolio_nav"]),
        participation_cap=float(execution["participation_cap"]),
        rebalance_stride=int(execution["rebalance_stride"]),
        min_daily_count=int(execution["min_daily_count"]),
    )
    return data, periods


def flatten_t1_summary(periods: pd.DataFrame, extra: dict[str, Any] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary = build_summary(periods)
    for split, split_summary in summary.items():
        for setting, metrics in split_summary.items():
            parts = setting.replace("top_", "").replace("x", "").split("_keep_")
            row = {
                "split": split,
                "setting": setting,
                "k": int(parts[0]),
                "keep_multiplier": float(parts[1]),
                "net_ann": metrics["net"]["annualized_return"],
                "net_ir": metrics["net"]["ir"],
                "net_mdd": metrics["net"]["max_drawdown"],
                "excess_benchmark_ann": metrics["excess_vs_benchmark"]["annualized_return"],
                "excess_exec_universe_ann": metrics["excess_vs_executable_universe"]["annualized_return"],
                "avg_filled_turnover": metrics["average_filled_turnover"],
                "avg_position_count": metrics["average_position_count"],
            }
            if extra:
                row.update(extra)
            rows.append(row)
    return pd.DataFrame(rows)


def build_time_summaries(periods: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = periods.copy()
    p["year"] = p["trade_date"].astype(str).str.slice(0, 4)
    p["month"] = p["trade_date"].astype(str).str.slice(0, 6)
    p["quarter"] = pd.PeriodIndex(pd.to_datetime(p["trade_date"]), freq="Q").astype(str)
    yearly = pd.DataFrame(
        [{"split": split, "year": year, **summarize_period_group(group)} for (split, year), group in p.groupby(["split", "year"], sort=True)]
    )
    monthly = pd.DataFrame(
        [{"split": split, "month": month, **summarize_period_group(group)} for (split, month), group in p.groupby(["split", "month"], sort=True)]
    )
    quarterly = pd.DataFrame(
        [{"split": split, "quarter": quarter, **summarize_period_group(group)} for (split, quarter), group in p.groupby(["split", "quarter"], sort=True)]
    )
    return yearly, monthly, quarterly


def run_k_keep_sensitivity(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    execution = config["execution"]
    sens = config["sensitivity"]
    periods = run_backtest(
        frame=data,
        k_values=[int(value) for value in sens["k_values"]],
        keep_multipliers=[float(value) for value in sens["keep_multipliers"]],
        cost_bps=float(execution["cost_bps"]),
        slippage_bps=float(execution["slippage_bps"]),
        portfolio_nav=float(execution["portfolio_nav"]),
        participation_cap=float(execution["participation_cap"]),
        rebalance_stride=int(execution["rebalance_stride"]),
        min_daily_count=int(execution["min_daily_count"]),
    )
    return flatten_t1_summary(periods)


def run_cost_sensitivity(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    execution = config["execution"]
    rows: list[pd.DataFrame] = []
    for cost in config["sensitivity"]["cost_bps_values"]:
        periods = run_backtest(
            frame=data,
            k_values=[int(execution["k"])],
            keep_multipliers=[float(execution["keep_multiplier"])],
            cost_bps=float(cost),
            slippage_bps=float(execution["slippage_bps"]),
            portfolio_nav=float(execution["portfolio_nav"]),
            participation_cap=float(execution["participation_cap"]),
            rebalance_stride=int(execution["rebalance_stride"]),
            min_daily_count=int(execution["min_daily_count"]),
        )
        rows.append(flatten_t1_summary(periods, {"cost_bps": float(cost)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_capacity_sensitivity(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    execution = config["execution"]
    rows: list[pd.DataFrame] = []
    for nav in config["sensitivity"]["portfolio_nav_values"]:
        for cap in config["sensitivity"]["participation_cap_values"]:
            periods = run_backtest(
                frame=data,
                k_values=[int(execution["k"])],
                keep_multipliers=[float(execution["keep_multiplier"])],
                cost_bps=float(execution["cost_bps"]),
                slippage_bps=float(execution["slippage_bps"]),
                portfolio_nav=float(nav),
                participation_cap=float(cap),
                rebalance_stride=int(execution["rebalance_stride"]),
                min_daily_count=int(execution["min_daily_count"]),
            )
            rows.append(flatten_t1_summary(periods, {"portfolio_nav": float(nav), "participation_cap": float(cap)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def style_and_industry_audit(
    selected: pd.DataFrame,
    predictions: pd.DataFrame,
    mart: pd.DataFrame,
    execution: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_detail = selected.merge(mart, on=["trade_date", "ts_code"], how="left")
    selected_detail = selected_detail.merge(execution, on=["trade_date", "ts_code"], how="left")
    universe = predictions[["trade_date", "ts_code", "split"]].merge(mart, on=["trade_date", "ts_code"], how="left")

    style_rows: list[dict[str, Any]] = []
    features = [column for column in STYLE_COLUMNS + LIQUIDITY_COLUMNS + TRADABILITY_COLUMNS if column in mart.columns]
    for split, selected_split in selected_detail.groupby("split", sort=True):
        dates = selected_split["trade_date"].unique()
        universe_split = universe[(universe["split"] == split) & (universe["trade_date"].isin(dates))]
        for feature in features:
            daily_rows: list[dict[str, float]] = []
            for date, selected_day in selected_split.groupby("trade_date", sort=True):
                universe_day = universe_split[universe_split["trade_date"] == date]
                if universe_day.empty:
                    continue
                daily_rows.append(
                    {
                        "selected_mean": pd.to_numeric(selected_day[feature], errors="coerce").mean(),
                        "universe_mean": pd.to_numeric(universe_day[feature], errors="coerce").mean(),
                    }
                )
            daily = pd.DataFrame(daily_rows)
            if daily.empty:
                continue
            diff = daily["selected_mean"] - daily["universe_mean"]
            style_rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "selected_mean": float(daily["selected_mean"].mean()),
                    "universe_mean": float(daily["universe_mean"].mean()),
                    "selected_minus_universe": float(diff.mean()),
                    "diff_ir": safe_ir(diff),
                    "abs_selected_minus_universe": float(abs(diff.mean())),
                }
            )
    style = pd.DataFrame(style_rows)

    industry_rows: list[dict[str, Any]] = []
    for (split, date), selected_day in selected_detail.groupby(["split", "trade_date"], sort=True):
        universe_day = universe[(universe["split"] == split) & (universe["trade_date"] == date)]
        selected_dist = selected_day["industry"].astype(str).value_counts(normalize=True)
        universe_dist = universe_day["industry"].astype(str).value_counts(normalize=True)
        for industry in sorted(set(selected_dist.index) | set(universe_dist.index)):
            industry_rows.append(
                {
                    "split": split,
                    "trade_date": date,
                    "industry": industry,
                    "selected_weight": float(selected_dist.get(industry, 0.0)),
                    "universe_weight": float(universe_dist.get(industry, 0.0)),
                    "active_weight": float(selected_dist.get(industry, 0.0) - universe_dist.get(industry, 0.0)),
                }
            )
    industry_daily = pd.DataFrame(industry_rows)
    industry_summary = (
        industry_daily.groupby(["split", "industry"], sort=True)
        .agg(
            selected_weight=("selected_weight", "mean"),
            universe_weight=("universe_weight", "mean"),
            active_weight=("active_weight", "mean"),
            max_abs_daily_active=("active_weight", lambda values: float(pd.Series(values).abs().max())),
        )
        .reset_index()
    )

    execution_rows: list[dict[str, Any]] = []
    for split, group in selected_detail.groupby("split", sort=True):
        execution_rows.append(
            {
                "split": split,
                "selected_rows": int(len(group)),
                "buy_executable_rate": float(group["buy_executable_t1_open"].mean()),
                "sell_executable_rate": float(group["sell_executable_t1_open"].mean()),
                "limit_up_rate": float(group.get("next_is_limit_up", pd.Series(False, index=group.index)).mean()),
                "limit_down_rate": float(group.get("next_is_limit_down", pd.Series(False, index=group.index)).mean()),
                "suspended_rate": float(group.get("next_is_suspended", pd.Series(False, index=group.index)).mean()),
                "avg_next_amount": float(pd.to_numeric(group["next_amount"], errors="coerce").mean()),
                "avg_exec_return": float(pd.to_numeric(group["execution_return_open_to_close5"], errors="coerce").mean()),
            }
        )
    execution_attr = pd.DataFrame(execution_rows)
    return selected_detail, style, industry_summary, execution_attr


def overlap_audit(
    selected: pd.DataFrame,
    comparison_predictions: dict[str, Path],
    periods: pd.DataFrame,
    k: int,
    keep: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_sets = {
        (split, date): set(group["ts_code"].astype(str))
        for (split, date), group in selected.groupby(["split", "trade_date"], sort=True)
    }
    for name, path in comparison_predictions.items():
        if not path.exists():
            continue
        preds = pd.read_parquet(path)
        preds["trade_date"] = preds["trade_date"].astype(str)
        preds["ts_code"] = preds["ts_code"].astype(str)
        preds["split"] = preds["split"].astype(str)
        other_selected = selected_codes_for_periods(preds, periods, k, keep)
        for (split, date), group in other_selected.groupby(["split", "trade_date"], sort=True):
            base = selected_sets.get((split, date), set())
            other = set(group["ts_code"].astype(str))
            if not base or not other:
                continue
            rows.append(
                {
                    "comparison": name,
                    "split": split,
                    "trade_date": date,
                    "overlap_count": len(base & other),
                    "overlap_rate": len(base & other) / k,
                    "jaccard": len(base & other) / len(base | other),
                }
            )
    overlap = pd.DataFrame(rows)
    if overlap.empty:
        return overlap
    return (
        overlap.groupby(["comparison", "split"], sort=True)
        .agg(
            days=("trade_date", "count"),
            mean_overlap_count=("overlap_count", "mean"),
            mean_overlap_rate=("overlap_rate", "mean"),
            median_overlap_rate=("overlap_rate", "median"),
            mean_jaccard=("jaccard", "mean"),
        )
        .reset_index()
    )


def write_markdown(
    out_dir: Path,
    mainline_summary: pd.DataFrame,
    yearly: pd.DataFrame,
    k_keep: pd.DataFrame,
    cost: pd.DataFrame,
    capacity: pd.DataFrame,
    style: pd.DataFrame,
    industry: pd.DataFrame,
    overlap: pd.DataFrame,
) -> None:
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    def table(frame: pd.DataFrame, max_rows: int = 12) -> str:
        if frame.empty:
            return "_No data._"
        display = frame.head(max_rows).copy()
        header = "| " + " | ".join(display.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
        body = ["| " + " | ".join(fmt(row[col]) for col in display.columns) + " |" for _, row in display.iterrows()]
        return "\n".join([header, sep, *body])

    validation = mainline_summary[mainline_summary["split"].eq("validation")]
    test = mainline_summary[mainline_summary["split"].eq("test")]
    lines = [
        "# Final Mainline T+1 Top10 Keep1x Compatibility Audit",
        "",
        "Scope: frozen L60 feature-style interaction GRU epoch-12 route only. Full62 is used only as an optional overlap comparator.",
        "",
        "## Mainline Summary",
        "",
        table(pd.concat([validation, test], ignore_index=True)),
        "",
        "## Yearly Stability",
        "",
        table(yearly),
        "",
        "## K/Keep Neighborhood",
        "",
        table(k_keep.sort_values(["split", "excess_exec_universe_ann"], ascending=[True, False])),
        "",
        "## Cost Sensitivity",
        "",
        table(cost.sort_values(["split", "cost_bps"])),
        "",
        "## Capacity Sensitivity",
        "",
        table(capacity.sort_values(["split", "portfolio_nav", "participation_cap"])),
        "",
        "## Largest Style/Liquidity Exposures",
        "",
        table(style.sort_values(["split", "abs_selected_minus_universe"], ascending=[True, False])),
        "",
        "## Largest Industry Active Weights",
        "",
        table(industry.assign(abs_active=industry["active_weight"].abs()).sort_values(["split", "abs_active"], ascending=[True, False]).drop(columns=["abs_active"])),
        "",
        "## Overlap",
        "",
        table(overlap),
        "",
        "## PM Readout",
        "",
        "- This route is suitable as the clean-feature research mainline candidate.",
        "- It should not be described as production-ready until stability and capacity weaknesses are resolved.",
        "- The most defensible claim is executable-universe excess under strict T+1 constraints, not standalone absolute return.",
    ]
    (out_dir / "clean_resid_mainline_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    inputs = config["inputs"]
    execution = config["execution"]
    out_dir = resolve_path(config["outputs"]["audit_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data, periods = run_mainline(config)
    predictions = pd.read_parquet(resolve_path(inputs["predictions"]))
    predictions["trade_date"] = predictions["trade_date"].astype(str)
    predictions["ts_code"] = predictions["ts_code"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    mart = read_mart(resolve_path(inputs["mart"]))
    execution_labels = read_execution(resolve_path(inputs["execution_labels"]))

    mainline_summary = flatten_t1_summary(periods)
    yearly, monthly, quarterly = build_time_summaries(periods)
    selected = selected_codes_for_periods(
        predictions,
        periods,
        int(execution["k"]),
        float(execution["keep_multiplier"]),
    )
    selected_detail, style, industry, execution_attr = style_and_industry_audit(
        selected,
        predictions,
        mart,
        execution_labels,
    )
    k_keep = run_k_keep_sensitivity(data, config)
    cost = run_cost_sensitivity(data, config)
    capacity = pd.DataFrame() if args.skip_heavy_sensitivity else run_capacity_sensitivity(data, config)
    comparison_predictions = {
        "clean_alpha_only": resolve_path(inputs["clean_alpha_only_predictions"]),
    }
    overlap = overlap_audit(selected, comparison_predictions, periods, int(execution["k"]), float(execution["keep_multiplier"]))

    periods.to_csv(out_dir / "mainline_periods.csv", index=False)
    mainline_summary.to_csv(out_dir / "mainline_summary.csv", index=False)
    yearly.to_csv(out_dir / "yearly_metrics.csv", index=False)
    monthly.to_csv(out_dir / "monthly_metrics.csv", index=False)
    quarterly.to_csv(out_dir / "quarterly_metrics.csv", index=False)
    k_keep.to_csv(out_dir / "k_keep_sensitivity.csv", index=False)
    cost.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    capacity.to_csv(out_dir / "capacity_participation_sensitivity.csv", index=False)
    selected_detail.to_parquet(out_dir / "selected_snapshots.parquet", index=False)
    style.to_csv(out_dir / "style_liquidity_vs_universe.csv", index=False)
    industry.to_csv(out_dir / "industry_exposure_selected.csv", index=False)
    execution_attr.to_csv(out_dir / "execution_attribution.csv", index=False)
    overlap.to_csv(out_dir / "topk_overlap.csv", index=False)
    worst_months = monthly.sort_values("net_ann").head(12)
    worst_months.to_csv(out_dir / "worst_months.csv", index=False)
    write_markdown(out_dir, mainline_summary, yearly, k_keep, cost, capacity, style, industry, overlap)

    manifest = {
        "mainline": config["mainline"],
        "config": str(config_path),
        "out_dir": str(out_dir),
        "rows": {
            "periods": int(len(periods)),
            "selected_snapshots": int(len(selected_detail)),
            "yearly": int(len(yearly)),
            "monthly": int(len(monthly)),
            "k_keep": int(len(k_keep)),
            "cost": int(len(cost)),
            "capacity": int(len(capacity)),
            "style": int(len(style)),
            "industry": int(len(industry)),
            "overlap": int(len(overlap)),
        },
        "method": "isolated_clean_resid_mainline_audit",
    }
    (out_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
