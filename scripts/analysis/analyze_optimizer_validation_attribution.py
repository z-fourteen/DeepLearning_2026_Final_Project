from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze optimizer validation weak periods and exposure diagnostics.")
    parser.add_argument("--periods", required=True, help="Path to soft_optimizer_grid_periods.csv.")
    parser.add_argument("--summary", required=True, help="Path to soft_optimizer_grid_summary.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-n", type=int, default=6)
    return parser.parse_args()


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def setting_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ["risk_control", "k", "style_penalty", "turnover_penalty"] if column in frame.columns]


def main() -> None:
    args = parse_args()
    periods = pd.read_csv(args.periods)
    summary = pd.read_csv(args.summary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "net_return",
        "excess_vs_benchmark",
        "excess_vs_executable_universe",
        "cash_weight",
        "invested_weight",
        "filled_turnover",
        "buy_capacity_slack",
        "min_invested_shortfall",
        "max_abs_exposure_z",
        "avg_abs_exposure_z",
        "max_exposure_slack",
        "avg_exposure_slack",
        "total_exposure_slack",
        "position_count",
        "min_invested_rule_ok",
    ]
    periods = numeric(periods, metric_cols)
    summary = numeric(
        summary,
        [
            "excess_exec_universe_ann",
            "net_ann",
            "excess_benchmark_ann",
            "net_max_drawdown",
            "avg_cash_weight",
            "avg_buy_capacity_slack",
            "min_invested_rule_pass_rate",
        ],
    )
    periods = periods[periods["split"].astype(str).eq(args.split)].copy()
    if periods.empty:
        raise ValueError(f"No rows found for split={args.split!r}")

    periods["trade_date"] = periods["trade_date"].astype(str)
    periods["month"] = periods["trade_date"].str.slice(0, 6)
    keys = setting_columns(periods)

    best_summary = (
        summary[summary["split"].astype(str).eq(args.split)]
        .sort_values("excess_exec_universe_ann", ascending=False)
        .head(args.top_n)
    )
    best_summary.to_csv(out_dir / "best_settings.csv", index=False)

    if keys:
        best_keys = best_summary[keys].drop_duplicates()
        selected = periods.merge(best_keys, on=keys, how="inner")
    else:
        selected = periods

    month_cols = [
        "net_return",
        "excess_vs_benchmark",
        "excess_vs_executable_universe",
        "cash_weight",
        "invested_weight",
        "filled_turnover",
        "buy_capacity_slack",
        "min_invested_shortfall",
        "max_abs_exposure_z",
        "avg_abs_exposure_z",
        "max_exposure_slack",
        "avg_exposure_slack",
        "total_exposure_slack",
        "position_count",
        "min_invested_rule_ok",
    ]
    monthly = (
        selected.groupby(["month", *keys], dropna=False)[month_cols]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    monthly.columns = ["_".join([str(part) for part in col if part != ""]) for col in monthly.columns]
    monthly = monthly.sort_values("excess_vs_executable_universe_mean")
    monthly.to_csv(out_dir / "monthly_setting_attribution.csv", index=False)

    month_overall = (
        selected.groupby("month", dropna=False)[month_cols]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    month_overall.columns = ["_".join([str(part) for part in col if part != ""]) for col in month_overall.columns]
    month_overall = month_overall.sort_values("excess_vs_executable_universe_mean")
    month_overall.to_csv(out_dir / "monthly_overall_attribution.csv", index=False)

    exposure_cols = [
        "max_abs_exposure_z",
        "avg_abs_exposure_z",
        "max_exposure_slack",
        "avg_exposure_slack",
        "total_exposure_slack",
        "cash_weight",
        "buy_capacity_slack",
        "filled_turnover",
        "position_count",
    ]
    exposure_corr = selected[["excess_vs_executable_universe", *exposure_cols]].corr(numeric_only=True)
    exposure_corr.to_csv(out_dir / "exposure_correlation.csv")

    manifest = {
        "periods": args.periods,
        "summary": args.summary,
        "output_dir": str(out_dir),
        "split": args.split,
        "top_n": args.top_n,
        "rows": int(len(selected)),
        "best_settings": str(out_dir / "best_settings.csv"),
        "monthly_setting_attribution": str(out_dir / "monthly_setting_attribution.csv"),
        "monthly_overall_attribution": str(out_dir / "monthly_overall_attribution.csv"),
        "exposure_correlation": str(out_dir / "exposure_correlation.csv"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
