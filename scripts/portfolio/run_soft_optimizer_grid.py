from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.portfolio.optimize_feasible_cash_buffer import (
    load_data,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    run_optimizer,
    summarize,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run soft exposure slack optimizer grid.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--mart", default="data/mart/datasets/core/dataset_v20260526.parquet")
    parser.add_argument("--labels", default="data/mart/labels/execution_labels_v20260526.parquet")
    parser.add_argument("--output-dir", default="outputs/backtest/soft_optimizer_grid")
    parser.add_argument("--risk-control", type=parse_str_list, required=True)
    parser.add_argument("--k", type=parse_int_list, required=True)
    parser.add_argument("--style-penalty", type=parse_float_list, required=True)
    parser.add_argument("--turnover-penalty", type=parse_float_list, required=True)
    parser.add_argument("--exposure-cap", type=parse_float_list, required=True)
    parser.add_argument("--min-invested", type=parse_float_list, required=True)
    parser.add_argument("--turnover-cap", type=parse_float_list, required=True)
    parser.add_argument("--participation-cap", type=parse_float_list, required=True)
    parser.add_argument("--single-name-cap", type=parse_float_list, required=True)
    parser.add_argument("--exposure-slack-penalty", type=parse_float_list, required=True)
    parser.add_argument("--buy-capacity-slack-penalty", type=parse_float_list, default=parse_float_list("1000"))
    parser.add_argument("--cash-penalty", type=parse_float_list, required=True)
    parser.add_argument("--min-invested-shortfall-penalty", type=parse_float_list, default=parse_float_list("0"))
    parser.add_argument("--candidate-multiplier", type=float, default=5.0)
    parser.add_argument("--portfolio-nav", type=float, default=10_000_000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--rebalance-stride", type=int, default=5)
    parser.add_argument("--min-daily-count", type=int, default=40)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-grid-runs", type=int, default=0, help="Optional smoke-test cap on outer grid paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)

    all_periods: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    grid = list(
        itertools.product(
            args.risk_control,
            args.k,
            args.style_penalty,
            args.turnover_penalty,
            args.exposure_cap,
            args.min_invested,
            args.turnover_cap,
            args.participation_cap,
            args.single_name_cap,
            args.exposure_slack_penalty,
            args.buy_capacity_slack_penalty,
            args.cash_penalty,
            args.min_invested_shortfall_penalty,
        )
    )
    if args.max_grid_runs > 0:
        grid = grid[: args.max_grid_runs]

    for run_id, (
        risk_control,
        k,
        style_penalty,
        turnover_penalty,
        exposure_cap,
        min_invested,
        turnover_cap,
        participation_cap,
        single_name_cap,
        exposure_slack_penalty,
        buy_capacity_slack_penalty,
        cash_penalty,
        min_invested_shortfall_penalty,
    ) in enumerate(grid, start=1):
        periods_path = output_dir / f"periods_grid_{run_id:04d}.csv"
        summary_path = output_dir / f"summary_grid_{run_id:04d}.csv"
        if periods_path.exists() and summary_path.exists():
            periods = pd.read_csv(periods_path)
            summary = pd.read_csv(summary_path)
            all_periods.append(periods)
            all_summary.append(summary)
            continue

        run_args = SimpleNamespace(
            predictions=args.predictions,
            mart=args.mart,
            labels=args.labels,
            output_dir=str(output_dir),
            risk_control=[risk_control],
            k=[int(k)],
            style_penalty=[float(style_penalty)],
            turnover_penalty=[float(turnover_penalty)],
            candidate_multiplier=args.candidate_multiplier,
            exposure_cap=float(exposure_cap),
            exposure_slack_penalty=float(exposure_slack_penalty),
            buy_capacity_slack_penalty=float(buy_capacity_slack_penalty),
            cash_penalty=float(cash_penalty),
            min_invested_shortfall_penalty=float(min_invested_shortfall_penalty),
            single_name_cap=float(single_name_cap),
            min_invested=float(min_invested),
            turnover_cap=float(turnover_cap),
            portfolio_nav=float(args.portfolio_nav),
            participation_cap=float(participation_cap),
            cost_bps=float(args.cost_bps),
            slippage_bps=float(args.slippage_bps),
            rebalance_stride=int(args.rebalance_stride),
            min_daily_count=int(args.min_daily_count),
            solver=args.solver,
        )
        periods = run_optimizer(run_args, data)
        if periods.empty:
            continue
        periods["grid_run_id"] = run_id
        summary = summarize(periods)
        summary["grid_run_id"] = run_id
        all_periods.append(periods)
        all_summary.append(summary)

        periods.to_csv(periods_path, index=False)
        summary.to_csv(summary_path, index=False)

    periods_all = pd.concat(all_periods, ignore_index=True) if all_periods else pd.DataFrame()
    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    periods_all.to_csv(output_dir / "soft_optimizer_grid_periods.csv", index=False)
    summary_all.to_csv(output_dir / "soft_optimizer_grid_summary.csv", index=False)
    manifest = {
        "predictions": args.predictions,
        "mart": args.mart,
        "labels": args.labels,
        "output_dir": str(output_dir),
        "risk_control": args.risk_control,
        "k": args.k,
        "style_penalty": args.style_penalty,
        "turnover_penalty": args.turnover_penalty,
        "exposure_cap": args.exposure_cap,
        "min_invested": args.min_invested,
        "turnover_cap": args.turnover_cap,
        "participation_cap": args.participation_cap,
        "single_name_cap": args.single_name_cap,
        "exposure_slack_penalty": args.exposure_slack_penalty,
        "buy_capacity_slack_penalty": args.buy_capacity_slack_penalty,
        "cash_penalty": args.cash_penalty,
        "min_invested_shortfall_penalty": args.min_invested_shortfall_penalty,
        "candidate_multiplier": float(args.candidate_multiplier),
        "portfolio_nav": float(args.portfolio_nav),
        "cost_bps": float(args.cost_bps),
        "slippage_bps": float(args.slippage_bps),
        "rebalance_stride": int(args.rebalance_stride),
        "min_daily_count": int(args.min_daily_count),
        "solver": args.solver,
        "grid_runs": int(len(grid)),
        "period_rows": int(len(periods_all)),
        "summary_rows": int(len(summary_all)),
        "method": "soft_exposure_slack_optimizer_grid",
    }
    (output_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
