from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_t1_fill_sim import build_summary, load_data, run_backtest  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return sorted(set(values))


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive floats.")
    return sorted(set(values))


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
    parser = argparse.ArgumentParser(
        description="Run NAV and participation sensitivity for T+1 fill simulation candidates."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet",
    )
    parser.add_argument(
        "--labels",
        default="data/mart/labels/labels_canonical_v20260526.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/backtest/t1_fill_sim/capacity_participation_matrix")
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("10,30"))
    parser.add_argument("--keep-multiplier", type=parse_float_list, default=parse_float_list("1,1.5,3"))
    parser.add_argument("--portfolio-nav", type=parse_float_list, default=parse_float_list("1000000,10000000,50000000,100000000"))
    parser.add_argument("--participation-cap", type=parse_float_list, default=parse_float_list("0.01,0.03,0.05,0.10"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--rebalance-stride", type=int, default=5)
    parser.add_argument("--min-daily-count", type=int, default=20)
    return parser.parse_args()


def flatten_summary(periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if periods.empty:
        return pd.DataFrame()
    summary = build_summary(periods)
    for split, split_summary in summary.items():
        for key, metrics in split_summary.items():
            parts = key.replace("top_", "").replace("x", "").split("_keep_")
            k = int(parts[0])
            keep = float(parts[1])
            rows.append(
                {
                    "split": split,
                    "k": k,
                    "keep_multiplier": keep,
                    "periods": metrics["net"]["period_count"],
                    "net_ann": metrics["net"]["annualized_return"],
                    "net_ir": metrics["net"]["ir"],
                    "net_max_drawdown": metrics["net"]["max_drawdown"],
                    "excess_benchmark_ann": metrics["excess_vs_benchmark"]["annualized_return"],
                    "excess_exec_universe_ann": metrics["excess_vs_executable_universe"]["annualized_return"],
                    "avg_desired_turnover": metrics["average_desired_turnover"],
                    "avg_filled_turnover": metrics["average_filled_turnover"],
                    "avg_transaction_cost": metrics["average_transaction_cost"],
                    "avg_buy_reject_count": metrics["average_buy_reject_count"],
                    "avg_sell_reject_count": metrics["average_sell_reject_count"],
                    "avg_partial_fill_count": metrics["average_partial_fill_count"],
                    "avg_position_count": metrics["average_position_count"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(Path(args.predictions), Path(args.labels))

    all_periods: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    for nav in args.portfolio_nav:
        for cap in args.participation_cap:
            periods = run_backtest(
                frame=data,
                k_values=args.k,
                keep_multipliers=args.keep_multiplier,
                cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                portfolio_nav=nav,
                participation_cap=cap,
                rebalance_stride=args.rebalance_stride,
                min_daily_count=args.min_daily_count,
            )
            periods["matrix_portfolio_nav"] = float(nav)
            periods["matrix_participation_cap"] = float(cap)
            all_periods.append(periods)
            flat = flatten_summary(periods)
            flat["portfolio_nav"] = float(nav)
            flat["participation_cap"] = float(cap)
            all_summary.append(flat)

    periods_all = pd.concat(all_periods, ignore_index=True) if all_periods else pd.DataFrame()
    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    periods_all.to_csv(output_dir / "capacity_participation_periods.csv", index=False)
    summary_all.to_csv(output_dir / "capacity_participation_summary.csv", index=False)

    manifest = {
        "predictions": args.predictions,
        "labels": args.labels,
        "output_dir": str(output_dir),
        "k": args.k,
        "keep_multiplier": args.keep_multiplier,
        "portfolio_nav": args.portfolio_nav,
        "participation_cap": args.participation_cap,
        "cost_bps": float(args.cost_bps),
        "slippage_bps": float(args.slippage_bps),
        "rebalance_stride": int(args.rebalance_stride),
        "period_rows": int(len(periods_all)),
        "summary_rows": int(len(summary_all)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
