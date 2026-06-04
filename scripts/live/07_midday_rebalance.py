from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import resolve_path, today_yyyymmdd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run midday rebalance planning from a manual 11:30 price snapshot."
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument("--feature-date", required=True)
    parser.add_argument("--midday-prices", required=True)
    parser.add_argument("--tag", default="midday")
    parser.add_argument("--predictions")
    parser.add_argument("--liquidity-parquet")
    parser.add_argument("--base-price-snapshot")
    parser.add_argument("--portfolio-nav", type=float)
    parser.add_argument("--overwrite-inputs", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="After planning, launch Stage 5 in same-day append mode.",
    )
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args()


def run_step(title: str, command: list[str]) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(" ".join(command))
    subprocess.run(command, cwd=resolve_path("."), check=True)


def main() -> None:
    args = parse_args()
    trade_date = str(args.trade_date)
    tag = str(args.tag)

    positions = Path("data") / "live" / "account" / f"positions_{trade_date}_{tag}.csv"
    quotes = Path("data") / "live" / "market" / f"quotes_{trade_date}_{tag}.csv"
    targets = Path("outputs") / "live_targets" / f"target_weights_{trade_date}_{tag}.csv"
    orders = Path("outputs") / "live_orders" / f"orders_{trade_date}_{tag}.csv"

    prepare_cmd = [
        sys.executable,
        "scripts/live/00_prepare_midday_rebalance_inputs.py",
        "--config",
        args.config,
        "--trade-date",
        trade_date,
        "--midday-prices",
        args.midday_prices,
        "--tag",
        tag,
    ]
    if args.base_price_snapshot:
        prepare_cmd.extend(["--base-price-snapshot", args.base_price_snapshot])
    if args.overwrite_inputs:
        prepare_cmd.append("--overwrite")
    run_step("Stage 0.m: prepare midday account and quote snapshots", prepare_cmd)

    opt_cmd = [
        sys.executable,
        "scripts/live/02_live_optimization.py",
        "--config",
        args.config,
        "--trade-date",
        trade_date,
        "--feature-date",
        args.feature_date,
        "--positions",
        str(positions),
        "--skip-position-inheritance-check",
        "--output-tag",
        tag,
    ]
    if args.predictions:
        opt_cmd.extend(["--predictions", args.predictions])
    if args.liquidity_parquet:
        opt_cmd.extend(["--liquidity-parquet", args.liquidity_parquet])
    if args.portfolio_nav is not None:
        opt_cmd.extend(["--portfolio-nav", str(args.portfolio_nav)])
    run_step("Stage 2.m: optimize midday target weights", opt_cmd)

    order_cmd = [
        sys.executable,
        "scripts/live/03_generate_target_orders.py",
        "--config",
        args.config,
        "--trade-date",
        trade_date,
        "--target-weights",
        str(targets),
        "--positions",
        str(positions),
        "--price-snapshot",
        str(quotes),
        "--output-tag",
        tag,
    ]
    if args.portfolio_nav is not None:
        order_cmd.extend(["--portfolio-nav", str(args.portfolio_nav)])
    run_step("Stage 3.m: generate midday rebalance orders", order_cmd)

    print(f"\nMidday orders ready: {resolve_path(orders)}")
    if not args.execute:
        print("Review the orders first. Rerun with --execute to launch Stage 5 append mode.")
        return

    exec_cmd = [
        sys.executable,
        "scripts/live/05_interactive_execution.py",
        "--config",
        args.config,
        "--trade-date",
        trade_date,
        "--orders-csv",
        str(orders),
        "--price-snapshot",
        str(quotes),
        "--same-day-mode",
        "append",
        "--execution-tag",
        tag,
    ]
    if args.no_push:
        exec_cmd.append("--no-push")
    run_step("Stage 5.m: append midday execution", exec_cmd)


if __name__ == "__main__":
    main()
