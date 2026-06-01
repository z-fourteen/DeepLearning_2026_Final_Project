"""Live stage 5: interactive execution, portfolio state, and Git handoff.

This stage ports the OXX2 daily script's manual fill workflow to the current
GRU live pipeline. It consumes the stage-3 order CSV, asks for actual fills,
updates portfolio_state.json, writes an execution log, and can commit/push the
live state files.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    PROJECT_ROOT,
    die,
    format_path,
    git_commit_and_push,
    load_yaml,
    normalize_code_column,
    previous_trading_day,
    price_column,
    resolve_path,
    today_yyyymmdd,
    write_json,
)

LOT_SIZE = 100
DEFAULT_NAV = 1_000_000.0


@dataclass
class Order:
    ts_code: str
    action: str
    target_shares: int
    target_weight: float
    close_price: float | None = None
    target_value: float = 0.0
    reason: str = ""


@dataclass
class ExecutionResult:
    ts_code: str
    action: str
    target_shares: int
    actual_shares: int
    actual_price: float
    actual_value: float
    status: str
    reason: str = ""


@dataclass
class Holding:
    shares: int = 0
    avg_cost: float = 0.0
    weight_at_entry: float = 0.0

    def market_value(self, price: float) -> float:
        return self.shares * price


@dataclass
class PortfolioState:
    last_signal_date: str = ""
    day_index: int = 0
    cash: float = DEFAULT_NAV
    initial_nav: float = DEFAULT_NAV
    holdings: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_orders: list[dict[str, Any]] = field(default_factory=list)

    def to_holdings(self) -> dict[str, Holding]:
        return {code: Holding(**vals) for code, vals in self.holdings.items()}

    def set_holdings(self, holdings: dict[str, Holding]) -> None:
        self.holdings = {code: asdict(holding) for code, holding in holdings.items()}

    def total_position_value(self, prices: dict[str, float]) -> float:
        return sum(
            float(item["shares"]) * prices.get(code, float(item.get("avg_cost", 0.0)))
            for code, item in self.holdings.items()
        )

    def total_nav(self, prices: dict[str, float]) -> float:
        return self.cash + self.total_position_value(prices)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self.last_signal_date = str(data.get("last_signal_date", ""))
        self.day_index = int(data.get("day_index", 0))
        self.cash = float(data.get("cash", DEFAULT_NAV))
        self.initial_nav = float(data.get("initial_nav", DEFAULT_NAV))
        self.holdings = data.get("holdings", {})
        self.pending_orders = data.get("pending_orders", [])
        return True

    def save(self, path: Path) -> None:
        write_json(
            path,
            {
                "last_signal_date": self.last_signal_date,
                "day_index": self.day_index,
                "cash": self.cash,
                "initial_nav": self.initial_nav,
                "holdings": self.holdings,
                "pending_orders": self.pending_orders,
                "updated_at": datetime.now().isoformat(),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live stage 5: interactively confirm fills and persist portfolio state."
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument("--feature-date", help="Ignored; accepted for compatibility with live_daily.py.")
    parser.add_argument("--orders-csv", help="Stage-3 orders CSV. Defaults to outputs.orders_dir/orders_DATE.csv.")
    parser.add_argument("--portfolio-state", help="Portfolio state JSON. Defaults to outputs/live/portfolio_state.json.")
    parser.add_argument("--execution-dir", help="Execution log directory. Defaults to outputs/live/orders.")
    parser.add_argument("--price-snapshot", help="Reference price snapshot. Defaults to live_inputs.price_snapshot.")
    parser.add_argument("--initial-nav", type=float, default=DEFAULT_NAV)
    parser.add_argument("--reset", action="store_true", help="Delete existing portfolio state before execution.")
    parser.add_argument("--no-push", action="store_true", help="Skip git commit/push after execution.")
    parser.add_argument("--push-branch", help="Branch to push. Defaults to current branch.")
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def ask_input(prompt: str, default: str = "") -> str:
    suffix = f" [default: {default}]" if default else ""
    try:
        value = input(f"    {prompt}{suffix}: ").strip()
        return value if value else default
    except (EOFError, KeyboardInterrupt):
        return default


def round_lot(shares: int | float) -> int:
    return max(0, int(shares) // LOT_SIZE * LOT_SIZE)


def load_reference_prices(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    frame = normalize_code_column(pd.read_csv(path))
    px_col = price_column(frame)
    frame["price"] = pd.to_numeric(frame[px_col], errors="coerce")
    return {
        str(row.ts_code): float(row.price)
        for row in frame[frame["price"].gt(0)].itertuples(index=False)
    }


def orders_csv_to_order_list(orders_df: pd.DataFrame) -> list[Order]:
    orders_df = normalize_code_column(orders_df.rename(columns={"code": "ts_code"}))
    orders: list[Order] = []
    for row in orders_df.itertuples(index=False):
        price = float(getattr(row, "price_ref", 0.0) or 0.0)
        delta_weight = float(getattr(row, "delta_weight", 0.0) or 0.0)
        target_shares = int(getattr(row, "target_volume", 0) or 0)
        if target_shares <= 0:
            continue
        action = str(getattr(row, "action", "")).upper()
        if action not in {"BUY", "SELL"}:
            die(f"unsupported order action={action!r}")
        orders.append(
            Order(
                ts_code=str(getattr(row, "ts_code")),
                action=action,
                target_shares=target_shares,
                target_weight=abs(delta_weight),
                close_price=price if price > 0 else None,
                target_value=float(getattr(row, "target_value", target_shares * price) or 0.0),
                reason=f"delta_weight={delta_weight:+.4f}",
            )
        )
    return orders


def execute_one_order(order: Order, state: PortfolioState, holdings: dict[str, Holding]) -> ExecutionResult:
    price_text = f"{order.close_price:.2f}" if order.close_price else ""
    print(f"\n  {order.action} {order.ts_code}")
    print(f"    target: {order.target_shares} shares @ {price_text or 'N/A'} ~= {order.target_value:,.2f}")
    print(f"    reason: {order.reason}")
    print("    choices: y=filled as planned, n=skip, p=partial shares, c=custom price and shares")
    choice = ask_input("execute", "y").lower()

    if choice == "n":
        print("    -> skipped")
        return ExecutionResult(order.ts_code, order.action, order.target_shares, 0, 0.0, 0.0, "skipped", "manual skip")

    price_input = ask_input("actual fill price", price_text)
    try:
        actual_price = float(price_input)
    except ValueError:
        actual_price = float(order.close_price or 0.0)
        print(f"    invalid price, using reference {actual_price:.2f}")
    if actual_price <= 0:
        return ExecutionResult(order.ts_code, order.action, order.target_shares, 0, 0.0, 0.0, "failed", "invalid price")

    if choice in {"p", "c"}:
        shares_input = ask_input("actual shares", str(order.target_shares))
        try:
            actual_shares = round_lot(int(shares_input))
        except ValueError:
            actual_shares = order.target_shares
            print(f"    invalid shares, using target {actual_shares}")
    else:
        actual_shares = order.target_shares
    actual_shares = round_lot(actual_shares)
    actual_value = actual_shares * actual_price

    if order.action == "SELL":
        current = holdings.get(order.ts_code, Holding())
        actual_shares = min(actual_shares, current.shares)
        actual_value = actual_shares * actual_price
        if actual_shares <= 0:
            return ExecutionResult(order.ts_code, "SELL", order.target_shares, 0, actual_price, 0.0, "failed", "no shares")
        current.shares -= actual_shares
        if current.shares <= 0:
            holdings.pop(order.ts_code, None)
        else:
            holdings[order.ts_code] = current
        state.cash += actual_value
    else:
        if actual_value > state.cash:
            actual_shares = round_lot(state.cash / actual_price)
            actual_value = actual_shares * actual_price
            if actual_shares <= 0:
                return ExecutionResult(order.ts_code, "BUY", order.target_shares, 0, actual_price, 0.0, "failed", "insufficient cash")
            print(f"    cash limited, adjusted to {actual_shares} shares")
        state.cash -= actual_value
        current = holdings.get(order.ts_code, Holding())
        total_cost = current.avg_cost * current.shares + actual_value
        current.shares += actual_shares
        current.avg_cost = total_cost / current.shares if current.shares > 0 else 0.0
        current.weight_at_entry = order.target_weight
        holdings[order.ts_code] = current

    state.set_holdings(holdings)
    status = "filled" if actual_shares >= order.target_shares else "partial"
    print(f"    -> {status}: {actual_shares} shares x {actual_price:.2f} = {actual_value:,.2f}")
    print(f"    cash: {state.cash:,.2f}")
    return ExecutionResult(order.ts_code, order.action, order.target_shares, actual_shares, actual_price, actual_value, status)


def interactive_execute_orders(orders: list[Order], state: PortfolioState) -> list[ExecutionResult]:
    print_header("Stage 5: Interactive Execution")
    if not orders:
        print("  No executable orders. Portfolio state will still be timestamped.")
        return []
    sell_orders = [order for order in orders if order.action == "SELL"]
    buy_orders = [order for order in orders if order.action == "BUY"]
    print(f"  Orders: {len(sell_orders)} sells + {len(buy_orders)} buys")
    print(f"  Starting cash: {state.cash:,.2f}")

    holdings = state.to_holdings()
    results: list[ExecutionResult] = []
    for order in [*sell_orders, *buy_orders]:
        results.append(execute_one_order(order, state, holdings))
    return results


def print_portfolio_summary(state: PortfolioState, prices: dict[str, float]) -> None:
    print_header("Portfolio Summary")
    holdings = state.to_holdings()
    nav = state.total_nav(prices)
    position_value = state.total_position_value(prices)
    position_ratio = position_value / nav if nav > 0 else 0.0
    total_return = (nav - state.initial_nav) / state.initial_nav if state.initial_nav > 0 else 0.0
    print(f"  NAV:             {nav:>14,.2f}")
    print(f"  Cash:            {state.cash:>14,.2f}")
    print(f"  Position value:  {position_value:>14,.2f} ({position_ratio:.1%})")
    print(f"  Total return:    {total_return:+.2%}")
    print(f"  Holdings:        {len(holdings)}")
    print(f"  Day index:       {state.day_index}")

    if not holdings:
        return
    print()
    print("  ts_code        shares    avg_cost    ref_price       value        pnl")
    for code in sorted(holdings):
        holding = holdings[code]
        price = prices.get(code, holding.avg_cost)
        value = holding.market_value(price)
        cost = holding.shares * holding.avg_cost
        pnl = (value - cost) / cost if cost > 0 else 0.0
        print(f"  {code:<12} {holding.shares:>8} {holding.avg_cost:>11.3f} {price:>12.3f} {value:>11,.2f} {pnl:>+9.1%}")


def save_execution_log(
    results: list[ExecutionResult],
    execution_dir: Path,
    trade_date: str,
    state: PortfolioState,
) -> Path:
    payload = {
        "signal_date": trade_date,
        "executions": [asdict(result) for result in results],
        "summary": {
            "buy_filled": sum(1 for r in results if r.action == "BUY" and r.status == "filled"),
            "buy_partial": sum(1 for r in results if r.action == "BUY" and r.status == "partial"),
            "buy_failed": sum(1 for r in results if r.action == "BUY" and r.status in {"skipped", "failed"}),
            "sell_filled": sum(1 for r in results if r.action == "SELL" and r.status == "filled"),
            "sell_partial": sum(1 for r in results if r.action == "SELL" and r.status == "partial"),
            "sell_failed": sum(1 for r in results if r.action == "SELL" and r.status in {"skipped", "failed"}),
            "total_buy_value": sum(r.actual_value for r in results if r.action == "BUY"),
            "total_sell_value": sum(r.actual_value for r in results if r.action == "SELL"),
            "post_cash": state.cash,
        },
        "logged_at": datetime.now().isoformat(),
    }
    path = execution_dir / f"execution_{trade_date}.json"
    write_json(path, payload)
    print(f"\n  Execution log saved: {path}")
    return path


def main() -> None:
    args = parse_args()
    trade_date = str(args.trade_date)
    config = load_yaml(args.config)
    prev_trade_date = previous_trading_day(config, trade_date)

    outputs = config["outputs"]
    live_inputs = config["live_inputs"]
    orders_dir = resolve_path(outputs["orders_dir"])
    orders_csv_path = resolve_path(args.orders_csv) if args.orders_csv else orders_dir / f"orders_{trade_date}.csv"
    portfolio_state_path = (
        resolve_path(args.portfolio_state)
        if args.portfolio_state
        else PROJECT_ROOT / "outputs" / "live" / "portfolio_state.json"
    )
    execution_dir = (
        resolve_path(args.execution_dir)
        if args.execution_dir
        else PROJECT_ROOT / "outputs" / "live" / "orders"
    )
    price_snapshot_path = (
        resolve_path(args.price_snapshot)
        if args.price_snapshot
        else format_path(live_inputs["price_snapshot"], trade_date=trade_date, prev_trade_date=prev_trade_date)
    )

    if args.reset and portfolio_state_path.exists():
        portfolio_state_path.unlink()

    state = PortfolioState(initial_nav=args.initial_nav, cash=args.initial_nav)
    loaded = state.load(portfolio_state_path)
    if not loaded:
        print(f"  No prior portfolio state. Starting with NAV={args.initial_nav:,.2f}")

    if not orders_csv_path.exists():
        die(f"missing orders CSV: {orders_csv_path}")
    orders = orders_csv_to_order_list(pd.read_csv(orders_csv_path))
    prices = load_reference_prices(price_snapshot_path)

    results = interactive_execute_orders(orders, state)
    state.last_signal_date = trade_date
    state.day_index += 1
    state.save(portfolio_state_path)
    print(f"\n  Portfolio state saved: {portfolio_state_path}")

    print_portfolio_summary(state, prices)
    execution_log = save_execution_log(results, execution_dir, trade_date, state)

    if not args.no_push:
        print_header("Git Handoff")
        git_commit_and_push(
            commit_msg=f"live: trade date {trade_date} execution records (day {state.day_index})",
            paths=[portfolio_state_path, execution_log],
            branch=args.push_branch,
            push=True,
        )
    else:
        print("\n  Git push skipped (--no-push).")


if __name__ == "__main__":
    main()
