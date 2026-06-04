from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    die,
    load_yaml,
    normalize_code_column,
    previous_trading_day,
    resolve_path,
    today_yyyymmdd,
    write_json,
)


@dataclass
class Holding:
    shares: int
    avg_cost: float
    weight_at_entry: float = 0.0


def fee_rates(config: dict[str, Any]) -> dict[str, float]:
    opt = config.get("optimizer", {}) or {}
    return {
        "commission": float(opt.get("commission_bps", 0.0)) / 10000.0,
        "transfer": float(opt.get("transfer_bps", 0.0)) / 10000.0,
        "sell_stamp_tax": float(opt.get("sell_stamp_tax_bps", 0.0)) / 10000.0,
    }


def transaction_fee(action: str, value: float, rates: dict[str, float]) -> float:
    fee = value * (rates["commission"] + rates["transfer"])
    if action == "SELL":
        fee += value * rates["sell_stamp_tax"]
    return float(fee)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild portfolio_state.json from previous close valuation and execution log."
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument("--execution-log")
    parser.add_argument("--portfolio-state", default="outputs/live/portfolio_state.json")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--actual-cash", type=float)
    return parser.parse_args()


def load_previous_state(config: dict[str, Any], trade_date: str) -> tuple[dict[str, Holding], float, dict[str, Any]]:
    prev_trade_date = previous_trading_day(config, trade_date)
    valuation_csv = resolve_path(Path("outputs") / "live" / "valuations" / f"valuation_{prev_trade_date}.csv")
    valuation_json = resolve_path(Path("outputs") / "live" / "valuations" / f"valuation_{prev_trade_date}.json")
    if not valuation_csv.exists() or not valuation_json.exists():
        die(f"missing previous close valuation: {valuation_csv}, {valuation_json}")
    summary = (json.loads(valuation_json.read_text(encoding="utf-8")).get("summary") or {})
    frame = normalize_code_column(pd.read_csv(valuation_csv))
    holdings: dict[str, Holding] = {}
    for row in frame.itertuples(index=False):
        shares = int(getattr(row, "shares", 0) or 0)
        if shares <= 0:
            continue
        holdings[str(row.ts_code)] = Holding(
            shares=shares,
            avg_cost=float(getattr(row, "avg_cost", 0.0) or 0.0),
            weight_at_entry=float(getattr(row, "weight", 0.0) or 0.0),
        )
    return holdings, float(summary.get("cash", 0.0) or 0.0), summary


def is_counted(record: dict[str, Any]) -> bool:
    return int(record.get("actual_shares", 0) or 0) > 0 and record.get("status") in {"filled", "partial"}


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    trade_date = str(args.trade_date)
    path = resolve_path(args.execution_log or Path("outputs") / "live" / "orders" / f"execution_{trade_date}.json")
    if not path.exists():
        die(f"missing execution log: {path}")
    execution = json.loads(path.read_text(encoding="utf-8"))
    holdings, cash, prev_summary = load_previous_state(config, trade_date)
    rates = fee_rates(config)

    total_buy = 0.0
    total_sell = 0.0
    total_fee = 0.0
    realized = 0.0
    rebuilt_records: list[dict[str, Any]] = []

    for record in execution.get("executions", []):
        if not is_counted(record):
            rebuilt_records.append(record)
            continue
        code = str(record["ts_code"])
        action = str(record["action"]).upper()
        shares = int(record.get("actual_shares", 0) or 0)
        price = float(record.get("actual_price", 0.0) or 0.0)
        value = shares * price
        fee = transaction_fee(action, value, rates)
        if action == "BUY":
            current = holdings.get(code, Holding(0, 0.0, 0.0))
            total_cost = current.avg_cost * current.shares + value + fee
            current.shares += shares
            current.avg_cost = total_cost / current.shares if current.shares > 0 else 0.0
            current.weight_at_entry = float(record.get("target_weight", 0.0) or current.weight_at_entry)
            holdings[code] = current
            cash -= value + fee
            total_buy += value
            cost_basis = value + fee
            realized_pnl = 0.0
        elif action == "SELL":
            current = holdings.get(code)
            if current is None or current.shares < shares:
                die(f"execution sells more shares than available: {code}")
            cost_basis = shares * current.avg_cost
            realized_pnl = value - fee - cost_basis
            current.shares -= shares
            if current.shares > 0:
                holdings[code] = current
            else:
                holdings.pop(code, None)
            cash += value - fee
            total_sell += value
            realized += realized_pnl
        else:
            die(f"unsupported action={action}")
        total_fee += fee
        updated = dict(record)
        updated["actual_value"] = value
        updated["fee"] = fee
        updated["cost_basis"] = cost_basis
        updated["realized_pnl"] = realized_pnl
        rebuilt_records.append(updated)

    if args.actual_cash is not None:
        adjustment = float(args.actual_cash) - cash
        cash = float(args.actual_cash)
    else:
        adjustment = 0.0

    state = {
        "last_valuation": {
            "trade_date": previous_trading_day(config, trade_date),
            "valuation_type": "close",
            "nav": prev_summary.get("nav"),
            "position_value": prev_summary.get("position_value"),
            "cash": prev_summary.get("cash"),
        },
        "last_signal_date": trade_date,
        "day_index": config.get("competition", {}).get("trading_days", []).index(trade_date) + 1,
        "cash": cash,
        "initial_nav": 1_000_000.0,
        "holdings": {code: asdict(holding) for code, holding in sorted(holdings.items())},
        "pending_orders": [],
        "cash_reconciliation": {
            "calculated_cash_before_adjustment": cash - adjustment,
            "actual_cash": args.actual_cash,
            "cash_adjustment": adjustment,
        },
    }
    summary = execution.get("summary", {}) or {}
    summary.update(
        {
            "total_buy_value": total_buy,
            "total_sell_value": total_sell,
            "total_fees": total_fee,
            "realized_sell_pnl": realized,
            "post_cash": cash,
        }
    )
    execution["executions"] = rebuilt_records
    execution["summary"] = summary

    print(
        f"rebuilt trade_date={trade_date} cash={cash:,.2f} "
        f"fees={total_fee:,.2f} buy={total_buy:,.2f} sell={total_sell:,.2f} "
        f"adjustment={adjustment:+,.2f}"
    )
    if not args.write:
        print("dry-run only; pass --write to update portfolio_state and execution log")
        return
    write_json(resolve_path(args.portfolio_state), state)
    write_json(path, execution)
    print(f"wrote: {resolve_path(args.portfolio_state)}")
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
