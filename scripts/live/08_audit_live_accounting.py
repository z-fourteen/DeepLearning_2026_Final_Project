from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "live") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "live"))

from common import load_yaml, previous_trading_day, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit live execution/state/valuation accounting consistency.")
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--portfolio-state", default="outputs/live/portfolio_state.json")
    parser.add_argument("--tolerance", type=float, default=0.02)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def holdings_from_previous_close(config: dict[str, Any], trade_date: str) -> tuple[dict[str, dict[str, float]], float]:
    prev_trade_date = previous_trading_day(config, trade_date)
    valuation_json = PROJECT_ROOT / "outputs" / "live" / "valuations" / f"valuation_{prev_trade_date}.json"
    valuation_csv = PROJECT_ROOT / "outputs" / "live" / "valuations" / f"valuation_{prev_trade_date}.csv"
    valuation = load_json(valuation_json)
    frame = pd.read_csv(valuation_csv)
    holdings: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        shares = int(getattr(row, "shares", 0) or 0)
        if shares <= 0:
            continue
        holdings[str(getattr(row, "ts_code"))] = {
            "shares": shares,
            "avg_cost": float(getattr(row, "avg_cost", 0.0) or 0.0),
        }
    return holdings, float((valuation.get("summary") or {}).get("cash", 0.0))


def apply_execution(
    holdings: dict[str, dict[str, float]],
    cash: float,
    execution: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], float]:
    holdings = {code: dict(value) for code, value in holdings.items()}
    for record in execution.get("executions", []):
        status = str(record.get("status", "")).lower()
        shares = int(record.get("actual_shares", 0) or 0)
        price = float(record.get("actual_price", 0.0) or 0.0)
        if status not in {"filled", "partial"} or shares <= 0 or price <= 0:
            continue
        code = str(record["ts_code"])
        value = shares * price
        fee = float(record.get("fee", 0.0) or 0.0)
        action = str(record.get("action", "")).upper()
        if action == "SELL":
            current = holdings.get(code, {"shares": 0, "avg_cost": 0.0})
            current["shares"] = int(current["shares"]) - shares
            cash += value - fee
            if int(current["shares"]) <= 0:
                holdings.pop(code, None)
            else:
                holdings[code] = current
        elif action == "BUY":
            current = holdings.get(code, {"shares": 0, "avg_cost": 0.0})
            old_shares = int(current["shares"])
            total_cost = old_shares * float(current["avg_cost"]) + value + fee
            new_shares = old_shares + shares
            holdings[code] = {
                "shares": new_shares,
                "avg_cost": total_cost / new_shares if new_shares > 0 else 0.0,
            }
            cash -= value + fee
    return holdings, cash


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> list[str]:
    if abs(actual - expected) <= tolerance:
        return []
    return [f"{label}: actual={actual:.6f}, expected={expected:.6f}, diff={actual - expected:+.6f}"]


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    trade_date = str(args.trade_date)
    tolerance = float(args.tolerance)

    state = load_json(resolve_path(args.portfolio_state))
    execution = load_json(PROJECT_ROOT / "outputs" / "live" / "orders" / f"execution_{trade_date}.json")
    valuation = load_json(PROJECT_ROOT / "outputs" / "live" / "valuations" / f"valuation_{trade_date}.json")
    valuation_csv = pd.read_csv(PROJECT_ROOT / "outputs" / "live" / "valuations" / f"valuation_{trade_date}.csv")

    expected_holdings, expected_cash = holdings_from_previous_close(config, trade_date)
    expected_holdings, expected_cash = apply_execution(expected_holdings, expected_cash, execution)

    errors: list[str] = []
    state_holdings = state.get("holdings") or {}
    for code in sorted(set(expected_holdings) | set(state_holdings)):
        expected = expected_holdings.get(code)
        actual = state_holdings.get(code)
        if expected is None:
            errors.append(f"{code}: unexpected state holding")
            continue
        if actual is None:
            errors.append(f"{code}: missing state holding")
            continue
        errors.extend(assert_close(float(actual.get("shares", 0)), float(expected["shares"]), 0.0, f"{code} shares"))
        errors.extend(assert_close(float(actual.get("avg_cost", 0.0)), float(expected["avg_cost"]), 1e-6, f"{code} avg_cost"))

    summary = valuation.get("summary") or {}
    position_value = float(valuation_csv["market_value"].sum())
    nav = float(summary.get("nav", 0.0))
    cash = float(summary.get("cash", 0.0))
    reconciliation = summary.get("cash_reconciliation") or {}
    cash_adjustment = float(reconciliation.get("cash_adjustment", 0.0) or 0.0)
    expected_cash_after_reconciliation = expected_cash + cash_adjustment
    errors.extend(assert_close(position_value, float(summary.get("position_value", 0.0)), tolerance, "valuation position_value"))
    errors.extend(assert_close(nav, cash + position_value, tolerance, "valuation nav=cash+position_value"))
    errors.extend(assert_close(float(state.get("cash", 0.0)), cash, tolerance, "state cash vs valuation cash"))
    errors.extend(assert_close(expected_cash_after_reconciliation, cash, tolerance, "execution cash plus reconciliation"))

    if errors:
        print("LIVE ACCOUNTING AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(2)

    print("LIVE ACCOUNTING AUDIT PASS")
    print(f"trade_date={trade_date} holdings={len(state_holdings)} cash={cash:.2f} nav={nav:.2f}")


if __name__ == "__main__":
    main()
