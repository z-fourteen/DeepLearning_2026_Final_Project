"""Live stage 6: close valuation from raw daily close prices.

Stage 5 records real fill prices as cost basis. Stage 6 runs after the raw
daily file is updated after market close, validates portfolio state against the
execution log, reads official close prices, and computes mark-to-market PnL.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from common import (
    assert_universe,
    format_path,
    load_yaml,
    normalize_code_column,
    previous_trading_day,
    resolve_path,
    today_yyyymmdd,
    write_json,
)

DEFAULT_NAV = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live stage 6: value current holdings from raw daily close prices."
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument(
        "--daily-csv",
        help='Raw daily CSV updated after close. Defaults to "A股数据/daily/TRADE_DATE.csv".',
    )
    parser.add_argument(
        "--portfolio-state", default="outputs/live/portfolio_state.json"
    )
    parser.add_argument(
        "--execution-log",
        help="Execution log used to validate state. Defaults to outputs/live/orders/execution_TRADE_DATE.json if it exists.",
    )
    parser.add_argument(
        "--rebuild-state-from-execution",
        action="store_true",
        help="Rebuild portfolio_state.json from filled/partial execution records before valuation.",
    )
    parser.add_argument("--output-dir", default="outputs/live/valuations")
    parser.add_argument(
        "--previous-nav",
        type=float,
        help="NAV used as the return base. Defaults to state.initial_nav on day 1, otherwise prior valuation if available.",
    )
    parser.add_argument(
        "--actual-nav",
        type=float,
        help="Broker-reported total assets after close. If set, cash is reconciled as actual_nav - position_value.",
    )
    parser.add_argument(
        "--actual-cash",
        type=float,
        help="Broker-reported cash after close. If set, overrides state cash for valuation.",
    )
    parser.add_argument(
        "--write-close-positions",
        action="store_true",
        help="Write data/live/account/close_positions_DATE.csv for next-day inheritance.",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing portfolio state: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def default_execution_log(trade_date: str) -> Path:
    return resolve_path(
        Path("outputs") / "live" / "orders" / f"execution_{trade_date}.json"
    )


def default_daily_csv(trade_date: str) -> Path:
    return resolve_path(Path("A\u80a1\u6570\u636e") / "daily" / f"{trade_date}.csv")


def load_daily_close_prices(
    path: Path, trade_date: str, holding_codes: set[str]
) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"missing raw daily CSV: {path}")
    frame = normalize_code_column(pd.read_csv(path))
    if "trade_date" in frame.columns:
        frame["trade_date"] = (
            frame["trade_date"].astype(str).str.replace("-", "", regex=False)
        )
        frame = frame[frame["trade_date"].eq(str(trade_date))].copy()
    if "close" not in frame.columns:
        raise ValueError(f"raw daily CSV must contain close column: {path}")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.drop_duplicates("ts_code")
    prices = {
        str(row.ts_code): float(row.close)
        for row in frame[frame["close"].gt(0)].itertuples(index=False)
    }
    missing = sorted(code for code in holding_codes if code not in prices)
    if missing:
        sample = ", ".join(missing[:10])
        raise ValueError(
            f"raw daily CSV missing valid close for {len(missing)} holdings: {sample}"
        )
    return {code: prices[code] for code in holding_codes}


def load_execution_log(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_counted_execution(record: dict[str, Any]) -> bool:
    status = str(record.get("status", "")).lower()
    shares = int(record.get("actual_shares", 0) or 0)
    price = float(record.get("actual_price", 0.0) or 0.0)
    return status in {"filled", "partial"} and shares > 0 and price > 0


def rebuild_state_from_execution(
    execution_log: dict[str, Any],
    base_state: dict[str, Any],
    trade_date: str,
) -> dict[str, Any]:
    initial_nav = float(
        execution_log.get("summary", {}).get(
            "initial_nav", base_state.get("initial_nav", DEFAULT_NAV)
        )
    )
    holdings: dict[str, dict[str, float | int]] = {}
    cash = initial_nav
    counted = 0
    excluded = 0
    for record in execution_log.get("executions", []):
        action = str(record.get("action", "")).upper()
        if not is_counted_execution(record):
            excluded += 1
            continue
        if action != "BUY":
            raise ValueError(
                "rebuild-state-from-execution only supports initial-build BUY logs; "
                "non-BUY fills require a prior-position snapshot"
            )
        code = str(record["ts_code"])
        shares = int(record["actual_shares"])
        price = float(record["actual_price"])
        value = float(record.get("actual_value", shares * price) or shares * price)

        current = holdings.get(
            code, {"shares": 0, "avg_cost": 0.0, "weight_at_entry": 0.0}
        )
        old_shares = int(current["shares"])
        total_cost = old_shares * float(current["avg_cost"]) + value
        new_shares = old_shares + shares
        current["shares"] = new_shares
        current["avg_cost"] = total_cost / new_shares if new_shares > 0 else 0.0
        old_state_holding = (base_state.get("holdings") or {}).get(code, {})
        current["weight_at_entry"] = float(
            old_state_holding.get("weight_at_entry", 0.0)
        )
        holdings[code] = current
        cash -= value
        counted += 1

    return {
        "last_signal_date": trade_date,
        "day_index": int(base_state.get("day_index", 1) or 1),
        "cash": cash,
        "initial_nav": initial_nav,
        "holdings": holdings,
        "pending_orders": [],
        "updated_at": datetime.now().isoformat(),
        "accounting_note": (
            f"Rebuilt from execution_{trade_date}.json counted executions only: "
            f"{counted} filled/partial records included, {excluded} skipped/failed records excluded."
        ),
    }


def compare_state_to_execution_state(
    actual_state: dict[str, Any],
    expected_state: dict[str, Any],
    cash_tol: float = 0.01,
    cost_tol: float = 1e-6,
) -> list[str]:
    errors: list[str] = []
    actual_holdings = actual_state.get("holdings") or {}
    expected_holdings = expected_state.get("holdings") or {}
    actual_codes = set(actual_holdings)
    expected_codes = set(expected_holdings)
    for code in sorted(expected_codes - actual_codes):
        errors.append(f"{code}: missing from portfolio_state")
    for code in sorted(actual_codes - expected_codes):
        errors.append(f"{code}: extra in portfolio_state")
    for code in sorted(actual_codes & expected_codes):
        actual = actual_holdings[code]
        expected = expected_holdings[code]
        if int(actual.get("shares", 0)) != int(expected.get("shares", 0)):
            errors.append(
                f"{code}: shares state={actual.get('shares')} execution={expected.get('shares')}"
            )
        actual_cost = float(actual.get("avg_cost", 0.0))
        expected_cost = float(expected.get("avg_cost", 0.0))
        if abs(actual_cost - expected_cost) > cost_tol:
            errors.append(
                f"{code}: avg_cost state={actual_cost:.6f} execution={expected_cost:.6f}"
            )
    actual_cash = float(actual_state.get("cash", 0.0))
    expected_cash = float(expected_state.get("cash", 0.0))
    if abs(actual_cash - expected_cash) > cash_tol:
        errors.append(f"cash: state={actual_cash:.2f} execution={expected_cash:.2f}")
    return errors


def validate_or_rebuild_state(
    state: dict[str, Any],
    execution_log: dict[str, Any] | None,
    trade_date: str,
    rebuild: bool,
) -> dict[str, Any]:
    if execution_log is None:
        if rebuild:
            raise FileNotFoundError(
                "cannot rebuild state because execution log is missing"
            )
        return state

    is_initial_build = (
        str(execution_log.get("day_classification", "")).lower() == "initial_build"
        or int(state.get("day_index", 0) or 0) <= 1
    )
    if not is_initial_build and not rebuild:
        return state

    rebuilt = rebuild_state_from_execution(execution_log, state, trade_date)
    if rebuild:
        return rebuilt
    if is_initial_build:
        errors = compare_state_to_execution_state(state, rebuilt)
        if errors:
            sample = "; ".join(errors[:10])
            raise ValueError(
                "portfolio_state.json is inconsistent with execution log. "
                "Rerun with --rebuild-state-from-execution to repair. "
                f"Sample mismatches: {sample}"
            )
    return state


def choose_previous_nav(
    state: dict[str, Any],
    config: dict[str, Any],
    trade_date: str,
    previous_nav: float | None,
) -> float:
    if previous_nav is not None:
        return float(previous_nav)
    last_valuation = state.get("last_valuation") or {}
    last_nav = last_valuation.get("nav")
    if last_nav is not None and str(last_valuation.get("trade_date")) != str(
        trade_date
    ):
        return float(last_nav)
    try:
        prev_trade_date = previous_trading_day(config, trade_date)
    except Exception:
        prev_trade_date = ""
    if prev_trade_date:
        prev_valuation_path = resolve_path(
            Path("outputs")
            / "live"
            / "valuations"
            / f"valuation_{prev_trade_date}.json"
        )
        if prev_valuation_path.exists():
            prev_valuation = json.loads(prev_valuation_path.read_text(encoding="utf-8"))
            nav = (prev_valuation.get("summary") or {}).get("nav")
            if nav is not None:
                return float(nav)
    return float(state.get("initial_nav", DEFAULT_NAV))


def build_valuation(
    state: dict[str, Any],
    close_prices: dict[str, float],
    trade_date: str,
    previous_nav: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    holdings = state.get("holdings", {})
    cash = float(state.get("cash", 0.0))
    for code in sorted(holdings):
        holding = holdings[code]
        shares = int(holding.get("shares", 0))
        avg_cost = float(holding.get("avg_cost", 0.0))
        close_price = float(close_prices[code])
        cost_value = shares * avg_cost
        market_value = shares * close_price
        pnl = market_value - cost_value
        rows.append(
            {
                "trade_date": trade_date,
                "ts_code": code,
                "shares": shares,
                "avg_cost": avg_cost,
                "close_price": close_price,
                "cost_value": cost_value,
                "market_value": market_value,
                "unrealized_pnl": pnl,
                "unrealized_return": pnl / cost_value if cost_value > 0 else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    position_value = float(frame["market_value"].sum()) if not frame.empty else 0.0
    cost_value = float(frame["cost_value"].sum()) if not frame.empty else 0.0
    nav = cash + position_value
    daily_pnl = nav - previous_nav
    summary = {
        "trade_date": trade_date,
        "cash": cash,
        "position_cost": cost_value,
        "position_value": position_value,
        "nav": nav,
        "previous_nav": previous_nav,
        "daily_pnl": daily_pnl,
        "daily_return": daily_pnl / previous_nav if previous_nav > 0 else None,
        "unrealized_pnl_vs_cost": position_value - cost_value,
        "unrealized_return_vs_cost": (
            (position_value - cost_value) / cost_value if cost_value > 0 else None
        ),
        "position_ratio": position_value / nav if nav > 0 else None,
        "holding_count": int(len(frame)),
    }
    if not frame.empty and nav > 0:
        frame["weight"] = frame["market_value"] / nav
    return frame, summary


def reconcile_broker_cash(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    actual_nav: float | None,
    actual_cash: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if actual_nav is None and actual_cash is None:
        return frame, summary

    summary = dict(summary)
    frame = frame.copy()
    original_cash = float(summary["cash"])
    original_nav = float(summary["nav"])
    position_value = float(summary["position_value"])

    if actual_cash is not None:
        cash = float(actual_cash)
        nav = cash + position_value
    else:
        nav = float(actual_nav)
        cash = nav - position_value

    previous_nav = float(summary["previous_nav"])
    summary["cash"] = cash
    summary["nav"] = nav
    summary["daily_pnl"] = nav - previous_nav
    summary["daily_return"] = (
        (nav - previous_nav) / previous_nav if previous_nav > 0 else None
    )
    summary["position_ratio"] = position_value / nav if nav > 0 else None
    summary["cash_reconciled"] = True
    summary["cash_reconciliation"] = {
        "original_cash": original_cash,
        "original_nav": original_nav,
        "actual_nav": float(actual_nav) if actual_nav is not None else None,
        "actual_cash": float(actual_cash) if actual_cash is not None else None,
        "cash_adjustment": cash - original_cash,
    }
    if not frame.empty and nav > 0:
        frame["weight"] = frame["market_value"] / nav
    return frame, summary


def write_close_positions(
    frame: pd.DataFrame, config: dict[str, Any], trade_date: str
) -> Path:
    path = format_path(
        config["live_inputs"]["previous_close_positions"],
        trade_date="unused",
        prev_trade_date=trade_date,
    )
    out = frame[["ts_code", "weight", "shares", "market_value", "close_price"]].copy()
    out = out.rename(columns={"shares": "volume"})
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out = out[(out["volume"] > 0) | (out["weight"] > 0)].copy()
    assert_universe(out, config, "close positions")
    if out.empty:
        raise ValueError("refuse to write empty close positions")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def print_summary(frame: pd.DataFrame, summary: dict[str, Any], source: Path) -> None:
    print_header("Stage 6: Close Valuation")
    print(f"  Source daily CSV: {source}")
    print_header("Close Valuation Summary")
    print(f"  NAV:             {summary['nav']:>14,.2f}")
    print(f"  Previous NAV:    {summary['previous_nav']:>14,.2f}")
    print(f"  Daily PnL:       {summary['daily_pnl']:>+14,.2f}")
    print(f"  Daily return:    {summary['daily_return']:+.2%}")
    print(f"  Cash:            {summary['cash']:>14,.2f}")
    print(
        f"  Position value:  {summary['position_value']:>14,.2f} ({summary['position_ratio']:.1%})"
    )
    print(f"  PnL vs cost:     {summary['unrealized_pnl_vs_cost']:>+14,.2f}")
    print()
    print("  ts_code        shares    avg_cost   close_price       value        pnl")
    for row in frame.itertuples(index=False):
        print(
            f"  {row.ts_code:<12} {row.shares:>8} {row.avg_cost:>11.3f} "
            f"{row.close_price:>13.3f} {row.market_value:>11,.2f} {row.unrealized_pnl:>+10,.2f}"
        )


def main() -> None:
    args = parse_args()
    trade_date = str(args.trade_date)
    config = load_yaml(args.config)
    state_path = resolve_path(args.portfolio_state)
    state = load_state(state_path)

    execution_log_path = (
        resolve_path(args.execution_log)
        if args.execution_log
        else default_execution_log(trade_date)
    )
    execution_log = load_execution_log(execution_log_path)
    state = validate_or_rebuild_state(
        state,
        execution_log,
        trade_date,
        args.rebuild_state_from_execution,
    )
    if args.rebuild_state_from_execution:
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    holdings = state.get("holdings", {})
    if not holdings:
        raise SystemExit("portfolio has no holdings to value")

    holding_codes = {str(code) for code in holdings}
    daily_csv = (
        resolve_path(args.daily_csv)
        if args.daily_csv
        else default_daily_csv(trade_date)
    )
    close_prices = load_daily_close_prices(daily_csv, trade_date, holding_codes)
    previous_nav = choose_previous_nav(state, config, trade_date, args.previous_nav)
    frame, summary = build_valuation(state, close_prices, trade_date, previous_nav)
    frame, summary = reconcile_broker_cash(
        frame, summary, args.actual_nav, args.actual_cash
    )

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"valuation_{trade_date}.csv"
    json_path = out_dir / f"valuation_{trade_date}.json"
    frame.to_csv(csv_path, index=False)
    write_json(
        json_path,
        {
            "trade_date": trade_date,
            "valuation_type": "close",
            "source": str(daily_csv),
            "execution_log": (
                str(execution_log_path) if execution_log is not None else None
            ),
            "state_rebuilt_from_execution": bool(args.rebuild_state_from_execution),
            "summary": summary,
            "positions_csv": str(csv_path),
            "created_at": datetime.now().isoformat(),
        },
    )

    state["last_valuation"] = {
        "trade_date": trade_date,
        "valuation_type": "close",
        "nav": summary["nav"],
        "daily_pnl": summary["daily_pnl"],
        "daily_return": summary["daily_return"],
        "position_value": summary["position_value"],
        "cash": summary["cash"],
        "cash_reconciled": bool(summary.get("cash_reconciled", False)),
        "cash_reconciliation": summary.get("cash_reconciliation"),
        "source": str(daily_csv),
        "execution_log": str(execution_log_path) if execution_log is not None else None,
        "state_rebuilt_from_execution": bool(args.rebuild_state_from_execution),
        "positions_csv": str(csv_path),
        "valuation_json": str(json_path),
        "updated_at": datetime.now().isoformat(),
    }
    state["cash"] = summary["cash"]
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    close_positions_path = None
    if args.write_close_positions:
        close_positions_path = write_close_positions(frame, config, trade_date)

    print_summary(frame, summary, daily_csv)
    print(f"\n  Valuation CSV saved: {csv_path}")
    print(f"  Valuation JSON saved: {json_path}")
    print(f"  Portfolio state updated: {state_path}")
    if close_positions_path is not None:
        print(f"  Close positions saved: {close_positions_path}")


if __name__ == "__main__":
    main()
