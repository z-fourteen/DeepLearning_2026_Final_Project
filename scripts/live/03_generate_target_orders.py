from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    die,
    format_path,
    load_positions,
    load_yaml,
    normalize_code_column,
    price_column,
    previous_trading_day,
    resolve_path,
    today_yyyymmdd,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live stage 3: convert target weights into executable order list.")
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument("--target-weights")
    parser.add_argument("--positions")
    parser.add_argument("--price-snapshot")
    return parser.parse_args()


def round_lot(shares: float, lot_size: int) -> int:
    if shares <= 0:
        return 0
    return int(np.floor(shares / lot_size) * lot_size)


def ceil_lots(value: float, lot_value: float) -> int:
    if value <= 0 or lot_value <= 0:
        return 0
    return int(np.ceil(value / lot_value))


def estimate_post_trade_values(
    frame: pd.DataFrame,
    orders: list[dict],
    nav: float,
) -> tuple[dict[str, float], float, float, float]:
    current = {
        str(row.ts_code): max(0.0, float(row.current_weight) * nav)
        for row in frame.itertuples(index=False)
    }
    buy_value = 0.0
    sell_value = 0.0
    for order in orders:
        code = str(order["code"])
        value = float(order["target_value"])
        current.setdefault(code, 0.0)
        if order["action"] == "BUY":
            current[code] += value
            buy_value += value
        else:
            current[code] = max(0.0, current[code] - value)
            sell_value += value
    invested = float(sum(current.values()))
    return current, invested, buy_value, sell_value


def add_min_invested_topup_orders(
    *,
    frame: pd.DataFrame,
    orders: list[dict],
    trade_date: str,
    nav: float,
    lot_size: int,
    min_invested: float,
    single_name_cap: float,
) -> dict[str, float | int]:
    post_values, invested, buy_value, sell_value = estimate_post_trade_values(frame, orders, nav)
    current_invested = float(
        (pd.to_numeric(frame["current_weight"], errors="coerce").fillna(0.0).clip(lower=0.0) * nav).sum()
    )
    cash_available = max(0.0, nav - current_invested + sell_value - buy_value)
    required_value = float(min_invested) * nav
    before_topup = invested
    added_value = 0.0
    added_orders = 0

    if invested + 1e-6 >= required_value:
        return {
            "required_value": required_value,
            "before_topup_invested_value": before_topup,
            "after_topup_invested_value": invested,
            "supplemental_buy_value": 0.0,
            "supplemental_order_count": 0,
        }

    candidates = frame[
        pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0).gt(0)
        & pd.to_numeric(frame["price"], errors="coerce").fillna(0.0).gt(0)
    ].copy()
    candidates["target_weight"] = pd.to_numeric(candidates["target_weight"], errors="coerce").fillna(0.0)
    candidates["pred_score"] = pd.to_numeric(candidates.get("pred_score", 0.0), errors="coerce").fillna(-np.inf)
    candidates = candidates.sort_values(["target_weight", "pred_score"], ascending=[False, False])

    for row in candidates.itertuples(index=False):
        if invested + 1e-6 >= required_value or cash_available <= 0:
            break
        code = str(row.ts_code)
        price = float(row.price)
        lot_value = price * lot_size
        if lot_value <= 0:
            continue
        current_value = float(post_values.get(code, 0.0))
        room_value = max(0.0, float(single_name_cap) * nav - current_value)
        max_lots_by_room = int(np.floor(room_value / lot_value))
        max_lots_by_cash = int(np.floor(cash_available / lot_value))
        max_lots = min(max_lots_by_room, max_lots_by_cash)
        if max_lots <= 0:
            continue
        needed_lots = max(1, ceil_lots(required_value - invested, lot_value))
        lots = min(max_lots, needed_lots)
        shares = lots * lot_size
        value = shares * price
        orders.append(
            {
                "trade_date": trade_date,
                "code": code,
                "action": "BUY",
                "price_ref": price,
                "target_value": float(value),
                "target_volume": int(shares),
                "delta_weight": float(value / nav),
                "reason": "round_lot_min_invested_topup",
            }
        )
        post_values[code] = current_value + value
        invested += value
        cash_available -= value
        added_value += value
        added_orders += 1

    if invested + 1e-6 < required_value:
        die(
            "rounded orders cannot satisfy min_invested after top-up: "
            f"invested={invested / nav:.2%}, required={float(min_invested):.2%}, "
            f"cash_available={cash_available:,.2f}. Consider relaxing single_name_cap/lot constraints."
        )

    return {
        "required_value": required_value,
        "before_topup_invested_value": before_topup,
        "after_topup_invested_value": invested,
        "supplemental_buy_value": added_value,
        "supplemental_order_count": added_orders,
    }


def main() -> None:
    args = parse_args()
    trade_date = str(args.trade_date)
    config = load_yaml(args.config)
    prev_trade_date = previous_trading_day(config, trade_date)
    paths = config["live_inputs"]
    guards = config["guards"]
    opt = config["optimizer"]

    target_path = resolve_path(args.target_weights) if args.target_weights else resolve_path(config["outputs"]["targets_dir"]) / f"target_weights_{trade_date}.csv"
    pos_path = resolve_path(args.positions) if args.positions else format_path(paths["positions"], trade_date=trade_date, prev_trade_date=prev_trade_date)
    price_path = resolve_path(args.price_snapshot) if args.price_snapshot else format_path(paths["price_snapshot"], trade_date=trade_date, prev_trade_date=prev_trade_date)

    if not target_path.exists():
        die(f"missing target weights: {target_path}")
    if not price_path.exists():
        die(f"missing 09:20 price snapshot: {price_path}")
    target = normalize_code_column(pd.read_csv(target_path))
    positions = load_positions(pos_path, "current positions")
    prices = normalize_code_column(pd.read_csv(price_path))
    px_col = price_column(prices)
    prices["price"] = pd.to_numeric(prices[px_col], errors="coerce")
    if prices["price"].le(0).any() or prices["price"].isna().any():
        die("price snapshot contains non-positive or missing prices")

    frame = target.merge(prices[["ts_code", "price"]], on="ts_code", how="left")
    frame = frame.merge(positions[["ts_code", *([c for c in ["volume"] if c in positions.columns])]], on="ts_code", how="left")
    if "volume" in frame.columns:
        frame["volume"] = frame["volume"].fillna(0).astype(int)
    else:
        # 若账户文件没有 volume，只能安全生成买单；卖单股数无法从权重反推真实可卖股数。
        frame["volume"] = 0
    frame["delta_weight"] = pd.to_numeric(frame["delta_weight"], errors="coerce").fillna(0.0)
    if "current_weight" not in frame.columns:
        frame["current_weight"] = 0.0
    if "target_weight" not in frame.columns:
        frame["target_weight"] = frame["current_weight"] + frame["delta_weight"]
    frame["current_weight"] = pd.to_numeric(frame["current_weight"], errors="coerce").fillna(0.0)
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0)

    nav = float(opt["portfolio_nav"])
    lot_size = int(guards.get("lot_size", 100))
    min_order_value = float(guards.get("min_order_value", 1000.0))
    min_invested = float(opt.get("min_invested", 0.0))
    single_name_cap = float(opt.get("single_name_cap", 1.0))
    orders: list[dict] = []
    for row in frame.itertuples(index=False):
        delta_value = float(row.delta_weight) * nav
        if abs(delta_value) < min_order_value:
            continue
        action = "BUY" if delta_value > 0 else "SELL"
        shares = round_lot(abs(delta_value) / float(row.price), lot_size)
        if action == "SELL":
            shares = min(shares, int(row.volume))
        if shares <= 0:
            continue
        orders.append(
            {
                "trade_date": trade_date,
                "code": str(row.ts_code),
                "action": action,
                "price_ref": float(row.price),
                "target_value": float(shares * float(row.price)),
                "target_volume": int(shares),
                "delta_weight": float(row.delta_weight),
                "reason": "target_delta",
            }
        )

    topup_stats = add_min_invested_topup_orders(
        frame=frame,
        orders=orders,
        trade_date=trade_date,
        nav=nav,
        lot_size=lot_size,
        min_invested=min_invested,
        single_name_cap=single_name_cap,
    )

    orders_frame = pd.DataFrame(orders)
    out_dir = resolve_path(config["outputs"]["orders_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"orders_{trade_date}.csv"
    orders_frame.to_csv(out_csv, index=False)
    buy_value = float(orders_frame.loc[orders_frame["action"].eq("BUY"), "target_value"].sum()) if not orders_frame.empty else 0.0
    sell_value = float(orders_frame.loc[orders_frame["action"].eq("SELL"), "target_value"].sum()) if not orders_frame.empty else 0.0
    write_json(
        out_dir / f"manifest_{trade_date}.json",
        {
            "trade_date": trade_date,
            "orders": str(out_csv),
            "order_count": int(len(orders_frame)),
            "buy_value": buy_value,
            "sell_value": sell_value,
            "price_snapshot": str(price_path),
            "min_invested": min_invested,
            "required_invested_value": float(topup_stats["required_value"]),
            "rounded_invested_value_before_topup": float(topup_stats["before_topup_invested_value"]),
            "rounded_invested_value_after_topup": float(topup_stats["after_topup_invested_value"]),
            "supplemental_buy_value": float(topup_stats["supplemental_buy_value"]),
            "supplemental_order_count": int(topup_stats["supplemental_order_count"]),
        },
    )

    print("\n【阶段三完成】目标调仓差分明细")
    print(f"trade_date={trade_date} output={out_csv}")
    print(f"BUY value={buy_value:,.2f} SELL value={sell_value:,.2f} order_count={len(orders_frame)}")
    print(
        "rounded invested: "
        f"before_topup={float(topup_stats['before_topup_invested_value']) / nav:.2%} "
        f"after_topup={float(topup_stats['after_topup_invested_value']) / nav:.2%} "
        f"required={min_invested:.2%}"
    )
    if orders_frame.empty:
        print("今日无超过最小金额阈值的调仓指令。")
        return
    print("\n【买入调仓看板】")
    print(orders_frame[orders_frame["action"].eq("BUY")][["code", "action", "target_value", "target_volume"]].to_string(index=False))
    print("\n【卖出调仓看板】")
    print(orders_frame[orders_frame["action"].eq("SELL")][["code", "action", "target_value", "target_volume"]].to_string(index=False))


if __name__ == "__main__":
    main()
