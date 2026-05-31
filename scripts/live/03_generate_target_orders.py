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

    nav = float(opt["portfolio_nav"])
    lot_size = int(guards.get("lot_size", 100))
    min_order_value = float(guards.get("min_order_value", 1000.0))
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
            }
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
        },
    )

    print("\n【阶段三完成】目标调仓差分明细")
    print(f"trade_date={trade_date} output={out_csv}")
    print(f"BUY value={buy_value:,.2f} SELL value={sell_value:,.2f} order_count={len(orders_frame)}")
    if orders_frame.empty:
        print("今日无超过最小金额阈值的调仓指令。")
        return
    print("\n【买入调仓看板】")
    print(orders_frame[orders_frame["action"].eq("BUY")][["code", "action", "target_value", "target_volume"]].to_string(index=False))
    print("\n【卖出调仓看板】")
    print(orders_frame[orders_frame["action"].eq("SELL")][["code", "action", "target_value", "target_volume"]].to_string(index=False))


if __name__ == "__main__":
    main()
