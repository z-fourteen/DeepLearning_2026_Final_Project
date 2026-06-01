from __future__ import annotations

import argparse
import time

import pandas as pd

from common import die, format_path, load_yaml, normalize_code_column, previous_trading_day, resolve_path, today_yyyymmdd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intraday execution monitor: check unfilled volume and suggest repricing.")
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument("--orders")
    parser.add_argument("--broker-status")
    parser.add_argument("--interval-minutes", type=float, default=5.0)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def load_status(path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        print(
            "\n未发现 broker status 文件，盘中监控等待以下合同：\n"
            "code, action, submitted_volume, filled_volume, order_price, best_bid, best_ask, tick_size\n"
            f"expected_path={path}\n"
        )
        return pd.DataFrame()
    frame = normalize_code_column(pd.read_csv(path))
    if "code" not in frame.columns:
        frame["code"] = frame["ts_code"]
    required = ["action", "submitted_volume", "filled_volume", "order_price", "best_bid", "best_ask"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        die(f"broker status missing columns: {missing}")
    frame["tick_size"] = pd.to_numeric(frame.get("tick_size", 0.01), errors="coerce").fillna(0.01)
    for column in ["submitted_volume", "filled_volume", "order_price", "best_bid", "best_ask"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def one_check(config: dict, trade_date: str, orders_path, status_path, ticks: int) -> pd.DataFrame:
    orders_path = resolve_path(orders_path)
    if not orders_path.exists():
        die(f"missing order file: {orders_path}")
    status = load_status(status_path)
    if status.empty:
        return status
    status["unfilled_volume"] = (status["submitted_volume"] - status["filled_volume"]).clip(lower=0)
    active = status[status["unfilled_volume"] > 0].copy()
    if active.empty:
        print("全部委托已成交或无未成交残股。")
        return active

    def suggested(row):
        if str(row.action).upper() == "BUY":
            return float(row.best_ask + ticks * row.tick_size)
        return float(max(0.0, row.best_bid - ticks * row.tick_size))

    active["suggested_price"] = active.apply(suggested, axis=1)
    active["advice"] = active.apply(
        lambda row: (
            f"当前报单量不足，触发撤单命令，建议 {row.action} {row.code} "
            f"残量 {int(row.unfilled_volume)} 股，向五档盘口让步 {ticks} tick，"
            f"原价 {row.order_price:.3f} -> 建议价 {row.suggested_price:.3f}"
        ),
        axis=1,
    )
    out_dir = resolve_path(config["outputs"]["monitor_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"intraday_advice_{trade_date}.csv"
    active.to_csv(out_csv, index=False)

    print("\n【盘中未成交残股监控】")
    print(active[["code", "action", "submitted_volume", "filled_volume", "unfilled_volume", "order_price", "best_bid", "best_ask", "suggested_price"]].to_string(index=False))
    print("\n【撤单重报建议】")
    for line in active["advice"].tolist():
        print(line)
    print(f"\n建议落盘：{out_csv}")
    return active


def main() -> None:
    args = parse_args()
    trade_date = str(args.trade_date)
    config = load_yaml(args.config)
    prev_trade_date = previous_trading_day(config, trade_date)
    paths = config["live_inputs"]
    orders_path = args.orders or resolve_path(config["outputs"]["orders_dir"]) / f"orders_{trade_date}.csv"
    status_path = args.broker_status or format_path(paths["broker_status"], trade_date=trade_date, prev_trade_date=prev_trade_date)

    # TWAP/VWAP 执行原则：
    # 1. 09:30-10:00 只完成 20%-30% 目标量，避免开盘冲击。
    # 2. 10:00-14:30 按成交量曲线匀速追踪，若连续 N 分钟未成交则撤单重报。
    # 3. 14:30 后优先保证 min_invested 规则与卖出风险释放，允许更激进地向五档盘口让价。
    # 4. 14:50 后仍有未成交买单且仓位低于 80%，应人工确认是否以更强价格追单。
    while True:
        one_check(config, trade_date, orders_path, status_path, int(args.ticks))
        if not args.loop:
            break
        time.sleep(max(1.0, float(args.interval_minutes)) * 60.0)


if __name__ == "__main__":
    main()
