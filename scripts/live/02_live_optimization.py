from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common import (
    assert_position_inheritance,
    die,
    dynamic_shortfall_penalty,
    format_path,
    load_positions,
    load_yaml,
    normalize_code_column,
    previous_trading_day,
    resolve_path,
    today_yyyymmdd,
    write_json,
)

from scripts.portfolio.optimize_feasible_cash_buffer import prepare_lp_universe, solve_day_lp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live stage 2: optimize target absolute weights from old_w and live scores.")
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument("--predictions")
    parser.add_argument("--positions")
    parser.add_argument("--previous-close-positions")
    parser.add_argument("--liquidity-parquet")
    return parser.parse_args()


def amount_column(frame: pd.DataFrame) -> str:
    for column in ["next_amount", "amount", "turnover_amount", "amt", "money"]:
        if column in frame.columns:
            return column
    die("live liquidity parquet must contain one of: next_amount, amount, turnover_amount, amt, money")
    raise AssertionError("unreachable")


def load_live_liquidity(path, trade_date: str) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        die(f"missing live liquidity/feature parquet: {path}")
    frame = pd.read_parquet(path)
    frame = normalize_code_column(frame)
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
        frame = frame[frame["trade_date"].eq(trade_date)].copy()
    amount_col = amount_column(frame)
    frame["next_amount"] = pd.to_numeric(frame[amount_col], errors="coerce").fillna(0.0)
    for src, dst in [
        ("buy_executable_t1_open", "buy_executable_t1_open"),
        ("sell_executable_t1_open", "sell_executable_t1_open"),
        ("is_suspended", "next_is_suspended"),
        ("limit_up", "next_is_limit_up"),
        ("limit_down", "next_is_limit_down"),
    ]:
        if src in frame.columns:
            frame[dst] = frame[src].fillna(False).astype(bool)
    frame["buy_executable_t1_open"] = frame.get("buy_executable_t1_open", True)
    frame["sell_executable_t1_open"] = frame.get("sell_executable_t1_open", True)
    return frame[["ts_code", "next_amount", "buy_executable_t1_open", "sell_executable_t1_open"]].drop_duplicates("ts_code")


def build_live_day(predictions: pd.DataFrame, liquidity: pd.DataFrame, positions: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    preds = normalize_code_column(predictions)
    preds = preds[preds["trade_date"].astype(str).str.replace("-", "", regex=False).eq(trade_date)].copy()
    if preds.empty:
        die(f"no live predictions for trade_date={trade_date}")
    preds["pred_score"] = pd.to_numeric(preds["pred_score"], errors="coerce")
    day = liquidity.merge(preds[["ts_code", "pred_score"]], on="ts_code", how="outer")
    current_codes = positions["ts_code"].astype(str).unique().tolist()
    missing_current = sorted(set(current_codes) - set(day["ts_code"].astype(str)))
    if missing_current:
        # 当前持仓必须进入候选宇宙，否则卖出/继承约束会断裂。
        day = pd.concat(
            [
                day,
                pd.DataFrame(
                    {
                        "ts_code": missing_current,
                        "next_amount": 0.0,
                        "buy_executable_t1_open": False,
                        "sell_executable_t1_open": True,
                        "pred_score": np.nan,
                    }
                ),
            ],
            ignore_index=True,
        )
    day["trade_date"] = trade_date
    day["execution_return_open_to_close5"] = 0.0
    day["benchmark_next_open_to_exit_close_return"] = 0.0
    day["next_amount"] = pd.to_numeric(day["next_amount"], errors="coerce").fillna(0.0)
    day["buy_executable_t1_open"] = day["buy_executable_t1_open"].fillna(False).astype(bool)
    day["sell_executable_t1_open"] = day["sell_executable_t1_open"].fillna(True).astype(bool)
    return day


def main() -> None:
    args = parse_args()
    trade_date = str(args.trade_date)
    config = load_yaml(args.config)
    prev_trade_date = previous_trading_day(config, trade_date)
    paths = config["live_inputs"]

    pred_path = resolve_path(args.predictions) if args.predictions else resolve_path(config["outputs"]["predictions_dir"]) / f"predictions_{trade_date}.parquet"
    pos_path = resolve_path(args.positions) if args.positions else format_path(paths["positions"], trade_date=trade_date, prev_trade_date=prev_trade_date)
    prev_pos_path = (
        resolve_path(args.previous_close_positions)
        if args.previous_close_positions
        else format_path(paths["previous_close_positions"], trade_date=trade_date, prev_trade_date=prev_trade_date)
    )
    liq_path = resolve_path(args.liquidity_parquet) if args.liquidity_parquet else format_path(paths["feature_panel"], trade_date=trade_date, prev_trade_date=prev_trade_date)

    if not pred_path.exists():
        die(f"missing live predictions: {pred_path}")
    predictions = pd.read_parquet(pred_path)
    current_positions = load_positions(pos_path, "current positions")
    previous_positions = load_positions(prev_pos_path, "previous close positions")
    assert_position_inheritance(current_positions, previous_positions, config)

    liquidity = load_live_liquidity(liq_path, trade_date)
    day = build_live_day(predictions, liquidity, current_positions, trade_date)
    current = dict(zip(current_positions["ts_code"].astype(str), current_positions["weight"].astype(float)))

    opt = config["optimizer"]
    shortfall_penalty = dynamic_shortfall_penalty(config, trade_date)
    risk_cols: list[str] = []
    universe = prepare_lp_universe(
        day=day,
        current=current,
        risk_cols=risk_cols,
        k=int(opt["k"]),
        candidate_multiplier=float(opt["candidate_multiplier"]),
        min_invested=float(opt["min_invested"]),
        single_name_cap=float(opt["single_name_cap"]),
        portfolio_nav=float(opt["portfolio_nav"]),
        participation_cap=float(opt["participation_cap"]),
    )
    weights, stats = solve_day_lp(
        universe=universe,
        current=current,
        risk_cols=risk_cols,
        k=int(opt["k"]),
        style_penalty=float(opt["style_penalty"]),
        turnover_penalty=float(opt["turnover_penalty"]),
        exposure_cap=float(opt["exposure_cap"]),
        single_name_cap=float(opt["single_name_cap"]),
        min_invested=float(opt["min_invested"]),
        turnover_cap=float(opt["turnover_cap"]),
        portfolio_nav=float(opt["portfolio_nav"]),
        participation_cap=float(opt["participation_cap"]),
        exposure_slack_penalty=float(opt["exposure_slack_penalty"]),
        buy_capacity_slack_penalty=float(opt["buy_capacity_slack_penalty"]),
        cash_penalty=float(opt["cash_penalty"]),
        min_invested_shortfall_penalty=shortfall_penalty,
        solver=str(opt["solver"]),
    )

    all_codes = sorted(set(current) | set(weights))
    target = pd.DataFrame({"ts_code": all_codes})
    target["current_weight"] = target["ts_code"].map(current).fillna(0.0)
    target["target_weight"] = target["ts_code"].map(weights).fillna(0.0)
    target["delta_weight"] = target["target_weight"] - target["current_weight"]
    target = target.merge(predictions[["ts_code", "pred_score"]], on="ts_code", how="left")
    target = target.sort_values("delta_weight", ascending=False)

    out_dir = resolve_path(config["outputs"]["targets_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"target_weights_{trade_date}.csv"
    target.to_csv(out_csv, index=False)
    write_json(
        out_dir / f"manifest_{trade_date}.json",
        {
            "trade_date": trade_date,
            "prev_trade_date": prev_trade_date,
            "target_weights": str(out_csv),
            "optimizer": opt,
            "time_decay_shortfall_penalty": shortfall_penalty,
            "stats": stats,
            "target_invested_weight": float(target["target_weight"].sum()),
            "rows": int(len(target)),
        },
    )

    print("\n【阶段二完成】CVXPY live optimizer")
    print(f"trade_date={trade_date} output={out_csv}")
    print(f"time_decay_shortfall_penalty={shortfall_penalty:.2f}")
    print(f"status={stats['optimizer_status']} invested={stats['invested_weight']:.4f} cash={stats['cash_weight']:.4f}")
    if stats.get("min_invested_shortfall", 0.0) > 1e-8:
        print("\a【仓位警报】min_invested 未完全达到，已用动态短缺惩罚生成最大可达组合，请优先执行买单。")
    print("\n目标权重变化 Top 20：")
    print(target.head(20)[["ts_code", "current_weight", "target_weight", "delta_weight", "pred_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
