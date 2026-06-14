from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATES = "20260601,20260602,20260603,20260604,20260605,20260608,20260609,20260610,20260611,20260612"


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return sorted(set(values))


def parse_str_list(value: str) -> list[str]:
    values = [item.strip().replace("-", "") for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated dates.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate same-day live predictions on the 2026-06-01 to 2026-06-12 trading window."
    )
    parser.add_argument("--predictions-dir", default="../Final-YXR/outputs/live_predictions")
    parser.add_argument("--daily-root", default="../Final-OXX2/A股数据/daily")
    parser.add_argument("--market-root", default="../Final-OXX2/A股数据/market")
    parser.add_argument("--benchmark-code", default="399006.SZ")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dates", type=parse_str_list, default=parse_str_list(DEFAULT_DATES))
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--strategy", choices=["daily", "5d", "both"], default="both")
    parser.add_argument("--rebalance-stride", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--min-daily-count", type=int, default=40)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    return parser.parse_args()


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


def normalize_date(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "")[:8]


def load_prediction_file(path: Path, trade_date: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported prediction file type: {path}")
    required = {"trade_date", "ts_code", "pred_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.copy()
    frame["trade_date"] = frame["trade_date"].map(normalize_date)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="coerce")
    frame = frame.dropna(subset=["ts_code", "pred_score"])
    return frame[frame["trade_date"].eq(trade_date)]


def load_predictions(predictions_dir: Path, dates: list[str], allow_missing: bool) -> pd.DataFrame:
    frames = []
    missing_dates = []
    for date in dates:
        csv_path = predictions_dir / f"predictions_{date}.csv"
        parquet_path = predictions_dir / f"predictions_{date}.parquet"
        if csv_path.exists():
            frames.append(load_prediction_file(csv_path, date))
        elif parquet_path.exists():
            frames.append(load_prediction_file(parquet_path, date))
        else:
            missing_dates.append(date)
    if missing_dates and not allow_missing:
        raise FileNotFoundError(
            "Missing live prediction files for dates: "
            + ",".join(missing_dates)
            + ". Generate predictions_YYYYMMDD.csv first, or pass --allow-missing-predictions for partial evaluation."
        )
    if not frames:
        raise FileNotFoundError(f"No prediction files found under {predictions_dir}")
    return pd.concat(frames, ignore_index=True)


def load_daily_bars(daily_root: Path, dates: list[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        path = daily_root / f"{date}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing daily bar file: {path}")
        frame = pd.read_csv(path)
        required = {"trade_date", "ts_code", "open", "close", "pre_close"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        frame = frame[["trade_date", "ts_code", "open", "close", "pre_close"]].copy()
        frame["trade_date"] = frame["trade_date"].map(normalize_date)
        frame["ts_code"] = frame["ts_code"].astype(str)
        for column in ["open", "close", "pre_close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)
    bars = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    bars = bars.dropna(subset=["trade_date", "ts_code", "open", "close", "pre_close"])
    return bars[(bars["open"] > 0) & (bars["close"] > 0) & (bars["pre_close"] > 0)]


def load_benchmark(market_root: Path, benchmark_code: str, dates: list[str]) -> dict[str, float]:
    path = market_root / f"{benchmark_code}.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {"trade_date", "close", "pre_close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame[["trade_date", "close", "pre_close"]].copy()
    frame["trade_date"] = frame["trade_date"].map(normalize_date)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["pre_close"] = pd.to_numeric(frame["pre_close"], errors="coerce")
    frame["benchmark_return"] = frame["close"] / frame["pre_close"] - 1.0
    return frame[frame["trade_date"].isin(dates)].set_index("trade_date")["benchmark_return"].to_dict()


def weighted_return(weights: dict[str, float], returns: dict[str, float]) -> float:
    return float(sum(weight * returns.get(code, 0.0) for code, weight in weights.items()))


def apply_return(weights: dict[str, float], returns: dict[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    updated = {code: weight * (1.0 + returns.get(code, 0.0)) for code, weight in weights.items()}
    total = sum(updated.values())
    if total <= 0:
        return {}
    return {code: weight / total for code, weight in updated.items() if weight > 1e-12}


def target_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    weight = 1.0 / len(codes)
    return {code: weight for code in codes}


def turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    codes = set(old_weights) | set(new_weights)
    return float(sum(abs(new_weights.get(code, 0.0) - old_weights.get(code, 0.0)) for code in codes))


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def summarize(periods: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    detail: dict[str, Any] = {}
    if periods.empty:
        return pd.DataFrame(), detail
    for key, group in periods.groupby(["strategy", "k"], sort=True):
        strategy, k = key
        returns = group["daily_return"].astype(float)
        benchmark = group["benchmark_return"].astype(float).fillna(0.0)
        cumulative_return = float((1.0 + returns).prod() - 1.0)
        benchmark_cumulative_return = float((1.0 + benchmark).prod() - 1.0)
        annualized_return = float((1.0 + cumulative_return) ** (252.0 / len(returns)) - 1.0)
        annualized_vol = float(returns.std(ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.0
        row = {
            "strategy": strategy,
            "k": int(k),
            "period_count": int(len(group)),
            "start_date": str(group["trade_date"].min()),
            "end_date": str(group["trade_date"].max()),
            "cumulative_return": cumulative_return,
            "benchmark_cumulative_return": benchmark_cumulative_return,
            "excess_cumulative_return": cumulative_return - benchmark_cumulative_return,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_like": annualized_return / annualized_vol if annualized_vol > 0 else None,
            "max_drawdown": max_drawdown(returns),
            "win_rate": float((returns > 0).mean()),
            "avg_turnover": float(group["turnover"].mean()),
            "avg_transaction_cost": float(group["transaction_cost"].mean()),
            "avg_position_count": float(group["position_count"].mean()),
        }
        rows.append(row)
        detail[f"{strategy}|k={k}"] = row
    return pd.DataFrame(rows), detail


def run_strategy(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    benchmark: dict[str, float],
    dates: list[str],
    k_values: list[int],
    strategy: str,
    rebalance_stride: int,
    min_daily_count: int,
    total_cost_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    bars_by_date = {date: day.copy() for date, day in bars.groupby("trade_date", sort=False)}
    predictions_by_date = {date: day.copy() for date, day in predictions.groupby("trade_date", sort=False)}

    for k in k_values:
        current_weights: dict[str, float] = {}
        for date_index, date in enumerate(dates):
            day_bars = bars_by_date.get(date)
            if day_bars is None or len(day_bars) < min_daily_count:
                continue
            overnight_returns = (day_bars.set_index("ts_code")["open"] / day_bars.set_index("ts_code")["pre_close"] - 1.0).to_dict()
            intraday_returns = (day_bars.set_index("ts_code")["close"] / day_bars.set_index("ts_code")["open"] - 1.0).to_dict()
            overnight_return = weighted_return(current_weights, overnight_returns)
            weights_at_open = apply_return(current_weights, overnight_returns)

            should_rebalance = strategy == "daily" or date_index % rebalance_stride == 0 or not weights_at_open
            selected_codes: list[str] = list(weights_at_open)
            transaction_cost = 0.0
            day_turnover = 0.0
            if should_rebalance:
                day_predictions = predictions_by_date.get(date)
                if day_predictions is None:
                    raise FileNotFoundError(f"Strategy {strategy} needs live predictions for rebalance date {date}.")
                ranked = (
                    day_predictions.merge(day_bars[["ts_code"]], on="ts_code", how="inner")
                    .sort_values(["pred_score", "ts_code"], ascending=[False, True])
                )
                if len(ranked) < min_daily_count:
                    continue
                selected_codes = ranked["ts_code"].head(k).astype(str).tolist()
                next_weights_at_open = target_weights(selected_codes)
                day_turnover = turnover(weights_at_open, next_weights_at_open)
                transaction_cost = day_turnover * total_cost_rate
            else:
                next_weights_at_open = weights_at_open

            intraday_return = weighted_return(next_weights_at_open, intraday_returns)
            daily_return = (1.0 + overnight_return) * (1.0 - transaction_cost) * (1.0 + intraday_return) - 1.0
            current_weights = apply_return(next_weights_at_open, intraday_returns)
            periods.append(
                {
                    "strategy": strategy,
                    "k": k,
                    "trade_date": date,
                    "is_rebalance_day": should_rebalance,
                    "position_count": len(current_weights),
                    "turnover": day_turnover,
                    "transaction_cost": transaction_cost,
                    "overnight_return": overnight_return,
                    "intraday_return": intraday_return,
                    "daily_return": daily_return,
                    "benchmark_return": benchmark.get(date, np.nan),
                    "selected_codes": ",".join(selected_codes),
                }
            )
            for code, weight in sorted(current_weights.items()):
                positions.append(
                    {
                        "strategy": strategy,
                        "k": k,
                        "trade_date": date,
                        "ts_code": code,
                        "close_weight": weight,
                    }
                )
    return pd.DataFrame(periods), pd.DataFrame(positions)


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.predictions_dir)
    daily_root = Path(args.daily_root)
    market_root = Path(args.market_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = args.dates
    predictions = load_predictions(predictions_dir, dates, args.allow_missing_predictions)
    available_dates = sorted(set(predictions["trade_date"]))
    if args.allow_missing_predictions:
        dates = [date for date in dates if date in available_dates]
    bars = load_daily_bars(daily_root, dates)
    benchmark = load_benchmark(market_root, args.benchmark_code, dates)
    total_cost_rate = (args.cost_bps + args.slippage_bps) / 10000.0
    strategies = ["daily", "5d"] if args.strategy == "both" else [args.strategy]

    all_periods = []
    all_positions = []
    for strategy in strategies:
        periods, positions = run_strategy(
            predictions=predictions,
            bars=bars,
            benchmark=benchmark,
            dates=dates,
            k_values=args.k,
            strategy=strategy,
            rebalance_stride=args.rebalance_stride,
            min_daily_count=args.min_daily_count,
            total_cost_rate=total_cost_rate,
        )
        all_periods.append(periods)
        all_positions.append(positions)
    periods = pd.concat(all_periods, ignore_index=True) if all_periods else pd.DataFrame()
    positions = pd.concat(all_positions, ignore_index=True) if all_positions else pd.DataFrame()
    summary_frame, summary = summarize(periods)

    periods.to_csv(output_dir / "live_same_day_periods.csv", index=False)
    positions.to_csv(output_dir / "live_same_day_positions.csv", index=False)
    summary_frame.to_csv(output_dir / "live_same_day_summary.csv", index=False)
    payload = {
        "inputs": {
            "predictions_dir": str(predictions_dir),
            "daily_root": str(daily_root),
            "market_root": str(market_root),
            "benchmark_code": args.benchmark_code,
            "dates": dates,
            "k": args.k,
            "strategy": args.strategy,
            "rebalance_stride": args.rebalance_stride,
            "cost_bps": args.cost_bps,
            "slippage_bps": args.slippage_bps,
            "min_daily_count": args.min_daily_count,
        },
        "summary": summary,
    }
    (output_dir / "live_same_day_metrics.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
