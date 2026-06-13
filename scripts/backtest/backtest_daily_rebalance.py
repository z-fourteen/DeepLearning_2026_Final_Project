from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = {"trade_date", "ts_code", "pred_score", "split"}
DAILY_COLUMNS = {"trade_date", "ts_code", "open", "close", "pre_close"}


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run daily T+1 open rebalance backtest from model predictions and raw A-share daily bars."
    )
    parser.add_argument("--predictions", required=True, help="Model predictions parquet/csv with trade_date, ts_code, pred_score, split.")
    parser.add_argument(
        "--daily-root",
        default="",
        help="Folder containing raw A-share daily CSV files named as YYYYMMDD.csv. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--market-root",
        default="",
        help="Folder containing benchmark market CSV files. Auto-detected when omitted.",
    )
    parser.add_argument("--benchmark-code", default="399006.SZ")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("10,20,30"))
    parser.add_argument(
        "--mode",
        choices=["full_topk", "rotate"],
        default="full_topk",
        help="full_topk fully rebalances to top K every day; rotate sells lowest-scored current names and refills.",
    )
    parser.add_argument("--replace-count", type=int, default=3, help="Daily replacement count for --mode rotate.")
    parser.add_argument("--split", default="test", help="Split to run, or all.")
    parser.add_argument("--start-date", default="", help="Inclusive signal date filter, e.g. 20240101.")
    parser.add_argument("--end-date", default="", help="Inclusive signal date filter, e.g. 20260529.")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--min-daily-count", type=int, default=40)
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


def validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalize_date(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "")[:8]


def load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported prediction file type: {path.suffix}")
    validate_columns(frame, PREDICTION_COLUMNS, "predictions")
    frame = frame.copy()
    frame["trade_date"] = frame["trade_date"].map(normalize_date)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=["trade_date", "ts_code", "split", "pred_score"])


def daily_dates(daily_root: Path) -> list[str]:
    dates = []
    for path in daily_root.glob("*.csv"):
        stem = path.stem
        if len(stem) == 8 and stem.isdigit():
            dates.append(stem)
    dates = sorted(set(dates))
    if not dates:
        raise FileNotFoundError(f"No daily csv files found under {daily_root}")
    return dates


def find_ashare_data_root(project_root: Path) -> Path:
    candidates = [
        project_root.parent / "Final-OXX2",
        project_root / "Final-OXX2",
        Path.cwd().parent / "Final-OXX2",
        Path.cwd() / "Final-OXX2",
    ]
    for base in candidates:
        if not base.exists():
            continue
        if (base / "daily").is_dir() and (base / "market").is_dir():
            return base
        for child in base.iterdir():
            if child.is_dir() and (child / "daily").is_dir() and (child / "market").is_dir():
                return child
    raise FileNotFoundError(
        "Could not auto-detect A-share data root. Please pass --daily-root and --market-root explicitly."
    )


def next_trade_date_map(signal_dates: list[str], trading_dates: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    j = 0
    for signal_date in sorted(set(signal_dates)):
        while j < len(trading_dates) and trading_dates[j] <= signal_date:
            j += 1
        if j < len(trading_dates):
            mapping[signal_date] = trading_dates[j]
    return mapping


def load_daily_bars(daily_root: Path, dates: list[str]) -> pd.DataFrame:
    frames = []
    for date in sorted(set(dates)):
        path = daily_root / f"{date}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        validate_columns(frame, DAILY_COLUMNS, str(path))
        frame = frame[list(DAILY_COLUMNS)].copy()
        frame["trade_date"] = frame["trade_date"].map(normalize_date)
        frame["ts_code"] = frame["ts_code"].astype(str)
        for column in ["open", "close", "pre_close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("None of the required execution-date daily csv files could be loaded.")
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.replace([np.inf, -np.inf], np.nan)
    bars = bars.dropna(subset=["trade_date", "ts_code", "open", "close", "pre_close"])
    return bars[(bars["open"] > 0) & (bars["close"] > 0) & (bars["pre_close"] > 0)]


def load_benchmark(market_root: Path, benchmark_code: str, execution_dates: list[str]) -> pd.DataFrame:
    path = market_root / f"{benchmark_code}.csv"
    if not path.exists():
        return pd.DataFrame({"execution_date": execution_dates, "benchmark_return": np.nan})
    frame = pd.read_csv(path)
    validate_columns(frame, {"trade_date", "close", "pre_close"}, str(path))
    frame = frame[["trade_date", "close", "pre_close"]].copy()
    frame["execution_date"] = frame["trade_date"].map(normalize_date)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["pre_close"] = pd.to_numeric(frame["pre_close"], errors="coerce")
    frame["benchmark_return"] = frame["close"] / frame["pre_close"] - 1.0
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame[["execution_date", "benchmark_return"]]


def weighted_return(weights: dict[str, float], returns: dict[str, float]) -> float:
    if not weights:
        return 0.0
    return float(sum(weight * returns.get(code, 0.0) for code, weight in weights.items()))


def apply_return(weights: dict[str, float], returns: dict[str, float]) -> dict[str, float]:
    updated = {code: weight * (1.0 + returns.get(code, 0.0)) for code, weight in weights.items()}
    total = sum(updated.values())
    if total <= 0:
        return {}
    return {code: weight / total for code, weight in updated.items() if weight > 1e-12}


def choose_target_codes(
    ranked_codes: list[str],
    current_codes: list[str],
    scores: dict[str, float],
    k: int,
    mode: str,
    replace_count: int,
) -> list[str]:
    if mode == "full_topk" or not current_codes:
        return ranked_codes[:k]

    current_with_scores = [(code, scores.get(code, -np.inf)) for code in current_codes if code in ranked_codes]
    current_with_scores = sorted(current_with_scores, key=lambda item: item[1])
    sell_count = min(max(replace_count, 0), len(current_with_scores), k)
    sell_codes = {code for code, _ in current_with_scores[:sell_count]}
    selected = [code for code in current_codes if code in ranked_codes and code not in sell_codes]
    selected_set = set(selected)
    for code in ranked_codes:
        if len(selected) >= k:
            break
        if code not in selected_set:
            selected.append(code)
            selected_set.add(code)
    return selected[:k]


def target_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    weight = 1.0 / len(codes)
    return {code: weight for code in codes}


def turnover_between(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    codes = set(old_weights) | set(new_weights)
    return float(sum(abs(new_weights.get(code, 0.0) - old_weights.get(code, 0.0)) for code in codes))


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def summarize(periods: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    detail: dict[str, Any] = {}
    if periods.empty:
        return pd.DataFrame(), detail
    group_cols = ["split", "mode", "k"]
    for key, group in periods.groupby(group_cols, sort=True):
        split, mode, k = key
        daily_return = group["daily_return"].astype(float)
        benchmark_return = group["benchmark_return"].astype(float)
        excess_return = daily_return - benchmark_return.fillna(0.0)
        cumulative_return = float((1.0 + daily_return).prod() - 1.0)
        benchmark_cumulative_return = float((1.0 + benchmark_return.fillna(0.0)).prod() - 1.0)
        std = float(daily_return.std(ddof=1)) if len(daily_return) > 1 else 0.0
        annualized_return = float((1.0 + cumulative_return) ** (252.0 / len(daily_return)) - 1.0)
        annualized_vol = float(std * math.sqrt(252.0))
        sharpe_like = annualized_return / annualized_vol if annualized_vol > 0 else None
        row = {
            "split": split,
            "mode": mode,
            "k": int(k),
            "period_count": int(len(group)),
            "cumulative_return": cumulative_return,
            "benchmark_cumulative_return": benchmark_cumulative_return,
            "excess_cumulative_return": cumulative_return - benchmark_cumulative_return,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_like": sharpe_like,
            "max_drawdown": max_drawdown(daily_return),
            "win_rate": float((daily_return > 0).mean()),
            "mean_daily_return": float(daily_return.mean()),
            "mean_excess_return": float(excess_return.mean()),
            "avg_turnover": float(group["turnover"].mean()),
            "avg_transaction_cost": float(group["transaction_cost"].mean()),
            "avg_position_count": float(group["position_count"].mean()),
        }
        rows.append(row)
        detail[f"{split}|{mode}|k={k}"] = row
    return pd.DataFrame(rows), detail


def run_backtest(
    predictions: pd.DataFrame,
    daily_bars: pd.DataFrame,
    benchmark: pd.DataFrame,
    signal_to_execution: dict[str, str],
    k_values: list[int],
    mode: str,
    replace_count: int,
    selected_split: str,
    min_daily_count: int,
    total_cost_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    bars_by_date = {date: day.copy() for date, day in daily_bars.groupby("trade_date", sort=False)}
    benchmark_by_date = benchmark.set_index("execution_date")["benchmark_return"].to_dict()

    if selected_split.lower() != "all":
        predictions = predictions[predictions["split"] == selected_split]
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    for split, split_frame in predictions.groupby("split", sort=True):
        signal_dates = sorted(date for date in split_frame["trade_date"].unique() if date in signal_to_execution)
        for k in k_values:
            current_weights: dict[str, float] = {}
            for signal_date in signal_dates:
                execution_date = signal_to_execution[signal_date]
                day_bars = bars_by_date.get(execution_date)
                if day_bars is None or len(day_bars) < min_daily_count:
                    continue

                day_signal = split_frame[split_frame["trade_date"] == signal_date]
                day_signal = day_signal.merge(day_bars, on="ts_code", how="inner", suffixes=("", "_bar"))
                if len(day_signal) < min_daily_count:
                    continue

                overnight_returns = (day_bars.set_index("ts_code")["open"] / day_bars.set_index("ts_code")["pre_close"] - 1.0).to_dict()
                intraday_returns = (day_bars.set_index("ts_code")["close"] / day_bars.set_index("ts_code")["open"] - 1.0).to_dict()
                overnight_return = weighted_return(current_weights, overnight_returns)
                weights_at_open = apply_return(current_weights, overnight_returns)

                ordered = day_signal.sort_values(["pred_score", "ts_code"], ascending=[False, True])
                ranked_codes = ordered["ts_code"].astype(str).tolist()
                scores = dict(zip(ordered["ts_code"].astype(str), ordered["pred_score"].astype(float)))
                selected_codes = choose_target_codes(
                    ranked_codes=ranked_codes,
                    current_codes=list(weights_at_open),
                    scores=scores,
                    k=k,
                    mode=mode,
                    replace_count=replace_count,
                )
                next_weights_at_open = target_weights(selected_codes)
                turnover = turnover_between(weights_at_open, next_weights_at_open)
                transaction_cost = turnover * total_cost_rate
                intraday_return = weighted_return(next_weights_at_open, intraday_returns)
                daily_return = (1.0 + overnight_return) * (1.0 - transaction_cost) * (1.0 + intraday_return) - 1.0
                current_weights = apply_return(next_weights_at_open, intraday_returns)

                periods.append(
                    {
                        "split": split,
                        "mode": mode,
                        "k": k,
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "position_count": len(current_weights),
                        "turnover": turnover,
                        "transaction_cost": transaction_cost,
                        "overnight_return": overnight_return,
                        "intraday_return": intraday_return,
                        "daily_return": daily_return,
                        "benchmark_return": benchmark_by_date.get(execution_date, np.nan),
                        "top_selected": ",".join(selected_codes),
                    }
                )
                for code, weight in sorted(current_weights.items()):
                    positions.append(
                        {
                            "split": split,
                            "mode": mode,
                            "k": k,
                            "signal_date": signal_date,
                            "execution_date": execution_date,
                            "ts_code": code,
                            "close_weight": weight,
                            "pred_score": scores.get(code, np.nan),
                        }
                    )

    return pd.DataFrame(periods), pd.DataFrame(positions)


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    project_root = Path(__file__).resolve().parents[2]
    data_root = find_ashare_data_root(project_root) if not args.daily_root or not args.market_root else None
    daily_root = Path(args.daily_root) if args.daily_root else data_root / "daily"
    market_root = Path(args.market_root) if args.market_root else data_root / "market"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(predictions_path)
    if args.start_date:
        predictions = predictions[predictions["trade_date"] >= normalize_date(args.start_date)]
    if args.end_date:
        predictions = predictions[predictions["trade_date"] <= normalize_date(args.end_date)]

    trading_dates = daily_dates(daily_root)
    signal_to_execution = next_trade_date_map(predictions["trade_date"].tolist(), trading_dates)
    execution_dates = sorted(set(signal_to_execution.values()))
    daily_bars = load_daily_bars(daily_root, execution_dates)
    benchmark = load_benchmark(market_root, args.benchmark_code, execution_dates)
    total_cost_rate = (args.cost_bps + args.slippage_bps) / 10000.0

    periods, positions = run_backtest(
        predictions=predictions,
        daily_bars=daily_bars,
        benchmark=benchmark,
        signal_to_execution=signal_to_execution,
        k_values=args.k,
        mode=args.mode,
        replace_count=args.replace_count,
        selected_split=args.split,
        min_daily_count=args.min_daily_count,
        total_cost_rate=total_cost_rate,
    )
    summary_frame, summary = summarize(periods)

    periods.to_csv(output_dir / "daily_rebalance_periods.csv", index=False)
    positions.to_csv(output_dir / "daily_rebalance_positions.csv", index=False)
    summary_frame.to_csv(output_dir / "daily_rebalance_summary.csv", index=False)
    payload = {
        "inputs": {
            "predictions": str(predictions_path),
            "daily_root": str(daily_root),
            "market_root": str(market_root),
            "benchmark_code": args.benchmark_code,
            "k": args.k,
            "mode": args.mode,
            "replace_count": args.replace_count,
            "split": args.split,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "cost_bps": args.cost_bps,
            "slippage_bps": args.slippage_bps,
            "min_daily_count": args.min_daily_count,
        },
        "summary": summary,
    }
    (output_dir / "daily_rebalance_metrics.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
