from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.portfolio.optimize_feasible_cash_buffer import prepare_lp_universe, solve_day_lp


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
        description="Counterfactual GRU live replay with internally rolled positions."
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--predictions-dir", default="outputs/live_predictions")
    parser.add_argument("--features-dir", default="data/live/features")
    parser.add_argument("--daily-root", default="A股数据/daily")
    parser.add_argument("--market-root", default="A股数据/market")
    parser.add_argument("--benchmark-code", default="399006.SZ")
    parser.add_argument("--output-dir", default="outputs/backtest/gru_live_strategy_replay")
    parser.add_argument("--dates", type=parse_str_list, default=parse_str_list(DEFAULT_DATES))
    parser.add_argument("--strategy", choices=["daily", "5d", "both"], default="both")
    parser.add_argument("--rebalance-stride", type=int, default=5)
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("10"))
    parser.add_argument("--initial-nav", type=float, default=1_000_000.0)
    parser.add_argument("--cost-bps", type=float)
    parser.add_argument("--slippage-bps", type=float)
    parser.add_argument("--min-daily-count", type=int)
    parser.add_argument("--solver")
    parser.add_argument(
        "--allow-feature-fallback",
        action="store_true",
        help="Use the latest available prior features_YYYYMMDD file when a requested date is missing.",
    )
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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_predictions(predictions_dir: Path, dates: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for date in dates:
        csv_path = predictions_dir / f"predictions_{date}.csv"
        parquet_path = predictions_dir / f"predictions_{date}.parquet"
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
        elif parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
        else:
            missing.append(date)
            continue
        required = {"trade_date", "ts_code", "pred_score"}
        absent = sorted(required - set(frame.columns))
        if absent:
            raise ValueError(f"{csv_path if csv_path.exists() else parquet_path} missing columns: {absent}")
        frame = frame.copy()
        frame["trade_date"] = frame["trade_date"].map(normalize_date)
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="coerce")
        frames.append(frame[frame["trade_date"].eq(date)].dropna(subset=["ts_code", "pred_score"]))
    if missing:
        raise FileNotFoundError(
            "Missing live prediction files for dates: "
            + ",".join(missing)
            + f" under {predictions_dir}. Run scripts/live/01_live_inference.py first."
        )
    if not frames:
        raise FileNotFoundError(f"No live prediction files found under {predictions_dir}")
    return pd.concat(frames, ignore_index=True)


def available_feature_dates(features_dir: Path) -> list[str]:
    dates: set[str] = set()
    for path in features_dir.glob("features_*.parquet"):
        stem = path.stem.replace("features_", "")
        if len(stem) == 8 and stem.isdigit():
            dates.add(stem)
    for path in features_dir.glob("features_*.csv"):
        stem = path.stem.replace("features_", "")
        if len(stem) == 8 and stem.isdigit():
            dates.add(stem)
    return sorted(dates)


def resolve_feature_file(features_dir: Path, date: str, allow_fallback: bool) -> tuple[Path, str]:
    for suffix in [".parquet", ".csv"]:
        path = features_dir / f"features_{date}{suffix}"
        if path.exists():
            return path, date
    if not allow_fallback:
        raise FileNotFoundError(f"Missing live feature file for date={date}: {features_dir}")
    prior = [item for item in available_feature_dates(features_dir) if item <= date]
    if not prior:
        raise FileNotFoundError(f"No fallback live feature file available for date={date}: {features_dir}")
    source_date = prior[-1]
    for suffix in [".parquet", ".csv"]:
        path = features_dir / f"features_{source_date}{suffix}"
        if path.exists():
            return path, source_date
    raise FileNotFoundError(f"No fallback live feature file available for date={date}: {features_dir}")


def load_features(features_dir: Path, dates: list[str], allow_fallback: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    feature_source_by_date: dict[str, str] = {}
    for date in dates:
        path, source_date = resolve_feature_file(features_dir, date, allow_fallback)
        feature_source_by_date[date] = source_date
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        frame = frame.copy()
        if "code" in frame.columns and "ts_code" not in frame.columns:
            frame = frame.rename(columns={"code": "ts_code"})
        if "trade_date" not in frame.columns or "ts_code" not in frame.columns:
            raise ValueError(f"Feature file for date={date} must contain trade_date and ts_code")
        frame["trade_date"] = frame["trade_date"].map(normalize_date)
        frame = frame[frame["trade_date"].eq(source_date)].copy()
        frame["trade_date"] = date
        frame["ts_code"] = frame["ts_code"].astype(str)
        if "amount" in frame.columns:
            frame["next_amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
        elif "next_amount" in frame.columns:
            frame["next_amount"] = pd.to_numeric(frame["next_amount"], errors="coerce").fillna(0.0)
        else:
            frame["next_amount"] = 0.0
        for column in ["buy_executable_t1_open", "sell_executable_t1_open"]:
            if column not in frame.columns:
                frame[column] = True
            frame[column] = frame[column].fillna(False).astype(bool)
        frames.append(
            frame[
                [
                    "trade_date",
                    "ts_code",
                    "next_amount",
                    "buy_executable_t1_open",
                    "sell_executable_t1_open",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True), feature_source_by_date


def load_daily_bars(daily_root: Path, dates: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date in dates:
        path = daily_root / f"{date}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing daily bar file: {path}")
        frame = pd.read_csv(path)
        required = {"trade_date", "ts_code", "open", "close", "pre_close"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
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
        raise ValueError(f"{path} missing columns: {missing}")
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


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def summarize(periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if periods.empty:
        return pd.DataFrame()
    for (strategy, k), group in periods.groupby(["strategy", "k"], sort=True):
        returns = group["daily_return"].astype(float)
        benchmark = group["benchmark_return"].astype(float).fillna(0.0)
        cumulative = float((1.0 + returns).prod() - 1.0)
        benchmark_cumulative = float((1.0 + benchmark).prod() - 1.0)
        annualized = float((1.0 + cumulative) ** (252.0 / len(returns)) - 1.0)
        vol = float(returns.std(ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.0
        rows.append(
            {
                "strategy": strategy,
                "k": int(k),
                "period_count": int(len(group)),
                "start_date": str(group["trade_date"].min()),
                "end_date": str(group["trade_date"].max()),
                "cumulative_return": cumulative,
                "benchmark_cumulative_return": benchmark_cumulative,
                "excess_cumulative_return": cumulative - benchmark_cumulative,
                "annualized_return": annualized,
                "annualized_vol": vol,
                "sharpe_like": annualized / vol if vol > 0 else None,
                "max_drawdown": max_drawdown(returns),
                "win_rate": float((returns > 0).mean()),
                "avg_desired_turnover": float(group["desired_turnover"].mean()),
                "avg_filled_turnover": float(group["filled_turnover"].mean()),
                "avg_transaction_cost": float(group["transaction_cost"].mean()),
                "avg_invested_weight": float(group["invested_weight"].mean()),
                "avg_cash_weight": float(group["cash_weight"].mean()),
                "avg_position_count": float(group["position_count"].mean()),
                "fallback_rate": float(group["fallback_used"].astype(bool).mean()),
                "final_nav": float(group["nav"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def run_replay(
    *,
    data: pd.DataFrame,
    bars: pd.DataFrame,
    benchmark: dict[str, float],
    dates: list[str],
    strategies: list[str],
    k_values: list[int],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    bars_by_date = {date: day.copy() for date, day in bars.groupby("trade_date", sort=False)}
    data_by_date = {date: day.copy() for date, day in data.groupby("trade_date", sort=False)}
    opt = dict(config["optimizer"])
    if args.cost_bps is not None:
        opt["cost_bps"] = args.cost_bps
    if args.slippage_bps is not None:
        opt["slippage_bps"] = args.slippage_bps
    if args.min_daily_count is not None:
        opt["min_daily_count"] = args.min_daily_count
    if args.solver is not None:
        opt["solver"] = args.solver
    total_cost_rate = (float(opt["cost_bps"]) + float(opt["slippage_bps"])) / 10000.0

    for strategy in strategies:
        for k in k_values:
            current: dict[str, float] = {}
            nav = float(args.initial_nav)
            for date_index, date in enumerate(dates):
                day_bars = bars_by_date.get(date)
                if day_bars is None or len(day_bars) < int(opt["min_daily_count"]):
                    continue
                overnight_returns = (
                    day_bars.set_index("ts_code")["open"] / day_bars.set_index("ts_code")["pre_close"] - 1.0
                ).to_dict()
                intraday_returns = (
                    day_bars.set_index("ts_code")["close"] / day_bars.set_index("ts_code")["open"] - 1.0
                ).to_dict()
                overnight_return = weighted_return(current, overnight_returns)
                weights_at_open = apply_return(current, overnight_returns)
                nav_at_open = nav * (1.0 + overnight_return)

                should_rebalance = strategy == "daily" or date_index % int(args.rebalance_stride) == 0 or not weights_at_open
                stats: dict[str, Any]
                if should_rebalance:
                    day = data_by_date.get(date)
                    if day is None or day["pred_score"].notna().sum() < int(opt["min_daily_count"]):
                        continue
                    day = day.merge(day_bars[["ts_code"]], on="ts_code", how="inner")
                    day = day.copy()
                    day["execution_return_open_to_close5"] = day["ts_code"].map(intraday_returns).fillna(0.0)
                    day["benchmark_next_open_to_exit_close_return"] = benchmark.get(date, np.nan)
                    current_invested = float(sum(max(0.0, weight) for weight in weights_at_open.values()))
                    turnover_cap = float(opt["turnover_cap"])
                    if current_invested <= 1e-8:
                        turnover_cap = max(turnover_cap, float(opt["min_invested"]))
                    universe = prepare_lp_universe(
                        day=day,
                        current=weights_at_open,
                        risk_cols=[],
                        k=int(k),
                        candidate_multiplier=float(opt["candidate_multiplier"]),
                        min_invested=float(opt["min_invested"]),
                        single_name_cap=float(opt["single_name_cap"]),
                        portfolio_nav=nav_at_open,
                        participation_cap=float(opt["participation_cap"]),
                    )
                    next_weights, stats = solve_day_lp(
                        universe=universe,
                        current=weights_at_open,
                        risk_cols=[],
                        k=int(k),
                        style_penalty=float(opt["style_penalty"]),
                        turnover_penalty=float(opt["turnover_penalty"]),
                        exposure_cap=float(opt["exposure_cap"]),
                        single_name_cap=float(opt["single_name_cap"]),
                        min_invested=float(opt["min_invested"]),
                        turnover_cap=turnover_cap,
                        portfolio_nav=nav_at_open,
                        participation_cap=float(opt["participation_cap"]),
                        exposure_slack_penalty=float(opt["exposure_slack_penalty"]),
                        buy_capacity_slack_penalty=float(opt["buy_capacity_slack_penalty"]),
                        cash_penalty=float(opt["cash_penalty"]),
                        min_invested_shortfall_penalty=0.0,
                        solver=str(opt["solver"]),
                    )
                else:
                    next_weights = weights_at_open
                    stats = {
                        "optimizer_status": "Hold",
                        "fallback_used": False,
                        "desired_turnover": 0.0,
                        "filled_turnover": 0.0,
                        "invested_weight": float(sum(next_weights.values())),
                        "cash_weight": float(max(0.0, 1.0 - sum(next_weights.values()))),
                        "position_count": int(len(next_weights)),
                    }

                cost = total_cost_rate * float(stats.get("filled_turnover", 0.0))
                intraday_return = weighted_return(next_weights, intraday_returns)
                daily_return = (1.0 + overnight_return) * (1.0 - cost) * (1.0 + intraday_return) - 1.0
                nav = nav * (1.0 + daily_return)
                current = apply_return(next_weights, intraday_returns)
                periods.append(
                    {
                        "strategy": strategy,
                        "k": int(k),
                        "trade_date": date,
                        "is_rebalance_day": bool(should_rebalance),
                        "nav": nav,
                        "overnight_return": overnight_return,
                        "intraday_return": intraday_return,
                        "daily_return": daily_return,
                        "benchmark_return": benchmark.get(date, np.nan),
                        "transaction_cost": cost,
                        **stats,
                    }
                )
                for code, weight in sorted(current.items()):
                    positions.append(
                        {
                            "strategy": strategy,
                            "k": int(k),
                            "trade_date": date,
                            "ts_code": code,
                            "close_weight": weight,
                            "nav": nav,
                        }
                    )
    return pd.DataFrame(periods), pd.DataFrame(positions)


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    dates = args.dates
    strategies = ["daily", "5d"] if args.strategy == "both" else [args.strategy]
    predictions = load_predictions(Path(args.predictions_dir), dates)
    features, feature_source_by_date = load_features(
        Path(args.features_dir),
        dates,
        allow_fallback=bool(args.allow_feature_fallback),
    )
    data = predictions.merge(features, on=["trade_date", "ts_code"], how="left")
    data["next_amount"] = pd.to_numeric(data["next_amount"], errors="coerce").fillna(0.0)
    bars = load_daily_bars(Path(args.daily_root), dates)
    benchmark = load_benchmark(Path(args.market_root), args.benchmark_code, dates)
    periods, positions = run_replay(
        data=data,
        bars=bars,
        benchmark=benchmark,
        dates=dates,
        strategies=strategies,
        k_values=args.k,
        config=config,
        args=args,
    )
    summary = summarize(periods)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    periods.to_csv(out_dir / "replay_periods.csv", index=False)
    positions.to_csv(out_dir / "replay_positions.csv", index=False)
    summary.to_csv(out_dir / "replay_summary.csv", index=False)
    manifest = {
        "config": args.config,
        "predictions_dir": args.predictions_dir,
        "features_dir": args.features_dir,
        "daily_root": args.daily_root,
        "market_root": args.market_root,
        "benchmark_code": args.benchmark_code,
        "dates": dates,
        "strategy": args.strategy,
        "k": args.k,
        "initial_nav": float(args.initial_nav),
        "feature_source_by_date": feature_source_by_date,
        "output_dir": str(out_dir),
        "period_rows": int(len(periods)),
        "position_rows": int(len(positions)),
        "summary_rows": int(len(summary)),
        "method": "gru_live_optimizer_counterfactual_replay",
    }
    (out_dir / "replay_metrics.json").write_text(
        json.dumps(json_safe({"manifest": manifest, "summary": summary.to_dict("records")}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
