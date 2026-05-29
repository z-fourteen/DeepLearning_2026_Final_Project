from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = {"trade_date", "ts_code", "pred_score", "split"}
EXECUTION_COLUMNS = {
    "trade_date",
    "ts_code",
    "next_trade_date",
    "exit_trade_date",
    "next_open",
    "next_amount",
    "buy_executable_t1_open",
    "sell_executable_t1_open",
    "execution_return_open_to_close5",
    "benchmark_next_open_to_exit_close_return",
}


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return sorted(set(values))


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated non-negative numbers.")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T+1 fill simulation backtest.")
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet",
    )
    parser.add_argument(
        "--execution-labels",
        default="data/mart/labels/execution_labels_v20260526.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/backtest/t1_fill_sim")
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--keep-multiplier", type=parse_float_list, default=parse_float_list("2"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Linear slippage bps applied to filled turnover. Conservative placeholder.",
    )
    parser.add_argument(
        "--portfolio-nav",
        type=float,
        default=10_000_000.0,
        help="Portfolio NAV in CNY-like units for participation checks.",
    )
    parser.add_argument(
        "--participation-cap",
        type=float,
        default=0.03,
        help="Max order notional as a fraction of next-day amount. Orders above cap are partially filled.",
    )
    parser.add_argument("--rebalance-stride", type=int, default=5)
    parser.add_argument("--min-daily-count", type=int, default=20)
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


def load_data(predictions_path: Path, execution_labels_path: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(predictions_path)
    labels = pd.read_parquet(execution_labels_path)
    validate_columns(predictions, PREDICTION_COLUMNS, "predictions")
    validate_columns(labels, EXECUTION_COLUMNS, "execution_labels")

    predictions = predictions.copy()
    labels = labels.copy()
    for frame in (predictions, labels):
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    predictions["pred_score"] = pd.to_numeric(predictions["pred_score"], errors="coerce")
    numeric_cols = [
        "next_open",
        "next_amount",
        "execution_return_open_to_close5",
        "benchmark_next_open_to_exit_close_return",
    ]
    for column in numeric_cols:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")
    for column in ["buy_executable_t1_open", "sell_executable_t1_open"]:
        labels[column] = labels[column].fillna(False).astype(bool)

    merged = predictions.merge(labels, on=["trade_date", "ts_code"], how="inner", validate="one_to_one")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.dropna(subset=["trade_date", "ts_code", "split", "pred_score"])


def selected_signal_dates(dates: list[str], stride: int) -> list[str]:
    if stride <= 0:
        raise ValueError("--rebalance-stride must be positive.")
    return dates[::stride]


def target_codes_with_rank_buffer(
    ordered_codes: list[str],
    previous_codes: list[str],
    k: int,
    keep_rank: int,
) -> list[str]:
    keep_set = set(ordered_codes[:keep_rank])
    selected = [code for code in previous_codes if code in keep_set]
    selected_set = set(selected)
    for code in ordered_codes:
        if len(selected) >= k:
            break
        if code not in selected_set:
            selected.append(code)
            selected_set.add(code)
    return selected[:k]


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {code: weight / total for code, weight in weights.items() if weight > 0}


def simulate_fill(
    current: dict[str, float],
    target_codes: list[str],
    day: pd.DataFrame,
    k: int,
    portfolio_nav: float,
    participation_cap: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    by_code = day.set_index("ts_code", drop=False)
    target_weight = 1.0 / k if k else 0.0
    desired = {code: target_weight for code in target_codes}
    all_codes = set(current) | set(desired)
    next_weights = dict(current)
    buy_reject = 0
    sell_reject = 0
    partial_fill_count = 0
    filled_buy_notional = 0.0
    filled_sell_notional = 0.0
    filled_turnover = 0.0
    desired_turnover = 0.0

    for code in sorted(all_codes):
        old_w = current.get(code, 0.0)
        target_w = desired.get(code, 0.0)
        delta = target_w - old_w
        if abs(delta) < 1e-12:
            next_weights[code] = old_w
            continue
        desired_turnover += abs(delta)
        if code not in by_code.index:
            if delta < 0:
                sell_reject += 1
                next_weights[code] = old_w
            continue
        row = by_code.loc[code]
        amount = float(row["next_amount"]) if pd.notna(row["next_amount"]) else 0.0
        max_weight_fill = (participation_cap * amount / portfolio_nav) if portfolio_nav > 0 else abs(delta)
        max_weight_fill = max(0.0, min(abs(delta), max_weight_fill))

        if delta > 0:
            if not bool(row["buy_executable_t1_open"]) or max_weight_fill <= 0:
                buy_reject += 1
                next_weights[code] = old_w
                continue
            fill = min(delta, max_weight_fill)
            if fill < delta - 1e-12:
                partial_fill_count += 1
            next_weights[code] = old_w + fill
            filled_buy_notional += fill
            filled_turnover += abs(fill)
        else:
            if not bool(row["sell_executable_t1_open"]) or max_weight_fill <= 0:
                sell_reject += 1
                next_weights[code] = old_w
                continue
            fill = min(-delta, max_weight_fill)
            if fill < -delta - 1e-12:
                partial_fill_count += 1
            next_weights[code] = old_w - fill
            filled_sell_notional += fill
            filled_turnover += abs(fill)

    next_weights = {code: weight for code, weight in next_weights.items() if weight > 1e-10}
    next_weights = normalize_weights(next_weights)
    stats = {
        "desired_turnover": desired_turnover,
        "filled_turnover": filled_turnover,
        "filled_buy_notional_weight": filled_buy_notional,
        "filled_sell_notional_weight": filled_sell_notional,
        "buy_reject_count": buy_reject,
        "sell_reject_count": sell_reject,
        "partial_fill_count": partial_fill_count,
        "position_count": len(next_weights),
        "cash_weight_before_normalize": max(0.0, 1.0 - sum(next_weights.values())),
    }
    return next_weights, stats


def weighted_return(weights: dict[str, float], day: pd.DataFrame) -> float:
    if not weights:
        return float("nan")
    by_code = day.set_index("ts_code")
    value = 0.0
    used_weight = 0.0
    for code, weight in weights.items():
        if code not in by_code.index:
            continue
        ret = by_code.loc[code, "execution_return_open_to_close5"]
        if pd.isna(ret):
            continue
        value += weight * float(ret)
        used_weight += weight
    if used_weight <= 0:
        return float("nan")
    return value / used_weight


def max_drawdown(returns: pd.Series) -> float:
    clean = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    equity = (1.0 + clean).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def summarize_returns(returns: pd.Series, periods_per_year: float) -> dict[str, float | int]:
    clean = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "period_count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "win_rate": float("nan"),
            "cumulative_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_vol": float("nan"),
            "sharpe_like": float("nan"),
            "max_drawdown": float("nan"),
        }
    std = clean.std(ddof=1) if len(clean) > 1 else float("nan")
    cumulative = float((1.0 + clean).prod() - 1.0)
    annualized_return = float((1.0 + cumulative) ** (periods_per_year / len(clean)) - 1.0)
    annualized_vol = float(std * math.sqrt(periods_per_year)) if math.isfinite(std) else float("nan")
    return {
        "period_count": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(std),
        "ir": float(clean.mean() / std) if std and math.isfinite(std) else float("nan"),
        "win_rate": float((clean > 0).mean()),
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe_like": float(annualized_return / annualized_vol)
        if annualized_vol and math.isfinite(annualized_vol)
        else float("nan"),
        "max_drawdown": max_drawdown(clean),
    }


def run_backtest(
    frame: pd.DataFrame,
    k_values: list[int],
    keep_multipliers: list[float],
    cost_bps: float,
    slippage_bps: float,
    portfolio_nav: float,
    participation_cap: float,
    rebalance_stride: int,
    min_daily_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_cost_rate = (cost_bps + slippage_bps) / 10000.0
    for split, split_frame in frame.groupby("split", sort=True):
        dates = selected_signal_dates(sorted(split_frame["trade_date"].unique()), rebalance_stride)
        current_by_key: dict[tuple[int, float], dict[str, float]] = {
            (k, keep): {} for k in k_values for keep in keep_multipliers
        }
        for trade_date in dates:
            day = split_frame[split_frame["trade_date"] == trade_date].sort_values(
                "pred_score", ascending=False
            )
            if len(day) < min_daily_count:
                continue
            ordered_codes = day["ts_code"].astype(str).tolist()
            benchmark_return = float(day["benchmark_next_open_to_exit_close_return"].dropna().iloc[0])
            executable_universe_return = float(
                day.loc[day["buy_executable_t1_open"], "execution_return_open_to_close5"].mean()
            )
            for k in k_values:
                if len(day) < k:
                    continue
                for keep in keep_multipliers:
                    key = (k, keep)
                    keep_rank = min(len(day), max(k, int(math.ceil(k * keep))))
                    previous_codes = list(current_by_key[key].keys())
                    target_codes = target_codes_with_rank_buffer(
                        ordered_codes, previous_codes, k, keep_rank
                    )
                    weights, fill_stats = simulate_fill(
                        current=current_by_key[key],
                        target_codes=target_codes,
                        day=day,
                        k=k,
                        portfolio_nav=portfolio_nav,
                        participation_cap=participation_cap,
                    )
                    current_by_key[key] = weights
                    gross_return = weighted_return(weights, day)
                    transaction_cost = total_cost_rate * fill_stats["filled_turnover"]
                    net_return = gross_return - transaction_cost if pd.notna(gross_return) else float("nan")
                    rows.append(
                        {
                            "split": split,
                            "trade_date": trade_date,
                            "k": int(k),
                            "keep_multiplier": float(keep),
                            "keep_rank": int(keep_rank),
                            "daily_count": int(len(day)),
                            "gross_return": gross_return,
                            "net_return": net_return,
                            "benchmark_return": benchmark_return,
                            "executable_universe_return": executable_universe_return,
                            "excess_vs_benchmark": net_return - benchmark_return
                            if pd.notna(net_return)
                            else float("nan"),
                            "excess_vs_executable_universe": net_return - executable_universe_return
                            if pd.notna(net_return)
                            else float("nan"),
                            "transaction_cost": transaction_cost,
                            "cost_bps": float(cost_bps),
                            "slippage_bps": float(slippage_bps),
                            "portfolio_nav": float(portfolio_nav),
                            "participation_cap": float(participation_cap),
                            **fill_stats,
                        }
                    )
    return pd.DataFrame(rows)


def build_summary(periods: pd.DataFrame, holding_days: int = 5) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if periods.empty:
        return summary
    periods_per_year = 252.0 / holding_days
    for split, split_frame in periods.groupby("split", sort=True):
        split_summary: dict[str, Any] = {}
        for (k, keep), group in split_frame.groupby(["k", "keep_multiplier"], sort=True):
            key = f"top_{int(k)}_keep_{keep:g}x"
            split_summary[key] = {
                "net": summarize_returns(group["net_return"], periods_per_year),
                "gross": summarize_returns(group["gross_return"], periods_per_year),
                "excess_vs_benchmark": summarize_returns(group["excess_vs_benchmark"], periods_per_year),
                "excess_vs_executable_universe": summarize_returns(
                    group["excess_vs_executable_universe"], periods_per_year
                ),
                "benchmark": summarize_returns(group["benchmark_return"], periods_per_year),
                "executable_universe": summarize_returns(
                    group["executable_universe_return"], periods_per_year
                ),
                "average_desired_turnover": float(group["desired_turnover"].mean()),
                "average_filled_turnover": float(group["filled_turnover"].mean()),
                "average_transaction_cost": float(group["transaction_cost"].mean()),
                "average_buy_reject_count": float(group["buy_reject_count"].mean()),
                "average_sell_reject_count": float(group["sell_reject_count"].mean()),
                "average_partial_fill_count": float(group["partial_fill_count"].mean()),
                "average_position_count": float(group["position_count"].mean()),
            }
        summary[str(split)] = split_summary
    return summary


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    execution_labels_path = Path(args.execution_labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(predictions_path, execution_labels_path)
    periods = run_backtest(
        frame=data,
        k_values=args.k,
        keep_multipliers=args.keep_multiplier,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        portfolio_nav=args.portfolio_nav,
        participation_cap=args.participation_cap,
        rebalance_stride=args.rebalance_stride,
        min_daily_count=args.min_daily_count,
    )
    summary = {
        "predictions": str(predictions_path),
        "execution_labels": str(execution_labels_path),
        "rows_after_merge": int(len(data)),
        "k": args.k,
        "keep_multiplier": args.keep_multiplier,
        "cost_bps": float(args.cost_bps),
        "slippage_bps": float(args.slippage_bps),
        "portfolio_nav": float(args.portfolio_nav),
        "participation_cap": float(args.participation_cap),
        "rebalance_stride": int(args.rebalance_stride),
        "method": "t1_open_fill_simulation_with_rank_buffer",
        "summary": build_summary(periods),
    }
    periods.to_csv(output_dir / "t1_fill_periods.csv", index=False)
    (output_dir / "t1_fill_metrics.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
