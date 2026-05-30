from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = {"trade_date", "ts_code", "pred_score", "split"}
LABEL_COLUMNS = {
    "trade_date",
    "ts_code",
    "future_return",
    "benchmark_future_return",
    "label_rel_return",
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
    parser = argparse.ArgumentParser(
        description="Backtest Top-K portfolios with rank-buffer turnover control."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", default="data/mart/labels/labels_v20260526.parquet")
    parser.add_argument("--output-dir", help="Defaults to prediction parent / turnover_control.")
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("10,20,30"))
    parser.add_argument("--cost-bps", type=parse_float_list, default=parse_float_list("0,10,20"))
    parser.add_argument("--keep-multiplier", type=parse_float_list, default=parse_float_list("1,1.5,2,3"))
    parser.add_argument("--holding-days", type=int, default=5)
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


def load_data(predictions_path: Path, labels_path: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(predictions_path)
    labels = pd.read_parquet(labels_path)
    validate_columns(predictions, PREDICTION_COLUMNS, "predictions")
    validate_columns(labels, LABEL_COLUMNS, "labels")

    predictions = predictions.copy()
    labels = labels.copy()
    for frame in (predictions, labels):
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    predictions["pred_score"] = pd.to_numeric(predictions["pred_score"], errors="coerce")
    for column in ["future_return", "benchmark_future_return", "label_rel_return"]:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")

    merged = predictions.merge(
        labels[["trade_date", "ts_code", "future_return", "benchmark_future_return", "label_rel_return"]],
        on=["trade_date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.dropna(subset=["trade_date", "ts_code", "split", "pred_score", "future_return"])


def selected_signal_dates(dates: list[str], stride: int) -> list[str]:
    if stride <= 0:
        raise ValueError("--rebalance-stride must be positive.")
    return dates[::stride]


def equal_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    weight = 1.0 / len(codes)
    return {code: weight for code in codes}


def portfolio_turnover(current: dict[str, float], previous: dict[str, float] | None) -> float:
    if previous is None:
        return float(sum(abs(weight) for weight in current.values()))
    names = set(current) | set(previous)
    return float(sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names))


def select_with_rank_buffer(
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
    cost_bps_values: list[float],
    keep_multipliers: list[float],
    min_daily_count: int,
    rebalance_stride: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_frame in frame.groupby("split", sort=True):
        dates = selected_signal_dates(sorted(split_frame["trade_date"].unique()), rebalance_stride)
        previous_top: dict[tuple[int, float], list[str]] = {}
        previous_bottom: dict[tuple[int, float], list[str]] = {}

        for trade_date in dates:
            group = split_frame[split_frame["trade_date"] == trade_date].sort_values(
                "pred_score", ascending=False
            )
            n = int(len(group))
            if n < min_daily_count:
                continue
            by_code = group.set_index("ts_code")
            ordered_top = group["ts_code"].astype(str).tolist()
            ordered_bottom = group.sort_values("pred_score", ascending=True)["ts_code"].astype(str).tolist()
            universe_return = float(group["future_return"].mean())
            benchmark_return = float(group["benchmark_future_return"].mean())

            for k in k_values:
                if n < k:
                    continue
                for keep_multiplier in keep_multipliers:
                    keep_rank = min(n, max(k, int(math.ceil(k * keep_multiplier))))
                    key = (k, keep_multiplier)
                    top_codes = select_with_rank_buffer(
                        ordered_top,
                        previous_top.get(key, []),
                        k,
                        keep_rank,
                    )
                    bottom_codes = select_with_rank_buffer(
                        ordered_bottom,
                        previous_bottom.get(key, []),
                        k,
                        keep_rank,
                    )
                    top_weights = equal_weights(top_codes)
                    bottom_weights = equal_weights(bottom_codes)
                    previous_top_weights = equal_weights(previous_top[key]) if key in previous_top else None
                    previous_bottom_weights = equal_weights(previous_bottom[key]) if key in previous_bottom else None
                    top_turnover = portfolio_turnover(top_weights, previous_top_weights)
                    bottom_turnover = portfolio_turnover(bottom_weights, previous_bottom_weights)
                    previous_top[key] = top_codes
                    previous_bottom[key] = bottom_codes

                    top_return_gross = float(by_code.loc[top_codes, "future_return"].mean())
                    bottom_return_gross = float(by_code.loc[bottom_codes, "future_return"].mean())
                    long_short_gross = top_return_gross - bottom_return_gross

                    for cost_bps in cost_bps_values:
                        cost_rate = cost_bps / 10000.0
                        top_transaction_cost = cost_rate * top_turnover
                        bottom_transaction_cost = cost_rate * bottom_turnover
                        long_short_transaction_cost = top_transaction_cost + bottom_transaction_cost
                        top_return_net = top_return_gross - top_transaction_cost
                        bottom_short_return_net = -bottom_return_gross - bottom_transaction_cost
                        long_short_net = long_short_gross - long_short_transaction_cost
                        rows.append(
                            {
                                "split": split,
                                "trade_date": trade_date,
                                "k": int(k),
                                "keep_multiplier": float(keep_multiplier),
                                "keep_rank": int(keep_rank),
                                "cost_bps": float(cost_bps),
                                "daily_count": n,
                                "top_turnover": top_turnover,
                                "bottom_turnover": bottom_turnover,
                                "long_short_turnover": top_turnover + bottom_turnover,
                                "top_transaction_cost": top_transaction_cost,
                                "bottom_transaction_cost": bottom_transaction_cost,
                                "long_short_transaction_cost": long_short_transaction_cost,
                                "top_return_gross": top_return_gross,
                                "top_return_net": top_return_net,
                                "bottom_return_gross": bottom_return_gross,
                                "bottom_short_return_net": bottom_short_return_net,
                                "long_short_gross": long_short_gross,
                                "long_short_net": long_short_net,
                                "benchmark_return": benchmark_return,
                                "universe_equal_return": universe_return,
                                "top_excess_vs_benchmark_net": top_return_net - benchmark_return,
                                "top_excess_vs_universe_net": top_return_net - universe_return,
                            }
                        )
    return pd.DataFrame(rows)


def build_summary(periods: pd.DataFrame, holding_days: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if periods.empty:
        return summary
    periods_per_year = 252.0 / holding_days
    for split, split_frame in periods.groupby("split", sort=True):
        split_summary: dict[str, Any] = {}
        for (k, keep_multiplier, cost_bps), group in split_frame.groupby(
            ["k", "keep_multiplier", "cost_bps"], sort=True
        ):
            key = f"top_{int(k)}_keep_{keep_multiplier:g}x_cost_{cost_bps:g}bps"
            split_summary[key] = {
                "top_net": summarize_returns(group["top_return_net"], periods_per_year),
                "top_excess_vs_benchmark_net": summarize_returns(
                    group["top_excess_vs_benchmark_net"], periods_per_year
                ),
                "top_excess_vs_universe_net": summarize_returns(
                    group["top_excess_vs_universe_net"], periods_per_year
                ),
                "long_short_gross": summarize_returns(group["long_short_gross"], periods_per_year),
                "long_short_net": summarize_returns(group["long_short_net"], periods_per_year),
                "bottom_short_net": summarize_returns(group["bottom_short_return_net"], periods_per_year),
                "benchmark": summarize_returns(group["benchmark_return"], periods_per_year),
                "universe_equal": summarize_returns(group["universe_equal_return"], periods_per_year),
                "average_turnover": float(group["top_turnover"].mean()),
                "average_bottom_turnover": float(group["bottom_turnover"].mean()),
                "average_long_short_turnover": float(group["long_short_turnover"].mean()),
                "average_top_transaction_cost": float(group["top_transaction_cost"].mean()),
                "average_long_short_transaction_cost": float(group["long_short_transaction_cost"].mean()),
            }
        summary[str(split)] = split_summary
    return summary


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent / "turnover_control"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(predictions_path, labels_path)
    periods = run_backtest(
        data,
        k_values=args.k,
        cost_bps_values=args.cost_bps,
        keep_multipliers=args.keep_multiplier,
        min_daily_count=args.min_daily_count,
        rebalance_stride=args.rebalance_stride,
    )
    summary = {
        "predictions": str(predictions_path),
        "labels": str(labels_path),
        "rows_after_label_merge": int(len(data)),
        "k_values": args.k,
        "cost_bps": args.cost_bps,
        "keep_multiplier": args.keep_multiplier,
        "holding_days": int(args.holding_days),
        "rebalance_stride": int(args.rebalance_stride),
        "min_daily_count": int(args.min_daily_count),
        "method": "rank_buffer_turnover_control",
        "selection_rule": "retain existing Top-K names while their rank stays within ceil(K * keep_multiplier), then fill vacancies by score rank",
        "summary": build_summary(periods, args.holding_days),
    }
    periods.to_csv(output_dir / "turnover_control_periods.csv", index=False)
    (output_dir / "turnover_control_metrics.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
