"""Staggered multi-bucket rebalancing backtest (rolling rebalance strategy).

Strategy overview:
    - Split portfolio into N sub-buckets (e.g., N=5)
    - Each day, rotate ONE bucket: sell old positions → buy new Top-K/N selections
    - Other N-1 buckets remain untouched
    - Result: daily signal utilization with ~1/N of full turnover cost per day

Comparison to baseline (stride=5):
    Baseline:     Day 0 [full rebalance] → hold 5 days → Day 5 [full rebalance] ...
    Staggered:   Every day [rebalance 1/N of portfolio] → each position held ~5 days

Key advantage:
    - Fresh signals used EVERY trading day (not every 5th day)
    - Same annual turnover cost as stride=5
    - Faster reaction to model prediction changes
    - More granular risk control (no single "bad rebalance day" wipes out everything)

Usage:
    python scripts/backtest/backtest_staggered_rebalance.py \
        --predictions outputs/runs/transformer_l60_clean_alpha_only_purgedwf/predictions.parquet \
        --output-dir outputs/backtest/staggered_l60 \
        --k 20 --num-buckets 5 --cost-bps 10 --slippage-bps 5
"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staggered multi-bucket rebalancing backtest.")
    parser.add_argument(
        "--predictions",
        default="outputs/runs/transformer_l60_clean_alpha_only_purgedwf/predictions.parquet",
    )
    parser.add_argument(
        "--execution-labels",
        default="data/mart/labels/execution_labels_v20260526.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/backtest/staggered")
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--portfolio-nav", type=float, default=10_000_000.0)
    parser.add_argument("--participation-cap", type=float, default=0.03)
    parser.add_argument("--num-buckets", type=int, default=5,
                        help="Number of staggered sub-buckets (default=5, matches holding period)")
    parser.add_argument("--min-daily-count", type=int, default=20)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return list(map(json_safe, value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_data(predictions_path: Path, execution_labels_path: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(predictions_path)
    labels = pd.read_parquet(execution_labels_path)
    predictions = predictions.copy()
    labels = labels.copy()
    for frame in (predictions, labels):
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    predictions["pred_score"] = pd.to_numeric(predictions["pred_score"], errors="coerce")
    numeric_cols = [
        "next_open", "next_amount", "execution_return_open_to_close5",
        "benchmark_next_open_to_exit_close_return",
    ]
    for col in numeric_cols:
        labels[col] = pd.to_numeric(labels[col], errors="coerce")
    for col in ["buy_executable_t1_open", "sell_executable_t1_open"]:
        labels[col] = labels[col].fillna(False).astype(bool)
    merged = predictions.merge(labels, on=["trade_date", "ts_code"], how="inner", validate="one_to_one")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.dropna(subset=["trade_date", "ts_code", "split", "pred_score"])


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {c: w / total for c, w in weights.items() if w > 0}


def simulate_bucket_fill(
    current_weights: dict[str, float],
    target_codes: list[str],
    day: pd.DataFrame,
    k_per_bucket: int,
    portfolio_nav: float,
    participation_cap: float,
    cost_rate: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Simulate T+1 fill for ONE bucket being rebalanced."""
    by_code = day.set_index("ts_code", drop=False)
    target_weight = 1.0 / k_per_bucket if k_per_bucket else 0.0
    desired = {code: target_weight for code in target_codes}
    all_codes = set(current_weights) | set(desired)
    next_weights = dict(current_weights)
    buy_reject = 0
    sell_reject = 0
    partial_fill = 0
    filled_turnover = 0.0
    desired_turnover = 0.0

    for code in sorted(all_codes):
        old_w = current_weights.get(code, 0.0)
        target_w = desired.get(code, 0.0)
        delta = target_w - old_w
        if abs(delta) < 1e-12:
            continue
        desired_turnover += abs(delta)
        if code not in by_code.index:
            if delta < 0:
                sell_reject += 1
                next_weights[code] = old_w
            continue
        row = by_code.loc[code]
        amount = float(row["next_amount"]) if pd.notna(row["next_amount"]) else 0.0
        max_fill = (participation_cap * amount / portfolio_nav) if portfolio_nav > 0 else abs(delta)
        max_fill = max(0.0, min(abs(delta), max_fill))
        if delta > 0:
            if not bool(row["buy_executable_t1_open"]) or max_fill <= 0:
                buy_reject += 1
                next_weights[code] = old_w
                continue
            fill = min(delta, max_fill)
            if fill < delta - 1e-12:
                partial_fill += 1
            next_weights[code] = old_w + fill
            filled_turnover += abs(fill)
        else:
            if not bool(row["sell_executable_t1_open"]) or max_fill <= 0:
                sell_reject += 1
                next_weights[code] = old_w
                continue
            fill = min(-delta, max_fill)
            if fill < -delta - 1e-12:
                partial_fill += 1
            next_weights[code] = old_w - fill
            filled_turnover += abs(fill)

    next_weights = {c: w for c, w in next_weights.items() if w > 1e-10}
    next_weights = normalize_weights(next_weights)
    transaction_cost = cost_rate * filled_turnover
    stats = {
        "filled_turnover": filled_turnover,
        "desired_turnover": desired_turnover,
        "transaction_cost": transaction_cost,
        "buy_reject": buy_reject,
        "sell_reject": sell_reject,
        "partial_fill": partial_fill,
        "position_count": len(next_weights),
    }
    return next_weights, stats


def weighted_return_for_bucket(weights: dict[str, float], day: pd.DataFrame) -> float:
    """Calculate weighted 5-day return for one bucket."""
    if not weights:
        return float("nan")
    by_code = day.set_index("ts_code")
    value = 0.0
    used = 0.0
    for code, w in weights.items():
        if code not in by_code.index:
            continue
        ret = by_code.loc[code, "execution_return_open_to_close5"]
        if pd.isna(ret):
            continue
        value += w * float(ret)
        used += w
    return value / used if used > 0 else float("nan")


def run_staggered_backtest(
    frame: pd.DataFrame,
    k_values: list[int],
    cost_bps: float,
    slippage_bps: float,
    portfolio_nav: float,
    participation_cap: float,
    num_buckets: int,
    min_daily_count: int,
) -> pd.DataFrame:
    """
    Core staggered rebalancing logic.

    For each split period:
    1. Get ALL trading days (not just every stride-th day)
    2. For each day d:
       a. Determine which bucket to rebalance: bucket_id = date_index % num_buckets
       b. For that bucket: select top-k/num_buckets stocks from day d's pred_score
       c. Simulate fill (T+1 open), update that bucket's weights
       d. All other buckets keep their existing positions (still earning returns)
    3. Portfolio return = equal-weighted average across all active buckets
    """
    rows: list[dict[str, Any]] = []
    total_cost_rate = (cost_bps + slippage_bps) / 10000.0

    for split, split_frame in frame.groupby("split", sort=True):
        all_dates = sorted(split_frame["trade_date"].unique())

        for k in k_values:
            k_per_bucket = max(1, k // num_buckets)
            # Each bucket tracks its own weights
            # buckets[bucket_id] = {ts_code: weight}
            buckets: list[dict[str, float]] = [{} for _ in range(num_buckets)]

            for date_idx, trade_date in enumerate(all_dates):
                day = split_frame[split_frame["trade_date"] == trade_date].sort_values(
                    "pred_score", ascending=False
                )
                if len(day) < min_daily_count:
                    continue

                ordered_codes = day["ts_code"].astype(str).tolist()
                bench_ret = float(day["benchmark_next_open_to_exit_close_return"].dropna().iloc[0])
                univ_ret = float(
                    day.loc[day["buy_executable_t1_open"], "execution_return_open_to_close5"].mean()
                )

                # Determine which bucket to rotate today
                bucket_id = date_idx % num_buckets
                target_codes = ordered_codes[:k_per_bucket]

                # Rebalance THIS bucket only
                new_weights, fill_stats = simulate_bucket_fill(
                    current_weights=buckets[bucket_id],
                    target_codes=target_codes,
                    day=day,
                    k_per_bucket=k_per_bucket,
                    portfolio_nav=portfolio_nav,
                    participation_cap=participation_cap,
                    cost_rate=total_cost_rate,
                )
                buckets[bucket_id] = new_weights

                # Calculate portfolio-level metrics
                # Each bucket contributes equally (1/num_buckets weight)
                bucket_returns = []
                bucket_costs = []
                total_positions = 0
                total_filled_turnover = 0.0
                total_desired_turnover = 0.0
                total_buy_reject = 0
                total_sell_reject = 0

                for bid in range(num_buckets):
                    br = weighted_return_for_bucket(buckets[bid], day)
                    if pd.notna(br):
                        bucket_returns.append(br)
                    total_positions += len(buckets[bid])

                # Only the rotated bucket incurred transaction costs
                bucket_costs.append(fill_stats["transaction_cost"])
                total_filled_turnover += fill_stats["filled_turnover"]
                total_desired_turnover += fill_stats["desired_turnover"]
                total_buy_reject += fill_stats["buy_reject"]
                total_sell_reject += fill_stats["sell_reject"]

                if not bucket_returns:
                    continue

                gross_return = np.mean(bucket_returns)
                total_cost = sum(bucket_costs)
                net_return = gross_return - total_cost

                rows.append({
                    "split": split,
                    "trade_date": trade_date,
                    "k": int(k),
                    "num_buckets": num_buckets,
                    "k_per_bucket": k_per_bucket,
                    "rotated_bucket": bucket_id,
                    "daily_count": int(len(day)),
                    "gross_return": float(gross_return),
                    "net_return": float(net_return),
                    "benchmark_return": bench_ret,
                    "executable_universe_return": univ_ret,
                    "excess_vs_benchmark": float(net_return - bench_ret),
                    "excess_vs_executable_universe": float(net_return - univ_ret),
                    "transaction_cost": float(total_cost),
                    "filled_turnover": float(total_filled_turnover),
                    "desired_turnover": float(total_desired_turnover),
                    "position_count": int(total_positions),
                    "buy_rejects": int(total_buy_reject),
                    "sell_rejects": int(total_sell_reject),
                    "active_buckets": len([b for b in buckets if b]),
                    "portfolio_nav": float(portfolio_nav),
                })

    return pd.DataFrame(rows)


def summarize_staggered(periods: pd.DataFrame, num_buckets: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if periods.empty:
        return summary
    periods_per_year = 252.0  # Daily frequency now

    def calc_stats(ser: pd.Series) -> dict:
        clean = ser.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return {"count": 0, "ann": float("nan"), "ir": float("nan"),
                    "max_dd": float("nan"), "cum": float("nan"), "win_rate": float("nan")}
        n = len(clean)
        cum = float((1 + clean).prod() - 1)
        ann = float((1 + cum) ** (periods_per_year / n) - 1)
        std = float(clean.std(ddof=1)) if n > 1 else float("nan")
        ir = float(clean.mean() / std) if std and std > 1e-12 else float("nan")
        eq = (1 + clean).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        return {
            "periods": n, "ann": ann, "ir": ir, "max_dd": dd,
            "cum": cum, "win_rate": float((clean > 0).mean()),
            "daily_mean": float(clean.mean()), "daily_std": float(std) if math.isfinite(std) else float("nan"),
        }

    for split, sf in periods.groupby("split", sort=True):
        split_sum: dict[str, Any] = {}
        for k, g in sf.groupby("k", sort=True):
            key = f"top_{int(k)}_buckets_{num_buckets}"
            split_sum[key] = {
                "net": calc_stats(g["net_return"]),
                "gross": calc_stats(g["gross_return"]),
                "excess_vs_benchmark": calc_stats(g["excess_vs_benchmark"]),
                "excess_vs_executable_universe": calc_stats(g["excess_vs_executable_universe"]),
                "benchmark": calc_stats(g["benchmark_return"]),
                "executable_universe": calc_stats(g["executable_universe_return"]),
                "avg_daily_cost": float(g["transaction_cost"].mean()),
                "avg_filled_turnover": float(g["filled_turnover"].mean()),
                "avg_position_count": float(g["position_count"].mean()),
                "avg_active_buckets": float(g["active_buckets"].mean()),
            }
        summary[str(split)] = split_sum
    return summary


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    execution_labels_path = Path(args.execution_labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_buckets = args.num_buckets

    data = load_data(predictions_path, execution_labels_path)
    print(f"Loaded {len(data)} rows, splits={data['split'].value_counts().to_dict()}")

    periods = run_staggered_backtest(
        frame=data,
        k_values=args.k,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        portfolio_nav=args.portfolio_nav,
        participation_cap=args.participation_cap,
        num_buckets=num_buckets,
        min_daily_count=args.min_daily_count,
    )

    summary = {
        "method": "staggered_multi_bucket_rebalancing",
        "predictions": str(predictions_path),
        "execution_labels": str(execution_labels_path),
        "rows_after_merge": int(len(data)),
        "k": args.k,
        "num_buckets": num_buckets,
        "k_per_bucket": [max(1, k // num_buckets) for k in args.k],
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "portfolio_nav": args.portfolio_nav,
        "participation_cap": args.participation_cap,
        "summary": summarize_staggered(periods, num_buckets),
    }

    periods.to_csv(output_dir / "staggered_periods.csv", index=False)
    (output_dir / "staggered_metrics.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))

    # Print comparison table
    print("\n" + "=" * 120)
    print(f"STAGGERED REBALANCE (N={num_buckets} buckets) vs BASELINE COMPARISON")
    print("=" * 120)
    for split in ["train", "validation", "test"]:
        sp = periods[periods["split"] == split]
        if sp.empty:
            continue
        print(f"\n--- {split.upper()} ({len(sp)} trading days) ---")
        for k in args.k:
            sk = sp[sp["k"] == k]
            if sk.empty:
                continue
            net = calc_quick(sk["net_return"])
            exc_bench = calc_quick(sk["excess_vs_benchmark"])
            exc_univ = calc_quick(sk["excess_vs_executable_universe"])
            hit_bench = float((sk["excess_vs_benchmark"] > 0).mean())
            hit_univ = float((sk["excess_vs_executable_universe"] > 0).mean())
            avg_cost = float(sk["transaction_cost"].mean()) * 10000  # in bps
            avg_turn = float(sk["filled_turnover"].mean())
            print(f"  K={k:3d}: Net_ann={net['ann']:+8.2%}  IR={net['ir']:+6.3f}  "
                  f"MaxDD={net['max_dd']:7.2%}  Win(vsBench)={hit_bench:.1%}  Win(vsUniv)={hit_univ:.1%}")
            print(f"         ExBench_ann={exc_bench['ann']:+8.2%}  ExUniv_ann={exc_univ['ann']:+8.2%}  "
                  f"AvgCost={avg_cost:.1f}bps  AvgTurn={avg_turn:.3f}")


def calc_quick(ser: pd.Series) -> dict:
    clean = ser.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return {"ann": float("nan"), "ir": float("nan"), "max_dd": float("nan"), "cum": float("nan")}
    n = len(clean)
    cum = float((1 + clean).prod() - 1)
    ppy = 252.0
    ann = float((1 + cum) ** (ppy / n) - 1)
    std = float(clean.std(ddof=1)) if n > 1 else float("nan")
    ir = float(clean.mean() / std) if std and std > 1e-12 else float("nan")
    eq = (1 + clean).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"ann": ann, "ir": ir, "max_dd": dd, "cum": cum}


if __name__ == "__main__":
    main()
