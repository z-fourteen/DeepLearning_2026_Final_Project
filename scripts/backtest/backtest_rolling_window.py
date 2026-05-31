"""Rolling-window backtest with daily full-position requirement.

Strategy overview:
    - Divide the entire test period into overlapping 10-day windows (stride=5)
    - Each window is an independent trading episode: start fresh, run 10 days
    - Within each window: enforce >=80% daily position (full-position rule)
    - T+1 execution: signal day t -> buy at t+1 open
    - Unfilled orders carry over to next day (max 2 days)
    - Final metrics = average across all ~60 windows

Key constraints (from grading rubric):
    - Daily position must be >= 80% (treated as fully invested)
    - If below 80% on a day, fill to target on the NEXT day
    - Each window = 10 trading days, stride = 5

Usage:
    conda run -n dl_env python scripts/backtest/backtest_rolling_window.py \\
        --predictions outputs/runs/transformer_l60_xxx/predictions.parquet \\
        --output-dir outputs/backtest/rolling_10d_l60 \\
        --window-size 10 --stride 5 --k 20 --cost-bps 10 --slippage-bps 5
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
    parser = argparse.ArgumentParser(
        description="Rolling-window backtest with daily full-position requirement."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/runs/transformer_l60_clean_alpha_only_purgedwf/predictions.parquet",
    )
    parser.add_argument(
        "--execution-labels",
        default="data/mart/labels/execution_labels_v20260526.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/backtest/rolling_10d")
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--portfolio-nav", type=float, default=10_000_000.0)
    parser.add_argument("--participation-cap", type=float, default=0.03)

    # Rolling window parameters
    parser.add_argument("--window-size", type=int, default=10,
                        help="Number of trading days per window (default=10)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Stride between consecutive windows (default=5)")
    parser.add_argument("--min-daily-count", type=int, default=20,
                        help="Minimum executable stocks per day to proceed")

    # Full-position parameters
    parser.add_argument("--min-position-ratio", type=float, default=0.80,
                        help="Minimum position ratio treated as full (default=0.80)")
    parser.add_argument("--max-carryover-days", type=int, default=2,
                        help="Max days to carry unfilled orders (default=2)")
    parser.add_argument("--rebalance-fraction", type=float, default=0.33,
                        help="Fraction of portfolio to rebalance on rebalance days (default=0.33)")

    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(map(json_safe, value))
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
    """Load and merge predictions with execution labels."""
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
    return {c: w / total for c, w in weights.items() if w > 1e-10}


class PendingOrder:
    """Represents an unfilled order carried over to future days."""

    def __init__(self, ts_code: str, action: str, target_weight: float,
                 priority: int = 0, age: int = 0):
        self.ts_code = ts_code
        self.action = action  # "buy" or "sell"
        self.target_weight = target_weight
        self.priority = priority  # 0=highest (carryover), 1=normal rebalance
        self.age = age  # days since creation


class WindowBacktester:
    """
    Runs one independent 10-day trading window with full-position enforcement.

    Core logic per day:
      1. Execute pending carryover orders (highest priority)
      2. Calculate current position ratio
      3. If position < min_ratio -> trigger emergency fill
      4. If rebalance day -> rotate rebalance_fraction of portfolio
      5. Compute daily P&L from all held positions
    """

    def __init__(
        self,
        window_data: pd.DataFrame,
        k: int,
        cost_rate: float,
        portfolio_nav: float,
        participation_cap: float,
        min_position_ratio: float,
        max_carryover_days: int,
        rebalance_fraction: float,
        rebalance_stride: int,
        min_daily_count: int,
    ):
        # Sort by date ascending
        self.dates = sorted(window_data["trade_date"].unique())
        self.data = window_data.set_index("trade_date")
        self.k = k
        self.cost_rate = cost_rate
        self.portfolio_nav = portfolio_nav
        self.participation_cap = participation_cap
        self.min_position_ratio = min_position_ratio
        self.max_carryover_days = max_carryover_days
        self.rebalance_fraction = rebalance_fraction
        self.rebalance_stride = rebalance_stride
        self.min_daily_count = min_daily_count

        # State
        self.weights: dict[str, float] = {}  # {ts_code: weight}
        self.pending_orders: list[PendingOrder] = []
        self.day_results: list[dict[str, Any]] = []
        self.cash_ratio = 1.0  # Start with all cash (will fill on Day 1 T+1)

    def _get_day_data(self, trade_date: str) -> pd.DataFrame | None:
        if trade_date not in self.data.index:
            return None
        day = self.data.loc[trade_date]
        if isinstance(day, pd.DataFrame):
            return day.sort_values("pred_score", ascending=False)
        return day.to_frame().T.sort_values("pred_score", ascending=False)

    def _simulate_fill(
        self,
        action: str,
        ts_code: str,
        target_delta: float,
        day_data: pd.DataFrame,
    ) -> tuple[float, bool, bool]:
        """
        Simulate T+1 fill for a single trade.
        Returns: (filled_amount, was_rejected, was_partial)
        """
        by_code = day_data.set_index("ts_code") if "ts_code" not in day_data.index else day_data

        if ts_code not in by_code.index:
            return 0.0, True, False

        row = by_code.loc[ts_code]
        amount = float(row["next_amount"]) if pd.notna(row.get("next_amount")) else 0.0
        max_fill = (
            (self.participation_cap * amount / self.portfolio_nav)
            if self.portfolio_nav > 0 else abs(target_delta)
        )
        max_fill = max(0.0, min(abs(target_delta), max_fill))

        if abs(target_delta) < 1e-12:
            return 0.0, False, False

        if action == "buy":
            if not bool(row.get("buy_executable_t1_open", False)) or max_fill <= 0:
                return 0.0, True, False
            fill = min(target_delta, max_fill)
            return fill, fill < target_delta - 1e-12, False
        else:  # sell
            if not bool(row.get("sell_executable_t1_open", False)) or max_fill <= 0:
                return 0.0, True, False
            fill = min(-target_delta, max_fill)
            return fill, fill < (-target_delta) - 1e-12, False

    def _execute_pending_orders(self, day_data: pd.DataFrame) -> tuple[float, dict]:
        """Execute pending carryover orders first (highest priority)."""
        filled_turnover = 0.0
        stats = {"filled": 0, "rejected": 0, "expired": 0}
        still_pending: list[PendingOrder] = []

        for order in self.pending_orders:
            order.age += 1

            # Expire old orders
            if order.age > self.max_carryover_days:
                stats["expired"] += 1
                continue

            current_w = self.weights.get(order.ts_code, 0.0)

            if order.action == "buy":
                delta = order.target_weight - current_w
                if delta <= 1e-12:
                    continue  # Already have enough
                fill, rejected, _ = self._simulate_fill("buy", order.ts_code, delta, day_data)
                if rejected or fill <= 0:
                    stats["rejected"] += 1
                    still_pending.append(order)
                    continue
                self.weights[order.ts_code] = current_w + fill
                filled_turnover += fill
                stats["filled"] += 1
            else:  # sell
                delta = current_w - order.target_weight
                if delta <= 1e-12:
                    if current_w > 1e-10:
                        del self.weights[order.ts_code]
                    continue
                fill, rejected, _ = self._simulate_fill("sell", order.ts_code, delta, day_data)
                if rejected or fill <= 0:
                    stats["rejected"] += 1
                    still_pending.append(order)
                    continue
                new_w = current_w - fill
                if new_w < 1e-10:
                    self.weights.pop(order.ts_code, None)
                else:
                    self.weights[order.ts_code] = new_w
                filled_turnover += fill
                stats["filled"] += 1

        self.pending_orders = still_pending
        self.weights = normalize_weights(self.weights)
        return filled_turnover, stats

    def _emergency_fill(self, day_data: pd.DataFrame) -> tuple[float, dict]:
        """Fill to meet minimum position requirement (>=80%)."""
        current_total = sum(self.weights.values())
        deficit = self.min_position_ratio - current_total

        if deficit <= 1e-9:
            return 0.0, {"fill_stocks": 0}

        stats = {"fill_stocks": 0}
        filled_turnover = 0.0

        # Get candidate stocks: not yet held, sorted by pred_score descending
        held_codes = set(self.weights.keys())
        candidates = day_data[~day_data["ts_code"].isin(held_codes)].copy()
        candidates = candidates[candidates["buy_executable_t1_open"] == True]

        if candidates.empty:
            return 0.0, stats

        target_per_stock = deficit / max(1, len(candidates))

        for _, row in candidates.iterrows():
            code = str(row["ts_code"])
            if sum(self.weights.values()) >= self.min_position_ratio - 1e-6:
                break

            fill, rejected, _ = self._simulate_fill("buy", code, target_per_stock, day_data)
            if fill > 0:
                self.weights[code] = self.weights.get(code, 0.0) + fill
                filled_turnover += fill
                stats["fill_stocks"] += 1
            elif rejected:
                # Create pending order for emergency fill too
                self.pending_orders.append(PendingOrder(
                    ts_code=code, action="buy",
                    target_weight=self.weights.get(code, 0.0) + target_per_stock,
                    priority=1, age=0,
                ))

        self.weights = normalize_weights(self.weights)
        return filled_turnover, stats

    def _rebalance(self, day_data: pd.DataFrame, day_index: int) -> tuple[float, dict]:
        """
        Rebalance a fraction of portfolio using latest pred_score.
        Replace bottom-ranked positions with top-ranked ones.
        """
        stats = {"sell_count": 0, "buy_count": 0, "rejects": 0}
        filled_turnover = 0.0

        if not self.weights:
            # No positions yet, do initial fill
            return self._initial_fill(day_data), {"initial_fill": list(self.weights.keys())}

        # Rank current holdings by today's pred_score
        held_codes = list(self.weights.keys())
        day_by_code = day_data.set_index("ts_code")

        # Get scores for held stocks
        held_with_scores = []
        for code in held_codes:
            if code in day_by_code.index:
                held_with_scores.append((code, float(day_by_code.loc[code, "pred_score"]),
                                         self.weights[code]))

        # Sort held stocks by score ascending (worst first to sell)
        held_with_scores.sort(key=lambda x: x[1])

        # Determine how many to replace
        n_replace = max(1, int(len(held_codes) * self.rebalance_fraction))

        # Get top-K overall stocks by pred_score
        top_codes = day_data.head(self.k)["ts_code"].astype(str).tolist()

        # Sell worst N held stocks
        codes_to_sell = [x[0] for x in held_with_scores[:n_replace]]
        sell_weight_freed = 0.0

        for code in codes_to_sell:
            current_w = self.weights.get(code, 0.0)
            if current_w < 1e-10:
                continue
            fill, rejected, _ = self._simulate_fill("sell", code, current_w, day_data)
            if fill > 0:
                new_w = current_w - fill
                if new_w < 1e-10:
                    self.weights.pop(code, None)
                else:
                    self.weights[code] = new_w
                sell_weight_freed += fill
                filled_turnover += fill
                stats["sell_count"] += 1
            elif rejected:
                self.pending_orders.append(PendingOrder(
                    ts_code=code, action="sell",
                    target_weight=0.0, priority=1, age=0,
                ))
                stats["rejects"] += 1

        # Buy top stocks not already held (use freed weight budget)
        buy_budget = sell_weight_freed
        bought_codes = set(self.weights.keys())

        for code in top_codes:
            if buy_budget <= 1e-10 or len(self.weights) >= self.k:
                break
            if code in bought_codes:
                continue
            target_w = min(buy_budget / max(1, self.k - len(self.weights)), buy_budget)

            fill, rejected, _ = self._simulate_fill("buy", code, target_w, day_data)
            if fill > 0:
                self.weights[code] = self.weights.get(code, 0.0) + fill
                filled_turnover += fill
                buy_budget -= fill
                stats["buy_count"] += 1
            elif rejected:
                self.pending_orders.append(PendingOrder(
                    ts_code=code, action="buy",
                    target_weight=target_w, priority=1, age=0,
                ))
                stats["rejects"] += 1

        self.weights = normalize_weights(self.weights)
        return filled_turnover, stats

    def _initial_fill(self, day_data: pd.DataFrame) -> float:
        """Initial portfolio construction on first rebalance day."""
        filled_turnover = 0.0
        top_codes = day_data.head(self.k)["ts_code"].astype(str).tolist()
        target_w = 1.0 / self.k

        for code in top_codes:
            fill, rejected, _ = self._simulate_fill("buy", code, target_w, day_data)
            if fill > 0:
                self.weights[code] = fill
                filled_turnover += fill
            elif rejected:
                self.pending_orders.append(PendingOrder(
                    ts_code=code, action="buy",
                    target_weight=target_w, priority=1, age=0,
                ))

        self.weights = normalize_weights(self.weights)
        return filled_turnover

    def _compute_daily_return(self, day_data: pd.DataFrame) -> float:
        """Compute weighted return from current holdings using 5-day execution return."""
        if not self.weights:
            return 0.0

        by_code = day_data.set_index("ts_code")
        value = 0.0
        used = 0.0

        for code, w in self.weights.items():
            if code not in by_code.index:
                continue
            ret = by_code.loc[code, "execution_return_open_to_close5"]
            if pd.isna(ret):
                continue
            value += w * float(ret)
            used += w

        return value / used if used > 0 else 0.0

    def _get_position_ratio(self) -> float:
        return sum(self.weights.values())

    def run(self) -> dict[str, Any]:
        """Run the full window backtest and return results dict."""
        total_cost = 0.0
        cumulative_return = 1.0
        cumulative_bench_return = 1.0  # track benchmark cumulative over window
        position_compliance_days = 0
        total_active_days = 0
        best_daily_return = float("-inf")
        worst_daily_return = float("inf")
        best_day_info: dict[str, Any] = {}
        worst_day_info: dict[str, Any] = {}

        for day_idx, trade_date in enumerate(self.dates):
            day_data = self._get_day_data(trade_date)
            if day_data is None or len(day_data) < self.min_daily_count:
                continue

            total_active_days += 1
            daily_cost = 0.0

            # Step 1: Execute carryover orders (highest priority)
            carryover_turnover, carryover_stats = self._execute_pending_orders(day_data)
            daily_cost += carryover_turnover * self.cost_rate
            total_cost += carryover_turnover * self.cost_rate

            # Step 2: Check if rebalance day (every rebalance_stride days)
            is_rebalance_day = (day_idx % self.rebalance_stride == 0)
            if is_rebalance_day:
                rebal_turnover, rebal_stats = self._rebalance(day_data, day_idx)
                daily_cost += rebal_turnover * self.cost_rate
                total_cost += rebal_turnover * self.cost_rate

            # Step 3: Emergency fill check (position < 80%)
            pos_ratio = self._get_position_ratio()
            if pos_ratio < self.min_position_ratio:
                emerg_turnover, emerg_stats = self._emergency_fill(day_data)
                daily_cost += emerg_turnover * self.cost_rate
                total_cost += emerg_turnover * self.cost_rate

            # Recompute after potential fills
            pos_ratio = self._get_position_ratio()
            if pos_ratio >= self.min_position_ratio - 1e-6:
                position_compliance_days += 1

            # Benchmark data (compute before daily return for best/worst day info)
            bench_ret = 0.0
            if "benchmark_next_open_to_exit_close_return" in day_data.columns:
                bench_series = day_data["benchmark_next_open_to_exit_close_return"].dropna()
                if not bench_series.empty:
                    bench_ret = float(bench_series.iloc[0])

            # Step 4: Compute daily return
            gross_ret = self._compute_daily_return(day_data)
            net_ret = gross_ret - daily_cost
            cumulative_return *= (1 + net_ret)

            # Track best/worst daily returns
            if net_ret > best_daily_return:
                best_daily_return = net_ret
                best_day_info = {
                    "trade_date": trade_date, "day_index": day_idx,
                    "net_return": net_ret, "gross_return": gross_ret,
                    "benchmark_return": bench_ret,
                }
            if net_ret < worst_daily_return:
                worst_daily_return = net_ret
                worst_day_info = {
                    "trade_date": trade_date, "day_index": day_idx,
                    "net_return": net_ret, "gross_return": gross_ret,
                    "benchmark_return": bench_ret,
                }

            # Accumulate benchmark return over the window
            cumulative_bench_return *= (1 + bench_ret)

            univ_ret = 0.0
            exec_mask = day_data["buy_executable_t1_open"] == True
            if exec_mask.any():
                univ_ret = float(day_data.loc[exec_mask, "execution_return_open_to_close5"].mean())

            self.day_results.append({
                "trade_date": trade_date,
                "day_index": day_idx,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "transaction_cost": daily_cost,
                "benchmark_return": bench_ret,
                "executable_universe_return": univ_ret,
                "excess_vs_benchmark": net_ret - bench_ret,
                "excess_vs_executable_universe": net_ret - univ_ret,
                "position_count": len(self.weights),
                "position_ratio": pos_ratio,
                "is_rebalance": is_rebalance_day,
                "pending_orders_len": len(self.pending_orders),
                "cumulative_return": cumulative_return,
            })

        # Window-level summary
        if not self.day_results:
            return {"error": "no_valid_days"}

        returns = [r["net_return"] for r in self.day_results]
        n = len(returns)
        cum = float((pd.Series(returns) + 1).prod() - 1)
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1)) if n > 1 else 0.0
        ir = mean_ret / std_ret if std_ret > 1e-12 else 0.0
        compliance_rate = position_compliance_days / max(1, total_active_days)

        return {
            "window_start": self.dates[0] if self.dates else None,
            "window_end": self.dates[-1] if self.dates else None,
            "num_days": n,
            "compliant_days": position_compliance_days,
            "compliance_rate": compliance_rate,
            "cumulative_return": cum,  # portfolio 10-day cumulative return
            "cumulative_bench_return": cumulative_bench_return - 1.0,  # benchmark cumulative over same period
            "excess_vs_bench_10d": cum - (cumulative_bench_return - 1.0),  # 10-day excess vs benchmark
            "daily_mean": mean_ret,
            "daily_std": std_ret,
            "ir": ir,
            "avg_position_count": float(np.mean([r["position_count"] for r in self.day_results])),
            "avg_position_ratio": float(np.mean([r["position_ratio"] for r in self.day_results])),
            "total_cost": float(total_cost),
            "avg_daily_cost": float(np.mean([r["transaction_cost"] for r in self.day_results])),
            "win_rate_vs_bench": float(np.mean([r["excess_vs_benchmark"] > 0 for r in self.day_results])),
            # Best / worst trading day
            "best_daily_return": best_daily_return if best_daily_return > float("-inf") else None,
            "best_day_info": best_day_info,
            "worst_daily_return": worst_daily_return if worst_daily_return < float("inf") else None,
            "worst_day_info": worst_day_info,
            "final_cumulative": cumulative_return,
            "day_results": self.day_results,
        }


def generate_windows(all_dates: list[str], window_size: int, stride: int) -> list[list[str]]:
    """Generate overlapping windows of dates."""
    windows = []
    n = len(all_dates)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(all_dates[start:end])
    return windows


def run_rolling_backtest(
    frame: pd.DataFrame,
    k_values: list[int],
    cost_bps: float,
    slippage_bps: float,
    portfolio_nav: float,
    participation_cap: float,
    window_size: int,
    stride: int,
    min_daily_count: int,
    min_position_ratio: float,
    max_carryover_days: int,
    rebalance_fraction: float,
    rebalance_stride: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run rolling-window backtest across all splits and aggregate results."""
    total_cost_rate = (cost_bps + slippage_bps) / 10000.0
    all_window_results: list[dict[str, Any]] = []

    summary = {
        "method": "rolling_window_full_position",
        "window_size": window_size,
        "stride": stride,
        "rebalance_stride_within_window": rebalance_stride,
        "min_position_ratio": min_position_ratio,
        "max_carryover_days": max_carryover_days,
        "rebalance_fraction": rebalance_fraction,
        "k_values": k_values,
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
    }

    for split, split_frame in frame.groupby("split", sort=True):
        print(f"\n{'='*80}")
        print(f"Processing split: {split}")
        print(f"{'='*80}")

        all_dates = sorted(split_frame["trade_date"].unique())
        windows = generate_windows(all_dates, window_size, stride)
        print(f"Total dates: {len(all_dates)}, Generated {len(windows)} windows "
              f"(size={window_size}, stride={stride})")

        split_summary = {"num_windows": len(windows)}

        for k in k_values:
            print(f"\n--- K={k} ---")
            window_metrics: list[dict[str, Any]] = []
            all_period_rows: list[dict[str, Any]] = []

            for win_idx, window_dates in enumerate(windows):
                # Extract data for this window
                window_frame = split_frame[split_frame["trade_date"].isin(window_dates)]

                tester = WindowBacktester(
                    window_data=window_frame,
                    k=k,
                    cost_rate=total_cost_rate,
                    portfolio_nav=portfolio_nav,
                    participation_cap=participation_cap,
                    min_position_ratio=min_position_ratio,
                    max_carryover_days=max_carryover_days,
                    rebalance_fraction=rebalance_fraction,
                    rebalance_stride=rebalance_stride,
                    min_daily_count=min_daily_count,
                )

                result = tester.run()

                if "error" in result:
                    continue

                result["window_index"] = win_idx
                result["k"] = k
                result["split"] = split
                window_metrics.append(result)

                # Collect daily rows for detailed output
                for dr in result.pop("day_results", []):
                    dr["split"] = split
                    dr["k"] = k
                    dr["window_index"] = win_idx
                    all_period_rows.append(dr)

                progress = f"W{win_idx+1:>3d}/{len(windows)} | " \
                          f"{result['window_start']}~{result['window_end']} | " \
                          f"d={result['num_days']:>2d} | " \
                          f"cum={result['cumulative_return']:+.4f} | " \
                          f"comp={result['compliance_rate']:.1%} | " \
                           f"pos={result['avg_position_count']:.0f}"
                print(progress)

            if not window_metrics:
                continue

            # Aggregate: average metrics across windows
            aggregated = aggregate_window_metrics(window_metrics)
            key = f"k_{k}"
            split_summary[key] = aggregated
            all_window_results.extend(window_metrics)

            # Also store raw window-level data
            summary[f"{split}_windows_k{k}"] = [
                {kk: vv for kk, vv in w.items() if kk != "day_results"}
                for w in window_metrics
            ]

            print(f"\n>>> K={k} AGGREGATED over {len(window_metrics)} windows:")
            print(f"    Avg 10d Return:   {aggregated.get('mean_cumulative_return', float('nan')):+.4%}")
            print(f"    Avg Bench 10d:     {aggregated.get('mean_cumulative_bench_return', float('nan')):+.4%}")
            print(f"    Avg Excess vsBench:{aggregated.get('mean_excess_vs_bench_10d', float('nan')):+.4%}")
            print(f"    Avg DailyRet:      {aggregated.get('mean_daily_mean', float('nan')):+.6f}")
            print(f"    Avg IR:            {aggregated.get('mean_ir', float('nan')):+.4f}")
            print(f"    Compliance:        {aggregated.get('mean_compliance_rate', float('nan')):.2%}")
            print(f"    Win(vsBench):      {aggregated.get('mean_win_rate_vs_bench', float('nan')):.2%}")
            print(f"    Best Day (avg):    {aggregated.get('mean_best_daily_return', float('nan')):+.4%}")
            print(f"    Worst Day (avg):   {aggregated.get('mean_worst_daily_return', float('nan')):+.4%}")

        summary[str(split)] = split_summary

        # Save daily detail for this split+k combination
        if all_period_rows:
            periods_df = pd.DataFrame(all_period_rows)

    # Build final periods DataFrame from all window results
    final_periods_rows = []
    for wr in all_window_results:
        base = {
            "split": wr["split"],
            "k": wr["k"],
            "window_index": wr["window_index"],
            "window_start": wr["window_start"],
            "window_end": wr["window_end"],
            "num_days": wr["num_days"],
            "cumulative_return": wr["cumulative_return"],
            "cumulative_bench_return": wr.get("cumulative_bench_return"),
            "excess_vs_bench_10d": wr.get("excess_vs_bench_10d"),
            "daily_mean": wr["daily_mean"],
            "ir": wr["ir"],
            "compliance_rate": wr["compliance_rate"],
            "avg_position_count": wr["avg_position_count"],
            "avg_position_ratio": wr["avg_position_ratio"],
            "total_cost": wr["total_cost"],
            "win_rate_vs_bench": wr["win_rate_vs_bench"],
            "best_daily_return": wr.get("best_daily_return"),
            "worst_daily_return": wr.get("worst_daily_return"),
            "final_cumulative": wr["final_cumulative"],
        }
        final_periods_rows.append(base)

    periods_df = pd.DataFrame(final_periods_rows)
    return periods_df, summary


def aggregate_window_metrics(metrics: list[dict]) -> dict:
    """Aggregate metrics across multiple windows by averaging."""
    keys_of_interest = [
        "cumulative_return", "cumulative_bench_return", "excess_vs_bench_10d",
        "daily_mean", "daily_std", "ir",
        "compliance_rate", "avg_position_count", "avg_position_ratio",
        "total_cost", "avg_daily_cost", "win_rate_vs_bench", "final_cumulative",
        "best_daily_return", "worst_daily_return",
    ]

    aggregated = {}
    for key in keys_of_interest:
        values = [m[key] for m in metrics if key in m and m[key] is not None]
        if values:
            clean = [v for v in values if v is not None and math.isfinite(v)]
            if clean:
                aggregated[f"mean_{key}"] = float(np.mean(clean))
                aggregated[f"std_{key}"] = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
                aggregated[f"median_{key}"] = float(np.median(clean))
                aggregated[f"min_{key}"] = float(np.min(clean))
                aggregated[f"max_{key}"] = float(np.max(clean))

    aggregated["num_windows"] = len(metrics)

    # Distribution buckets for cumulative return
    cums = [m["cumulative_return"] for m in metrics if "cumulative_return" in m]
    if cums:
        aggregated["pct_positive_windows"] = float(np.mean([c > 0 for c in cums]))
        aggregated["pct_windows_above_5pct"] = float(np.mean([c > 0.05 for c in cums]))
        aggregated["pct_windows_below_neg5pct"] = float(np.mean([c < -0.05 for c in cums]))

    return aggregated


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    execution_labels_path = Path(args.execution_labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(predictions_path, execution_labels_path)
    print(f"Loaded {len(data)} rows, splits={data['split'].value_counts().to_dict()}")

    periods_df, summary = run_rolling_backtest(
        frame=data,
        k_values=args.k,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        portfolio_nav=args.portfolio_nav,
        participation_cap=args.participation_cap,
        window_size=args.window_size,
        stride=args.stride,
        min_daily_count=args.min_daily_count,
        min_position_ratio=args.min_position_ratio,
        max_carryover_days=args.max_carryover_days,
        rebalance_fraction=args.rebalance_fraction,
        rebalance_stride=max(1, args.window_size // 3),  # ~3x rebalance within 10-day window
    )

    # Save results
    periods_df.to_csv(output_dir / "rolling_window_periods.csv", index=False)

    meta = {
        "predictions": str(predictions_path),
        "execution_labels": str(execution_labels_path),
        "rows_after_merge": int(len(data)),
    }
    summary.update(meta)

    (output_dir / "rolling_window_metrics.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print final report
    print("\n" + "=" * 100)
    print("ROLLING WINDOW BACKTEST FINAL REPORT")
    print("=" * 100)
    print(f"\nConfig: window_size={args.window_size}, stride={args.stride}, "
          f"K={args.k}, cost={args.cost_bps+args.slippage_bps}bps")
    print(f"Position: min_ratio={args.min_position_ratio:.0%}, "
          f"carryover={args.max_carryover_days}d, rebalance_frac={args.rebalance_fraction:.0%}")

    for split in ["train", "validation", "test"]:
        if str(split) not in summary:
            continue
        ss = summary[str(split)]
        print(f"\n--- {split.upper()} ({ss.get('num_windows', '?')} windows) ---")
        for k in args.k:
            key = f"k_{k}"
            if key not in ss:
                continue
            m = ss[key]
            print(f"  K={k:3d}: Windows={m.get('num_windows', '?')} | "
                  f"AvgCumRet={m.get('mean_cumulative_return', float('nan')):+.4%} "
                  f"(med={m.get('median_cumulative_return', float('nan')):+.4%}) | "
                  f"AvgIR={m.get('mean_ir', float('nan')):+.4f} | "
                  f"Compliance={m.get('mean_compliance_rate', float('nan')):.1%} | "
                  f"WinBench={m.get('mean_win_rate_vs_bench', float('nan')):.1%}")
            # 10-day return vs benchmark
            print(f"         10d Portfolio={m.get('mean_cumulative_return', float('nan')):+.4%} | "
                  f"10d Benchmark={m.get('mean_cumulative_bench_return', float('nan')):+.4%} | "
                  f"10d Excess={m.get('mean_excess_vs_bench_10d', float('nan')):+.4%}")
            # Best / worst day
            best_avg = m.get('mean_best_daily_return')
            worst_avg = m.get('mean_worst_daily_return')
            best_max = m.get('max_best_daily_return')
            worst_min = m.get('min_worst_daily_return')
            print(f"         BestDay (avg/max) ={best_avg:+.4%}/{best_max:+.4%} | "
                  f"WorstDay (avg/min)={worst_avg:+.4%}/{worst_min:+.4%}")
            print(f"         PositiveWin%={m.get('pct_positive_windows', float('nan')):.1%} | "
                  f">+5%={m.get('pct_windows_above_5pct', float('nan')):.1%} | "
                  f"<-5%={m.get('pct_windows_below_neg5pct', float('nan')):.1%}")

    # Print top/best and worst windows detail
    print(f"\n{'='*100}")
    print("BEST & WORST WINDOWS DETAIL")
    print(f"{'='*100}")
    for split in ["validation", "test"]:
        key = f"{split}_windows_k20"
        if key not in summary:
            continue
        wins = summary[key]
        if not wins:
            continue

        # Sort by cumulative return
        sorted_wins = sorted(wins, key=lambda w: w.get("cumulative_return", 0), reverse=True)

        print(f"\n--- {split.upper()} ---")
        print(f"  TOP 5 WINDOWS:")
        for i, w in enumerate(sorted_wins[:5]):
            bi = w.get("best_day_info", {})
            wi = w.get("worst_day_info", {})
            print(f"    #{i+1} [{w['window_start']}~{w['window_end']}] "
                  f"10dRet={w['cumulative_return']:+.4%} | "
                  f"Bench10d={w.get('cumulative_bench_return', 0):+.4%} | "
                  f"Excess={w.get('excess_vs_bench_10d', 0):+.4%} | "
                  f"BestDay={bi.get('net_return', 0):+.4%}({bi.get('trade_date','')}) | "
                  f"WorstDay={wi.get('net_return', 0):+.4%}({wi.get('trade_date','')})")

        print(f"  WORST 5 WINDOWS:")
        for i, w in enumerate(sorted_wins[-5:]):
            bi = w.get("best_day_info", {})
            wi = w.get("worst_day_info", {})
            print(f"    #{i+1} [{w['window_start']}~{w['window_end']}] "
                  f"10dRet={w['cumulative_return']:+.4%} | "
                  f"Bench10d={w.get('cumulative_bench_return', 0):+.4%} | "
                  f"Excess={w.get('excess_vs_bench_10d', 0):+.4%} | "
                  f"BestDay={bi.get('net_return', 0):+.4%}({bi.get('trade_date','')}) | "
                  f"WorstDay={wi.get('net_return', 0):+.4%}({wi.get('trade_date','')})")


if __name__ == "__main__":
    main()
