from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STYLE_SETS = {
    "industry_proxy": [
        "lag1_industry_turnover_rank",
        "lag1_industry_amount_rank",
        "lag1_industry_pb_rank",
        "lag1_industry_mv_rank",
    ],
    "industry_size": [
        "lag1_industry_turnover_rank",
        "lag1_industry_amount_rank",
        "lag1_industry_pb_rank",
        "lag1_industry_mv_rank",
        "lag1_log_circ_mv",
        "lag1_log_total_mv",
    ],
    "full_style": [
        "lag1_industry_turnover_rank",
        "lag1_industry_amount_rank",
        "lag1_industry_pb_rank",
        "lag1_industry_mv_rank",
        "lag1_log_circ_mv",
        "lag1_log_total_mv",
        "lag1_amount_log",
        "lag1_amount_rank_pct",
        "lag1_turnover_rate_f",
        "lag1_turnover_20d_mean",
        "lag1_ret_20d_std",
        "lag1_ret_60d_std",
        "lag1_ret_20d",
        "lag1_ret_60d_mean",
        "lag1_beta_20d",
        "lag1_beta_60d",
    ],
}

LABEL_COLUMNS = [
    "trade_date",
    "ts_code",
    "next_amount",
    "buy_executable",
    "sell_executable",
    "next_open_return_5d",
    "benchmark_next_open_return_5d",
]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive ints.")
    return sorted(set(values))


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated non-negative floats.")
    return sorted(set(values))


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hard-constrained Barra-lite portfolio construction.")
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet",
    )
    parser.add_argument("--mart", default="data/mart/datasets/dataset_v20260526.parquet")
    parser.add_argument("--labels", default="data/mart/labels/labels_canonical_v20260526.parquet")
    parser.add_argument("--output-dir", default="outputs/backtest/optimizer/hard_constraints")
    parser.add_argument("--style-set", type=parse_str_list, default=parse_str_list("industry_proxy,industry_size,full_style"))
    parser.add_argument("--exposure-cap", type=parse_float_list, default=parse_float_list("0.15,0.25,0.35,0.50"))
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("10,30"))
    parser.add_argument("--candidate-multiplier", type=float, default=8.0)
    parser.add_argument("--fallback-soft-cap", type=float, default=2.0)
    parser.add_argument("--portfolio-nav", type=float, default=10_000_000.0)
    parser.add_argument("--participation-cap", type=float, default=0.03)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--rebalance-stride", type=int, default=5)
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


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    values = values.fillna(values.median())
    std = values.std(ddof=0)
    if not std or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def max_drawdown(returns: pd.Series) -> float:
    clean = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    equity = (1.0 + clean).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def summarize_returns(returns: pd.Series, periods_per_year: float = 252.0 / 5.0) -> dict[str, float | int]:
    clean = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"period_count": 0, "annualized_return": float("nan"), "ir": float("nan"), "max_drawdown": float("nan")}
    std = clean.std(ddof=1)
    cumulative = float((1.0 + clean).prod() - 1.0)
    ann = float((1.0 + cumulative) ** (periods_per_year / len(clean)) - 1.0)
    return {
        "period_count": int(len(clean)),
        "annualized_return": ann,
        "ir": float(clean.mean() / std) if std and pd.notna(std) else float("nan"),
        "max_drawdown": max_drawdown(clean),
    }


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    predictions = pd.read_parquet(args.predictions)
    style_cols = sorted({column for cols in STYLE_SETS.values() for column in cols})
    mart_cols = set(pd.read_parquet(args.mart, engine="pyarrow").columns)
    mart_selected = ["trade_date", "ts_code", *[column for column in style_cols if column in mart_cols]]
    mart = pd.read_parquet(args.mart, columns=mart_selected)
    labels = pd.read_parquet(args.labels, columns=LABEL_COLUMNS)
    data = predictions.merge(mart, on=["trade_date", "ts_code"], how="inner")
    data = data.merge(labels, on=["trade_date", "ts_code"], how="inner")
    for column in ["trade_date", "ts_code", "split"]:
        data[column] = data[column].astype(str)
    for column in ["pred_score", "next_amount", "next_open_return_5d", "benchmark_next_open_return_5d", *mart_selected[2:]]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["buy_executable", "sell_executable"]:
        data[column] = data[column].fillna(False).astype(bool)
    return data.replace([np.inf, -np.inf], np.nan)


def add_style_z(day: pd.DataFrame, style_cols: list[str]) -> pd.DataFrame:
    result = day.copy()
    for column in style_cols:
        if column in result.columns:
            result[f"z__{column}"] = zscore(result[column])
    return result


def portfolio_exposure(codes: list[str], day: pd.DataFrame, z_cols: list[str]) -> pd.Series:
    if not codes or not z_cols:
        return pd.Series(0.0, index=z_cols)
    selected = day[day["ts_code"].isin(codes)]
    if selected.empty:
        return pd.Series(0.0, index=z_cols)
    return selected[z_cols].mean()


def exposure_ok(codes: list[str], day: pd.DataFrame, z_cols: list[str], cap: float) -> bool:
    if not z_cols:
        return True
    exposure = portfolio_exposure(codes, day, z_cols)
    return bool((exposure.abs() <= cap).all())


def select_hard_constrained(
    day: pd.DataFrame,
    previous_codes: list[str],
    k: int,
    style_cols: list[str],
    cap: float,
    candidate_multiplier: float,
    fallback_soft_cap: float,
) -> tuple[list[str], dict[str, Any]]:
    candidate_count = min(len(day), max(k, int(math.ceil(k * candidate_multiplier))))
    candidates = day.sort_values("pred_score", ascending=False).head(candidate_count).copy()
    z_cols = [f"z__{column}" for column in style_cols if f"z__{column}" in day.columns]
    selected: list[str] = []
    reject_count = 0

    kept_previous = [code for code in previous_codes if code in set(candidates["ts_code"])]
    for code in kept_previous:
        trial = [*selected, code]
        if len(trial) <= k and exposure_ok(trial, day, z_cols, cap):
            selected = trial

    for code in candidates["ts_code"].astype(str):
        if len(selected) >= k:
            break
        if code in selected:
            continue
        trial = [*selected, code]
        if exposure_ok(trial, day, z_cols, cap):
            selected = trial
        else:
            reject_count += 1

    used_fallback = False
    if len(selected) < k:
        used_fallback = True
        for code in candidates["ts_code"].astype(str):
            if len(selected) >= k:
                break
            if code in selected:
                continue
            trial = [*selected, code]
            if exposure_ok(trial, day, z_cols, fallback_soft_cap):
                selected = trial

    exposure = portfolio_exposure(selected, day, z_cols)
    return selected[:k], {
        "constraint_reject_count": int(reject_count),
        "used_fallback": bool(used_fallback),
        "max_abs_target_exposure_z": float(exposure.abs().max()) if len(exposure) else 0.0,
        "avg_abs_target_exposure_z": float(exposure.abs().mean()) if len(exposure) else 0.0,
        "target_count": int(len(selected[:k])),
    }


def simulate_fill(
    current: dict[str, float],
    target_codes: list[str],
    day: pd.DataFrame,
    portfolio_nav: float,
    participation_cap: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    target_weight = 1.0 / len(target_codes) if target_codes else 0.0
    desired = {code: target_weight for code in target_codes}
    by_code = day.set_index("ts_code", drop=False)
    next_weights = dict(current)
    desired_turnover = 0.0
    filled_turnover = 0.0
    buy_reject = 0
    sell_reject = 0
    partial_fill = 0
    for code in sorted(set(current) | set(desired)):
        old_w = current.get(code, 0.0)
        target_w = desired.get(code, 0.0)
        delta = target_w - old_w
        if abs(delta) < 1e-12:
            continue
        desired_turnover += abs(delta)
        if code not in by_code.index:
            if delta < 0:
                sell_reject += 1
            continue
        row = by_code.loc[code]
        amount = float(row["next_amount"]) if pd.notna(row["next_amount"]) else 0.0
        max_fill = min(abs(delta), max(0.0, participation_cap * amount / portfolio_nav))
        if delta > 0:
            if not bool(row["buy_executable"]) or max_fill <= 0:
                buy_reject += 1
                continue
            fill = min(delta, max_fill)
            next_weights[code] = old_w + fill
        else:
            if not bool(row["sell_executable"]) or max_fill <= 0:
                sell_reject += 1
                continue
            fill = min(-delta, max_fill)
            next_weights[code] = old_w - fill
        if fill < abs(delta) - 1e-12:
            partial_fill += 1
        filled_turnover += abs(fill)
    next_weights = {code: weight for code, weight in next_weights.items() if weight > 1e-10}
    total = sum(next_weights.values())
    if total > 0:
        next_weights = {code: weight / total for code, weight in next_weights.items()}
    return next_weights, {
        "desired_turnover": desired_turnover,
        "filled_turnover": filled_turnover,
        "buy_reject_count": buy_reject,
        "sell_reject_count": sell_reject,
        "partial_fill_count": partial_fill,
        "position_count": len(next_weights),
    }


def weighted_return(weights: dict[str, float], day: pd.DataFrame) -> float:
    if not weights:
        return float("nan")
    by_code = day.set_index("ts_code")
    value = 0.0
    used = 0.0
    for code, weight in weights.items():
        if code not in by_code.index:
            continue
        ret = by_code.loc[code, "next_open_return_5d"]
        if pd.isna(ret):
            continue
        value += weight * float(ret)
        used += weight
    return float(value / used) if used > 0 else float("nan")


def run(args: argparse.Namespace, data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_cost_rate = (args.cost_bps + args.slippage_bps) / 10000.0
    for split, split_frame in data.groupby("split", sort=True):
        dates = sorted(split_frame["trade_date"].unique())[:: args.rebalance_stride]
        state: dict[tuple[Any, ...], dict[str, float]] = {}
        for trade_date in dates:
            base_day = split_frame[split_frame["trade_date"] == trade_date].dropna(subset=["pred_score", "next_open_return_5d"])
            if len(base_day) < args.min_daily_count:
                continue
            benchmark_return = float(base_day["benchmark_next_open_return_5d"].dropna().iloc[0])
            executable_universe_return = float(base_day.loc[base_day["buy_executable"], "next_open_return_5d"].mean())
            for style_set in args.style_set:
                style_cols = [column for column in STYLE_SETS[style_set] if column in base_day.columns]
                day = add_style_z(base_day, style_cols)
                for k in args.k:
                    if len(day) < k:
                        continue
                    for cap in args.exposure_cap:
                        key = (style_set, k, cap)
                        current = state.get(key, {})
                        selected, constraint_stats = select_hard_constrained(
                            day=day,
                            previous_codes=list(current.keys()),
                            k=k,
                            style_cols=style_cols,
                            cap=cap,
                            candidate_multiplier=args.candidate_multiplier,
                            fallback_soft_cap=args.fallback_soft_cap,
                        )
                        weights, fill_stats = simulate_fill(
                            current=current,
                            target_codes=selected,
                            day=day,
                            portfolio_nav=args.portfolio_nav,
                            participation_cap=args.participation_cap,
                        )
                        state[key] = weights
                        gross = weighted_return(weights, day)
                        cost = total_cost_rate * fill_stats["filled_turnover"]
                        net = gross - cost if pd.notna(gross) else float("nan")
                        rows.append(
                            {
                                "split": split,
                                "trade_date": trade_date,
                                "style_set": style_set,
                                "k": int(k),
                                "exposure_cap": float(cap),
                                "gross_return": gross,
                                "net_return": net,
                                "benchmark_return": benchmark_return,
                                "executable_universe_return": executable_universe_return,
                                "excess_vs_benchmark": net - benchmark_return if pd.notna(net) else float("nan"),
                                "excess_vs_executable_universe": net - executable_universe_return if pd.notna(net) else float("nan"),
                                "transaction_cost": cost,
                                **constraint_stats,
                                **fill_stats,
                            }
                        )
    return pd.DataFrame(rows)


def summarize(periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in periods.groupby(["split", "style_set", "k", "exposure_cap"], sort=True):
        split, style_set, k, cap = keys
        net = summarize_returns(group["net_return"])
        bench = summarize_returns(group["excess_vs_benchmark"])
        execu = summarize_returns(group["excess_vs_executable_universe"])
        rows.append(
            {
                "split": split,
                "style_set": style_set,
                "k": int(k),
                "exposure_cap": float(cap),
                "periods": net["period_count"],
                "net_ann": net["annualized_return"],
                "net_ir": net["ir"],
                "net_max_drawdown": net["max_drawdown"],
                "excess_benchmark_ann": bench["annualized_return"],
                "excess_exec_universe_ann": execu["annualized_return"],
                "avg_desired_turnover": float(group["desired_turnover"].mean()),
                "avg_filled_turnover": float(group["filled_turnover"].mean()),
                "avg_constraint_reject_count": float(group["constraint_reject_count"].mean()),
                "fallback_rate": float(group["used_fallback"].astype(bool).mean()),
                "avg_max_abs_target_exposure_z": float(group["max_abs_target_exposure_z"].mean()),
                "avg_abs_target_exposure_z": float(group["avg_abs_target_exposure_z"].mean()),
                "avg_partial_fill_count": float(group["partial_fill_count"].mean()),
                "avg_position_count": float(group["position_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    periods = run(args, data)
    summary = summarize(periods)
    periods.to_csv(out_dir / "hard_constraint_periods.csv", index=False)
    summary.to_csv(out_dir / "hard_constraint_summary.csv", index=False)
    manifest = {
        "predictions": args.predictions,
        "mart": args.mart,
        "labels": args.labels,
        "output_dir": str(out_dir),
        "style_set": args.style_set,
        "exposure_cap": args.exposure_cap,
        "k": args.k,
        "portfolio_nav": float(args.portfolio_nav),
        "participation_cap": float(args.participation_cap),
        "period_rows": int(len(periods)),
        "summary_rows": int(len(summary)),
        "method": "hard_constrained_greedy_barra_lite",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
