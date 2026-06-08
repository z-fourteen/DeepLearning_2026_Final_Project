from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


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


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated non-negative floats.")
    return sorted(set(values))


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive ints.")
    return sorted(set(values))


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LP feasible-set optimizer with cash allowed.")
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_resid_industry_proxy_full_style/predictions.parquet",
    )
    parser.add_argument("--mart", default="data/mart/datasets/dataset_v20260526.parquet")
    parser.add_argument("--labels", default="data/mart/labels/labels_canonical_v20260526.parquet")
    parser.add_argument("--output-dir", default="outputs/backtest/optimizer/lp_feasible_set")
    parser.add_argument("--style-set", type=parse_str_list, default=parse_str_list("industry_proxy,industry_size,full_style"))
    parser.add_argument("--exposure-cap", type=parse_float_list, default=parse_float_list("0.10,0.15,0.25,0.35"))
    parser.add_argument("--candidate-count", type=parse_int_list, default=parse_int_list("80,120"))
    parser.add_argument("--single-name-cap", type=float, default=0.05)
    parser.add_argument("--min-invested", type=parse_float_list, default=parse_float_list("0.6,0.8,1.0"))
    parser.add_argument("--turnover-cap", type=parse_float_list, default=parse_float_list("0.3,0.6,1.0"))
    parser.add_argument("--turnover-penalty", type=float, default=0.002)
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
    annualized = float((1.0 + cumulative) ** (periods_per_year / len(clean)) - 1.0)
    return {
        "period_count": int(len(clean)),
        "annualized_return": annualized,
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


def prepare_candidates(day: pd.DataFrame, style_cols: list[str], candidate_count: int) -> pd.DataFrame:
    selected = day.dropna(subset=["pred_score", "next_open_return_5d"]).copy()
    selected = selected.sort_values("pred_score", ascending=False).head(candidate_count).copy()
    selected["alpha_z"] = zscore(selected["pred_score"])
    for column in style_cols:
        if column in day.columns:
            universe_z = zscore(day[column])
            z_map = pd.Series(universe_z.values, index=day["ts_code"].astype(str))
            selected[f"z__{column}"] = selected["ts_code"].astype(str).map(z_map).fillna(0.0)
    return selected.reset_index(drop=True)


def optimize_weights(
    candidates: pd.DataFrame,
    current: dict[str, float],
    style_cols: list[str],
    exposure_cap: float,
    single_name_cap: float,
    min_invested: float,
    turnover_cap: float,
    turnover_penalty: float,
    portfolio_nav: float,
    participation_cap: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    n = len(candidates)
    if n == 0:
        return {}, {"solver_success": False, "solver_status": "empty_candidates"}
    codes = candidates["ts_code"].astype(str).tolist()
    old = np.array([current.get(code, 0.0) for code in codes], dtype="float64")
    alpha = candidates["alpha_z"].to_numpy(dtype="float64")
    buyable = candidates["buy_executable"].to_numpy(dtype=bool)
    sellable = candidates["sell_executable"].to_numpy(dtype=bool)
    next_amount = candidates["next_amount"].fillna(0.0).to_numpy(dtype="float64")

    # Variables: w_i, buy_i, sell_i. Minimize -alpha'w + tc_penalty * (buy + sell).
    c = np.concatenate([-alpha, np.full(n, turnover_penalty), np.full(n, turnover_penalty)])
    bounds = []
    max_adv_weight = participation_cap * next_amount / portfolio_nav
    for idx in range(n):
        upper = min(single_name_cap, 1.0)
        if not buyable[idx] and old[idx] <= 0:
            upper = 0.0
        bounds.append((0.0, upper))
    for idx in range(n):
        upper = max(0.0, max_adv_weight[idx]) if buyable[idx] else 0.0
        bounds.append((0.0, upper))
    for idx in range(n):
        upper = max(0.0, max_adv_weight[idx]) if sellable[idx] else 0.0
        bounds.append((0.0, upper))

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    # w - buy <= old, old - w - sell <= 0.
    for idx in range(n):
        row = np.zeros(3 * n)
        row[idx] = 1.0
        row[n + idx] = -1.0
        a_ub.append(row)
        b_ub.append(float(old[idx]))

        row = np.zeros(3 * n)
        row[idx] = -1.0
        row[2 * n + idx] = -1.0
        a_ub.append(row)
        b_ub.append(float(-old[idx]))

    # Invested weight: sum(w) <= 1, sum(w) >= min_invested.
    row = np.zeros(3 * n)
    row[:n] = 1.0
    a_ub.append(row)
    b_ub.append(1.0)

    row = np.zeros(3 * n)
    row[:n] = -1.0
    a_ub.append(row)
    b_ub.append(-min_invested)

    # Turnover cap on explicit buy+sell variables.
    row = np.zeros(3 * n)
    row[n:] = 1.0
    a_ub.append(row)
    b_ub.append(turnover_cap)

    # Existing holdings outside candidate set are assumed liquidated only when not represented here.
    # This first LP constrains the candidate sleeve and records off-candidate old weight separately.
    for column in style_cols:
        z_col = f"z__{column}"
        if z_col not in candidates.columns:
            continue
        exposure = candidates[z_col].to_numpy(dtype="float64")
        row = np.zeros(3 * n)
        row[:n] = exposure
        a_ub.append(row)
        b_ub.append(exposure_cap)

        row = np.zeros(3 * n)
        row[:n] = -exposure
        a_ub.append(row)
        b_ub.append(exposure_cap)

    result = linprog(
        c=c,
        A_ub=np.vstack(a_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return {}, {
            "solver_success": False,
            "solver_status": str(result.message),
            "invested_weight": 0.0,
            "cash_weight": 1.0,
            "turnover": 0.0,
            "max_abs_exposure_z": float("nan"),
            "avg_abs_exposure_z": float("nan"),
        }
    weights = result.x[:n]
    buys = result.x[n : 2 * n]
    sells = result.x[2 * n :]
    exposure_values = []
    for column in style_cols:
        z_col = f"z__{column}"
        if z_col in candidates.columns:
            exposure_values.append(float(np.dot(weights, candidates[z_col].to_numpy(dtype="float64"))))
    portfolio = {code: float(weight) for code, weight in zip(codes, weights) if weight > 1e-8}
    invested = float(weights.sum())
    return portfolio, {
        "solver_success": True,
        "solver_status": "optimal",
        "invested_weight": invested,
        "cash_weight": float(max(0.0, 1.0 - invested)),
        "turnover": float(buys.sum() + sells.sum()),
        "buy_turnover": float(buys.sum()),
        "sell_turnover": float(sells.sum()),
        "max_abs_exposure_z": float(np.max(np.abs(exposure_values))) if exposure_values else 0.0,
        "avg_abs_exposure_z": float(np.mean(np.abs(exposure_values))) if exposure_values else 0.0,
        "nonzero_names": int(len(portfolio)),
        "objective_value": float(-result.fun),
    }


def weighted_return(weights: dict[str, float], day: pd.DataFrame) -> float:
    if not weights:
        return 0.0
    by_code = day.set_index("ts_code")
    value = 0.0
    for code, weight in weights.items():
        if code not in by_code.index:
            continue
        ret = by_code.loc[code, "next_open_return_5d"]
        if pd.notna(ret):
            value += weight * float(ret)
    return float(value)


def run(args: argparse.Namespace, data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cost_rate = (args.cost_bps + args.slippage_bps) / 10000.0
    for split, split_frame in data.groupby("split", sort=True):
        dates = sorted(split_frame["trade_date"].unique())[:: args.rebalance_stride]
        states: dict[tuple[Any, ...], dict[str, float]] = {}
        for trade_date in dates:
            day = split_frame[split_frame["trade_date"] == trade_date].copy()
            if len(day) < args.min_daily_count:
                continue
            benchmark_return = float(day["benchmark_next_open_return_5d"].dropna().iloc[0])
            exec_universe_return = float(day.loc[day["buy_executable"], "next_open_return_5d"].mean())
            for style_set in args.style_set:
                style_cols = [column for column in STYLE_SETS[style_set] if column in day.columns]
                for candidate_count in args.candidate_count:
                    candidates = prepare_candidates(day, style_cols, candidate_count)
                    if len(candidates) < args.min_daily_count:
                        continue
                    for exposure_cap in args.exposure_cap:
                        for min_invested in args.min_invested:
                            for turnover_cap in args.turnover_cap:
                                key = (style_set, candidate_count, exposure_cap, min_invested, turnover_cap)
                                current = states.get(key, {})
                                weights, stats = optimize_weights(
                                    candidates=candidates,
                                    current=current,
                                    style_cols=style_cols,
                                    exposure_cap=exposure_cap,
                                    single_name_cap=args.single_name_cap,
                                    min_invested=min_invested,
                                    turnover_cap=turnover_cap,
                                    turnover_penalty=args.turnover_penalty,
                                    portfolio_nav=args.portfolio_nav,
                                    participation_cap=args.participation_cap,
                                )
                                states[key] = weights
                                gross = weighted_return(weights, candidates)
                                cost = cost_rate * stats.get("turnover", 0.0)
                                net = gross - cost
                                rows.append(
                                    {
                                        "split": split,
                                        "trade_date": trade_date,
                                        "style_set": style_set,
                                        "candidate_count": int(candidate_count),
                                        "exposure_cap": float(exposure_cap),
                                        "min_invested": float(min_invested),
                                        "turnover_cap": float(turnover_cap),
                                        "gross_return": gross,
                                        "net_return": net,
                                        "benchmark_return": benchmark_return,
                                        "executable_universe_return": exec_universe_return,
                                        "excess_vs_benchmark": net - benchmark_return,
                                        "excess_vs_executable_universe": net - exec_universe_return,
                                        "transaction_cost": cost,
                                        **stats,
                                    }
                                )
    return pd.DataFrame(rows)


def summarize(periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["split", "style_set", "candidate_count", "exposure_cap", "min_invested", "turnover_cap"]
    for key_values, group in periods.groupby(keys, sort=True):
        net = summarize_returns(group["net_return"])
        bench = summarize_returns(group["excess_vs_benchmark"])
        execu = summarize_returns(group["excess_vs_executable_universe"])
        row = dict(zip(keys, key_values))
        row.update(
            {
                "periods": net["period_count"],
                "net_ann": net["annualized_return"],
                "net_ir": net["ir"],
                "net_max_drawdown": net["max_drawdown"],
                "excess_benchmark_ann": bench["annualized_return"],
                "excess_exec_universe_ann": execu["annualized_return"],
                "solver_success_rate": float(group["solver_success"].astype(bool).mean()),
                "avg_invested_weight": float(group["invested_weight"].mean()),
                "avg_cash_weight": float(group["cash_weight"].mean()),
                "avg_turnover": float(group["turnover"].mean()),
                "avg_max_abs_exposure_z": float(group["max_abs_exposure_z"].mean()),
                "avg_abs_exposure_z": float(group["avg_abs_exposure_z"].mean()),
                "avg_nonzero_names": float(group["nonzero_names"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    periods = run(args, data)
    summary = summarize(periods)
    periods.to_csv(out_dir / "lp_optimizer_periods.csv", index=False)
    summary.to_csv(out_dir / "lp_optimizer_summary.csv", index=False)
    manifest = {
        "predictions": args.predictions,
        "mart": args.mart,
        "labels": args.labels,
        "output_dir": str(out_dir),
        "style_set": args.style_set,
        "exposure_cap": args.exposure_cap,
        "candidate_count": args.candidate_count,
        "min_invested": args.min_invested,
        "turnover_cap": args.turnover_cap,
        "single_name_cap": float(args.single_name_cap),
        "portfolio_nav": float(args.portfolio_nav),
        "participation_cap": float(args.participation_cap),
        "period_rows": int(len(periods)),
        "summary_rows": int(len(summary)),
        "method": "linear_program_feasible_set_optimizer_cash_allowed",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
