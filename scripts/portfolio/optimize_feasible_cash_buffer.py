from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


RISK_CONTROL_SETS = {
    "none": [],
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
    "industry_size_liquidity_vol_mom": [
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
    "buy_executable_t1_open",
    "sell_executable_t1_open",
    "execution_return_open_to_close5",
    "benchmark_next_open_to_exit_close_return",
]
PREDICTION_COLUMNS = {"trade_date", "ts_code", "pred_score", "split"}


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return sorted(set(values))


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated non-negative floats.")
    return sorted(set(values))


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LP feasible-set optimizer with cash buffer and T+1 fills.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--mart", default="data/mart/datasets/core/dataset_v20260526.parquet")
    parser.add_argument("--labels", default="data/mart/labels/execution_labels_v20260526.parquet")
    parser.add_argument("--output-dir", default="outputs/backtest/optimizer/feasible_cash_buffer")
    parser.add_argument(
        "--risk-control",
        type=parse_str_list,
        default=parse_str_list("none,industry_proxy,industry_size,industry_size_liquidity_vol_mom"),
    )
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("20"))
    parser.add_argument("--style-penalty", type=parse_float_list, default=parse_float_list("0,0.05,0.10,0.20"))
    parser.add_argument("--turnover-penalty", type=parse_float_list, default=parse_float_list("0,0.02"))
    parser.add_argument("--candidate-multiplier", type=float, default=5.0)
    parser.add_argument("--exposure-cap", type=float, default=0.35)
    parser.add_argument("--single-name-cap", type=float, default=0.0)
    parser.add_argument("--min-invested", type=float, default=0.0)
    parser.add_argument("--turnover-cap", type=float, default=1.0)
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
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    predictions = predictions.copy()
    predictions["trade_date"] = predictions["trade_date"].astype(str)
    predictions["ts_code"] = predictions["ts_code"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    predictions["pred_score"] = pd.to_numeric(predictions["pred_score"], errors="coerce")
    date_split = predictions[["trade_date", "split"]].drop_duplicates("trade_date")
    active_dates = set(date_split["trade_date"])

    risk_cols = sorted({column for cols in RISK_CONTROL_SETS.values() for column in cols})
    mart_cols = set(pd.read_parquet(args.mart, engine="pyarrow").columns)
    mart_selected = ["trade_date", "ts_code", *[column for column in risk_cols if column in mart_cols]]
    mart = pd.read_parquet(args.mart, columns=mart_selected)
    labels = pd.read_parquet(args.labels, columns=LABEL_COLUMNS)

    for frame in (mart, labels):
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
    labels = labels[labels["trade_date"].isin(active_dates)].copy()
    mart = mart[mart["trade_date"].isin(active_dates)].copy()

    data = labels.merge(mart, on=["trade_date", "ts_code"], how="left")
    data = data.merge(predictions, on=["trade_date", "ts_code"], how="left")
    data = data.merge(date_split, on="trade_date", how="left", suffixes=("", "_date"))
    data["split"] = data["split"].fillna(data["split_date"]).astype(str)
    data = data.drop(columns=[column for column in ["split_date"] if column in data.columns])

    numeric = [
        "pred_score",
        "next_amount",
        "execution_return_open_to_close5",
        "benchmark_next_open_to_exit_close_return",
        *mart_selected[2:],
    ]
    for column in numeric:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["buy_executable_t1_open", "sell_executable_t1_open"]:
        data[column] = data[column].fillna(False).astype(bool)
    return data.replace([np.inf, -np.inf], np.nan)


def prepare_lp_universe(
    day: pd.DataFrame,
    current: dict[str, float],
    risk_cols: list[str],
    k: int,
    candidate_multiplier: float,
) -> pd.DataFrame:
    scored = day.dropna(subset=["pred_score"]).sort_values("pred_score", ascending=False)
    candidate_count = min(len(scored), max(k, int(math.ceil(k * candidate_multiplier))))
    candidate_codes = set(scored.head(candidate_count)["ts_code"].astype(str))
    candidate_codes.update(current)
    universe = day[day["ts_code"].astype(str).isin(candidate_codes)].copy()
    universe = universe.dropna(subset=["execution_return_open_to_close5"]).reset_index(drop=True)
    if universe.empty:
        return universe

    scored_mask = universe["pred_score"].notna()
    alpha = pd.Series(-3.0, index=universe.index, dtype="float64")
    if scored_mask.any():
        alpha.loc[scored_mask] = zscore(universe.loc[scored_mask, "pred_score"])
    universe["alpha_z"] = alpha
    for column in risk_cols:
        if column in day.columns:
            z_by_code = pd.Series(zscore(day[column]).values, index=day["ts_code"].astype(str))
            universe[f"z__{column}"] = universe["ts_code"].astype(str).map(z_by_code).fillna(0.0)
    return universe


def solve_day_lp(
    universe: pd.DataFrame,
    current: dict[str, float],
    risk_cols: list[str],
    k: int,
    style_penalty: float,
    turnover_penalty: float,
    exposure_cap: float,
    single_name_cap: float,
    min_invested: float,
    turnover_cap: float,
    portfolio_nav: float,
    participation_cap: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    if universe.empty:
        return current, {"optimizer_status": "EmptyUniverse", "fallback_used": True}

    codes = universe["ts_code"].astype(str).tolist()
    n = len(codes)
    e = len(risk_cols)
    old = np.array([current.get(code, 0.0) for code in codes], dtype="float64")
    alpha = universe["alpha_z"].to_numpy(dtype="float64")
    next_amount = universe["next_amount"].fillna(0.0).to_numpy(dtype="float64")
    buyable = universe["buy_executable_t1_open"].to_numpy(dtype=bool)
    sellable = universe["sell_executable_t1_open"].to_numpy(dtype=bool)

    effective_cap = single_name_cap if single_name_cap > 0 else 1.0 / k
    capacity = np.maximum(0.0, participation_cap * next_amount / portfolio_nav)

    # Variables: final weights w, buys b, sells s, absolute exposure slacks e.
    c = np.concatenate(
        [
            -alpha,
            np.full(n, turnover_penalty),
            np.full(n, turnover_penalty),
            np.full(e, style_penalty),
        ]
    )
    bounds: list[tuple[float, float | None]] = []
    bounds.extend((0.0, effective_cap) for _ in range(n))
    bounds.extend((0.0, float(capacity[i]) if buyable[i] else 0.0) for i in range(n))
    bounds.extend((0.0, float(capacity[i]) if sellable[i] else 0.0) for i in range(n))
    bounds.extend((0.0, None) for _ in range(e))

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    var_count = 3 * n + e

    for i in range(n):
        row = np.zeros(var_count)
        row[i] = 1.0
        row[n + i] = -1.0
        rows.append(row)
        rhs.append(float(old[i]))

        row = np.zeros(var_count)
        row[i] = -1.0
        row[2 * n + i] = -1.0
        rows.append(row)
        rhs.append(float(-old[i]))

    row = np.zeros(var_count)
    row[:n] = 1.0
    rows.append(row)
    rhs.append(1.0)

    row = np.zeros(var_count)
    row[:n] = -1.0
    rows.append(row)
    rhs.append(-float(min_invested))

    row = np.zeros(var_count)
    row[n : 3 * n] = 1.0
    rows.append(row)
    rhs.append(float(turnover_cap))

    exposure_values = []
    for j, column in enumerate(risk_cols):
        z_col = f"z__{column}"
        if z_col not in universe.columns:
            continue
        exposure = universe[z_col].to_numpy(dtype="float64")
        exposure_values.append(exposure)

        row = np.zeros(var_count)
        row[:n] = exposure
        row[3 * n + j] = -1.0
        rows.append(row)
        rhs.append(0.0)

        row = np.zeros(var_count)
        row[:n] = -exposure
        row[3 * n + j] = -1.0
        rows.append(row)
        rhs.append(0.0)

        row = np.zeros(var_count)
        row[3 * n + j] = 1.0
        rows.append(row)
        rhs.append(float(exposure_cap))

    try:
        result = linprog(
            c=c,
            A_ub=np.vstack(rows),
            b_ub=np.array(rhs),
            bounds=bounds,
            method="highs",
        )
    except Exception as exc:  # pragma: no cover - defensive solver boundary
        return current, {
            "optimizer_status": "SolverError",
            "solver_message": str(exc),
            "fallback_used": True,
        }

    if not result.success:
        return current, {
            "optimizer_status": "Infeasible",
            "solver_message": str(result.message),
            "fallback_used": True,
        }

    weights = result.x[:n]
    buys = result.x[n : 2 * n]
    sells = result.x[2 * n : 3 * n]
    portfolio = {code: float(weight) for code, weight in zip(codes, weights) if weight > 1e-8}
    abs_exposures = [float(abs(np.dot(weights, exposure))) for exposure in exposure_values]
    invested = float(weights.sum())
    desired_turnover = float(sum(abs(current.get(code, 0.0) - portfolio.get(code, 0.0)) for code in set(current) | set(portfolio)))
    filled_turnover = float(buys.sum() + sells.sum())
    status = "Optimal" if result.status == 0 else "Feasible"
    return portfolio, {
        "optimizer_status": status,
        "solver_message": str(result.message),
        "fallback_used": False,
        "invested_weight": invested,
        "cash_weight": float(max(0.0, 1.0 - invested)),
        "desired_turnover": desired_turnover,
        "filled_turnover": filled_turnover,
        "filled_desired_ratio": float(filled_turnover / desired_turnover) if desired_turnover > 1e-12 else 1.0,
        "buy_turnover": float(buys.sum()),
        "sell_turnover": float(sells.sum()),
        "position_count": int(len(portfolio)),
        "max_abs_exposure_z": float(max(abs_exposures)) if abs_exposures else 0.0,
        "avg_abs_exposure_z": float(np.mean(abs_exposures)) if abs_exposures else 0.0,
        "objective_value": float(-result.fun),
    }


def weighted_return(weights: dict[str, float], day: pd.DataFrame) -> float:
    if not weights:
        return 0.0
    by_code = day.set_index("ts_code")
    value = 0.0
    for code, weight in weights.items():
        if code in by_code.index and pd.notna(by_code.loc[code, "execution_return_open_to_close5"]):
            value += weight * float(by_code.loc[code, "execution_return_open_to_close5"])
    return float(value)


def run_optimizer(args: argparse.Namespace, data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_cost_rate = (args.cost_bps + args.slippage_bps) / 10000.0
    for split, split_frame in data.groupby("split", sort=True):
        dates = sorted(split_frame["trade_date"].unique())[:: args.rebalance_stride]
        states: dict[tuple[str, int, float, float], dict[str, float]] = {}
        for trade_date in dates:
            day = split_frame[split_frame["trade_date"] == trade_date].copy()
            if day["pred_score"].notna().sum() < args.min_daily_count:
                continue
            benchmark_return = float(day["benchmark_next_open_to_exit_close_return"].dropna().iloc[0])
            exec_universe_return = float(day.loc[day["buy_executable_t1_open"], "execution_return_open_to_close5"].mean())
            for risk_control in args.risk_control:
                risk_cols = [column for column in RISK_CONTROL_SETS.get(risk_control, []) if column in day.columns]
                for k in args.k:
                    for style_penalty in args.style_penalty:
                        for turnover_penalty in args.turnover_penalty:
                            key = (risk_control, k, style_penalty, turnover_penalty)
                            current = states.get(key, {})
                            universe = prepare_lp_universe(day, current, risk_cols, k, args.candidate_multiplier)
                            weights, stats = solve_day_lp(
                                universe=universe,
                                current=current,
                                risk_cols=risk_cols,
                                k=k,
                                style_penalty=style_penalty,
                                turnover_penalty=turnover_penalty,
                                exposure_cap=args.exposure_cap,
                                single_name_cap=args.single_name_cap,
                                min_invested=args.min_invested,
                                turnover_cap=args.turnover_cap,
                                portfolio_nav=args.portfolio_nav,
                                participation_cap=args.participation_cap,
                            )
                            states[key] = weights
                            gross = weighted_return(weights, day)
                            cost = total_cost_rate * float(stats.get("filled_turnover", 0.0))
                            net = gross - cost
                            rows.append(
                                {
                                    "split": split,
                                    "trade_date": trade_date,
                                    "risk_control": risk_control,
                                    "k": int(k),
                                    "style_penalty": float(style_penalty),
                                    "turnover_penalty": float(turnover_penalty),
                                    "gross_return": gross,
                                    "net_return": net,
                                    "benchmark_return": benchmark_return,
                                    "executable_universe_return": exec_universe_return,
                                    "excess_vs_benchmark": net - benchmark_return,
                                    "excess_vs_executable_universe": net - exec_universe_return,
                                    "transaction_cost": cost,
                                    "portfolio_nav": float(args.portfolio_nav),
                                    "participation_cap": float(args.participation_cap),
                                    "turnover_cap": float(args.turnover_cap),
                                    "single_name_cap": float(args.single_name_cap if args.single_name_cap > 0 else 1.0 / k),
                                    "min_invested": float(args.min_invested),
                                    **stats,
                                }
                            )
    return pd.DataFrame(rows)


def summarize(periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["split", "risk_control", "k", "style_penalty", "turnover_penalty"]
    for values, group in periods.groupby(keys, sort=True):
        net = summarize_returns(group["net_return"])
        bench = summarize_returns(group["excess_vs_benchmark"])
        execu = summarize_returns(group["excess_vs_executable_universe"])
        row = dict(zip(keys, values))
        row.update(
            {
                "periods": net["period_count"],
                "net_ann": net["annualized_return"],
                "net_ir": net["ir"],
                "net_max_drawdown": net["max_drawdown"],
                "excess_benchmark_ann": bench["annualized_return"],
                "excess_exec_universe_ann": execu["annualized_return"],
                "avg_desired_turnover": float(group["desired_turnover"].mean()),
                "avg_filled_turnover": float(group["filled_turnover"].mean()),
                "avg_filled_desired_ratio": float(group["filled_desired_ratio"].mean()),
                "avg_position_count": float(group["position_count"].mean()),
                "avg_cash_weight": float(group["cash_weight"].mean()),
                "avg_invested_weight": float(group["invested_weight"].mean()),
                "optimal_rate": float(group["optimizer_status"].eq("Optimal").mean()),
                "feasible_rate": float(group["optimizer_status"].isin(["Optimal", "Feasible"]).mean()),
                "solver_error_rate": float(group["optimizer_status"].eq("SolverError").mean()),
                "fallback_rate": float(group["fallback_used"].astype(bool).mean()),
                "avg_max_abs_exposure_z": float(group["max_abs_exposure_z"].mean()),
                "avg_abs_exposure_z": float(group["avg_abs_exposure_z"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    periods = run_optimizer(args, data)
    summary = summarize(periods)
    periods.to_csv(out_dir / "optimizer_periods.csv", index=False)
    summary.to_csv(out_dir / "optimizer_summary.csv", index=False)
    manifest = {
        "predictions": args.predictions,
        "mart": args.mart,
        "labels": args.labels,
        "output_dir": str(out_dir),
        "risk_control": args.risk_control,
        "k": args.k,
        "style_penalty": args.style_penalty,
        "turnover_penalty": args.turnover_penalty,
        "candidate_multiplier": float(args.candidate_multiplier),
        "exposure_cap": float(args.exposure_cap),
        "single_name_cap": float(args.single_name_cap),
        "min_invested": float(args.min_invested),
        "turnover_cap": float(args.turnover_cap),
        "portfolio_nav": float(args.portfolio_nav),
        "participation_cap": float(args.participation_cap),
        "period_rows": int(len(periods)),
        "summary_rows": int(len(summary)),
        "method": "linear_program_feasible_set_optimizer_cash_buffer_t1",
    }
    (out_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
