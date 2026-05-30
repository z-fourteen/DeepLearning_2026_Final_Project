from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cvxpy as cp


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
COMPETITION_MIN_INVESTED = 0.80


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
    parser.add_argument("--exposure-slack-penalty", type=float, default=25.0)
    parser.add_argument(
        "--buy-capacity-slack-penalty",
        type=float,
        default=1000.0,
        help=(
            "Penalty for overriding T+1 buy participation capacity. The >=80% invested rule stays hard; "
            "this slack records any capacity override needed to satisfy it."
        ),
    )
    parser.add_argument("--cash-penalty", type=float, default=0.0)
    parser.add_argument("--min-invested-shortfall-penalty", type=float, default=0.0)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--single-name-cap", type=float, default=0.0)
    parser.add_argument(
        "--min-invested",
        type=float,
        default=COMPETITION_MIN_INVESTED,
        help="Hard minimum invested weight. Default 0.80 matches the competition exposure rule.",
    )
    parser.add_argument("--turnover-cap", type=float, default=1.0)
    parser.add_argument("--portfolio-nav", type=float, default=10_000_000.0)
    parser.add_argument("--participation-cap", type=float, default=0.03)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--rebalance-stride", type=int, default=5)
    parser.add_argument("--min-daily-count", type=int, default=40)
    args = parser.parse_args()
    if not 0.0 <= args.min_invested <= 1.0:
        raise ValueError(f"--min-invested must be in [0, 1], got {args.min_invested}")
    if args.min_invested_shortfall_penalty > 0:
        print(
            "WARNING: --min-invested-shortfall-penalty softens the minimum invested constraint; "
            "competition runs should keep it at 0 for a hard >=80% invested rule."
        )
    if args.buy_capacity_slack_penalty <= 0:
        raise ValueError(
            f"--buy-capacity-slack-penalty must be positive, got {args.buy_capacity_slack_penalty}"
        )
    return args


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
    min_invested: float,
    single_name_cap: float,
    portfolio_nav: float,
    participation_cap: float,
) -> pd.DataFrame:
    scored = day.dropna(subset=["pred_score"]).sort_values("pred_score", ascending=False)
    candidate_count = min(len(scored), max(k, int(math.ceil(k * candidate_multiplier))))
    if min_invested > 0 and not scored.empty:
        effective_cap = float(single_name_cap if single_name_cap > 0 else 1.0 / k)
        current_invested = float(sum(current.values()))
        selected_codes: set[str] = set(current)
        for row in scored.itertuples(index=False):
            selected_codes.add(str(row.ts_code))
            candidate_count = max(candidate_count, len(selected_codes))
            if len(selected_codes) < candidate_count:
                continue
            selected = scored[scored["ts_code"].astype(str).isin(selected_codes)]
            buyable = selected["buy_executable_t1_open"].fillna(False).astype(bool).to_numpy()
            amount = selected["next_amount"].fillna(0.0).astype("float64").to_numpy()
            buy_capacity = np.where(
                buyable,
                np.maximum(0.0, float(participation_cap) * amount / float(portfolio_nav)),
                0.0,
            )
            old_weight = selected["ts_code"].astype(str).map(current).fillna(0.0).astype("float64").to_numpy()
            max_reachable = float(np.minimum(effective_cap, old_weight + buy_capacity).sum())
            if max(current_invested, max_reachable) >= float(min_invested):
                break
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


def portfolio_stats(
    *,
    status: str,
    message: str,
    fallback_used: bool,
    codes: list[str],
    weights: np.ndarray,
    current: dict[str, float],
    exposure_values: list[np.ndarray],
    exposure_cap: float,
    min_invested: float,
    buys_value: np.ndarray | None = None,
    sells_value: np.ndarray | None = None,
    exposure_slacks: np.ndarray | None = None,
    buy_capacity_slack: np.ndarray | None = None,
    min_invested_shortfall: float | None = None,
    objective_value: float = float("nan"),
) -> tuple[dict[str, float], dict[str, Any]]:
    buys = np.zeros_like(weights) if buys_value is None else buys_value
    sells = np.zeros_like(weights) if sells_value is None else sells_value
    slacks = np.array([], dtype="float64") if exposure_slacks is None else exposure_slacks
    buy_slack = np.array([], dtype="float64") if buy_capacity_slack is None else buy_capacity_slack

    portfolio = {code: float(weight) for code, weight in zip(codes, weights) if weight > 1e-8}
    invested = float(weights.sum())
    desired_turnover = float(
        sum(abs(current.get(code, 0.0) - portfolio.get(code, 0.0)) for code in set(current) | set(portfolio))
    )
    filled_turnover = float(buys.sum() + sells.sum())
    abs_exposures = [float(abs(np.dot(weights, exposure))) for exposure in exposure_values]
    cap_breach = [max(0.0, exposure - float(exposure_cap)) for exposure in abs_exposures]
    shortfall = (
        max(0.0, float(min_invested) - invested)
        if min_invested_shortfall is None
        else float(max(0.0, min_invested_shortfall))
    )

    return portfolio, {
        "optimizer_status": status,
        "solver_message": message,
        "fallback_used": fallback_used,
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
        "max_exposure_slack": float(slacks.max()) if len(slacks) else (float(max(cap_breach)) if cap_breach else 0.0),
        "avg_exposure_slack": float(slacks.mean()) if len(slacks) else (float(np.mean(cap_breach)) if cap_breach else 0.0),
        "total_exposure_slack": float(slacks.sum()) if len(slacks) else (float(sum(cap_breach)) if cap_breach else 0.0),
        "buy_capacity_slack": float(buy_slack.sum()) if len(buy_slack) else 0.0,
        "min_invested_shortfall": shortfall,
        "min_invested_rule_ok": float(shortfall <= 1e-6),
        "objective_value": objective_value,
    }


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
    exposure_slack_penalty: float,
    buy_capacity_slack_penalty: float,
    cash_penalty: float,
    min_invested_shortfall_penalty: float,
    solver: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    if universe.empty:
        codes = sorted(current)
        weights = np.array([current[code] for code in codes], dtype="float64")
        return portfolio_stats(
            status="MinInvestedUnreachable",
            message="empty universe",
            fallback_used=True,
            codes=codes,
            weights=weights,
            current=current,
            exposure_values=[],
            exposure_cap=exposure_cap,
            min_invested=min_invested,
        )

    codes = universe["ts_code"].astype(str).tolist()
    n = len(codes)
    old = np.array([current.get(code, 0.0) for code in codes], dtype="float64")
    alpha = universe["alpha_z"].to_numpy(dtype="float64")
    next_amount = universe["next_amount"].fillna(0.0).to_numpy(dtype="float64")
    buyable = universe["buy_executable_t1_open"].to_numpy(dtype=bool)
    sellable = universe["sell_executable_t1_open"].to_numpy(dtype=bool)

    effective_cap = single_name_cap if single_name_cap > 0 else 1.0 / k
    capacity = np.maximum(0.0, participation_cap * next_amount / portfolio_nav)

    exposure_values: list[np.ndarray] = []
    for column in risk_cols:
        z_col = f"z__{column}"
        if z_col in universe.columns:
            exposure_values.append(universe[z_col].to_numpy(dtype="float64"))
    e_active = len(exposure_values)

    def legal_fallback(status: str, message: str) -> tuple[dict[str, float], dict[str, Any]]:
        weights = old.copy()
        buys_value = np.zeros(n, dtype="float64")
        sell_value = np.zeros(n, dtype="float64")
        remaining_turnover = float(turnover_cap)
        buy_capacity = np.where(buyable, capacity, 0.0)

        for idx in np.argsort(-alpha):
            invested = float(weights.sum())
            if invested >= float(min_invested) - 1e-8 or remaining_turnover <= 1e-12:
                break
            room = max(0.0, float(effective_cap) - float(weights[idx]))
            fill = min(room, float(buy_capacity[idx]), remaining_turnover, float(min_invested) - invested)
            if fill <= 1e-12:
                continue
            weights[idx] += fill
            buys_value[idx] += fill
            remaining_turnover -= fill

        fallback_status = "LegalFallback" if float(weights.sum()) >= float(min_invested) - 1e-6 else "MinInvestedUnreachable"
        return portfolio_stats(
            status=fallback_status,
            message=message if fallback_status == "LegalFallback" else f"{message}; min invested unreachable",
            fallback_used=True,
            codes=codes,
            weights=weights,
            current=current,
            exposure_values=exposure_values,
            exposure_cap=exposure_cap,
            min_invested=min_invested,
            buys_value=buys_value,
            sells_value=sell_value,
        )

    try:
        w = cp.Variable(n, nonneg=True)
        buys = cp.Variable(n, nonneg=True)
        sells = cp.Variable(n, nonneg=True)
        abs_exposure = cp.Variable(e_active, nonneg=True) if e_active else None
        slack_pos = cp.Variable(e_active, nonneg=True) if e_active else None
        slack_neg = cp.Variable(e_active, nonneg=True) if e_active else None
        invested_shortfall = cp.Variable(nonneg=True) if min_invested_shortfall_penalty > 0 else None

        buy_capacity = np.where(buyable, capacity, 0.0)
        sell_capacity = np.where(sellable, capacity, 0.0)
        buy_capacity_slack = cp.Variable(n, nonneg=True)
        constraints = [
            w <= effective_cap,
            buys <= buy_capacity + buy_capacity_slack,
            sells <= sell_capacity,
            w == old + buys - sells,
            cp.sum(w) <= 1.0,
            cp.sum(buys + sells) <= float(turnover_cap),
        ]
        if invested_shortfall is not None:
            constraints.append(cp.sum(w) + invested_shortfall >= float(min_invested))
        else:
            constraints.append(cp.sum(w) >= float(min_invested))

        style_cost = 0.0
        slack_cost = 0.0
        if e_active and abs_exposure is not None and slack_pos is not None and slack_neg is not None:
            for j, exposure in enumerate(exposure_values):
                exp_j = exposure @ w
                constraints.extend(
                    [
                        abs_exposure[j] >= exp_j,
                        abs_exposure[j] >= -exp_j,
                        exp_j <= float(exposure_cap) + slack_pos[j],
                        -exp_j <= float(exposure_cap) + slack_neg[j],
                    ]
                )
            style_cost = float(style_penalty) * cp.sum(abs_exposure)
            slack_cost = float(exposure_slack_penalty) * cp.sum(slack_pos + slack_neg)

        objective = cp.Maximize(
            alpha @ w
            + float(cash_penalty) * cp.sum(w)
            - float(turnover_penalty) * cp.sum(buys + sells)
            - style_cost
            - slack_cost
            - float(buy_capacity_slack_penalty) * cp.sum(buy_capacity_slack)
            - (
                float(min_invested_shortfall_penalty) * invested_shortfall
                if invested_shortfall is not None
                else 0.0
            )
        )
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver.upper())
    except Exception as exc:  # pragma: no cover - defensive solver boundary
        return legal_fallback("SolverError", str(exc))

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return legal_fallback(str(problem.status), str(problem.status))

    weights = np.asarray(w.value, dtype="float64")
    buys_value = np.asarray(buys.value, dtype="float64")
    sells_value = np.asarray(sells.value, dtype="float64")
    pos_slacks = np.asarray(slack_pos.value, dtype="float64") if e_active and slack_pos is not None else np.array([], dtype="float64")
    neg_slacks = np.asarray(slack_neg.value, dtype="float64") if e_active and slack_neg is not None else np.array([], dtype="float64")
    exposure_slacks = pos_slacks + neg_slacks
    buy_capacity_slack_value = np.asarray(buy_capacity_slack.value, dtype="float64")
    shortfall_value = float(invested_shortfall.value) if min_invested_shortfall_penalty > 0 and invested_shortfall is not None else 0.0
    invested = float(weights.sum())
    if min_invested_shortfall_penalty <= 0 and invested < float(min_invested) - 1e-6:
        return legal_fallback(str(problem.status), "solver returned portfolio below hard min invested")
    status = "Optimal" if problem.status == cp.OPTIMAL else "Feasible"
    return portfolio_stats(
        status=status,
        message=str(problem.status),
        fallback_used=False,
        codes=codes,
        weights=weights,
        current=current,
        exposure_values=exposure_values,
        exposure_cap=exposure_cap,
        min_invested=min_invested,
        buys_value=buys_value,
        sells_value=sells_value,
        exposure_slacks=exposure_slacks,
        buy_capacity_slack=buy_capacity_slack_value,
        min_invested_shortfall=shortfall_value,
        objective_value=float(problem.value) if problem.value is not None else float("nan"),
    )


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
                            universe = prepare_lp_universe(
                                day,
                                current,
                                risk_cols,
                                k,
                                args.candidate_multiplier,
                                args.min_invested,
                                args.single_name_cap,
                                args.portfolio_nav,
                                args.participation_cap,
                            )
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
                                exposure_slack_penalty=args.exposure_slack_penalty,
                                buy_capacity_slack_penalty=args.buy_capacity_slack_penalty,
                                cash_penalty=args.cash_penalty,
                                min_invested_shortfall_penalty=args.min_invested_shortfall_penalty,
                                solver=args.solver,
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
                                    "exposure_cap": float(args.exposure_cap),
                                    "exposure_slack_penalty": float(args.exposure_slack_penalty),
                                    "buy_capacity_slack_penalty": float(args.buy_capacity_slack_penalty),
                                    "cash_penalty": float(args.cash_penalty),
                                    "min_invested_shortfall_penalty": float(args.min_invested_shortfall_penalty),
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
                "avg_max_exposure_slack": float(group["max_exposure_slack"].mean()),
                "avg_exposure_slack": float(group["avg_exposure_slack"].mean()),
                "avg_total_exposure_slack": float(group["total_exposure_slack"].mean()),
                "avg_buy_capacity_slack": float(group["buy_capacity_slack"].mean()),
                "avg_min_invested_shortfall": float(group["min_invested_shortfall"].mean()),
                "min_invested_rule_pass_rate": float(group["min_invested_rule_ok"].mean()),
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
        "exposure_slack_penalty": float(args.exposure_slack_penalty),
        "buy_capacity_slack_penalty": float(args.buy_capacity_slack_penalty),
        "cash_penalty": float(args.cash_penalty),
        "min_invested_shortfall_penalty": float(args.min_invested_shortfall_penalty),
        "solver": args.solver,
        "single_name_cap": float(args.single_name_cap),
        "min_invested": float(args.min_invested),
        "turnover_cap": float(args.turnover_cap),
        "portfolio_nav": float(args.portfolio_nav),
        "participation_cap": float(args.participation_cap),
        "period_rows": int(len(periods)),
        "summary_rows": int(len(summary)),
        "method": "cvxpy_soft_exposure_slack_optimizer_cash_buffer_t1",
    }
    (out_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
