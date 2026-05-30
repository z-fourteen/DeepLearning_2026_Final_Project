from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    "industry_size_liquidity": [
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

FULL_STYLE_EVAL_COLUMNS = [
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
]


PREDICTION_COLUMNS = {"trade_date", "ts_code", "pred_score", "split"}
LABEL_COLUMNS = [
    "trade_date",
    "ts_code",
    "next_amount",
    "buy_executable_t1_open",
    "sell_executable_t1_open",
    "execution_return_open_to_close5",
    "benchmark_next_open_to_exit_close_return",
]


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
    parser = argparse.ArgumentParser(description="Run a Barra-lite constrained portfolio optimizer.")
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet",
    )
    parser.add_argument("--mart", default="data/mart/datasets/dataset_v20260526.parquet")
    parser.add_argument("--labels", default="data/mart/labels/execution_labels_v20260526.parquet")
    parser.add_argument("--output-dir", default="outputs/backtest/optimizer/barra_lite")
    parser.add_argument("--risk-control", type=parse_str_list, default=parse_str_list("none,industry_proxy,industry_size,industry_size_liquidity_vol_mom"))
    parser.add_argument("--k", type=parse_int_list, default=parse_int_list("30"))
    parser.add_argument("--candidate-multiplier", type=float, default=5.0)
    parser.add_argument("--style-penalty", type=parse_float_list, default=parse_float_list("0,0.05,0.10,0.20"))
    parser.add_argument("--turnover-penalty", type=parse_float_list, default=parse_float_list("0,0.02"))
    parser.add_argument("--portfolio-nav", type=float, default=10_000_000.0)
    parser.add_argument("--participation-cap", type=float, default=0.03)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--rebalance-stride", type=int, default=5)
    parser.add_argument("--single-name-cap", type=float, default=0.05)
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
    median = values.median()
    values = values.fillna(median)
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
        return {
            "period_count": 0,
            "mean": float("nan"),
            "ir": float("nan"),
            "annualized_return": float("nan"),
            "max_drawdown": float("nan"),
        }
    std = clean.std(ddof=1)
    cumulative = float((1.0 + clean).prod() - 1.0)
    annualized = float((1.0 + cumulative) ** (periods_per_year / len(clean)) - 1.0)
    return {
        "period_count": int(len(clean)),
        "mean": float(clean.mean()),
        "ir": float(clean.mean() / std) if std and pd.notna(std) else float("nan"),
        "annualized_return": annualized,
        "max_drawdown": max_drawdown(clean),
    }


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    predictions = pd.read_parquet(args.predictions)
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    risk_cols = sorted({column for cols in RISK_CONTROL_SETS.values() for column in cols})
    mart_cols = set(pd.read_parquet(args.mart, engine="pyarrow").columns)
    mart_selected = ["trade_date", "ts_code", *[column for column in risk_cols if column in mart_cols]]
    mart = pd.read_parquet(args.mart, columns=mart_selected)
    labels = pd.read_parquet(args.labels, columns=LABEL_COLUMNS)
    data = predictions.merge(mart, on=["trade_date", "ts_code"], how="inner")
    data = data.merge(labels, on=["trade_date", "ts_code"], how="inner")
    for column in ["trade_date", "ts_code", "split"]:
        data[column] = data[column].astype(str)
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


def select_signal_dates(dates: list[str], stride: int) -> list[str]:
    if stride <= 0:
        raise ValueError("--rebalance-stride must be positive")
    return dates[::stride]


def optimize_codes(
    day: pd.DataFrame,
    previous_codes: list[str],
    k: int,
    risk_cols: list[str],
    style_penalty: float,
    turnover_penalty: float,
    candidate_multiplier: float,
) -> list[str]:
    candidate_count = min(len(day), max(k, int(math.ceil(k * candidate_multiplier))))
    candidates = day.sort_values("pred_score", ascending=False).head(candidate_count).copy()
    candidates["alpha_z"] = zscore(candidates["pred_score"])
    risk_penalty = pd.Series(0.0, index=candidates.index)
    for column in risk_cols:
        if column not in candidates.columns:
            continue
        risk_penalty += zscore(candidates[column]).abs()
    if risk_cols:
        risk_penalty = risk_penalty / len(risk_cols)
    candidates["risk_penalty"] = risk_penalty
    candidates["turnover_penalty"] = (~candidates["ts_code"].isin(previous_codes)).astype(float)
    candidates["objective"] = (
        candidates["alpha_z"]
        - style_penalty * candidates["risk_penalty"]
        - turnover_penalty * candidates["turnover_penalty"]
    )
    selected = candidates.sort_values("objective", ascending=False)["ts_code"].astype(str).head(k).tolist()
    return selected


def simulate_fill(
    current: dict[str, float],
    target_codes: list[str],
    day: pd.DataFrame,
    single_name_cap: float,
    portfolio_nav: float,
    participation_cap: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    target_weight = min(single_name_cap, 1.0 / len(target_codes)) if target_codes else 0.0
    desired = {code: target_weight for code in target_codes}
    if sum(desired.values()) > 0:
        scale = 1.0 / sum(desired.values())
        desired = {code: weight * scale for code, weight in desired.items()}
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
            if not bool(row["buy_executable_t1_open"]) or max_fill <= 0:
                buy_reject += 1
                continue
            fill = min(delta, max_fill)
            next_weights[code] = old_w + fill
        else:
            if not bool(row["sell_executable_t1_open"]) or max_fill <= 0:
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
        ret = by_code.loc[code, "execution_return_open_to_close5"]
        if pd.isna(ret):
            continue
        value += weight * float(ret)
        used += weight
    return float(value / used) if used > 0 else float("nan")


def exposure_value(weights: dict[str, float], day: pd.DataFrame, risk_cols: list[str]) -> float:
    if not weights or not risk_cols:
        return 0.0
    selected = day[day["ts_code"].isin(weights)].copy()
    exposures = []
    for column in risk_cols:
        if column not in day.columns:
            continue
        z = zscore(day[column])
        z_by_code = pd.Series(z.values, index=day["ts_code"].astype(str))
        exposures.append(abs(sum(weights.get(code, 0.0) * z_by_code.get(code, 0.0) for code in weights)))
    return float(np.mean(exposures)) if exposures else 0.0


def exposure_snapshot(weights: dict[str, float], day: pd.DataFrame, risk_cols: list[str]) -> dict[str, float]:
    full_style_cols = [column for column in FULL_STYLE_EVAL_COLUMNS if column in day.columns]
    return {
        "avg_abs_control_exposure_z": exposure_value(weights, day, risk_cols),
        "avg_abs_full_style_exposure_z": exposure_value(weights, day, full_style_cols),
    }


def run_optimizer(args: argparse.Namespace, data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_cost_rate = (args.cost_bps + args.slippage_bps) / 10000.0
    for split, split_frame in data.groupby("split", sort=True):
        dates = select_signal_dates(sorted(split_frame["trade_date"].unique()), args.rebalance_stride)
        state: dict[tuple[Any, ...], dict[str, float]] = {}
        for trade_date in dates:
            day = split_frame[split_frame["trade_date"] == trade_date].copy()
            if len(day) < args.min_daily_count:
                continue
            day = day.dropna(subset=["pred_score", "execution_return_open_to_close5"])
            if len(day) < args.min_daily_count:
                continue
            benchmark_return = float(day["benchmark_next_open_to_exit_close_return"].dropna().iloc[0])
            exec_universe_return = float(
                day.loc[day["buy_executable_t1_open"], "execution_return_open_to_close5"].mean()
            )
            for risk_control in args.risk_control:
                risk_cols = [column for column in RISK_CONTROL_SETS.get(risk_control, []) if column in day.columns]
                for k in args.k:
                    if len(day) < k:
                        continue
                    for style_penalty in args.style_penalty:
                        for turnover_penalty in args.turnover_penalty:
                            key = (risk_control, k, style_penalty, turnover_penalty)
                            current = state.get(key, {})
                            previous_codes = list(current.keys())
                            target_codes = optimize_codes(
                                day=day,
                                previous_codes=previous_codes,
                                k=k,
                                risk_cols=risk_cols,
                                style_penalty=style_penalty,
                                turnover_penalty=turnover_penalty,
                                candidate_multiplier=args.candidate_multiplier,
                            )
                            weights, stats = simulate_fill(
                                current=current,
                                target_codes=target_codes,
                                day=day,
                                single_name_cap=args.single_name_cap,
                                portfolio_nav=args.portfolio_nav,
                                participation_cap=args.participation_cap,
                            )
                            state[key] = weights
                            gross = weighted_return(weights, day)
                            cost = total_cost_rate * stats["filled_turnover"]
                            net = gross - cost if pd.notna(gross) else float("nan")
                            exposure = exposure_snapshot(weights, day, risk_cols)
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
                                    "excess_vs_benchmark": net - benchmark_return if pd.notna(net) else float("nan"),
                                    "excess_vs_executable_universe": net - exec_universe_return if pd.notna(net) else float("nan"),
                                    "transaction_cost": cost,
                                    **stats,
                                    **exposure,
                                }
                            )
    return pd.DataFrame(rows)


def summarize(periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in periods.groupby(["split", "risk_control", "k", "style_penalty", "turnover_penalty"], sort=True):
        split, risk_control, k, style_penalty, turnover_penalty = keys
        net = summarize_returns(group["net_return"])
        bench = summarize_returns(group["excess_vs_benchmark"])
        execu = summarize_returns(group["excess_vs_executable_universe"])
        rows.append(
            {
                "split": split,
                "risk_control": risk_control,
                "k": int(k),
                "style_penalty": float(style_penalty),
                "turnover_penalty": float(turnover_penalty),
                "periods": net["period_count"],
                "net_ann": net["annualized_return"],
                "net_ir": net["ir"],
                "net_max_drawdown": net["max_drawdown"],
                "excess_benchmark_ann": bench["annualized_return"],
                "excess_exec_universe_ann": execu["annualized_return"],
                "avg_desired_turnover": float(group["desired_turnover"].mean()),
                "avg_filled_turnover": float(group["filled_turnover"].mean()),
                "avg_abs_control_exposure_z": float(group["avg_abs_control_exposure_z"].mean()),
                "avg_abs_full_style_exposure_z": float(group["avg_abs_full_style_exposure_z"].mean()),
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
        "portfolio_nav": float(args.portfolio_nav),
        "participation_cap": float(args.participation_cap),
        "period_rows": int(len(periods)),
        "summary_rows": int(len(summary)),
        "method": "barra_lite_constrained_reranker_with_t1_fill_simulation",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
