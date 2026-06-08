from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STYLE_COLUMNS = [
    "lag1_log_circ_mv",
    "lag1_log_total_mv",
    "lag1_beta_20d",
    "lag1_beta_60d",
    "lag1_ret_20d",
    "lag1_ret_5d_mean",
    "lag1_ret_20d_mean",
    "lag1_ret_60d_mean",
    "lag1_ret_20d_std",
    "lag1_ret_60d_std",
    "lag1_amplitude",
    "lag1_vol_log",
]

LIQUIDITY_COLUMNS = [
    "lag1_amount_log",
    "lag1_amount_rank_pct",
    "lag1_amount_20d_mean",
    "lag1_amount_60d_mean",
    "lag1_turnover_rate",
    "lag1_turnover_rate_f",
    "lag1_turnover_20d_mean",
    "lag1_turnover_60d_mean",
    "lag1_turnover_cost_proxy",
    "lag1_illiquidity_proxy",
]

EXECUTION_COLUMNS = [
    "buy_executable_t1_open",
    "sell_executable_t1_open",
    "next_is_limit_up",
    "next_is_limit_down",
    "next_is_suspended",
    "next_amount",
    "next_vol",
]


def parse_candidate(value: str) -> tuple[int, float]:
    if ":" in value:
        left, right = value.split(":", 1)
    elif "," in value:
        left, right = value.split(",", 1)
    else:
        raise argparse.ArgumentTypeError("Candidate must be K:keep, e.g. 10:1.5")
    return int(left), float(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit old full62 style, liquidity, and execution attribution."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet",
    )
    parser.add_argument("--mart", default="data/mart/datasets/dataset_v20260526.parquet")
    parser.add_argument(
        "--execution-labels",
        default="data/mart/labels/execution_labels_v20260526.parquet",
    )
    parser.add_argument(
        "--periods",
        default=(
            "outputs/backtest/t1_fill_sim/"
            "gru_l20_slope0005_k_keep_matrix_nav10m_part3pct/t1_fill_periods.csv"
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        default=None,
        help="Candidate as K:keep. Repeatable. Defaults to 10:1.5, 30:3, 30:1.",
    )
    parser.add_argument("--out-dir", default="outputs/audit/full62_attribution")
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


def safe_ir(series: pd.Series) -> float:
    clean = pd.Series(series, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / std) if std and pd.notna(std) else float("nan")


def read_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(args.predictions)
    feature_cols = [
        "trade_date",
        "ts_code",
        *STYLE_COLUMNS,
        *LIQUIDITY_COLUMNS,
    ]
    mart_columns = pd.read_parquet(args.mart, engine="pyarrow").columns
    mart_selected = [column for column in feature_cols if column in set(mart_columns)]
    mart = pd.read_parquet(args.mart, columns=mart_selected)
    exec_columns = ["trade_date", "ts_code", *EXECUTION_COLUMNS, "execution_return_open_to_close5"]
    execution = pd.read_parquet(args.execution_labels, columns=exec_columns)
    periods = pd.read_csv(args.periods)

    for frame in [predictions, mart, execution, periods]:
        if "trade_date" in frame.columns:
            frame["trade_date"] = frame["trade_date"].astype(str)
        if "ts_code" in frame.columns:
            frame["ts_code"] = frame["ts_code"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    for column in [*STYLE_COLUMNS, *LIQUIDITY_COLUMNS]:
        if column in mart.columns:
            mart[column] = pd.to_numeric(mart[column], errors="coerce")
    for column in ["next_amount", "next_vol", "execution_return_open_to_close5"]:
        if column in execution.columns:
            execution[column] = pd.to_numeric(execution[column], errors="coerce")
    for column in [
        "buy_executable_t1_open",
        "sell_executable_t1_open",
        "next_is_limit_up",
        "next_is_limit_down",
        "next_is_suspended",
    ]:
        if column in execution.columns:
            execution[column] = execution[column].fillna(False).astype(bool)
    return predictions, mart, execution, periods


def candidate_key(k: int, keep: float) -> str:
    return f"k{k}_keep{keep:g}"


def select_codes(group: pd.DataFrame, k: int, keep: float, previous_codes: list[str]) -> list[str]:
    ordered = group.sort_values("pred_score", ascending=False)["ts_code"].astype(str).tolist()
    keep_rank = min(len(ordered), max(k, int(math.ceil(k * keep))))
    keep_set = set(ordered[:keep_rank])
    selected = [code for code in previous_codes if code in keep_set]
    selected_set = set(selected)
    for code in ordered:
        if len(selected) >= k:
            break
        if code not in selected_set:
            selected.append(code)
            selected_set.add(code)
    return selected[:k]


def build_candidate_snapshots(
    predictions: pd.DataFrame,
    mart: pd.DataFrame,
    execution: pd.DataFrame,
    periods: pd.DataFrame,
    candidates: list[tuple[int, float]],
) -> pd.DataFrame:
    data = predictions.merge(mart, on=["trade_date", "ts_code"], how="left")
    data = data.merge(execution, on=["trade_date", "ts_code"], how="left")
    rows: list[dict[str, Any]] = []
    for split, split_periods in periods.groupby("split", sort=True):
        split_predictions = data[data["split"] == split]
        for k, keep in candidates:
            p = split_periods[
                (split_periods["k"].astype(int) == int(k))
                & np.isclose(split_periods["keep_multiplier"].astype(float), float(keep))
            ].sort_values("trade_date")
            previous_codes: list[str] = []
            for _, period in p.iterrows():
                date = str(period["trade_date"])
                group = split_predictions[split_predictions["trade_date"] == date]
                if len(group) < k:
                    continue
                selected = select_codes(group, k, keep, previous_codes)
                previous_codes = selected
                selected_frame = group[group["ts_code"].isin(selected)].copy()
                selected_frame["candidate"] = candidate_key(k, keep)
                selected_frame["k"] = int(k)
                selected_frame["keep"] = float(keep)
                selected_frame["in_selected"] = True
                rows.extend(selected_frame.to_dict(orient="records"))
    return pd.DataFrame(rows)


def style_exposure(snapshots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, split), group in snapshots.groupby(["candidate", "split"], sort=True):
        for column in STYLE_COLUMNS:
            if column not in group.columns:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "style_col": column,
                    "selected_mean": float(pd.to_numeric(group[column], errors="coerce").mean()),
                    "selected_std": float(pd.to_numeric(group[column], errors="coerce").std(ddof=1)),
                    "daily_mean_ir": safe_ir(
                        group.groupby("trade_date")[column].mean()
                    ),
                    "non_null_rate": float(group[column].notna().mean()),
                }
            )
    return pd.DataFrame(rows)


def style_vs_universe(
    snapshots: pd.DataFrame,
    predictions: pd.DataFrame,
    mart: pd.DataFrame,
    candidates: list[tuple[int, float]],
) -> pd.DataFrame:
    universe = predictions.merge(mart, on=["trade_date", "ts_code"], how="left")
    rows: list[dict[str, Any]] = []
    for (candidate, split), selected in snapshots.groupby(["candidate", "split"], sort=True):
        dates = selected["trade_date"].unique()
        split_universe = universe[(universe["split"] == split) & (universe["trade_date"].isin(dates))]
        for column in STYLE_COLUMNS + LIQUIDITY_COLUMNS:
            if column not in selected.columns or column not in split_universe.columns:
                continue
            daily_rows = []
            for date, selected_day in selected.groupby("trade_date", sort=True):
                universe_day = split_universe[split_universe["trade_date"] == date]
                if universe_day.empty:
                    continue
                daily_rows.append(
                    {
                        "selected_mean": pd.to_numeric(selected_day[column], errors="coerce").mean(),
                        "universe_mean": pd.to_numeric(universe_day[column], errors="coerce").mean(),
                    }
                )
            daily = pd.DataFrame(daily_rows)
            if daily.empty:
                continue
            diff = daily["selected_mean"] - daily["universe_mean"]
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "feature": column,
                    "selected_mean": float(daily["selected_mean"].mean()),
                    "universe_mean": float(daily["universe_mean"].mean()),
                    "selected_minus_universe": float(diff.mean()),
                    "diff_ir": safe_ir(diff),
                }
            )
    return pd.DataFrame(rows)


def liquidity_bucket_returns(
    predictions: pd.DataFrame,
    mart: pd.DataFrame,
    execution: pd.DataFrame,
) -> pd.DataFrame:
    data = predictions.merge(mart, on=["trade_date", "ts_code"], how="left")
    data = data.merge(execution, on=["trade_date", "ts_code"], how="left")
    rows: list[dict[str, Any]] = []
    bucket_features = [
        "lag1_amount_20d_mean",
        "lag1_amount_rank_pct",
        "lag1_log_circ_mv",
        "lag1_turnover_rate_f",
        "lag1_ret_20d_std",
    ]
    for split, split_frame in data.groupby("split", sort=True):
        for feature in bucket_features:
            if feature not in split_frame.columns:
                continue
            daily_parts: list[pd.DataFrame] = []
            for date, group in split_frame.groupby("trade_date", sort=True):
                values = pd.to_numeric(group[feature], errors="coerce")
                valid = group[values.notna()].copy()
                if len(valid) < 30 or values.nunique(dropna=True) < 3:
                    continue
                try:
                    valid["bucket"] = pd.qcut(
                        pd.to_numeric(valid[feature], errors="coerce"),
                        q=3,
                        labels=["low", "mid", "high"],
                        duplicates="drop",
                    )
                except ValueError:
                    continue
                daily_parts.append(valid)
            if not daily_parts:
                continue
            bucketed = pd.concat(daily_parts, ignore_index=True)
            for bucket, bucket_frame in bucketed.groupby("bucket", observed=True, sort=True):
                ret = pd.to_numeric(bucket_frame["execution_return_open_to_close5"], errors="coerce")
                rows.append(
                    {
                        "split": split,
                        "bucket_feature": feature,
                        "bucket": str(bucket),
                        "rows": int(len(bucket_frame)),
                        "mean_exec_return": float(ret.mean()),
                        "return_ir": safe_ir(
                            bucket_frame.groupby("trade_date")[
                                "execution_return_open_to_close5"
                            ].mean()
                        ),
                        "buy_executable_rate": float(bucket_frame["buy_executable_t1_open"].mean()),
                        "limit_up_rate": float(bucket_frame["next_is_limit_up"].mean()),
                        "limit_down_rate": float(bucket_frame["next_is_limit_down"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def execution_attribution(periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, k, keep), group in periods.groupby(["split", "k", "keep_multiplier"], sort=True):
        gross = pd.to_numeric(group["gross_return"], errors="coerce")
        net = pd.to_numeric(group["net_return"], errors="coerce")
        bench_excess = pd.to_numeric(group["excess_vs_benchmark"], errors="coerce")
        exec_excess = pd.to_numeric(group["excess_vs_executable_universe"], errors="coerce")
        rows.append(
            {
                "candidate": candidate_key(int(k), float(keep)),
                "split": split,
                "periods": int(len(group)),
                "gross_mean": float(gross.mean()),
                "net_mean": float(net.mean()),
                "cost_drag_mean": float((gross - net).mean()),
                "net_ir": safe_ir(net),
                "excess_benchmark_mean": float(bench_excess.mean()),
                "excess_executable_universe_mean": float(exec_excess.mean()),
                "desired_turnover": float(group["desired_turnover"].mean()),
                "filled_turnover": float(group["filled_turnover"].mean()),
                "fill_ratio": float(
                    group["filled_turnover"].sum() / group["desired_turnover"].sum()
                )
                if group["desired_turnover"].sum() > 0
                else float("nan"),
                "avg_buy_reject": float(group["buy_reject_count"].mean()),
                "avg_sell_reject": float(group["sell_reject_count"].mean()),
                "avg_partial_fill": float(group["partial_fill_count"].mean()),
                "avg_positions": float(group["position_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    out_dir: Path,
    style_universe: pd.DataFrame,
    liquidity: pd.DataFrame,
    execution: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No data._"
        display = frame.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
                )
            else:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else str(value)
                )
        header = "| " + " | ".join(display.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
        body = [
            "| " + " | ".join(str(row[column]) for column in display.columns) + " |"
            for _, row in display.iterrows()
        ]
        return "\n".join([header, sep, *body])

    lines = [
        "# Full62 Attribution Audit",
        "",
        "Scope: old full62 GRU score model with T+1 fill-simulation candidates.",
        "",
        "## Execution Attribution",
        "",
    ]
    exec_sorted = execution.sort_values(
        ["split", "excess_executable_universe_mean"], ascending=[True, False]
    )
    lines.append(markdown_table(exec_sorted))
    lines.extend(["", "## Largest Style And Liquidity Exposures", ""])
    exposure = style_universe.copy()
    if not exposure.empty:
        exposure["abs_diff"] = exposure["selected_minus_universe"].abs()
        top = exposure.sort_values(["split", "candidate", "abs_diff"], ascending=[True, True, False])
        top = top.groupby(["split", "candidate"], as_index=False).head(8)
        lines.append(markdown_table(top.drop(columns=["abs_diff"])))
    lines.extend(["", "## Liquidity Bucket Returns", ""])
    lines.append(markdown_table(liquidity) if not liquidity.empty else "_No liquidity bucket data._")
    (out_dir / "full62_attribution_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidates = args.candidate or [(10, 1.5), (30, 3.0), (30, 1.0)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions, mart, execution, periods = read_inputs(args)
    snapshots = build_candidate_snapshots(predictions, mart, execution, periods, candidates)
    style = style_exposure(snapshots)
    style_universe = style_vs_universe(snapshots, predictions, mart, candidates)
    liquidity = liquidity_bucket_returns(predictions, mart, execution)
    exec_attr = execution_attribution(
        periods[
            periods.apply(
                lambda row: (int(row["k"]), float(row["keep_multiplier"])) in candidates,
                axis=1,
            )
        ]
    )

    snapshots.to_parquet(out_dir / "candidate_selected_snapshots.parquet", index=False)
    style.to_csv(out_dir / "style_exposure_selected.csv", index=False)
    style_universe.to_csv(out_dir / "style_liquidity_vs_universe.csv", index=False)
    liquidity.to_csv(out_dir / "liquidity_bucket_returns.csv", index=False)
    exec_attr.to_csv(out_dir / "execution_attribution.csv", index=False)
    write_markdown(out_dir, style_universe, liquidity, exec_attr)

    summary = {
        "out_dir": str(out_dir),
        "candidates": [{"k": k, "keep": keep} for k, keep in candidates],
        "snapshot_rows": int(len(snapshots)),
        "style_rows": int(len(style_universe)),
        "liquidity_rows": int(len(liquidity)),
        "execution_rows": int(len(exec_attr)),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
