from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTROL_SETS = {
    "none": [],
    "size": ["lag1_log_circ_mv", "lag1_log_total_mv"],
    "size_liquidity": [
        "lag1_log_circ_mv",
        "lag1_log_total_mv",
        "lag1_amount_log",
        "lag1_amount_rank_pct",
        "lag1_amount_20d_mean",
    ],
    "full_style": [
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
        "lag1_pb_winsor",
        "lag1_pe_ttm_winsor",
    ],
    "industry_proxy_full_style": [
        "lag1_industry_turnover_rank",
        "lag1_industry_amount_rank",
        "lag1_industry_pb_rank",
        "lag1_industry_mv_rank",
        "lag1_log_circ_mv",
        "lag1_log_total_mv",
        "lag1_amount_log",
        "lag1_amount_rank_pct",
        "lag1_turnover_rate_f",
        "lag1_ret_20d_std",
        "lag1_ret_20d",
        "lag1_beta_60d",
        "lag1_pb_winsor",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Barra-lite residual alpha audit.")
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_clean_alpha_resid_style_purgedwf_strictmask_leaky0005/predictions.parquet",
    )
    parser.add_argument("--mart", default="data/mart/datasets/core/dataset_v20260526.parquet")
    parser.add_argument("--labels", default="data/mart/labels/labels_canonical_v20260526.parquet")
    parser.add_argument("--target", default="execution_excess_open_to_close5")
    parser.add_argument("--out-dir", default="outputs/audit/barra_lite_residual_alpha")
    parser.add_argument("--min-count", type=int, default=40)
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


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.astype("float64").replace([np.inf, -np.inf], np.nan)
    result = result.fillna(result.median(numeric_only=True))
    std = result.std(ddof=0).replace(0, np.nan)
    return ((result - result.mean()) / std).fillna(0.0)


def residualize(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    yv = pd.to_numeric(y, errors="coerce").astype("float64")
    valid = yv.notna()
    if x.empty:
        return yv - yv.mean()
    xv = zscore(x).loc[yv.index]
    valid &= xv.notna().all(axis=1)
    if valid.sum() <= xv.shape[1] + 2:
        return pd.Series(np.nan, index=y.index)
    design = np.column_stack([np.ones(valid.sum()), xv.loc[valid].to_numpy(dtype="float64")])
    coef = np.linalg.lstsq(design, yv.loc[valid].to_numpy(dtype="float64"), rcond=None)[0]
    fitted = design @ coef
    residual = pd.Series(np.nan, index=y.index, dtype="float64")
    residual.loc[valid] = yv.loc[valid].to_numpy(dtype="float64") - fitted
    return residual


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    predictions = pd.read_parquet(args.predictions)
    needed = sorted({column for cols in CONTROL_SETS.values() for column in cols})
    mart_cols = set(pd.read_parquet(args.mart, engine="pyarrow").columns)
    selected = ["trade_date", "ts_code", *[column for column in needed if column in mart_cols]]
    mart = pd.read_parquet(args.mart, columns=selected)
    labels = pd.read_parquet(args.labels, columns=["trade_date", "ts_code", args.target])
    data = predictions.merge(mart, on=["trade_date", "ts_code"], how="inner")
    data = data.merge(labels, on=["trade_date", "ts_code"], how="inner")
    for column in ["trade_date", "ts_code", "split"]:
        data[column] = data[column].astype(str)
    for column in ["pred_score", args.target, *selected[2:]]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.replace([np.inf, -np.inf], np.nan)


def score_coef(group: pd.DataFrame, controls: list[str], target: str) -> float:
    cols = ["pred_score", target, *controls]
    frame = group[cols].dropna()
    if len(frame) <= len(controls) + 4:
        return float("nan")
    x_controls = zscore(frame[controls]) if controls else pd.DataFrame(index=frame.index)
    score = zscore(frame[["pred_score"]])
    design = np.column_stack(
        [
            np.ones(len(frame)),
            score.to_numpy(dtype="float64"),
            x_controls.to_numpy(dtype="float64") if not x_controls.empty else np.empty((len(frame), 0)),
        ]
    )
    coef = np.linalg.lstsq(design, frame[target].to_numpy(dtype="float64"), rcond=None)[0]
    return float(coef[1])


def run_audit(data: pd.DataFrame, target: str, min_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    for (split, trade_date), group in data.groupby(["split", "trade_date"], sort=True):
        if len(group) < min_count:
            continue
        for set_name, controls in CONTROL_SETS.items():
            available = [column for column in controls if column in group.columns]
            frame = group[["pred_score", target, *available]].dropna(subset=["pred_score", target])
            if len(frame) < max(min_count, len(available) + 5):
                continue
            alpha_resid = residualize(frame["pred_score"], frame[available]) if available else frame["pred_score"]
            target_resid = residualize(frame[target], frame[available]) if available else frame[target]
            daily_rows.append(
                {
                    "split": split,
                    "trade_date": trade_date,
                    "control_set": set_name,
                    "available_controls": ",".join(available),
                    "n": int(len(frame)),
                    "raw_ic": safe_corr(frame["pred_score"], frame[target]),
                    "residual_ic": safe_corr(alpha_resid, target_resid),
                    "score_coef_after_controls": score_coef(frame, available, target),
                }
            )
            ranked = frame.assign(alpha_resid=alpha_resid, target_resid=target_resid).dropna(
                subset=["alpha_resid", "target_resid"]
            )
            if len(ranked) >= 30:
                ranked["raw_decile"] = pd.qcut(
                    ranked["pred_score"].rank(method="first"), 10, labels=False, duplicates="drop"
                )
                ranked["resid_decile"] = pd.qcut(
                    ranked["alpha_resid"].rank(method="first"), 10, labels=False, duplicates="drop"
                )
                for decile_col, ret_col, mode in [
                    ("raw_decile", target, "raw_score"),
                    ("resid_decile", "target_resid", "residual_score"),
                ]:
                    for decile, part in ranked.groupby(decile_col):
                        decile_rows.append(
                            {
                                "split": split,
                                "trade_date": trade_date,
                                "control_set": set_name,
                                "mode": mode,
                                "decile": int(decile),
                                "mean_return": float(part[ret_col].mean()),
                                "count": int(len(part)),
                            }
                        )
    daily = pd.DataFrame(daily_rows)
    deciles = pd.DataFrame(decile_rows)
    summary_rows: list[dict[str, Any]] = []
    if not daily.empty:
        for (split, control_set), group in daily.groupby(["split", "control_set"], sort=True):
            summary_rows.append(
                {
                    "split": split,
                    "control_set": control_set,
                    "days": int(len(group)),
                    "mean_raw_ic": float(group["raw_ic"].mean()),
                    "mean_residual_ic": float(group["residual_ic"].mean()),
                    "mean_score_coef_after_controls": float(group["score_coef_after_controls"].mean()),
                    "positive_residual_ic_rate": float((group["residual_ic"] > 0).mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    return daily, deciles, summary


def write_markdown(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = ["# Barra-lite Residual Alpha Audit", ""]
    if summary.empty:
        lines.append("No valid summary rows.")
    else:
        lines.extend(
            [
                "| split | control_set | days | mean_raw_ic | mean_residual_ic | score_coef_after_controls | positive_residual_ic_rate |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary.to_dict(orient="records"):
            lines.append(
                "| {split} | {control_set} | {days} | {mean_raw_ic:.6f} | {mean_residual_ic:.6f} | "
                "{mean_score_coef_after_controls:.6f} | {positive_residual_ic_rate:.4f} |".format(**row)
            )
    (out_dir / "residual_alpha_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    daily, deciles, summary = run_audit(data, args.target, args.min_count)
    daily.to_csv(out_dir / "daily_residual_ic.csv", index=False)
    deciles.to_csv(out_dir / "decile_returns.csv", index=False)
    summary.to_csv(out_dir / "residual_summary.csv", index=False)
    write_markdown(out_dir, summary)
    manifest = {
        "predictions": args.predictions,
        "mart": args.mart,
        "labels": args.labels,
        "target": args.target,
        "rows_after_merge": int(len(data)),
        "summary_rows": int(len(summary)),
        "out_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
