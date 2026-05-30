from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


RUNS = {
    "base_alpha18_resid": {
        "run_dir": Path("outputs/runs/gru_l20_clean_alpha_resid_style_strictmask_leaky0005"),
        "dropped_feature": "none",
    },
    "drop_turnover_cost_proxy_resid": {
        "run_dir": Path("outputs/runs/gru_l20_alpha_resid_style_drop_turnover_cost_proxy_resid_leaky0005"),
        "dropped_feature": "lag1_turnover_cost_proxy__resid_style",
    },
    "drop_turnover_20d_std_resid": {
        "run_dir": Path("outputs/runs/gru_l20_alpha_resid_style_drop_turnover_20d_std_resid_leaky0005"),
        "dropped_feature": "lag1_turnover_20d_std__resid_style",
    },
    "drop_turnover_60d_std_resid": {
        "run_dir": Path("outputs/runs/gru_l20_alpha_resid_style_drop_turnover_60d_std_resid_leaky0005"),
        "dropped_feature": "lag1_turnover_60d_std__resid_style",
    },
    "drop_amount_rank_pct_resid": {
        "run_dir": Path("outputs/runs/gru_l20_alpha_resid_style_drop_amount_rank_pct_resid_leaky0005"),
        "dropped_feature": "lag1_amount_rank_pct__resid_style",
    },
    "drop_amount_log_resid": {
        "run_dir": Path("outputs/runs/gru_l20_alpha_resid_style_drop_amount_log_resid_leaky0005"),
        "dropped_feature": "lag1_amount_log__resid_style",
    },
}

OUT_DIR = Path("outputs/analysis/resid_style_ablation")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_ir(series: pd.Series) -> float:
    clean = pd.Series(series, dtype="float64").dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / std) if std and math.isfinite(std) else float("nan")


def daily_ic(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, date), group in frame.groupby(["split", "trade_date"], sort=True):
        if group["pred_score"].nunique() <= 1 or group["label_rel_return"].nunique() <= 1:
            continue
        rows.append(
            {
                "split": split,
                "trade_date": date,
                "year": str(date)[:4],
                "ic": group["pred_score"].corr(group["label_rel_return"], method="pearson"),
                "rank_ic": group["pred_score"].corr(group["label_rel_return"], method="spearman"),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def ic_summary(name: str, dropped_feature: str, run_dir: Path) -> list[dict[str, Any]]:
    metrics = load_json(run_dir / "metrics.json")
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    daily = daily_ic(pred)
    rows: list[dict[str, Any]] = []
    for split, group in daily.groupby("split", sort=True):
        rows.append(
            {
                "run": name,
                "dropped_feature": dropped_feature,
                "split": split,
                "best_epoch": metrics["best_epoch"],
                "best_val_rank_ic": metrics["best_metric"],
                "days": int(len(group)),
                "ic_mean": float(group["ic"].mean()),
                "icir": safe_ir(group["ic"]),
                "rank_ic_mean": float(group["rank_ic"].mean()),
                "rank_icir": safe_ir(group["rank_ic"]),
                "rank_ic_positive_rate": float((group["rank_ic"] > 0).mean()),
            }
        )
    return rows


def yearly_ic_summary(name: str, dropped_feature: str, run_dir: Path) -> list[dict[str, Any]]:
    daily = daily_ic(pd.read_parquet(run_dir / "predictions.parquet"))
    rows: list[dict[str, Any]] = []
    for (split, year), group in daily.groupby(["split", "year"], sort=True):
        rows.append(
            {
                "run": name,
                "dropped_feature": dropped_feature,
                "split": split,
                "year": year,
                "days": int(len(group)),
                "ic_mean": float(group["ic"].mean()),
                "rank_ic_mean": float(group["rank_ic"].mean()),
                "rank_icir": safe_ir(group["rank_ic"]),
            }
        )
    return rows


def topk_summary(name: str, dropped_feature: str, run_dir: Path) -> list[dict[str, Any]]:
    data = load_json(run_dir / "topk_metrics.json")["summary"]
    rows: list[dict[str, Any]] = []
    for split, split_data in data.items():
        for k in (10, 20, 30):
            item = split_data[f"top_{k}"]
            rows.append(
                {
                    "run": name,
                    "dropped_feature": dropped_feature,
                    "split": split,
                    "k": k,
                    "top_mean": item["top"]["mean"],
                    "bottom_mean": item["bottom"]["mean"],
                    "spread": item["long_short_spread"]["mean"],
                    "spread_ir": item["long_short_spread"]["ir"],
                    "spread_positive_rate": item["long_short_spread"]["positive_rate"],
                    "quantile_high_minus_low": split_data.get("quantile_high_minus_low"),
                }
            )
    return rows


def backtest_summary(name: str, dropped_feature: str, run_dir: Path) -> list[dict[str, Any]]:
    data = load_json(run_dir / "backtest_metrics.json")["summary"]
    rows: list[dict[str, Any]] = []
    for split, split_data in data.items():
        for k in (10, 20, 30):
            item = split_data[f"top_{k}_cost_10bps"]
            rows.append(
                {
                    "run": name,
                    "dropped_feature": dropped_feature,
                    "split": split,
                    "k": k,
                    "top_ann": item["top_net"]["annualized_return"],
                    "top_cum": item["top_net"]["cumulative_return"],
                    "excess_bench_ann": item["top_excess_vs_benchmark_net"]["annualized_return"],
                    "excess_universe_ann": item["top_excess_vs_universe_net"]["annualized_return"],
                    "long_short_ann": item["long_short_net"]["annualized_return"],
                    "long_short_cum": item["long_short_net"]["cumulative_return"],
                    "avg_turnover": item["average_turnover"],
                    "avg_ls_turnover": item["average_long_short_turnover"],
                }
            )
    return rows


def add_delta(frame: pd.DataFrame, metrics: list[str], keys: list[str]) -> pd.DataFrame:
    base = frame[frame["run"].eq("base_alpha18_resid")][keys + metrics].copy()
    base = base.rename(columns={metric: f"base_{metric}" for metric in metrics})
    out = frame.merge(base, on=keys, how="left")
    for metric in metrics:
        out[f"delta_{metric}"] = out[metric] - out[f"base_{metric}"]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ic_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    backtest_rows: list[dict[str, Any]] = []

    for name, spec in RUNS.items():
        run_dir = spec["run_dir"]
        dropped_feature = spec["dropped_feature"]
        ic_rows.extend(ic_summary(name, dropped_feature, run_dir))
        yearly_rows.extend(yearly_ic_summary(name, dropped_feature, run_dir))
        topk_rows.extend(topk_summary(name, dropped_feature, run_dir))
        backtest_rows.extend(backtest_summary(name, dropped_feature, run_dir))

    ic = add_delta(
        pd.DataFrame(ic_rows),
        ["ic_mean", "rank_ic_mean", "rank_icir"],
        ["split"],
    )
    yearly = add_delta(
        pd.DataFrame(yearly_rows),
        ["ic_mean", "rank_ic_mean", "rank_icir"],
        ["split", "year"],
    )
    topk = add_delta(
        pd.DataFrame(topk_rows),
        ["spread", "spread_ir", "quantile_high_minus_low"],
        ["split", "k"],
    )
    backtest = add_delta(
        pd.DataFrame(backtest_rows),
        ["top_ann", "excess_universe_ann", "long_short_ann", "avg_turnover"],
        ["split", "k"],
    )

    ic.to_csv(OUT_DIR / "ic_summary.csv", index=False)
    yearly.to_csv(OUT_DIR / "yearly_ic_summary.csv", index=False)
    topk.to_csv(OUT_DIR / "topk_summary.csv", index=False)
    backtest.to_csv(OUT_DIR / "backtest_10bps_summary.csv", index=False)

    focus = (
        ic[ic["split"].eq("test")]
        .sort_values("delta_ic_mean", ascending=False)
        [
            [
                "run",
                "dropped_feature",
                "best_epoch",
                "ic_mean",
                "delta_ic_mean",
                "rank_ic_mean",
                "delta_rank_ic_mean",
                "rank_icir",
            ]
        ]
    )
    top30 = topk[(topk["split"].eq("test")) & (topk["k"].eq(30))].sort_values(
        "delta_spread", ascending=False
    )
    bt30 = backtest[(backtest["split"].eq("test")) & (backtest["k"].eq(30))].sort_values(
        "delta_long_short_ann", ascending=False
    )

    print("\nTest IC focus")
    print(focus.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nTest Top30 focus")
    print(
        top30[
            [
                "run",
                "dropped_feature",
                "spread",
                "delta_spread",
                "spread_ir",
                "quantile_high_minus_low",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\nTest Backtest Top30 10bps focus")
    print(
        bt30[
            [
                "run",
                "dropped_feature",
                "top_ann",
                "delta_top_ann",
                "long_short_ann",
                "delta_long_short_ann",
                "avg_turnover",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
