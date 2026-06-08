from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


RUNS = {
    "old_full62": Path("outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005"),
    "new_alpha13": Path("outputs/runs/gru_l20_clean_alpha_only_strictmask_leaky0005"),
    "new_alpha18_resid": Path("outputs/runs/gru_l20_clean_alpha_resid_style_strictmask_leaky0005"),
}


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b and math.isfinite(b) else float("nan")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def daily_ic(frame: pd.DataFrame) -> dict[str, float | int]:
    rows: list[tuple[float, float]] = []
    const_days = 0
    for _, group in frame.groupby("trade_date", sort=True):
        if group["pred_score"].nunique() <= 1:
            const_days += 1
            continue
        if group["label_rel_return"].nunique() <= 1:
            continue
        rows.append(
            (
                group["pred_score"].corr(group["label_rel_return"], method="pearson"),
                group["pred_score"].corr(group["label_rel_return"], method="spearman"),
            )
        )

    ic = pd.Series([row[0] for row in rows], dtype="float64").dropna()
    rank_ic = pd.Series([row[1] for row in rows], dtype="float64").dropna()
    return {
        "samples": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "pred_std": float(frame["pred_score"].std()),
        "ic_mean": float(ic.mean()),
        "icir": safe_ratio(float(ic.mean()), float(ic.std(ddof=1))),
        "rank_ic_mean": float(rank_ic.mean()),
        "rank_icir": safe_ratio(float(rank_ic.mean()), float(rank_ic.std(ddof=1))),
        "const_days": int(const_days),
    }


def flat_topk(run_name: str, run_dir: Path) -> list[dict[str, Any]]:
    data = load_json(run_dir / "topk_metrics.json")["summary"]
    rows: list[dict[str, Any]] = []
    for split, split_data in data.items():
        for k in (10, 20, 30):
            item = split_data[f"top_{k}"]
            rows.append(
                {
                    "run": run_name,
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


def flat_backtest(run_name: str, run_dir: Path) -> list[dict[str, Any]]:
    data = load_json(run_dir / "backtest_metrics.json")["summary"]
    rows: list[dict[str, Any]] = []
    for split, split_data in data.items():
        for k in (10, 20, 30):
            item = split_data[f"top_{k}_cost_10bps"]
            rows.append(
                {
                    "run": run_name,
                    "split": split,
                    "k": k,
                    "periods": item["top_net"]["period_count"],
                    "top_ann": item["top_net"]["annualized_return"],
                    "top_cum": item["top_net"]["cumulative_return"],
                    "max_drawdown": item["top_net"]["max_drawdown"],
                    "excess_bench_ann": item["top_excess_vs_benchmark_net"]["annualized_return"],
                    "excess_universe_ann": item["top_excess_vs_universe_net"]["annualized_return"],
                    "long_short_ann": item["long_short_net"]["annualized_return"],
                    "long_short_cum": item["long_short_net"]["cumulative_return"],
                    "avg_turnover": item["average_turnover"],
                    "avg_ls_turnover": item["average_long_short_turnover"],
                }
            )
    return rows


def main() -> None:
    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_rows: list[dict[str, Any]] = []
    for run_name, run_dir in RUNS.items():
        metrics = load_json(run_dir / "metrics.json")
        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        for split, frame in predictions.groupby("split", sort=True):
            row = {
                "run": run_name,
                "split": split,
                "best_epoch": metrics["best_epoch"],
                "val_best_rank_ic": metrics["best_metric"],
                "train_samples": metrics["summary"]["train_samples"],
                "validation_samples": metrics["summary"]["validation_samples"],
                "test_samples": metrics["summary"]["test_samples"],
                "num_features": metrics["summary"]["num_features"],
            }
            row.update(daily_ic(frame))
            ic_rows.append(row)

    ic = pd.DataFrame(ic_rows).sort_values(["run", "split"])
    topk = pd.DataFrame(
        row for run_name, run_dir in RUNS.items() for row in flat_topk(run_name, run_dir)
    ).sort_values(["run", "split", "k"])
    backtest = pd.DataFrame(
        row for run_name, run_dir in RUNS.items() for row in flat_backtest(run_name, run_dir)
    ).sort_values(["run", "split", "k"])

    ic.to_csv(out_dir / "gru_clean_vs_old_ic.csv", index=False)
    topk.to_csv(out_dir / "gru_clean_vs_old_topk.csv", index=False)
    backtest.to_csv(out_dir / "gru_clean_vs_old_backtest_10bps.csv", index=False)

    print("\nIC / RankIC")
    print(ic.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nTop-K proxy")
    print(topk.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nBacktest 10bps")
    print(backtest.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
