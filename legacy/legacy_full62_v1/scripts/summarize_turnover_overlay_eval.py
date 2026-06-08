from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RUNS = {
    "old_full62": Path("outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005"),
    "old_full62_strictmask_overlay": Path(
        "outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay"
    ),
}


def safe_ir(series: pd.Series) -> float | None:
    clean = pd.Series(series, dtype="float64").dropna()
    std = clean.std(ddof=1)
    if std and pd.notna(std):
        return float(clean.mean() / std)
    return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def daily_ic(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, date), group in pred.groupby(["split", "trade_date"], sort=True):
        if group["pred_score"].nunique() <= 1:
            continue
        if group["label_rel_return"].nunique() <= 1:
            continue
        rows.append(
            {
                "split": split,
                "trade_date": date,
                "ic": group["pred_score"].corr(
                    group["label_rel_return"], method="pearson"
                ),
                "rank_ic": group["pred_score"].corr(
                    group["label_rel_return"], method="spearman"
                ),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def collect_ic() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run, run_dir in RUNS.items():
        pred = pd.read_parquet(run_dir / "predictions.parquet")
        pred["trade_date"] = pred["trade_date"].astype(str)
        pred["ts_code"] = pred["ts_code"].astype(str)
        daily = daily_ic(pred)
        for split, group in daily.groupby("split", sort=True):
            rows.append(
                {
                    "run": run,
                    "split": split,
                    "rows": int((pred["split"] == split).sum()),
                    "dates": int(group["trade_date"].nunique()),
                    "avg_n": float(group["n"].mean()),
                    "ic_mean": float(group["ic"].mean()),
                    "icir": safe_ir(group["ic"]),
                    "rank_ic_mean": float(group["rank_ic"].mean()),
                    "rank_icir": safe_ir(group["rank_ic"]),
                    "rank_ic_pos": float((group["rank_ic"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def collect_topk() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run, run_dir in RUNS.items():
        path = run_dir / "topk_metrics.json"
        if not path.exists():
            continue
        summary = read_json(path)["summary"]["test"]
        for k in (10, 20, 30):
            item = summary[f"top_{k}"]
            rows.append(
                {
                    "run": run,
                    "k": k,
                    "top_mean": item["top"]["mean"],
                    "spread_mean": item["long_short_spread"]["mean"],
                    "spread_ir": item["long_short_spread"]["ir"],
                    "top_excess_mean": item["top_excess_vs_daily_mean"]["mean"],
                }
            )
    return pd.DataFrame(rows)


def collect_backtest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run, run_dir in RUNS.items():
        path = run_dir / "backtest_metrics.json"
        if not path.exists():
            continue
        summary = read_json(path)["summary"]["test"]
        for k in (10, 20, 30):
            item = summary[f"top_{k}_cost_10bps"]
            rows.append(
                {
                    "run": run,
                    "k": k,
                    "top_ann": item["top_net"]["annualized_return"],
                    "excess_univ_ann": item["top_excess_vs_universe_net"][
                        "annualized_return"
                    ],
                    "ls_ann": item["long_short_net"]["annualized_return"],
                    "top_turnover": item["average_top_turnover"],
                    "ls_turnover": item["average_long_short_turnover"],
                }
            )
    return pd.DataFrame(rows)


def collect_buffer() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run, run_dir in RUNS.items():
        path = run_dir / "turnover_control" / "turnover_control_metrics.json"
        if not path.exists():
            continue
        summary = read_json(path)["summary"]["test"]
        for k in (10, 20, 30):
            for keep in ("1", "1.5", "2", "3"):
                item = summary[f"top_{k}_keep_{keep}x_cost_10bps"]
                rows.append(
                    {
                        "run": run,
                        "k": k,
                        "keep": keep,
                        "top_ann": item["top_net"]["annualized_return"],
                        "excess_univ_ann": item["top_excess_vs_universe_net"][
                            "annualized_return"
                        ],
                        "ls_ann": item["long_short_net"]["annualized_return"],
                        "top_turnover": item.get(
                            "average_top_turnover", item["average_turnover"]
                        ),
                        "ls_turnover": item["average_long_short_turnover"],
                        "maxdd": item["top_net"]["max_drawdown"],
                    }
                )
    return pd.DataFrame(rows)


def print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def main() -> None:
    ic = collect_ic()
    topk = collect_topk()
    backtest = collect_backtest()
    buffer = collect_buffer()

    out_dir = Path("outputs/analysis/turnover_overlay_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    ic.to_csv(out_dir / "ic.csv", index=False)
    topk.to_csv(out_dir / "topk_proxy.csv", index=False)
    backtest.to_csv(out_dir / "normal_backtest_10bps.csv", index=False)
    buffer.to_csv(out_dir / "turnover_buffer_10bps.csv", index=False)

    print_frame("IC test", ic[ic["split"] == "test"])
    print_frame("TopK proxy test", topk)
    print_frame("Normal backtest test 10bps", backtest)
    print_frame(
        "Turnover buffer test 10bps - overlay",
        buffer[buffer["run"] == "old_full62_strictmask_overlay"],
    )
    candidates = buffer[
        (buffer["excess_univ_ann"] > 0.0) & (buffer["top_turnover"] < 0.8)
    ].sort_values(["excess_univ_ann", "top_ann"], ascending=False)
    print_frame("Best positive-excess candidates with turnover < 0.8", candidates.head(12))


if __name__ == "__main__":
    main()
