from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNS = {
    "final_mainline_l60_ckptscore_e12": PROJECT_ROOT
    / "outputs"
    / "runs"
    / "feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean"
    / "predictions.parquet",
    "clean_alpha_only": PROJECT_ROOT
    / "outputs"
    / "runs"
    / "gru_l20_clean_alpha_only_purgedwf_strictmask_leaky0005"
    / "predictions.parquet",
    "legacy_l20_clean_alpha_resid_style": PROJECT_ROOT
    / "outputs"
    / "runs"
    / "gru_l20_clean_alpha_resid_style_purgedwf_strictmask_leaky0005"
    / "predictions.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clean_dataset GRU predictions with T+1 fill simulation and an optional optimizer."
    )
    parser.add_argument("--output-dir", default="outputs/backtest/clean_dataset_execution_stack")
    parser.add_argument("--execution-labels", default="data/mart/labels/execution_labels_v20260526.parquet")
    parser.add_argument("--mart", default="data/mart/datasets/core/dataset_v20260526.parquet")
    parser.add_argument("--k", default="10,20,30")
    parser.add_argument("--keep-multiplier", default="1,1.5,2,3")
    parser.add_argument(
        "--risk-control",
        default="none,industry_proxy,industry_size,industry_size_liquidity_vol_mom",
    )
    parser.add_argument("--style-penalty", default="0,0.05,0.10,0.20")
    parser.add_argument("--turnover-penalty", default="0,0.02")
    parser.add_argument(
        "--optimizer-script",
        help=(
            "Optional production optimizer entry point. Omit to keep this clean stack limited "
            "to T+1 fill simulation."
        ),
    )
    parser.add_argument("--portfolio-nav", default="10000000")
    parser.add_argument("--participation-cap", default="0.03")
    parser.add_argument("--cost-bps", default="10")
    parser.add_argument("--slippage-bps", default="5")
    parser.add_argument("--rebalance-stride", default="5")
    parser.add_argument("--min-daily-count", default="40")
    parser.add_argument("--buy-capacity-slack-penalty", default="1000")
    parser.add_argument(
        "--min-invested",
        default="0.80",
        help="Minimum invested weight passed to the optimizer. Default follows the competition rule.",
    )
    parser.add_argument(
        "--min-invested-shortfall-penalty",
        default="0",
        help="Keep at 0 for a hard minimum invested constraint.",
    )
    parser.add_argument("--only-existing", action="store_true", help="Skip runs whose prediction file is missing.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_t1_metrics(run_name: str, path: Path) -> list[dict[str, Any]]:
    metrics = read_json(path)
    rows: list[dict[str, Any]] = []
    for split, split_summary in metrics.get("summary", {}).items():
        for key, values in split_summary.items():
            rows.append(
                {
                    "run": run_name,
                    "engine": "t1_fill_rank_buffer",
                    "split": split,
                    "setting": key,
                    "net_ann": values["net"]["annualized_return"],
                    "net_ir": values["net"]["ir"],
                    "net_mdd": values["net"]["max_drawdown"],
                    "excess_benchmark_ann": values["excess_vs_benchmark"]["annualized_return"],
                    "excess_exec_universe_ann": values["excess_vs_executable_universe"]["annualized_return"],
                    "avg_filled_turnover": values["average_filled_turnover"],
                    "avg_position_count": values["average_position_count"],
                }
            )
    return rows


def summarize_optimizer(run_name: str, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict(orient="records"):
        rows.append(
            {
                "run": run_name,
                "engine": "barra_lite_optimizer",
                "split": row["split"],
                "setting": (
                    f"{row['risk_control']}_k{int(row['k'])}_"
                    f"style{row['style_penalty']:g}_turn{row['turnover_penalty']:g}"
                ),
                "net_ann": row["net_ann"],
                "net_ir": row["net_ir"],
                "net_mdd": row["net_max_drawdown"],
                "excess_benchmark_ann": row["excess_benchmark_ann"],
                "excess_exec_universe_ann": row["excess_exec_universe_ann"],
                "avg_filled_turnover": row["avg_filled_turnover"],
                "avg_position_count": row["avg_position_count"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    executed: dict[str, dict[str, str]] = {}
    for run_name, predictions in RUNS.items():
        if not predictions.exists():
            if args.only_existing:
                continue
            raise FileNotFoundError(f"Missing predictions for {run_name}: {predictions}")

        run_out = out_dir / run_name
        t1_out = run_out / "t1_fill"
        opt_out = run_out / "optimizer"
        t1_out.mkdir(parents=True, exist_ok=True)
        opt_out.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                sys.executable,
                "scripts/backtest/backtest_t1_fill_sim.py",
                "--predictions",
                str(predictions),
                "--execution-labels",
                args.execution_labels,
                "--output-dir",
                str(t1_out),
                "--k",
                args.k,
                "--keep-multiplier",
                args.keep_multiplier,
                "--portfolio-nav",
                args.portfolio_nav,
                "--participation-cap",
                args.participation_cap,
                "--cost-bps",
                args.cost_bps,
                "--slippage-bps",
                args.slippage_bps,
                "--rebalance-stride",
                args.rebalance_stride,
                "--min-daily-count",
                args.min_daily_count,
            ]
        )
        if args.optimizer_script:
            run_command(
                [
                    sys.executable,
                    args.optimizer_script,
                    "--predictions",
                    str(predictions),
                    "--mart",
                    args.mart,
                    "--labels",
                    args.execution_labels,
                    "--output-dir",
                    str(opt_out),
                    "--risk-control",
                    args.risk_control,
                    "--k",
                    args.k,
                    "--style-penalty",
                    args.style_penalty,
                    "--turnover-penalty",
                    args.turnover_penalty,
                    "--portfolio-nav",
                    args.portfolio_nav,
                    "--participation-cap",
                    args.participation_cap,
                    "--cost-bps",
                    args.cost_bps,
                    "--slippage-bps",
                    args.slippage_bps,
                    "--rebalance-stride",
                    args.rebalance_stride,
                    "--min-daily-count",
                    args.min_daily_count,
                    "--buy-capacity-slack-penalty",
                    args.buy_capacity_slack_penalty,
                    "--min-invested",
                    args.min_invested,
                    "--min-invested-shortfall-penalty",
                    args.min_invested_shortfall_penalty,
                ]
            )

        all_rows.extend(summarize_t1_metrics(run_name, t1_out / "t1_fill_metrics.json"))
        if args.optimizer_script:
            all_rows.extend(summarize_optimizer(run_name, opt_out / "optimizer_summary.csv"))
        executed[run_name] = {
            "predictions": str(predictions),
            "t1_fill_dir": str(t1_out),
            "optimizer_dir": str(opt_out) if args.optimizer_script else "",
        }

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "output_dir": str(out_dir),
        "runs": executed,
        "summary": str(summary_path),
        "method": "clean_dataset_strictmask_t1_fill_optional_optimizer",
        "optimizer_script": args.optimizer_script or "",
        "buy_capacity_slack_penalty": args.buy_capacity_slack_penalty,
        "min_invested": args.min_invested,
        "min_invested_shortfall_penalty": args.min_invested_shortfall_penalty,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
