from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.backtest_t1_fill_sim import build_summary, load_data, run_backtest  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def resolve_path(path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated clean-resid T+1 Top20 keep2x mainline.")
    parser.add_argument("--config", default="configs/backtest/clean_resid_t1_top20_keep2.yaml")
    parser.add_argument("--output-dir", help="Override outputs.backtest_dir from config.")
    return parser.parse_args()


def flatten_summary(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_summary in metrics.items():
        for setting, values in split_summary.items():
            rows.append(
                {
                    "split": split,
                    "setting": setting,
                    "net_ann": values["net"]["annualized_return"],
                    "net_ir": values["net"]["ir"],
                    "net_mdd": values["net"]["max_drawdown"],
                    "gross_ann": values["gross"]["annualized_return"],
                    "excess_benchmark_ann": values["excess_vs_benchmark"]["annualized_return"],
                    "excess_exec_universe_ann": values["excess_vs_executable_universe"]["annualized_return"],
                    "benchmark_ann": values["benchmark"]["annualized_return"],
                    "executable_universe_ann": values["executable_universe"]["annualized_return"],
                    "avg_desired_turnover": values["average_desired_turnover"],
                    "avg_filled_turnover": values["average_filled_turnover"],
                    "avg_transaction_cost": values["average_transaction_cost"],
                    "avg_buy_reject_count": values["average_buy_reject_count"],
                    "avg_sell_reject_count": values["average_sell_reject_count"],
                    "avg_partial_fill_count": values["average_partial_fill_count"],
                    "avg_position_count": values["average_position_count"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    inputs = config["inputs"]
    execution = config["execution"]
    out_dir = resolve_path(args.output_dir or config["outputs"]["backtest_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(resolve_path(inputs["predictions"]), resolve_path(inputs["execution_labels"]))
    periods = run_backtest(
        frame=data,
        k_values=[int(execution["k"])],
        keep_multipliers=[float(execution["keep_multiplier"])],
        cost_bps=float(execution["cost_bps"]),
        slippage_bps=float(execution["slippage_bps"]),
        portfolio_nav=float(execution["portfolio_nav"]),
        participation_cap=float(execution["participation_cap"]),
        rebalance_stride=int(execution["rebalance_stride"]),
        min_daily_count=int(execution["min_daily_count"]),
    )
    summary = build_summary(periods)
    flat = flatten_summary(summary)

    periods.to_csv(out_dir / "mainline_periods.csv", index=False)
    flat.to_csv(out_dir / "mainline_summary.csv", index=False)
    manifest = {
        "mainline": config["mainline"],
        "config": str(config_path),
        "predictions": str(resolve_path(inputs["predictions"])),
        "execution_labels": str(resolve_path(inputs["execution_labels"])),
        "output_dir": str(out_dir),
        "rows_after_merge": int(len(data)),
        "execution": execution,
        "summary": summary,
        "method": "isolated_clean_resid_t1_open_top20_keep2x",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
