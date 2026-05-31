from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.portfolio.optimize_feasible_cash_buffer import load_data, run_optimizer, summarize  # noqa: E402


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen final mainline optimizer.")
    parser.add_argument("--config", default="configs/portfolio/final_mainline_optimizer.yaml")
    parser.add_argument("--output-dir", help="Override outputs.output_dir from config.")
    return parser.parse_args()


def optimizer_args(config: dict[str, Any], output_dir: Path) -> SimpleNamespace:
    inputs = config["inputs"]
    opt = config["optimizer"]
    return SimpleNamespace(
        predictions=str(resolve(inputs["predictions"])),
        mart=str(resolve(inputs["mart"])),
        labels=str(resolve(inputs["labels"])),
        output_dir=str(output_dir),
        risk_control=[str(opt["risk_control"])],
        k=[int(opt["k"])],
        style_penalty=[float(opt["style_penalty"])],
        turnover_penalty=[float(opt["turnover_penalty"])],
        candidate_multiplier=float(opt["candidate_multiplier"]),
        exposure_cap=float(opt["exposure_cap"]),
        exposure_slack_penalty=float(opt["exposure_slack_penalty"]),
        buy_capacity_slack_penalty=float(opt["buy_capacity_slack_penalty"]),
        cash_penalty=float(opt["cash_penalty"]),
        min_invested_shortfall_penalty=float(opt["min_invested_shortfall_penalty"]),
        solver=str(opt["solver"]),
        single_name_cap=float(opt["single_name_cap"]),
        min_invested=float(opt["min_invested"]),
        turnover_cap=float(opt["turnover_cap"]),
        portfolio_nav=float(opt["portfolio_nav"]),
        participation_cap=float(opt["participation_cap"]),
        cost_bps=float(opt["cost_bps"]),
        slippage_bps=float(opt["slippage_bps"]),
        rebalance_stride=int(opt["rebalance_stride"]),
        min_daily_count=int(opt["min_daily_count"]),
    )


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    out_dir = resolve(args.output_dir or config["outputs"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    run_args = optimizer_args(config, out_dir)
    data = load_data(run_args)
    periods = run_optimizer(run_args, data)
    summary = summarize(periods)

    periods.to_csv(out_dir / "final_optimizer_periods.csv", index=False)
    summary.to_csv(out_dir / "final_optimizer_summary.csv", index=False)
    manifest = {
        "mainline": config["mainline"],
        "config": str(config_path),
        "inputs": config["inputs"],
        "optimizer": config["optimizer"],
        "evidence": config.get("evidence", {}),
        "output_dir": str(out_dir),
        "period_rows": int(len(periods)),
        "summary_rows": int(len(summary)),
        "method": "frozen_final_mainline_optimizer",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
