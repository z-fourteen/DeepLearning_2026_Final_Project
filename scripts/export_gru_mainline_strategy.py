from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.ingest.agent import load_yaml


DEFAULT_CONFIG = Path("configs/gru_mainline_strategy.yaml")


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
    parser = argparse.ArgumentParser(description="Export the promoted GRU mainline strategy artifact.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def load_strategy(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    strategy = config.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError(f"Missing strategy block in {config_path}")
    return strategy


def selected_periods(strategy: dict[str, Any]) -> pd.DataFrame:
    overlay_run = Path(strategy["prediction_overlay_run"])
    periods_path = overlay_run / "turnover_control" / "turnover_control_periods.csv"
    if not periods_path.exists():
        raise FileNotFoundError(f"Missing turnover-control periods: {periods_path}")

    portfolio = strategy["portfolio"]
    periods = pd.read_csv(periods_path)
    mask = (
        periods["k"].eq(int(portfolio["k"]))
        & np.isclose(periods["keep_multiplier"], float(portfolio["keep_multiplier"]))
        & np.isclose(periods["cost_bps"], float(portfolio["cost_bps"]))
    )
    result = periods[mask].copy()
    if result.empty:
        raise ValueError("No periods matched the configured mainline portfolio.")
    return result.sort_values(["split", "trade_date"]).reset_index(drop=True)


def selected_summary(strategy: dict[str, Any]) -> dict[str, Any]:
    overlay_run = Path(strategy["prediction_overlay_run"])
    metrics_path = overlay_run / "turnover_control" / "turnover_control_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing turnover-control metrics: {metrics_path}")

    portfolio = strategy["portfolio"]
    key = (
        f"top_{int(portfolio['k'])}_keep_{float(portfolio['keep_multiplier']):g}x_"
        f"cost_{float(portfolio['cost_bps']):g}bps"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = {
        split: split_summary[key]
        for split, split_summary in metrics["summary"].items()
        if key in split_summary
    }
    if not summary:
        raise ValueError(f"Missing mainline summary key: {key}")
    return {
        "metric_key": key,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    strategy = load_strategy(config_path)
    out_dir = Path(strategy["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    periods = selected_periods(strategy)
    summary = selected_summary(strategy)
    manifest = {
        "strategy": strategy,
        "config": str(config_path),
        "period_rows": int(len(periods)),
        "splits": periods["split"].value_counts().sort_index().to_dict(),
        **summary,
    }

    periods.to_csv(out_dir / "mainline_periods.csv", index=False)
    (out_dir / "mainline_summary.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
