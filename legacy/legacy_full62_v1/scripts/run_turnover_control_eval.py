from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Top-K and turnover-control evaluations for a run directory.")
    parser.add_argument("--run-dir", required=True, help="Directory containing predictions.parquet.")
    parser.add_argument("--labels", default="data/mart/labels/labels_v20260526.parquet")
    parser.add_argument("--k", default="10,20,30")
    parser.add_argument("--cost-bps", default="0,10,20")
    parser.add_argument("--keep-multiplier", default="1,1.5,2,3")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    predictions = run_dir / "predictions.parquet"
    if not predictions.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions}")

    run(
        [
            sys.executable,
            "scripts/evaluate_topk.py",
            "--predictions",
            str(predictions),
            "--k",
            args.k,
        ]
    )
    run(
        [
            sys.executable,
            "scripts/backtest_topk.py",
            "--predictions",
            str(predictions),
            "--labels",
            args.labels,
            "--k",
            args.k,
            "--cost-bps",
            args.cost_bps,
        ]
    )
    run(
        [
            sys.executable,
            "scripts/backtest_topk_turnover_control.py",
            "--predictions",
            str(predictions),
            "--labels",
            args.labels,
            "--k",
            args.k,
            "--cost-bps",
            args.cost_bps,
            "--keep-multiplier",
            args.keep_multiplier,
            "--output-dir",
            str(run_dir / "turnover_control"),
        ]
    )


if __name__ == "__main__":
    main()
