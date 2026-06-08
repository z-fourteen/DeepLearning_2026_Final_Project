from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live stage 0: prepare feature panel and account input files."
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--data-version", default="v20260526")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--feature-date", required=True)
    parser.add_argument("--features-parquet", default="")
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--mart-features")
    parser.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help="Allow feature preparation to fall back to raw daily data.",
    )
    parser.add_argument(
        "--skip-prepare-features",
        action="store_true",
        help="Use an existing feature panel and only prepare account inputs.",
    )
    parser.add_argument(
        "--skip-prepare-account-inputs",
        action="store_true",
        help="Only prepare the feature panel.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    print()
    print("=" * 88)
    print(name)
    print("=" * 88)
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    feature_panel = args.features_parquet or f"data/live/features/features_{args.feature_date}.parquet"

    if not args.skip_prepare_features:
        command = [
            sys.executable,
            "scripts/live/00_prepare_live_features.py",
            "--config",
            args.config,
            "--data-version",
            args.data_version,
            "--trade-date",
            args.trade_date,
            "--feature-date",
            args.feature_date,
            "--output",
            feature_panel,
        ]
        if args.lookback is not None:
            command.extend(["--lookback", str(args.lookback)])
        if args.mart_features:
            command.extend(["--mart-features", args.mart_features])
        if args.allow_raw_fallback:
            command.append("--allow-raw-fallback")
        run_step("Stage 0a: prepare live features", command)
    else:
        print(f"Stage 0a skipped; using existing feature panel: {feature_panel}")

    if not args.skip_prepare_account_inputs:
        command = [
            sys.executable,
            "scripts/live/00_prepare_live_account_inputs.py",
            "--config",
            args.config,
            "--trade-date",
            args.trade_date,
            "--feature-date",
            args.feature_date,
            "--features-parquet",
            feature_panel,
        ]
        if args.overwrite:
            command.append("--overwrite")
        run_step("Stage 0b: prepare live account inputs", command)
    else:
        print("Stage 0b skipped by request.")


if __name__ == "__main__":
    main()
