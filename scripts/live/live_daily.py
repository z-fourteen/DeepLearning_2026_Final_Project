from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live.common import format_path, load_yaml, resolve_path, today_yyyymmdd


def print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def run_command(name: str, args: list[str], interactive: bool = False) -> None:
    print_header(name)
    print(" ".join(args))
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=not interactive,
        text=not interactive,
        encoding="utf-8" if not interactive else None,
        errors="replace" if not interactive else None,
    )
    if completed.returncode != 0:
        if not interactive:
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            if stdout.strip():
                print(stdout[-2000:])
            if stderr.strip():
                print(stderr[-2000:], file=sys.stderr)
        raise SystemExit(completed.returncode)
    if not interactive and completed.stdout:
        print(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command live pipeline for the current frozen GRU mainline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full build through the latest available feature date + live order generation:
    python scripts/live/live_daily.py --run-dag --full-dag --data-version v20260526 --end-date 20260529 --trade-date 20260601

  Daily auto-incremental update + live order generation:
    python scripts/live/live_daily.py --run-dag --data-version v20260526 --end-date 20260529 --trade-date 20260601

  Use existing data and only rerun live stages:
    python scripts/live/live_daily.py --skip-dag --trade-date 20260601 --feature-date 20260529
        """,
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default="")
    parser.add_argument("--run-dag", action="store_true", help="Run data DAG before live inference.")
    parser.add_argument("--skip-dag", action="store_true", help="Skip data DAG and use existing live inputs.")
    parser.add_argument("--full-dag", action="store_true", help="Force full DAG. Without this, --run-dag uses auto incremental mode.")
    parser.add_argument("--data-version", default="v20260526")
    parser.add_argument("--start-date", default="20160104")
    parser.add_argument(
        "--end-date",
        default="",
        help="Latest available raw/feature date for the DAG, for example 20260529 before the 20260601 session.",
    )
    parser.add_argument(
        "--feature-date",
        default="",
        help="As-of date used for model sequences. Defaults to --end-date when set, otherwise --trade-date.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--features-parquet", help="Override live feature panel for stage 1.")
    parser.add_argument("--skip-prepare-features", action="store_true", help="Skip stage 0.5 live feature materialization.")
    parser.add_argument("--skip-prepare-account-inputs", action="store_true", help="Skip current positions and quote snapshot preparation.")
    parser.add_argument(
        "--allow-raw-feature-fallback",
        action="store_true",
        help="Allow stage 0.5 to use raw daily fallback if mart features are unavailable.",
    )
    parser.add_argument("--sequence-npz", help="Override live sequence NPZ for stage 1.")
    parser.add_argument("--positions", help="Override current positions CSV for stages 2/3.")
    parser.add_argument("--previous-close-positions", help="Override previous close positions CSV for stage 2.")
    parser.add_argument("--liquidity-parquet", help="Override liquidity/feature panel for stage 2.")
    parser.add_argument("--price-snapshot", help="Override 09:20 price snapshot CSV for stages 3/5.")
    parser.add_argument("--execute", action="store_true", help="Run interactive execution stage after generating orders.")
    parser.add_argument("--reset", action="store_true", help="Reset portfolio state in interactive execution stage.")
    parser.add_argument("--initial-nav", type=float, default=1_000_000.0)
    parser.add_argument("--no-push", action="store_true", help="Skip git push in interactive execution stage.")
    parser.add_argument("--push-branch", help="Branch to push in interactive execution stage.")
    return parser.parse_args()


def append_optional(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def run_data_dag(args: argparse.Namespace, feature_date: str) -> None:
    if args.skip_dag:
        print_header("Data DAG skipped")
        print("Using existing data/live inputs.")
        return
    if not args.run_dag:
        print_header("Data DAG not requested")
        print("Pass --run-dag for full/incremental data refresh, or --skip-dag to make this explicit.")
        return

    command = [
        sys.executable,
        "scripts/run_daily_dag.py",
        "--data-version",
        args.data_version,
        "--start-date",
        args.start_date,
        "--end-date",
        feature_date,
    ]
    if not args.full_dag and not args.reset:
        command.append("--incremental")
    run_command("Stage 0: data DAG", command)


def default_feature_panel_path(config_path: str, feature_date: str) -> str:
    live_config = load_yaml(config_path)
    path = format_path(
        live_config["live_inputs"]["feature_panel"],
        trade_date=feature_date,
        prev_trade_date="NA",
    )
    return str(path.relative_to(PROJECT_ROOT))


def prepare_live_inputs(args: argparse.Namespace, trade_date: str, feature_date: str, feature_panel: str) -> None:
    skip_features = bool(args.skip_prepare_features or args.features_parquet)
    skip_account = bool(args.skip_prepare_account_inputs or (args.positions and args.price_snapshot))
    if skip_features and skip_account:
        print_header("Stage 0: live input preparation skipped")
        if args.features_parquet:
            print("--features-parquet override provided")
        if args.positions and args.price_snapshot:
            print("--positions and --price-snapshot overrides provided")
        return

    command = [
        sys.executable,
        "scripts/live/00_prepare_live_inputs.py",
        "--config",
        str(resolve_path(args.config).relative_to(PROJECT_ROOT)),
        "--data-version",
        args.data_version,
        "--trade-date",
        trade_date,
        "--feature-date",
        feature_date,
        "--features-parquet",
        feature_panel,
    ]
    if skip_features:
        command.append("--skip-prepare-features")
    if skip_account:
        command.append("--skip-prepare-account-inputs")
    if args.allow_raw_feature_fallback:
        command.append("--allow-raw-fallback")
    if args.reset:
        command.append("--overwrite")
    run_command("Stage 0: prepare live inputs", command)


def run_live_stages(args: argparse.Namespace, trade_date: str, feature_date: str) -> None:
    config = str(resolve_path(args.config).relative_to(PROJECT_ROOT))
    feature_panel = args.features_parquet or default_feature_panel_path(args.config, feature_date)

    inference = [
        sys.executable,
        "scripts/live/01_live_inference.py",
        "--config",
        config,
        "--trade-date",
        trade_date,
        "--feature-date",
        feature_date,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    append_optional(inference, "--features-parquet", feature_panel)
    append_optional(inference, "--sequence-npz", args.sequence_npz)
    run_command("Stage 1: frozen GRU inference", inference)

    optimization = [
        sys.executable,
        "scripts/live/02_live_optimization.py",
        "--config",
        config,
        "--trade-date",
        trade_date,
        "--feature-date",
        feature_date,
    ]
    append_optional(optimization, "--positions", args.positions)
    append_optional(optimization, "--previous-close-positions", args.previous_close_positions)
    append_optional(optimization, "--liquidity-parquet", args.liquidity_parquet or feature_panel)
    run_command("Stage 2: live optimizer", optimization)

    orders = [
        sys.executable,
        "scripts/live/03_generate_target_orders.py",
        "--config",
        config,
        "--trade-date",
        trade_date,
    ]
    append_optional(orders, "--positions", args.positions)
    append_optional(orders, "--price-snapshot", args.price_snapshot)
    run_command("Stage 3: target orders", orders)

    if args.execute:
        execution = [
            sys.executable,
            "scripts/live/05_interactive_execution.py",
            "--config",
            config,
            "--trade-date",
            trade_date,
            "--initial-nav",
            str(args.initial_nav),
        ]
        append_optional(execution, "--price-snapshot", args.price_snapshot)
        if args.reset:
            execution.append("--reset")
        if args.no_push:
            execution.append("--no-push")
        append_optional(execution, "--push-branch", args.push_branch)
        run_command("Stage 5: interactive execution", execution, interactive=True)


def main() -> None:
    args = parse_args()
    trade_date = args.trade_date or today_yyyymmdd()
    feature_date = args.feature_date or args.end_date or trade_date

    print_header(f"Current GRU live daily pipeline - {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"project_root={PROJECT_ROOT}")
    print(f"trade_date={trade_date}")
    print(f"feature_date/end_date={feature_date}")
    print(f"config={args.config}")
    print("model=current frozen GRU mainline from configs/live/live_trading.yaml")

    run_data_dag(args, feature_date)
    feature_panel = args.features_parquet or default_feature_panel_path(args.config, feature_date)
    prepare_live_inputs(args, trade_date, feature_date, feature_panel)
    run_live_stages(args, trade_date, feature_date)

    print_header("Live pipeline completed")
    print(f"predictions: outputs/live_predictions/predictions_{trade_date}.parquet")
    print(f"targets:     outputs/live_targets/target_weights_{trade_date}.csv")
    print(f"orders:      outputs/live_orders/orders_{trade_date}.csv")
    print()
    print("One-command forms:")
    print(
        f"  full:        python scripts/live/live_daily.py --run-dag --full-dag "
        f"--data-version {args.data_version} --end-date {feature_date} --trade-date {trade_date}"
    )
    print(
        f"  incremental: python scripts/live/live_daily.py --run-dag "
        f"--data-version {args.data_version} --end-date {feature_date} --trade-date {trade_date}"
    )


if __name__ == "__main__":
    main()
