from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_CONFIGS = [
    "configs/sequence_gru_l20_mse_ic_leaky_head_slope_0005.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GRU LeakyReLU head robustness grid sequentially."
    )
    parser.add_argument("--device", default="cuda", help="Device passed to train_sequence.py.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch child scripts.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run Top-K proxy and holding-period backtest after each successful training run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build train objects for each config without training.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun training even when metrics.json and predictions.parquet already exist.",
    )
    parser.add_argument(
        "--force-evaluate",
        action="store_true",
        help="Rerun evaluation even when topk_metrics.json and backtest_metrics.json already exist.",
    )
    parser.add_argument("--max-epochs", type=int, help="Optional training epoch override.")
    parser.add_argument("--max-train-batches", type=int, help="Optional smoke-test train batch cap.")
    parser.add_argument("--max-val-batches", type=int, help="Optional smoke-test validation batch cap.")
    parser.add_argument("--max-test-batches", type=int, help="Optional smoke-test test batch cap.")
    parser.add_argument(
        "--labels",
        default="data/mart/labels/labels_v20260526.parquet",
        help="Labels parquet used by backtest_topk.py when --evaluate is enabled.",
    )
    parser.add_argument("--k", default="10,20,30", help="Comma-separated Top-K values for evaluation.")
    parser.add_argument(
        "--cost-bps",
        default="0,10,20",
        help="Comma-separated one-way cost bps values for backtest evaluation.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(f"\n[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def load_output_dir(config_path: Path) -> Path:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    output_dir = Path(config["run"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def training_complete(output_dir: Path) -> bool:
    return (output_dir / "metrics.json").exists() and (output_dir / "predictions.parquet").exists()


def evaluation_complete(output_dir: Path) -> bool:
    return (output_dir / "topk_metrics.json").exists() and (output_dir / "backtest_metrics.json").exists()


def train_command(args: argparse.Namespace, config: str) -> list[str]:
    command = [
        args.python,
        "scripts/train_sequence.py",
        "--config",
        config,
        "--device",
        args.device,
    ]
    if args.dry_run:
        command.append("--dry-run")
    optional_flags = {
        "--max-epochs": args.max_epochs,
        "--max-train-batches": args.max_train_batches,
        "--max-val-batches": args.max_val_batches,
        "--max-test-batches": args.max_test_batches,
    }
    for flag, value in optional_flags.items():
        if value is not None:
            command.extend([flag, str(value)])
    return command


def evaluate_run(args: argparse.Namespace, config: str) -> None:
    output_dir = load_output_dir(PROJECT_ROOT / config)
    predictions = output_dir / "predictions.parquet"
    if not predictions.exists():
        raise FileNotFoundError(f"Missing predictions file after training: {predictions}")

    run_command(
        [
            args.python,
            "scripts/evaluate_topk.py",
            "--predictions",
            str(predictions.relative_to(PROJECT_ROOT)),
            "--k",
            args.k,
        ]
    )
    run_command(
        [
            args.python,
            "scripts/backtest_topk.py",
            "--predictions",
            str(predictions.relative_to(PROJECT_ROOT)),
            "--labels",
            args.labels,
            "--k",
            args.k,
            "--cost-bps",
            args.cost_bps,
        ]
    )


def main() -> None:
    args = parse_args()
    for config in GRID_CONFIGS:
        output_dir = load_output_dir(PROJECT_ROOT / config)
        if args.dry_run:
            run_command(train_command(args, config))
            continue

        if training_complete(output_dir) and not args.force_rerun:
            print(
                f"\n[skip] training already complete for {config}: {output_dir.relative_to(PROJECT_ROOT)}",
                flush=True,
            )
        else:
            run_command(train_command(args, config))

        if args.evaluate and not args.dry_run:
            if evaluation_complete(output_dir) and not args.force_evaluate:
                print(
                    f"[skip] evaluation already complete for {config}: {output_dir.relative_to(PROJECT_ROOT)}",
                    flush=True,
                )
            else:
                evaluate_run(args, config)

    print("\nAll LeakyReLU robustness grid runs completed.", flush=True)


if __name__ == "__main__":
    main()
