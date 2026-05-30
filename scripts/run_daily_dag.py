from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, args: list[str]) -> dict:
    command = [sys.executable, *args]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    json_start = stdout.find("{")
    payload = json.loads(stdout[json_start:]) if json_start >= 0 else {"stdout": stdout}
    return {"step": name, "result": payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily quant data DAG.")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--start-date", default="20160104")
    parser.add_argument("--end-date", default="20260525")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-pool", action="store_true")
    parser.add_argument("--skip-state-build", action="store_true")
    parser.add_argument("--skip-mart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[dict] = []
    if not args.skip_ingest:
        results.append(run_step("ingest_raw", ["scripts/data/run_ingest_raw.py", "--data-version", args.data_version]))
    if not args.skip_pool:
        results.append(run_step("build_pool", ["scripts/data/run_build_pool.py", "--data-version", args.data_version]))
    if not args.skip_state_build:
        results.append(
            run_step(
                "build_market_state",
                ["scripts/data/run_build_market_state.py", "--data-version", args.data_version, "--incremental"],
            )
        )
    results.append(
        run_step(
            "validate_market_state_coverage",
            [
                "scripts/data/validate_market_state_coverage.py",
                "--data-version",
                args.data_version,
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--strict",
            ],
        )
    )
    if not args.skip_mart:
        results.append(
            run_step(
                "build_mart",
                [
                    "scripts/data/run_build_mart.py",
                    "--data-version",
                    args.data_version,
                    "--start-date",
                    args.start_date,
                    "--end-date",
                    args.end_date,
                ],
            )
        )
    print(json.dumps({"data_version": args.data_version, "dag_status": "PASS", "steps": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
