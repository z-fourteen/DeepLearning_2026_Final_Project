from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def try_parse_json(text: str) -> dict | list:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            return json.loads(text[idx:].strip())
        except json.JSONDecodeError:
            continue
    return {"stdout": text[-500:] if len(text) > 500 else text}


def run_step(name: str, args: list[str]) -> dict:
    command = [sys.executable, *args]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Step {name!r} failed with exit code {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-500:]}"
        )
    stdout = completed.stdout.strip()
    return {"step": name, "result": try_parse_json(stdout)}


def skipped_step(name: str, reason: str = "skipped by flag") -> dict:
    return {"step": name, "result": {"skipped": True, "reason": reason}}


def pool_output_path() -> Path:
    config_path = PROJECT_ROOT / "configs/data/data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    pool = config.get("pool", {})
    output_dir = pool.get("output_dir", "data/lake/core/chinext_pool")
    scd2_file = pool.get("scd2_file", "chinext_pool_scd2.parquet")
    return PROJECT_ROOT / output_dir / scd2_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily quant data DAG.")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--start-date", default="20160104")
    parser.add_argument("--end-date", default="20260525")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-pool", action="store_true")
    parser.add_argument("--skip-state-build", action="store_true")
    parser.add_argument("--skip-mart", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Daily incremental mode: ingest detects changed source files, "
            "pool is skipped, state builds missing/raw-newer dates, and mart is rebuilt."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[dict] = []
    pool_path = pool_output_path()
    pool_missing = not pool_path.exists()

    if args.incremental and not args.skip_pool and not pool_missing:
        args.skip_pool = True

    if not args.skip_ingest:
        results.append(run_step("ingest_raw", ["scripts/data/run_ingest_raw.py", "--data-version", args.data_version]))
    else:
        results.append(skipped_step("ingest_raw"))

    if not args.skip_pool:
        pool_args = ["scripts/data/run_build_pool.py", "--data-version", args.data_version]
        if pool_missing:
            pool_args.extend(["--backfill", "--overwrite"])
        results.append(run_step("build_pool", pool_args))
    else:
        reason = "incremental mode; existing pool SCD2 found" if args.incremental else "skipped by flag"
        results.append(skipped_step("build_pool", reason))

    if not args.skip_state_build:
        results.append(
            run_step(
                "build_market_state",
                ["scripts/data/run_build_market_state.py", "--data-version", args.data_version, "--incremental"],
            )
        )
    else:
        results.append(skipped_step("build_market_state"))

    if not args.skip_validate:
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
    else:
        results.append(skipped_step("validate_market_state_coverage"))

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
    else:
        results.append(skipped_step("build_mart"))

    mode = "incremental" if args.incremental else "full"
    print(
        json.dumps(
            {"data_version": args.data_version, "dag_mode": mode, "dag_status": "PASS", "steps": results},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
