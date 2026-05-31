from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, args: list[str], capture: bool = True) -> dict:
    command = [sys.executable, *args]
    kwargs: dict = {"cwd": PROJECT_ROOT, "check": False}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
        kwargs["encoding"] = "utf-8"
        kwargs["errors"] = "replace"
    else:
        kwargs["capture_output"] = False
    completed = subprocess.run(command, **kwargs)
    if completed.returncode != 0:
        stderr = getattr(completed, "stderr", "") or ""
        raise RuntimeError(f"Step '{name}' failed (exit {completed.returncode}): {stderr[-300:]}")
    if capture:
        stdout = completed.stdout.strip()
        payload = _try_parse_json(stdout)
    else:
        payload = {}
    return {"step": name, "result": payload}


def _try_parse_json(text: str) -> dict | list:
    """尝试从文本中提取最后一个完整 JSON 对象，失败则返回原始文本。"""
    # 找最后一个 '{' 或 '[' 作为 JSON 起始点
    for start_marker in ("{", "["):
        idx = text.rfind(start_marker)
        if idx < 0:
            continue
        candidate = text[idx:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {"stdout": text[-300:] if len(text) > 300 else text}


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
    parser.add_argument("--incremental", action="store_true",
                        help="智能增量模式: 自动跳过 pool (极少变化), "
                             "ingest/state 使用增量, mart 全量 (因 rolling 特征依赖完整历史)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[dict] = []

    if args.incremental:
        args.skip_pool = True  # 成分股池几乎不变，跳过

    if not args.skip_ingest:
        results.append(run_step("ingest_raw", ["scripts/data/run_ingest_raw.py", "--data-version", args.data_version]))
    else:
        results.append({"step": "ingest_raw", "result": {"skipped": True}})

    if not args.skip_pool:
        results.append(run_step("build_pool", ["scripts/data/run_build_pool.py", "--data-version", args.data_version]))
    else:
        results.append({"step": "build_pool", "result": {"skipped": True, "reason": "incremental mode"}})

    if not args.skip_state_build:
        results.append(
            run_step(
                "build_market_state",
                ["scripts/data/run_build_market_state.py", "--data-version", args.data_version, "--incremental"],
            )
        )
    else:
        results.append({"step": "build_market_state", "result": {"skipped": True}})

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
        results.append({"step": "validate_market_state_coverage", "result": {"skipped": True}})

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
        results.append({"step": "build_mart", "result": {"skipped": True}})

    mode = "incremental" if args.incremental else "full"
    print(json.dumps({
        "data_version": args.data_version,
        "dag_mode": mode,
        "dag_status": "PASS",
        "steps": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
