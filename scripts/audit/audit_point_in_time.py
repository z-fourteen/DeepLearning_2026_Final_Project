from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.ingest.agent import load_yaml  # noqa: E402


SEVERITY_ORDER = {"BLOCKER": 0, "WARNING": 1, "INFO": 2, "PASS": 3}


@dataclass
class Finding:
    check_id: str
    severity: str
    component: str
    status: str
    evidence: str
    recommendation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run point-in-time data audit.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-version", default="v20260526")
    parser.add_argument(
        "--dataset",
        default="data/mart/datasets/dataset_v20260526.parquet",
        help="Main mart dataset parquet.",
    )
    parser.add_argument(
        "--labels",
        default="data/mart/labels/labels_v20260526.parquet",
        help="Label parquet used by backtests.",
    )
    parser.add_argument(
        "--filter-log",
        default=(
            "data/mart/datasets/"
            "dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026_filter_log.csv"
        ),
        help="Strict tradable mask filter log.",
    )
    parser.add_argument(
        "--features-config",
        default="configs/features.yaml",
        help="Feature configuration yaml.",
    )
    parser.add_argument(
        "--clean-config",
        default="configs/features/advanced_sequence_clean_v1.yaml",
        help="Clean feature role configuration yaml.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/audit/point_in_time",
        help="Audit output directory.",
    )
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def add_finding(
    findings: list[Finding],
    check_id: str,
    severity: str,
    component: str,
    status: str,
    evidence: str,
    recommendation: str,
) -> None:
    findings.append(
        Finding(
            check_id=check_id,
            severity=severity,
            component=component,
            status=status,
            evidence=evidence,
            recommendation=recommendation,
        )
    )


def read_columns(path: Path) -> list[str]:
    return list(pd.read_parquet(path, engine="pyarrow").columns)


def safe_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    available = set(read_columns(path))
    selected = [column for column in columns if column in available]
    return pd.read_parquet(path, columns=selected)


def scan_negative_shifts(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if re.search(r"\.shift\(\s*-\s*\d+", line):
            window_start = max(0, line_no - 4)
            context = "\n".join(lines[window_start:line_no])
            allowed = "LEAKAGE_ALLOWED_LABEL_SHIFT" in context
            rows.append(
                {
                    "path": str(path),
                    "line": line_no,
                    "allowed_label_shift": allowed,
                    "code": line.strip(),
                }
            )
    return rows


def audit_configs(root: Path, args: argparse.Namespace, findings: list[Finding]) -> None:
    features_config_path = root / args.features_config
    clean_config_path = root / args.clean_config
    labels_config_path = root / "configs" / "labels.yaml"

    features_config = load_yaml(features_config_path)
    clean_config = load_yaml(clean_config_path)
    labels_config = load_yaml(labels_config_path)

    future_shift_allowed = features_config.get("future_shift_allowed")
    availability = features_config.get("feature_availability")
    strict_lag = features_config.get("validation", {}).get("dataset_requires_lagged_features_only")
    if future_shift_allowed is False and strict_lag is True:
        add_finding(
            findings,
            "CFG001",
            "PASS",
            "features_config",
            "lag_policy_declared",
            (
                f"{rel(features_config_path, root)} declares future_shift_allowed=false, "
                f"dataset_requires_lagged_features_only=true, feature_availability={availability!r}."
            ),
            "Keep this policy and verify it against actual dataset columns.",
        )
    else:
        add_finding(
            findings,
            "CFG001",
            "BLOCKER",
            "features_config",
            "lag_policy_missing_or_weak",
            (
                f"future_shift_allowed={future_shift_allowed!r}, "
                f"dataset_requires_lagged_features_only={strict_lag!r}."
            ),
            "Set future_shift_allowed=false and require lagged feature columns only.",
        )

    horizon = labels_config.get("default_horizon")
    label_mode = labels_config.get("label_mode")
    add_finding(
        findings,
        "CFG002",
        "INFO",
        "labels_config",
        "label_policy_declared",
        f"{rel(labels_config_path, root)} declares default_horizon={horizon}, label_mode={label_mode}.",
        "Document exact signal timestamp and execution timestamp for this horizon.",
    )

    raw_controls = clean_config.get("model_input_policy", {}).get("do_not_feed_raw_controls")
    strict_mask = clean_config.get("strict_tradable_mask", {})
    if raw_controls is True and strict_mask.get("enabled") is True:
        add_finding(
            findings,
            "CFG003",
            "PASS",
            "clean_feature_config",
            "controls_separated",
            (
                f"{rel(clean_config_path, root)} separates raw controls from model tensor "
                "and enables strict tradable mask."
            ),
            "Keep controls out of model input unless a specific ablation opts in.",
        )
    else:
        add_finding(
            findings,
            "CFG003",
            "WARNING",
            "clean_feature_config",
            "controls_policy_incomplete",
            f"do_not_feed_raw_controls={raw_controls!r}, strict_mask_enabled={strict_mask.get('enabled')!r}.",
            "Keep risk and tradability controls separated from alpha tensor inputs.",
        )


def audit_static_code(root: Path, findings: list[Finding]) -> pd.DataFrame:
    code_paths = [
        root / "pipelines" / "mart" / "agent.py",
        root / "pipelines" / "mart" / "dataset.py",
        root / "pipelines" / "mart" / "clean_dataset.py",
        root / "scripts" / "backtest_topk.py",
        root / "scripts" / "backtest_topk_turnover_control.py",
        root / "scripts" / "build_strictmask_prediction_overlay.py",
    ]
    rows: list[dict[str, Any]] = []
    for path in code_paths:
        if path.exists():
            rows.extend(scan_negative_shifts(path))

    unapproved = [row for row in rows if not row["allowed_label_shift"]]
    if unapproved:
        add_finding(
            findings,
            "CODE001",
            "BLOCKER",
            "source_code",
            "unapproved_negative_shift_found",
            json.dumps(unapproved, ensure_ascii=False),
            "Inspect every negative shift and move any future-looking operation into label-only code.",
        )
    else:
        add_finding(
            findings,
            "CODE001",
            "PASS",
            "source_code",
            "negative_shift_scan_clean",
            "No unapproved negative shift found in audited mart/backtest scripts.",
            "Continue scanning new feature scripts before accepting results.",
        )

    agent_path = root / "pipelines" / "mart" / "agent.py"
    if agent_path.exists():
        text = agent_path.read_text(encoding="utf-8")
        label_close_to_close = (
            'shift(-horizon) / df["close"] - 1' in text
            and 'shift(-horizon) / market["close"] - 1' in text
        )
        if label_close_to_close:
            add_finding(
                findings,
                "CODE002",
                "INFO",
                "label_construction",
                "close_to_close_forward_label",
                (
                    "Labels are constructed as future close-to-close returns: "
                    "stock close.shift(-horizon)/close - 1 and benchmark close.shift(-horizon)/close - 1."
                ),
                (
                    "This is acceptable as a supervised target when production-like evaluation uses "
                    "canonical execution labels and T+1 fill simulation."
                ),
            )

        lag_shift_found = re.search(r"result\[lagged_name\]\s*=\s*grouped\[column\]\.shift\(1\)", text)
        if lag_shift_found:
            add_finding(
                findings,
                "CODE003",
                "PASS",
                "feature_construction",
                "lagged_feature_shift_found",
                "add_lagged_features uses grouped[column].shift(1) for lag1_ features.",
                "Keep raw same-day features out of model-ready datasets.",
            )
        else:
            add_finding(
                findings,
                "CODE003",
                "WARNING",
                "feature_construction",
                "lagged_feature_shift_not_verified",
                "Could not verify grouped shift(1) pattern in add_lagged_features.",
                "Manually inspect feature construction before accepting point-in-time status.",
            )

    return pd.DataFrame(rows)


def audit_dataset(root: Path, args: argparse.Namespace, findings: list[Finding]) -> pd.DataFrame:
    dataset_path = root / args.dataset
    if not dataset_path.exists():
        add_finding(
            findings,
            "DATA001",
            "BLOCKER",
            "mart_dataset",
            "dataset_missing",
            f"Missing dataset: {rel(dataset_path, root)}.",
            "Build mart dataset before running point-in-time audit.",
        )
        return pd.DataFrame()

    columns = read_columns(dataset_path)
    feature_columns = [column for column in columns if column not in {"trade_date", "ts_code", "future_return", "benchmark_future_return", "label_rel_return"}]
    non_lag_features = [column for column in feature_columns if not column.startswith("lag1_")]
    if non_lag_features:
        add_finding(
            findings,
            "DATA001",
            "BLOCKER",
            "mart_dataset",
            "non_lag_feature_columns_present",
            f"Non-lag model dataset columns: {non_lag_features[:50]}",
            "Remove or rename non-lag feature columns from model-ready datasets.",
        )
    else:
        add_finding(
            findings,
            "DATA001",
            "PASS",
            "mart_dataset",
            "all_feature_columns_lagged",
            f"All {len(feature_columns)} feature columns use lag1_ prefix.",
            "Still verify that lag1_ columns are created with shift(1), not just named lag1_.",
        )

    key_cols = ["trade_date", "ts_code"]
    sample_cols = [column for column in [*key_cols, "label_rel_return"] if column in columns]
    frame = safe_read_parquet(dataset_path, columns=sample_cols)
    for column in key_cols:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    duplicate_count = int(frame.duplicated(key_cols).sum()) if set(key_cols).issubset(frame.columns) else -1
    date_min = frame["trade_date"].min() if "trade_date" in frame.columns and len(frame) else None
    date_max = frame["trade_date"].max() if "trade_date" in frame.columns and len(frame) else None
    if duplicate_count:
        add_finding(
            findings,
            "DATA002",
            "BLOCKER",
            "mart_dataset",
            "duplicate_trade_date_ts_code",
            f"Found {duplicate_count} duplicated trade_date+ts_code keys.",
            "Deduplicate mart dataset before training or backtesting.",
        )
    else:
        add_finding(
            findings,
            "DATA002",
            "PASS",
            "mart_dataset",
            "unique_trade_date_ts_code",
            f"Rows={len(frame)}, date_range={date_min}..{date_max}, duplicate_keys=0.",
            "Keep key uniqueness checks in CI for every dataset rebuild.",
        )

    suspicious_patterns = [
        "amount",
        "turnover",
        "vol",
        "circ_mv",
        "total_mv",
        "limit",
        "gap_open",
        "intraday",
        "weekday",
        "month",
        "benchmark",
    ]
    suspicious = [
        column
        for column in feature_columns
        if any(pattern in column for pattern in suspicious_patterns)
    ]
    severity = "WARNING" if suspicious else "PASS"
    add_finding(
        findings,
        "DATA003",
        severity,
        "mart_dataset",
        "style_or_microstructure_features_detected" if suspicious else "no_suspicious_feature_names",
        f"Suspicious feature count={len(suspicious)}. Examples={suspicious[:80]}",
        (
            "These fields are not necessarily leaked, but they require style, liquidity, "
            "and execution attribution before production use."
            if suspicious
            else "No suspicious style or microstructure feature names detected."
        ),
    )
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "starts_with_lag1": [column.startswith("lag1_") for column in feature_columns],
            "suspicious_style_or_microstructure": [column in suspicious for column in feature_columns],
        }
    )


def audit_labels(root: Path, args: argparse.Namespace, findings: list[Finding]) -> None:
    labels_path = root / args.labels
    if not labels_path.exists():
        add_finding(
            findings,
            "LBL001",
            "BLOCKER",
            "labels",
            "labels_missing",
            f"Missing labels: {rel(labels_path, root)}.",
            "Build labels before running point-in-time audit.",
        )
        return

    available_columns = set(read_columns(labels_path))
    base_columns = ["trade_date", "ts_code", "future_return", "benchmark_future_return", "label_rel_return"]
    labels = safe_read_parquet(labels_path, columns=base_columns)
    labels["trade_date"] = labels["trade_date"].astype(str)
    labels["ts_code"] = labels["ts_code"].astype(str)
    duplicate_count = int(labels.duplicated(["trade_date", "ts_code"]).sum())
    null_counts = labels[["future_return", "benchmark_future_return", "label_rel_return"]].isna().sum().to_dict()
    date_min = labels["trade_date"].min() if len(labels) else None
    date_max = labels["trade_date"].max() if len(labels) else None
    if duplicate_count:
        add_finding(
            findings,
            "LBL001",
            "BLOCKER",
            "labels",
            "duplicate_label_keys",
            f"Found {duplicate_count} duplicated trade_date+ts_code label keys.",
            "Deduplicate label table before training or backtesting.",
        )
    else:
        add_finding(
            findings,
            "LBL001",
            "PASS",
            "labels",
            "unique_label_keys",
            f"Rows={len(labels)}, date_range={date_min}..{date_max}, null_counts={null_counts}.",
            "Keep label key uniqueness checks for every rebuild.",
        )

    diff = (
        pd.to_numeric(labels["future_return"], errors="coerce")
        - pd.to_numeric(labels["benchmark_future_return"], errors="coerce")
        - pd.to_numeric(labels["label_rel_return"], errors="coerce")
    )
    max_abs_diff = float(diff.abs().dropna().max()) if diff.notna().any() else float("nan")
    if pd.notna(max_abs_diff) and max_abs_diff < 1e-8:
        add_finding(
            findings,
            "LBL002",
            "PASS",
            "labels",
            "relative_label_identity_pass",
            f"max_abs(future_return - benchmark_future_return - label_rel_return)={max_abs_diff:.3g}.",
            "Identity check passes; execution timing still needs separate audit.",
        )
    else:
        add_finding(
            findings,
            "LBL002",
            "BLOCKER",
            "labels",
            "relative_label_identity_fail",
            f"max_abs_diff={max_abs_diff}.",
            "Fix label_rel_return construction before accepting any IC or backtest result.",
        )

    required_execution_columns = {
        "next_open_return_5d",
        "next_vwap_return_5d",
        "buy_executable",
        "sell_executable",
        "next_open",
        "next_vwap",
        "next_amount",
        "next_vol",
        "next_is_limit_up",
        "next_is_limit_down",
        "execution_excess_open_to_close5",
    }
    missing_execution = sorted(required_execution_columns - available_columns)
    if missing_execution:
        add_finding(
            findings,
            "LBL003",
            "WARNING",
            "labels",
            "execution_label_missing",
            (
                "Current label table has close-to-close research labels but is missing canonical "
                f"execution fields: {missing_execution}."
            ),
            "Use the canonical label table with T+1 execution returns and executable flags.",
        )
        return

    execution_sample = safe_read_parquet(
        labels_path,
        columns=[
            "trade_date",
            "ts_code",
            "next_open_return_5d",
            "next_vwap_return_5d",
            "buy_executable",
            "sell_executable",
            "next_open",
            "next_vwap",
            "next_amount",
            "next_vol",
            "next_is_limit_up",
            "next_is_limit_down",
            "execution_excess_open_to_close5",
        ],
    )
    for column in ["buy_executable", "sell_executable", "next_is_limit_up", "next_is_limit_down"]:
        execution_sample[column] = execution_sample[column].fillna(False).astype(bool)
    coverage = {
        "next_open_return_5d": float(execution_sample["next_open_return_5d"].notna().mean()),
        "next_vwap_return_5d": float(execution_sample["next_vwap_return_5d"].notna().mean()),
        "execution_excess_open_to_close5": float(
            execution_sample["execution_excess_open_to_close5"].notna().mean()
        ),
        "buy_executable_rate": float(execution_sample["buy_executable"].mean()),
        "sell_executable_rate": float(execution_sample["sell_executable"].mean()),
        "limit_up_rate": float(execution_sample["next_is_limit_up"].mean()),
        "limit_down_rate": float(execution_sample["next_is_limit_down"].mean()),
    }
    add_finding(
        findings,
        "LBL003",
        "PASS",
        "labels",
        "canonical_execution_labels_present",
        f"Canonical execution fields are present. Coverage={coverage}.",
        "Use this upgraded label table for PIT audit, T+1 fill simulation, and production-like evaluation.",
    )


def audit_filter_log(root: Path, args: argparse.Namespace, findings: list[Finding]) -> None:
    filter_path = root / args.filter_log
    if not filter_path.exists():
        add_finding(
            findings,
            "MSK001",
            "WARNING",
            "strict_mask",
            "filter_log_missing",
            f"Missing filter log: {rel(filter_path, root)}.",
            "Build clean dataset filter log before auditing strict tradable mask.",
        )
        return

    filter_log = pd.read_csv(filter_path)
    required = {"trade_date", "ts_code", "split", "strict_tradable"}
    missing = sorted(required - set(filter_log.columns))
    if missing:
        add_finding(
            findings,
            "MSK001",
            "BLOCKER",
            "strict_mask",
            "filter_log_missing_columns",
            f"Missing columns: {missing}.",
            "Regenerate filter log with trade_date, ts_code, split, and strict_tradable.",
        )
        return

    filter_log["trade_date"] = filter_log["trade_date"].astype(str)
    filter_log["ts_code"] = filter_log["ts_code"].astype(str)
    filter_log["split"] = filter_log["split"].astype(str)
    duplicate_count = int(filter_log.duplicated(["trade_date", "ts_code", "split"]).sum())
    keep_rate = float(filter_log["strict_tradable"].astype(bool).mean()) if len(filter_log) else float("nan")
    mask_cols = [column for column in filter_log.columns if column.startswith("mask_")]
    mask_rates = {
        column: float(filter_log[column].astype(bool).mean())
        for column in mask_cols
    }
    if duplicate_count:
        add_finding(
            findings,
            "MSK001",
            "BLOCKER",
            "strict_mask",
            "duplicate_filter_keys",
            f"Found {duplicate_count} duplicated trade_date+ts_code+split keys.",
            "Deduplicate filter log before applying overlay.",
        )
    else:
        add_finding(
            findings,
            "MSK001",
            "PASS",
            "strict_mask",
            "filter_keys_unique",
            f"Rows={len(filter_log)}, keep_rate={keep_rate:.4f}, mask_rates={mask_rates}.",
            "Keep filter-key uniqueness checks for every rebuild.",
        )

    if "mask_locked_limit" in filter_log.columns:
        add_finding(
            findings,
            "MSK002",
            "INFO",
            "strict_mask",
            "same_day_limit_filter_documented",
            (
                "filter_log contains mask_locked_limit. In current clean_dataset implementation, "
                "this is derived from same trade_date state. Canonical labels now provide next-session "
                "buy/sell executable flags for production-like backtests."
            ),
            (
                "Use strict mask as a conservative sample filter only; use canonical T+1 fields for execution."
            ),
        )


def write_outputs(
    out_dir: Path,
    findings: list[Finding],
    feature_audit: pd.DataFrame,
    shift_audit: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    findings_df = pd.DataFrame([asdict(item) for item in findings])
    if not findings_df.empty:
        findings_df["severity_rank"] = findings_df["severity"].map(SEVERITY_ORDER).fillna(99)
        findings_df = findings_df.sort_values(["severity_rank", "check_id"]).drop(columns=["severity_rank"])
    findings_df.to_csv(out_dir / "field_audit.csv", index=False, encoding="utf-8-sig")
    feature_audit.to_csv(out_dir / "feature_column_audit.csv", index=False, encoding="utf-8-sig")
    shift_audit.to_csv(out_dir / "negative_shift_audit.csv", index=False, encoding="utf-8-sig")

    suspect_features = []
    if not feature_audit.empty and "suspicious_style_or_microstructure" in feature_audit.columns:
        suspect_features = feature_audit.loc[
            feature_audit["suspicious_style_or_microstructure"].astype(bool),
            "feature",
        ].astype(str).tolist()
    (out_dir / "suspect_features.txt").write_text(
        "\n".join(suspect_features) + ("\n" if suspect_features else ""),
        encoding="utf-8",
    )

    blocker_count = int((findings_df["severity"] == "BLOCKER").sum()) if not findings_df.empty else 0
    warning_count = int((findings_df["severity"] == "WARNING").sum()) if not findings_df.empty else 0
    pass_count = int((findings_df["severity"] == "PASS").sum()) if not findings_df.empty else 0
    verdict = "FAIL" if blocker_count else ("PASS_WITH_WARNINGS" if warning_count else "PASS")

    lines = [
        "# Point-In-Time Audit Findings",
        "",
        f"Verdict: `{verdict}`",
        "",
        f"- Blockers: {blocker_count}",
        f"- Warnings: {warning_count}",
        f"- Pass checks: {pass_count}",
        "",
        "## Findings",
        "",
    ]
    for row in findings_df.to_dict(orient="records"):
        lines.extend(
            [
                f"### {row['check_id']} - {row['severity']} - {row['status']}",
                "",
                f"- Component: `{row['component']}`",
                f"- Evidence: {row['evidence']}",
                f"- Recommendation: {row['recommendation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Generated Files",
            "",
            "- `field_audit.csv`",
            "- `feature_column_audit.csv`",
            "- `negative_shift_audit.csv`",
            "- `suspect_features.txt`",
            "",
        ]
    )
    (out_dir / "leakage_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    findings: list[Finding] = []

    audit_configs(root, args, findings)
    shift_audit = audit_static_code(root, findings)
    feature_audit = audit_dataset(root, args, findings)
    audit_labels(root, args, findings)
    audit_filter_log(root, args, findings)

    out_dir = root / args.out_dir
    write_outputs(out_dir, findings, feature_audit, shift_audit)

    summary = {
        "out_dir": str(out_dir),
        "blockers": sum(item.severity == "BLOCKER" for item in findings),
        "warnings": sum(item.severity == "WARNING" for item in findings),
        "passes": sum(item.severity == "PASS" for item in findings),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
