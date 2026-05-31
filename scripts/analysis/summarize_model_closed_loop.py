from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a model's train -> execution -> optimizer loop.")
    parser.add_argument(
        "--run-name",
        default="feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean",
    )
    parser.add_argument(
        "--optimizer-suffix",
        default="core80",
        help="Suffix used by outputs/backtest/optimizer/<run_name>_<suffix>.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/feature_style_interaction_gru_l20_topk10_wide30_clean_closed_loop",
    )
    parser.add_argument(
        "--compare-run",
        action="append",
        default=[
            "feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean",
            "regime_gated_gru_l20_clean_alpha_resid_style_topk10_wide30",
        ],
        help="Optional run name to include in T+1 comparison if its metrics exist.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_t1_metrics(path: Path, run_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    metrics = load_json(path)
    rows: list[dict[str, Any]] = []
    for split, split_summary in metrics.get("summary", {}).items():
        for setting, values in split_summary.items():
            rows.append(
                {
                    "run_name": run_name,
                    "split": split,
                    "setting": setting,
                    "net_ann": values["net"]["annualized_return"],
                    "net_ir": values["net"]["ir"],
                    "net_mdd": values["net"]["max_drawdown"],
                    "win_rate": values["net"]["win_rate"],
                    "excess_benchmark_ann": values["excess_vs_benchmark"]["annualized_return"],
                    "excess_exec_universe_ann": values["excess_vs_executable_universe"][
                        "annualized_return"
                    ],
                    "avg_desired_turnover": values["average_desired_turnover"],
                    "avg_filled_turnover": values["average_filled_turnover"],
                    "avg_transaction_cost": values["average_transaction_cost"],
                    "avg_position_count": values["average_position_count"],
                }
            )
    return pd.DataFrame(rows)


def summarize_training(metrics_path: Path) -> pd.DataFrame:
    metrics = load_json(metrics_path)
    best_epoch = int(metrics["best_epoch"])
    best_rows = [row for row in metrics["history"] if int(row["epoch"]) == best_epoch]
    best = best_rows[0] if best_rows else {}
    summary = metrics.get("summary", {})
    return pd.DataFrame(
        [
            {
                "run_name": summary.get("run_name"),
                "best_epoch": best_epoch,
                "best_rank_ic_mean": metrics.get("best_metric"),
                "best_ic_mean": best.get("ic_mean"),
                "best_rank_icir": best.get("rank_icir"),
                "best_val_loss": best.get("val_loss"),
                "best_pred_std": best.get("pred_std"),
                "stop_reason": metrics.get("stop_reason"),
                "train_samples": summary.get("train_samples"),
                "validation_samples": summary.get("validation_samples"),
                "test_samples": summary.get("test_samples"),
                "prediction_rows": metrics.get("prediction_rows"),
                "lookback": summary.get("lookback"),
                "num_features": summary.get("num_features"),
                "model": summary.get("model"),
            }
        ]
    )


def prediction_diagnostics(predictions_path: Path) -> pd.DataFrame:
    preds = pd.read_parquet(predictions_path)
    rows: list[dict[str, Any]] = []
    for split, group in preds.groupby("split", sort=True):
        score = pd.to_numeric(group["pred_score"], errors="coerce")
        label = pd.to_numeric(group.get("label_rel_return"), errors="coerce")
        rows.append(
            {
                "split": split,
                "rows": int(len(group)),
                "dates": int(group["trade_date"].nunique()),
                "score_mean": float(score.mean()),
                "score_std": float(score.std(ddof=1)),
                "score_min": float(score.min()),
                "score_max": float(score.max()),
                "label_mean": float(label.mean()),
                "label_std": float(label.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def top_by_split(frame: pd.DataFrame, metric: str, n: int = 8) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for _, group in frame.groupby("split", sort=True):
        rows.append(group.sort_values(metric, ascending=False).head(n))
    return pd.concat(rows, ignore_index=True)


def markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No data._"
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *body])


def write_markdown(
    out_dir: Path,
    run_name: str,
    training: pd.DataFrame,
    pred_diag: pd.DataFrame,
    t1_top: pd.DataFrame,
    optimizer_top: pd.DataFrame,
    t1_compare: pd.DataFrame,
) -> None:
    best = training.iloc[0].to_dict()
    validation_t1 = t1_top[t1_top["split"].eq("validation")]
    test_t1 = t1_top[t1_top["split"].eq("test")]
    validation_opt = optimizer_top[optimizer_top["split"].eq("validation")]
    test_opt = optimizer_top[optimizer_top["split"].eq("test")]

    lines = [
        f"# {run_name} Closed-Loop Analysis",
        "",
        "## Executive Readout",
        "",
        (
            f"- Training selected epoch {int(best['best_epoch'])} with validation rank IC "
            f"{best['best_rank_ic_mean']:.6f} and rank ICIR {best['best_rank_icir']:.6f}."
        ),
        "- The executable loop is complete: predictions, T+1 fill simulation, soft optimizer grid, and this summary are all materialized.",
        "- The key risk is validation/test divergence: validation execution metrics are negative, while test absolute returns are strong. Treat this as research evidence, not a promotion signal.",
        "- Test-period gains remain mostly beta/market-regime assisted: T+1 test absolute returns are positive, but excess versus benchmark is still negative for the best absolute-return rows.",
        "",
        "## Training",
        "",
        markdown_table(training),
        "",
        "## Prediction Diagnostics",
        "",
        markdown_table(pred_diag),
        "",
        "## T+1 Fill Simulation: Best Rows By Split",
        "",
        markdown_table(pd.concat([validation_t1, test_t1], ignore_index=True)),
        "",
        "## Soft Optimizer Core80: Best Rows By Split",
        "",
        markdown_table(pd.concat([validation_opt, test_opt], ignore_index=True)),
        "",
        "## T+1 Comparator Snapshot",
        "",
        markdown_table(t1_compare),
        "",
        "## Decision",
        "",
        "- Do not promote as production mainline on the current evidence.",
        "- Keep as a candidate architecture because the TopK wide-band loss improves test T+1 absolute return and Top10 robustness, but require a validation-positive rerun or rolling-window confirmation.",
        "- Next useful experiment: keep architecture fixed and test stronger validation alignment, for example lower model capacity or add explicit excess-return/benchmark-relative selection in the loss/evaluation gate.",
    ]
    (out_dir / "closed_loop_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_name = args.run_name
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = resolve(Path("outputs/runs") / run_name)
    t1_metrics = resolve(Path("outputs/backtest/t1_fill_sim") / run_name / "t1_fill_metrics.json")
    optimizer_dir = resolve(Path("outputs/backtest/optimizer") / f"{run_name}_{args.optimizer_suffix}")
    optimizer_summary_path = optimizer_dir / "soft_optimizer_grid_summary.csv"

    training = summarize_training(run_dir / "metrics.json")
    pred_diag = prediction_diagnostics(run_dir / "predictions.parquet")
    t1_summary = flatten_t1_metrics(t1_metrics, run_name)
    optimizer_summary = pd.read_csv(optimizer_summary_path) if optimizer_summary_path.exists() else pd.DataFrame()
    if not optimizer_summary.empty:
        optimizer_summary.insert(0, "run_name", run_name)

    t1_top = top_by_split(t1_summary, "net_ann")
    optimizer_top = top_by_split(optimizer_summary, "net_ann")

    comparison_frames = [t1_summary]
    for compare_run in args.compare_run or []:
        compare_path = resolve(Path("outputs/backtest/t1_fill_sim") / compare_run / "t1_fill_metrics.json")
        comparison_frames.append(flatten_t1_metrics(compare_path, compare_run))
    t1_compare_all = pd.concat([frame for frame in comparison_frames if not frame.empty], ignore_index=True)
    t1_compare = top_by_split(t1_compare_all, "net_ann", n=6)

    training.to_csv(out_dir / "training_summary.csv", index=False)
    pred_diag.to_csv(out_dir / "prediction_diagnostics.csv", index=False)
    t1_summary.to_csv(out_dir / "t1_fill_summary.csv", index=False)
    t1_top.to_csv(out_dir / "t1_fill_top_rows.csv", index=False)
    optimizer_summary.to_csv(out_dir / "soft_optimizer_grid_summary.csv", index=False)
    optimizer_top.to_csv(out_dir / "soft_optimizer_top_rows.csv", index=False)
    t1_compare.to_csv(out_dir / "t1_comparator_top_rows.csv", index=False)

    write_markdown(out_dir, run_name, training, pred_diag, t1_top, optimizer_top, t1_compare)
    manifest = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "t1_metrics": str(t1_metrics),
        "optimizer_summary": str(optimizer_summary_path),
        "output_dir": str(out_dir),
        "rows": {
            "training": int(len(training)),
            "prediction_diagnostics": int(len(pred_diag)),
            "t1_summary": int(len(t1_summary)),
            "optimizer_summary": int(len(optimizer_summary)),
            "t1_compare": int(len(t1_compare)),
        },
        "method": "closed_loop_model_summary",
    }
    (out_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
