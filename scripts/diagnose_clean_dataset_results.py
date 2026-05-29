from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUN_DIRS = {
    "old_full62": Path("outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005"),
    "new_alpha13": Path("outputs/runs/gru_l20_clean_alpha_only_strictmask_leaky0005"),
    "new_alpha18_resid": Path("outputs/runs/gru_l20_clean_alpha_resid_style_strictmask_leaky0005"),
}

SIDECAR = Path(
    "data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026_sidecar.parquet"
)
FILTER_LOG = Path(
    "data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026_filter_log.csv"
)
MART = Path("data/mart/datasets/dataset_v20260526.parquet")
FEATURE_CONFIG = Path("configs/feature_sets/advanced_sequence_clean_v1.yaml")
OUT_DIR = Path("outputs/analysis/clean_dataset_diagnosis")

STYLE_COLUMNS = [
    "lag1_log_circ_mv",
    "lag1_amount_20d_mean",
    "lag1_amount_rank_pct",
    "lag1_turnover_rate_f",
    "lag1_turnover_cost_proxy",
    "lag1_ret_20d_std",
    "lag1_ret_60d_std",
    "lag1_amplitude",
    "lag1_limit_position",
    "lag1_limit_touch_up",
    "lag1_limit_touch_down",
    "lag1_near_limit_up_2pct",
    "lag1_near_limit_down_2pct",
]


def read_alpha_features() -> list[str]:
    lines = FEATURE_CONFIG.read_text(encoding="utf-8").splitlines()
    features: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == "alpha_features:":
            in_block = True
            continue
        if in_block and stripped and not stripped.startswith("- "):
            break
        if in_block and stripped.startswith("- "):
            features.append(stripped[2:].strip())
    return features


def load_predictions() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, run_dir in RUN_DIRS.items():
        frame = pd.read_parquet(run_dir / "predictions.parquet").copy()
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["year"] = frame["trade_date"].str.slice(0, 4)
        frames[name] = frame
    return frames


def safe_ir(series: pd.Series) -> float:
    clean = pd.Series(series, dtype="float64").dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / std) if std and np.isfinite(std) else float("nan")


def daily_ic(frame: pd.DataFrame, score_col: str = "pred_score") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, date), group in frame.groupby(["split", "trade_date"], sort=True):
        if group[score_col].nunique() <= 1 or group["label_rel_return"].nunique() <= 1:
            continue
        rows.append(
            {
                "split": split,
                "trade_date": date,
                "year": str(date)[:4],
                "ic": group[score_col].corr(group["label_rel_return"], method="pearson"),
                "rank_ic": group[score_col].corr(group["label_rel_return"], method="spearman"),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic_by_year(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run, frame in preds.items():
        daily = daily_ic(frame)
        for (split, year), group in daily.groupby(["split", "year"], sort=True):
            rows.append(
                {
                    "run": run,
                    "split": split,
                    "year": year,
                    "days": int(len(group)),
                    "avg_daily_n": float(group["n"].mean()),
                    "ic_mean": float(group["ic"].mean()),
                    "icir": safe_ir(group["ic"]),
                    "rank_ic_mean": float(group["rank_ic"].mean()),
                    "rank_icir": safe_ir(group["rank_ic"]),
                    "rank_ic_positive_rate": float((group["rank_ic"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def common_universe_metrics(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    old = preds["old_full62"][
        ["trade_date", "ts_code", "split", "pred_score", "label_rel_return"]
    ].rename(columns={"pred_score": "old_score"})
    rows: list[dict[str, Any]] = []
    for run in ["new_alpha13", "new_alpha18_resid"]:
        merged = preds[run][["trade_date", "ts_code", "split", "pred_score"]].merge(
            old, on=["trade_date", "ts_code", "split"], how="inner", validate="one_to_one"
        )
        for split, group in merged.groupby("split", sort=True):
            for score_col, label in [("old_score", "old_on_clean_common"), ("pred_score", run)]:
                daily = daily_ic(
                    group.rename(columns={score_col: "score_for_eval"}), score_col="score_for_eval"
                )
                rows.append(
                    {
                        "comparison": f"{run}_common_with_old",
                        "run": label,
                        "split": split,
                        "rows": int(len(group)),
                        "dates": int(group["trade_date"].nunique()),
                        "score_corr_with_other": float(
                            group["pred_score"].corr(group["old_score"], method="spearman")
                        ),
                        "ic_mean": float(daily["ic"].mean()),
                        "rank_ic_mean": float(daily["rank_ic"].mean()),
                        "rank_icir": safe_ir(daily["rank_ic"]),
                    }
                )
    return pd.DataFrame(rows)


def topk_overlap(preds: dict[str, pd.DataFrame], k_values: tuple[int, ...] = (10, 20, 30)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [("old_full62", "new_alpha13"), ("old_full62", "new_alpha18_resid"), ("new_alpha13", "new_alpha18_resid")]
    for left_name, right_name in pairs:
        left = preds[left_name][["trade_date", "ts_code", "split", "pred_score"]].rename(
            columns={"pred_score": "left_score"}
        )
        right = preds[right_name][["trade_date", "ts_code", "split", "pred_score"]].rename(
            columns={"pred_score": "right_score"}
        )
        merged = left.merge(right, on=["trade_date", "ts_code", "split"], how="inner")
        for split, split_frame in merged.groupby("split", sort=True):
            for k in k_values:
                overlaps: list[float] = []
                corrs: list[float] = []
                for _, group in split_frame.groupby("trade_date", sort=True):
                    if len(group) < k:
                        continue
                    left_top = set(group.nlargest(k, "left_score")["ts_code"])
                    right_top = set(group.nlargest(k, "right_score")["ts_code"])
                    overlaps.append(len(left_top & right_top) / k)
                    corrs.append(group["left_score"].corr(group["right_score"], method="spearman"))
                rows.append(
                    {
                        "pair": f"{left_name}__{right_name}",
                        "split": split,
                        "k": k,
                        "days": len(overlaps),
                        "top_overlap_mean": float(np.mean(overlaps)),
                        "top_overlap_p25": float(np.quantile(overlaps, 0.25)),
                        "top_overlap_p75": float(np.quantile(overlaps, 0.75)),
                        "daily_score_spearman_mean": float(np.nanmean(corrs)),
                    }
                )
    return pd.DataFrame(rows)


def style_exposure(preds: dict[str, pd.DataFrame], sidecar: pd.DataFrame, k_values: tuple[int, ...] = (10, 30)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    controls = ["trade_date", "ts_code", "split", *STYLE_COLUMNS]
    side = sidecar[controls].copy()
    for run, pred in preds.items():
        merged = pred[["trade_date", "ts_code", "split", "pred_score"]].merge(
            side, on=["trade_date", "ts_code", "split"], how="inner", validate="one_to_one"
        )
        for split, split_frame in merged.groupby("split", sort=True):
            for k in k_values:
                daily_rows: list[dict[str, float]] = []
                for _, group in split_frame.groupby("trade_date", sort=True):
                    if len(group) < k:
                        continue
                    ordered = group.sort_values("pred_score", ascending=False)
                    top = ordered.head(k)
                    bottom = ordered.tail(k)
                    daily_rows.append(
                        {
                            f"{col}_top": float(top[col].mean())
                            for col in STYLE_COLUMNS
                            if col in top
                        }
                        | {
                            f"{col}_bottom": float(bottom[col].mean())
                            for col in STYLE_COLUMNS
                            if col in bottom
                        }
                        | {
                            f"{col}_spread": float(top[col].mean() - bottom[col].mean())
                            for col in STYLE_COLUMNS
                            if col in top
                        }
                    )
                daily = pd.DataFrame(daily_rows)
                for col in STYLE_COLUMNS:
                    rows.append(
                        {
                            "run": run,
                            "split": split,
                            "k": k,
                            "style_col": col,
                            "top_mean": float(daily[f"{col}_top"].mean()),
                            "bottom_mean": float(daily[f"{col}_bottom"].mean()),
                            "top_minus_bottom": float(daily[f"{col}_spread"].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def alpha_feature_ic(sidecar: pd.DataFrame) -> pd.DataFrame:
    features = read_alpha_features()
    keys = sidecar[["trade_date", "ts_code", "split"]].copy()
    mart_cols = ["trade_date", "ts_code", "label_rel_return", *features]
    mart = pd.read_parquet(MART, columns=mart_cols)
    mart["trade_date"] = mart["trade_date"].astype(str)
    mart["ts_code"] = mart["ts_code"].astype(str)
    merged = keys.merge(mart, on=["trade_date", "ts_code"], how="inner", validate="one_to_one")

    rows: list[dict[str, Any]] = []
    for split, split_frame in merged.groupby("split", sort=True):
        for feature in features:
            daily_rows: list[dict[str, float]] = []
            for date, group in split_frame.groupby("trade_date", sort=True):
                if group[feature].nunique() <= 1 or group["label_rel_return"].nunique() <= 1:
                    continue
                daily_rows.append(
                    {
                        "trade_date": date,
                        "ic": group[feature].corr(group["label_rel_return"], method="pearson"),
                        "rank_ic": group[feature].corr(group["label_rel_return"], method="spearman"),
                    }
                )
            daily = pd.DataFrame(daily_rows)
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "days": int(len(daily)),
                    "ic_mean": float(daily["ic"].mean()),
                    "rank_ic_mean": float(daily["rank_ic"].mean()),
                    "rank_icir": safe_ir(daily["rank_ic"]),
                    "rank_ic_positive_rate": float((daily["rank_ic"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def filter_diagnostics() -> pd.DataFrame:
    frame = pd.read_csv(FILTER_LOG)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["year"] = frame["trade_date"].str.slice(0, 4)
    mask_cols = [col for col in frame.columns if col.startswith("mask_")]
    rows: list[dict[str, Any]] = []
    for (split, year), group in frame.groupby(["split", "year"], sort=True):
        row: dict[str, Any] = {
            "split": split,
            "year": year,
            "rows": int(len(group)),
            "kept": int(group["strict_tradable"].sum()),
            "drop_rate": float(1.0 - group["strict_tradable"].mean()),
        }
        for col in mask_cols:
            row[f"{col}_rate"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_predictions()
    sidecar = pd.read_parquet(SIDECAR)
    sidecar["trade_date"] = sidecar["trade_date"].astype(str)
    sidecar["ts_code"] = sidecar["ts_code"].astype(str)

    outputs = {
        "yearly_ic": summarize_ic_by_year(preds),
        "common_universe_ic": common_universe_metrics(preds),
        "topk_overlap": topk_overlap(preds),
        "style_exposure_top_bottom": style_exposure(preds, sidecar),
        "alpha_feature_ic_strict_universe": alpha_feature_ic(sidecar),
        "filter_by_split_year": filter_diagnostics(),
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"\n{name}")
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
