"""
Standalone builder: l60 alpha_only sequence dataset from existing core mart.
Mimics clean_dataset.py logic but reads state-equivalent columns from mart parquet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ──────────────────────────────────────────────────────
LOOKBACK = 60
DATA_VERSION = "v20260526"
FEATURE_CONFIG = "configs/features/advanced_sequence_clean_v1.yaml"
SPLIT_CONFIG = "configs/data/splits.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data/mart/datasets/clean_purged_wf"

ALPHA_FEATURES = [
    "lag1_net_mf_strength_20d_mean",
    "lag1_net_mf_strength_60d_mean",
    "lag1_close_position",
    "lag1_excess_ret_10d_mean",
    "lag1_excess_ret_1d",
    "lag1_excess_ret_5d_mean",
    "lag1_industry_neutral_ret_1d",
    "lag1_ret_1d",
    "lag1_ret_20d",
    "lag1_ret_5d_mean",
    "lag1_bollinger_z_20d",
    "lag1_ma_ratio_20_60",
    "lag1_macd_hist",
]

LABEL_COL = "label_rel_return"

# strict_tradable_mask equivalents from mart columns
LIQUIDITY_COL = "lag1_amount_20d_mean"
SIZE_COL = "lag1_log_circ_mv"
MIN_AMOUNT = 70_000.0
BOTTOM_Q_LIQ = 0.05
BOTTOM_Q_SIZE = 0.05


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def add_split(df: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    result = df.copy()
    result["split"] = "unused"
    split_name = split_cfg.get("default_split")
    split_entry = split_cfg.get("splits", {}).get(split_name, {})
    # Handle nested folds structure
    active_fold_name = split_entry.get("active_fold")
    if active_fold_name and "folds" in split_entry:
        active = split_entry["folds"].get(active_fold_name, {})
    else:
        active = split_entry
    for sp in ["train", "validation", "test"]:
        if sp not in active:
            continue
        start = active[sp]["start_date"]
        end = active[sp]["end_date"]
        result.loc[result["trade_date"].between(start, end), "split"] = sp
    for pr in active.get("purge_ranges", []):
        mask = result["trade_date"].between(pr["start_date"], pr["end_date"])
        result.loc[mask, "split"] = "purged"
    return result[result["split"].isin(["train", "validation", "test"])].reset_index(drop=True)


def add_industry(df: pd.DataFrame) -> pd.DataFrame:
    if "industry" in df.columns:
        return df
    pool_path = PROJECT_ROOT / "data/lake/core/chinext_pool/chinext_pool_scd2.parquet"
    if not pool_path.exists():
        df["industry"] = "UNKNOWN"
        return df
    pool = pd.read_parquet(pool_path, columns=["ts_code", "industry", "effective_from", "effective_to"])
    pool["effective_from"] = pool["effective_from"].astype(str)
    pool["effective_to"] = pool["effective_to"].fillna("99991231").astype(str)
    result = df.copy()
    result["_idx"] = range(len(result))
    mg = result[["_idx", "trade_date", "ts_code"]].merge(pool, on="ts_code", how="left")
    mg["trade_date"] = mg["trade_date"].astype(str)
    ok = mg[(mg["trade_date"] >= mg["effective_from"]) & (mg["trade_date"] <= mg["effective_to"])]
    ok = ok.sort_values("_idx").drop_duplicates("_idx", keep="last")
    result = result.merge(ok[["_idx", "industry"]], on="_idx", how="left").drop(columns="_idx")
    result["industry"] = result["industry"].astype(str).fillna("UNKNOWN")
    return result


def apply_strict_tradable_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate strict_tradable_mask using available mart columns."""
    d = df.copy()

    # Locked limit (touching up or down)
    d["mask_locked_limit"] = (
        pd.to_numeric(d.get("lag1_limit_touch_up", 0), errors="coerce").fillna(0).astype(bool) |
        pd.to_numeric(d.get("lag1_limit_touch_down", 0), errors="coerce").fillna(0).astype(bool)
    )

    # Low amount
    d["_amt"] = pd.to_numeric(d[LIQUIDITY_COL], errors="coerce")
    lo_amt = d["_amt"] < MIN_AMOUNT
    q_thresh = d.groupby("trade_date", sort=False)["_amt"].transform(
        lambda x: x.quantile(BOTTOM_Q_LIQ) if x.notna().any() else np.nan
    )
    d["mask_low_amount"] = lo_amt | d["_amt"].fillna(np.nan).lt(q_thresh)

    # Microcap (bottom quantile by date of size)
    d["_sz"] = pd.to_numeric(d[SIZE_COL], errors="coerce")
    q_sz = d.groupby("trade_date", sort=False)["_sz"].transform(
        lambda x: x.quantile(BOTTOM_Q_SIZE) if x.notna().any() else np.nan
    )
    d["mask_microcap"] = d["_sz"].fillna(np.nan).lt(q_sz)

    d["strict_tradable"] = ~(d["mask_locked_limit"] | d["mask_low_amount"] | d["mask_microcap"])

    kept = d["strict_tradable"].sum()
    total = len(d)
    print(f"[mask] kept={kept}/{total} ({kept/total:.4f})")
    print(f"[mask]  locked_limit={d['mask_locked_limit'].sum()}, "
          f"low_amount={d['mask_low_amount'].sum()}, microcap={d['mask_microcap'].sum()}")

    return d[d["strict_tradable"]].reset_index(drop=True)


def build_sequences(panel: pd.DataFrame, features: list[str], lookback: int):
    sequences = []
    labels = []
    records = []
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    for _ts_code, group in panel.groupby("ts_code", sort=True):
        vals = group[features].to_numpy(np.float32)
        y = pd.to_numeric(group[LABEL_COL], errors="coerce").to_numpy(np.float32)
        for i in range(lookback - 1, len(group)):
            win = vals[i - lookback + 1:i + 1]
            lab = y[i]
            if np.isnan(lab) or np.isnan(win).any():
                continue
            sequences.append(win)
            labels.append(float(lab))
            row = group.iloc[i]
            records.append({
                "trade_date": str(row["trade_date"]),
                "ts_code": str(row["ts_code"]),
                "split": str(row["split"]),
                LABEL_COL: float(lab),
            })

    if not sequences:
        return (
            np.empty((0, lookback, len(features)), np.float32),
            np.empty((0,), np.float32),
            pd.DataFrame(),
        )

    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    idx = pd.DataFrame(records)
    return X, y, idx


def main():
    print("=== Building l60 alpha_only dataset ===")

    # 1) Load mart
    mart_path = PROJECT_ROOT / f"data/mart/datasets/core/dataset_{DATA_VERSION}.parquet"
    print(f"[1/5] Loading mart: {mart_path}")
    df = pd.read_parquet(mart_path)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"  shape={df.shape}")

    # 2) Add industry + split
    print("[2/5] Adding industry + split ...")
    df = add_industry(df)
    df = add_split(df, load_yaml(PROJECT_ROOT / SPLIT_CONFIG))
    print(f"  after split: {len(df)} rows")
    print(f"  split counts:\n{df['split'].value_counts()}")

    # 3) Filter to required cols + drop NaN label
    required = ["trade_date", "ts_code", "split", LABEL_COL, *ALPHA_FEATURES,
                LIQUIDITY_COL, SIZE_COL,
                "lag1_limit_touch_up", "lag1_limit_touch_down"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns {missing}")
    avail = [c for c in required if c in df.columns]
    panel = df[avail].dropna(subset=[LABEL_COL]).copy()
    for f in ALPHA_FEATURES:
        if f in panel.columns:
            panel[f] = pd.to_numeric(panel[f], errors="coerce")
            panel[f] = panel[f].replace([np.inf, -np.inf], np.nan)

    # 4) Strict tradable mask
    print("[3/5] Applying strict_tradable_mask ...")
    panel = apply_strict_tradable_mask(panel)
    print(f"  after mask: {len(panel)} rows")

    # 5) Build sequences
    print(f"[4/5] Building lookback-{LOOKBACK} sequences ...")
    X, y, sample_idx = build_sequences(panel, ALPHA_FEATURES, LOOKBACK)
    print(f"  X.shape={X.shape}, y.shape={y.shape}")

    # 6) Save
    print("[5/5] Saving ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"dataset_seq_l{LOOKBACK}_adv_clean_v1_alpha_only_chinext_purged_walk_forward"

    np.savez_compressed(
        OUTPUT_DIR / f"{stem}.npz",
        X=X, y=y,
        trade_date=sample_idx["trade_date"].to_numpy() if len(sample_idx) else np.array([]),
        ts_code=sample_idx["ts_code"].to_numpy() if len(sample_idx) else np.array([]),
        split=sample_idx["split"].to_numpy() if len(sample_idx) else np.array([]),
        feature_names=np.array(ALPHA_FEATURES),
        build_mode=np.array(["alpha_only"]),
    )

    manifest = {
        "dataset_type": "clean_sequence",
        "path": str(OUTPUT_DIR / f"{stem}.npz"),
        "sidecar_path": str(OUTPUT_DIR / f"{stem}_sidecar.parquet"),
        "filter_log_path": str(OUTPUT_DIR / f"{stem}_filter_log.csv"),
        "data_version": DATA_VERSION,
        "split_name": "chinext_purged_walk_forward_v1",
        "lookback": LOOKBACK,
        "feature_set": "advanced_sequence_clean_v1",
        "build_mode": "alpha_only",
        "output_stem": stem,
        "samples": int(len(y)),
        "model_features": ALPHA_FEATURES,
        "model_feature_count": len(ALPHA_FEATURES),
        "split_counts": sample_idx["split"].value_counts().to_dict() if len(sample_idx) else {},
    }
    with open(OUTPUT_DIR / f"{stem}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved to {OUTPUT_DIR / stem}.npz")
    print(f"Samples: {len(y)}, Features: {len(ALPHA_FEATURES)}, Lookback: {LOOKBACK}")
    print(f"Splits: {manifest['split_counts']}")


if __name__ == "__main__":
    main()
