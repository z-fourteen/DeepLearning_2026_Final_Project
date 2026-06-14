from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill StdTF live predictions from an existing live_sequence anchor "
            "and raw daily CSVs. This avoids depending on parquet readers."
        )
    )
    parser.add_argument("--yxr-root", default="../Final-YXR")
    parser.add_argument("--daily-root", default="../Final-OXX2/A股数据/daily")
    parser.add_argument("--config", default="configs/live/live_trading_StdTF.yaml")
    parser.add_argument("--anchor-npz", default="data/live/features/live_sequence_20260603_l60.npz")
    parser.add_argument("--anchor-feature-date", default="20260603")
    parser.add_argument(
        "--trade-dates",
        nargs="+",
        default=["20260605", "20260608", "20260609", "20260610", "20260611", "20260612"],
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def yyyymmdd(value: object) -> str:
    return str(value).replace("-", "")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_daily_panel(daily_root: Path, max_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(daily_root.glob("*.csv")):
        date = path.stem
        if not date.isdigit() or date > max_date:
            continue
        frame = pd.read_csv(path)
        if "trade_date" not in frame.columns:
            frame["trade_date"] = date
        frame["trade_date"] = frame["trade_date"].map(yyyymmdd)
        frame["ts_code"] = frame["ts_code"].astype(str)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no daily CSVs found in {daily_root} up to {max_date}")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def add_raw_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for column in ["open", "high", "low", "close", "pre_close", "amount"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    grouped = panel.groupby("ts_code", group_keys=False)
    ret = panel["close"] / panel["pre_close"].replace(0.0, np.nan) - 1.0
    panel["_ret_1d"] = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_ret = panel.groupby("trade_date")["_ret_1d"].transform("mean")
    panel["_excess_ret_1d"] = panel["_ret_1d"] - market_ret

    span = (panel["high"] - panel["low"]).replace(0.0, np.nan)
    panel["lag1_close_position"] = ((panel["close"] - panel["low"]) / span).clip(0.0, 1.0).fillna(0.5)
    panel["lag1_ret_1d"] = panel["_ret_1d"]
    panel["lag1_excess_ret_1d"] = panel["_excess_ret_1d"]
    panel["lag1_industry_neutral_ret_1d"] = panel["_excess_ret_1d"]

    panel["lag1_ret_5d_mean"] = grouped["_ret_1d"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    panel["lag1_excess_ret_5d_mean"] = grouped["_excess_ret_1d"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    panel["lag1_excess_ret_10d_mean"] = grouped["_excess_ret_1d"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    panel["lag1_ret_20d"] = grouped["close"].transform(lambda s: s / s.shift(20) - 1.0)

    ma20 = grouped["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    ma60 = grouped["close"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    std20 = grouped["close"].transform(lambda s: s.rolling(20, min_periods=10).std())
    panel["lag1_bollinger_z_20d"] = (panel["close"] - ma20) / std20.replace(0.0, np.nan)
    panel["lag1_ma_ratio_20_60"] = ma20 / ma60.replace(0.0, np.nan)

    ema12 = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
    ema26 = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
    dif = ema12 - ema26
    dea = dif.groupby(panel["ts_code"]).transform(lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
    panel["lag1_macd_hist"] = dif - dea

    return panel


def previous_trading_day(config: dict, trade_date: str) -> str:
    days = [str(item) for item in config["competition"]["trading_days"]]
    index = days.index(trade_date)
    if index == 0:
        raise ValueError(f"{trade_date} has no previous competition trading day")
    return days[index - 1]


def update_sequence(
    anchor_x: np.ndarray,
    codes: list[str],
    feature_names: list[str],
    raw_features: pd.DataFrame,
    start_after: str,
    end_date: str,
) -> np.ndarray:
    x = anchor_x.copy()
    code_to_index = {code: idx for idx, code in enumerate(codes)}
    rows = raw_features[(raw_features["trade_date"] > start_after) & (raw_features["trade_date"] <= end_date)]
    dates = sorted(rows["trade_date"].unique())
    carry_forward = {"lag1_net_mf_strength_20d_mean", "lag1_net_mf_strength_60d_mean"}

    for date in dates:
        today = rows[rows["trade_date"].eq(date)].set_index("ts_code")
        new_rows = x[:, -1, :].copy()
        for code, idx in code_to_index.items():
            if code not in today.index:
                continue
            row = today.loc[code]
            values = []
            for feature_idx, feature in enumerate(feature_names):
                if feature in carry_forward:
                    values.append(float(x[idx, -1, feature_idx]))
                elif feature in row and pd.notna(row[feature]) and math.isfinite(float(row[feature])):
                    values.append(float(row[feature]))
                else:
                    values.append(float(x[idx, -1, feature_idx]))
            new_rows[idx, :] = np.asarray(values, dtype="float32")
        x = np.concatenate([x[:, 1:, :], new_rows[:, None, :]], axis=1)
    return x


def build_model(yxr_root: Path, config: dict, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(yxr_root))
    sys.path.insert(0, str(yxr_root / "scripts" / "live"))
    from common import build_frozen_model  # type: ignore

    old_cwd = Path.cwd()
    try:
        # common.resolve_path is rooted at Final-YXR because common.py lives there.
        return build_frozen_model(config, device)
    finally:
        if Path.cwd() != old_cwd:
            pass


@torch.no_grad()
def infer(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        outputs.append(model(batch).detach().cpu().view(-1).numpy())
    return np.concatenate(outputs, axis=0)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    yxr_root = Path(args.yxr_root).resolve()
    daily_root = Path(args.daily_root).resolve()
    config_path = yxr_root / args.config
    config = load_yaml(config_path)
    feature_names = [str(item) for item in config["model"]["expected_features"]]

    anchor = np.load(yxr_root / args.anchor_npz, allow_pickle=True)
    anchor_x = anchor["X"].astype("float32", copy=False)
    codes = anchor["ts_code"].astype(str).tolist()
    anchor_features = anchor["feature_names"].astype(str).tolist()
    if anchor_features != feature_names:
        raise ValueError(f"anchor feature order mismatch: {anchor_features} != {feature_names}")

    max_feature_date = max(previous_trading_day(config, date) for date in args.trade_dates)
    raw_panel = add_raw_features(load_daily_panel(daily_root, max_feature_date))

    device = torch.device(args.device)
    model = build_model(yxr_root, config, device)
    model.to(device).eval()

    seq_dir = yxr_root / "data" / "live" / "features"
    pred_dir = yxr_root / config["outputs"]["predictions_dir"]
    seq_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for trade_date in args.trade_dates:
        feature_date = previous_trading_day(config, trade_date)
        out_csv = pred_dir / f"predictions_{trade_date}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"SKIP {trade_date}: {out_csv} exists")
            continue

        x = update_sequence(anchor_x, codes, feature_names, raw_panel, args.anchor_feature_date, feature_date)
        if not np.isfinite(x).all():
            raise ValueError(f"sequence for trade_date={trade_date} contains NaN/Inf")

        seq_path = seq_dir / f"live_sequence_{feature_date}_l{anchor_x.shape[1]}.npz"
        np.savez_compressed(
            seq_path,
            X=x,
            ts_code=np.asarray(codes, dtype=object),
            feature_names=np.asarray(feature_names, dtype=object),
            trade_date=np.asarray([feature_date] * len(codes), dtype=object),
        )

        scores = infer(model, x, args.batch_size, device)
        predictions = pd.DataFrame(
            {
                "trade_date": trade_date,
                "ts_code": codes,
                "pred_score": scores,
                "split": "live",
                "model_name": config["model"]["name"],
            }
        ).sort_values("pred_score", ascending=False)
        predictions.to_csv(out_csv, index=False)
        write_json(
            pred_dir / f"manifest_{trade_date}.json",
            {
                "trade_date": trade_date,
                "feature_date": feature_date,
                "source": str(seq_path),
                "output_csv": str(out_csv),
                "rows": int(len(predictions)),
                "score_mean": float(predictions["pred_score"].mean()),
                "score_std": float(predictions["pred_score"].std(ddof=1)),
                "note": (
                    "Backfilled from anchor NPZ plus raw daily CSV features. "
                    "Money-flow features are carried forward because raw daily CSVs do not contain them."
                ),
            },
        )
        print(f"OK {trade_date}: feature_date={feature_date} rows={len(predictions)} -> {out_csv}")
        print(predictions.head(5)[["ts_code", "pred_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
