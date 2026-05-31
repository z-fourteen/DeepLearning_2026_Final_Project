from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    assert_market_coverage,
    build_frozen_model,
    die,
    dynamic_shortfall_penalty,
    ensure_columns,
    format_path,
    json_safe,
    load_yaml,
    normalize_code_column,
    normalize_date_column,
    resolve_device,
    resolve_path,
    today_yyyymmdd,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live stage 1: feature validation and frozen-model inference.")
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument("--features-parquet")
    parser.add_argument("--sequence-npz")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def load_live_npz(path: Path, config: dict, trade_date: str) -> tuple[np.ndarray, list[str]]:
    if not path.exists():
        die(f"live sequence npz not found: {path}")
    data = np.load(path, allow_pickle=True)
    for key in ["X", "ts_code", "feature_names"]:
        if key not in data.files:
            die(f"live sequence npz missing key={key}: {path}")
    feature_names = data["feature_names"].astype(str).tolist()
    expected = config["model"]["expected_features"]
    if feature_names != expected:
        die(f"live npz feature order mismatch; expected={expected}, actual={feature_names}")
    x = data["X"].astype("float32", copy=False)
    if x.ndim != 3:
        die(f"live npz X must be [N,T,F], got shape={x.shape}")
    if x.shape[1] != int(config["model"]["lookback"]) or x.shape[2] != len(expected):
        die(f"live npz shape mismatch: got={x.shape}, expected T={config['model']['lookback']}, F={len(expected)}")
    if not np.isfinite(x).all():
        die("live npz contains NaN/Inf; refuse to infer")
    codes = data["ts_code"].astype(str).tolist()
    frame = pd.DataFrame({"ts_code": codes, "trade_date": trade_date})
    assert_market_coverage(frame, config, "live npz")
    return x, codes


def build_sequences_from_panel(path: Path, config: dict, trade_date: str) -> tuple[np.ndarray, list[str]]:
    if not path.exists():
        die(f"live feature parquet not found: {path}")
    panel = pd.read_parquet(path)
    panel = normalize_code_column(normalize_date_column(panel))
    expected_features = config["model"]["expected_features"]
    ensure_columns(panel, ["trade_date", "ts_code", *expected_features], "live feature panel")

    today_rows = panel[panel["trade_date"].eq(trade_date)].copy()
    assert_market_coverage(today_rows, config, f"live feature panel trade_date={trade_date}")

    lookback = int(config["model"]["lookback"])
    sequences: list[np.ndarray] = []
    codes: list[str] = []
    panel = panel[panel["trade_date"].le(trade_date)].sort_values(["ts_code", "trade_date"])

    for code, group in panel.groupby("ts_code", sort=True):
        if group["trade_date"].iloc[-1] != trade_date:
            continue
        tail = group.tail(lookback)
        if len(tail) != lookback:
            continue
        values = tail[expected_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
        if not np.isfinite(values).all():
            continue
        sequences.append(values)
        codes.append(str(code))

    if not sequences:
        die("no valid live sequences after lookback/NaN filtering")
    result = np.stack(sequences, axis=0)
    coverage_frame = pd.DataFrame({"ts_code": codes})
    assert_market_coverage(coverage_frame, config, "valid live sequences")
    return result, codes


@torch.no_grad()
def run_inference(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    preds: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        pred = model(batch).detach().cpu().view(-1).numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0)


def main() -> None:
    args = parse_args()
    trade_date = str(args.trade_date)
    config = load_yaml(args.config)
    prev_trade_date = "NA"
    input_cfg = config["live_inputs"]

    npz_path = resolve_path(args.sequence_npz) if args.sequence_npz else format_path(input_cfg["sequence_npz"], trade_date=trade_date, prev_trade_date=prev_trade_date)
    parquet_path = resolve_path(args.features_parquet) if args.features_parquet else format_path(input_cfg["feature_panel"], trade_date=trade_date, prev_trade_date=prev_trade_date)

    # 优先使用 parquet，因为比赛数据通常以日更 parquet 到达；若 parquet 缺失，则尝试使用预构造 NPZ。
    if parquet_path.exists():
        x, codes = build_sequences_from_panel(parquet_path, config, trade_date)
        source = str(parquet_path)
    else:
        x, codes = load_live_npz(npz_path, config, trade_date)
        source = str(npz_path)

    device = resolve_device(args.device)
    model = build_frozen_model(config, device)
    scores = run_inference(model, x, int(args.batch_size), device)

    out_dir = resolve_path(config["outputs"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "trade_date": trade_date,
            "ts_code": codes,
            "pred_score": scores,
            "split": "live",
            "model_name": config["model"]["name"],
        }
    ).sort_values("pred_score", ascending=False)

    out_parquet = out_dir / f"predictions_{trade_date}.parquet"
    out_csv = out_dir / f"predictions_{trade_date}.csv"
    predictions.to_parquet(out_parquet, index=False)
    predictions.to_csv(out_csv, index=False)
    write_json(
        out_dir / f"manifest_{trade_date}.json",
        {
            "trade_date": trade_date,
            "source": source,
            "output_parquet": str(out_parquet),
            "output_csv": str(out_csv),
            "rows": int(len(predictions)),
            "score_mean": float(predictions["pred_score"].mean()),
            "score_std": float(predictions["pred_score"].std(ddof=1)),
            "shortfall_penalty_today": dynamic_shortfall_penalty(config, trade_date),
        },
    )

    print("\n【阶段一完成】冻结模型 live inference")
    print(f"trade_date={trade_date} rows={len(predictions)} output={out_parquet}")
    print("\nTop 20 预测分数：")
    print(predictions.head(20)[["ts_code", "pred_score"]].to_string(index=False))
    print("\nBottom 10 风险尾部：")
    print(predictions.tail(10)[["ts_code", "pred_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
